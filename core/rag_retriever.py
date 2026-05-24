"""
================================================================================
 RAG 检索器 —— 针对 Spider 2.0-Lite 的本地 Markdown 知识库
================================================================================
功能：
  1. 将 resource/documents/ 目录下的 .md 文件按 # / ## 标题智能切块
  2. 使用 BAAI/bge-small-en-v1.5 生成向量
  3. 构建 FAISS 索引并持久化到磁盘
  4. 提供 EKRetriever 类，加载索引并执行 top-k 语义检索

依赖：sentence-transformers, faiss-cpu, numpy（无 LangChain）
================================================================================
"""

import json
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 常量配置
# ──────────────────────────────────────────────────────────────────────────────

# 项目路径
_BASE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_DOC_DIR = _BASE_DIR / "resource" / "documents"
_DEFAULT_INDEX_DIR = _BASE_DIR / "data" / "ek_index"

# 模型名（首次运行自动下载到本地 cache）
_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# 切块控制
_MAX_CHUNK_CHARS = 800        # 超过此长度的 chunk 将被二次切分（按段落边界）
_MIN_CHUNK_CHARS = 50         # 短于此长度的 chunk 合并到上一个（避免零碎标题残留）

# FAISS 索引类型
_INDEX_FILE = "ek_index.faiss"
_CHUNKS_FILE = "ek_chunks.pkl"


# ──────────────────────────────────────────────────────────────────────────────
# 第一部分：Markdown 智能切块器
# ──────────────────────────────────────────────────────────────────────────────

def _split_markdown_by_headers(
    content: str, source_file: str
) -> List[Dict[str, object]]:
    """将 Markdown 文本按 # / ## / ### 标题层级切分为语义块。

    策略：
      1. 逐行扫描，遇到标题行（以 # 开头且非代码块内）则开始新 chunk。
      2. 维护一个标题栈（header stack），记录当前 chunk 的完整层级路径。
         - 遇到同级或上级标题时，弹出栈顶后再压入新标题。
         - 遇到子级标题时（如 ## 下面跟 ###），直接压入。
      3. 扫描结束后，对每个 chunk 检查长度：
         - 若超过 _MAX_CHUNK_CHARS，在段落边界（\\n\\n）处二次切分。
         - 若短于 _MIN_CHUNK_CHARS（且无实质内容），跳过。

    参数：
        content: Markdown 原始文本
        source_file: 来源文件名（仅用于元数据标记）
    返回：
        chunks: [{"text": str, "source": str, "header_path": str}, ...]
    """
    lines = content.split("\n")
    chunks: List[Dict[str, object]] = []
    header_stack: List[Tuple[int, str]] = []  # [(level, title), ...]
    current_lines: List[str] = []

    # 检测是否在代码块内（``` 包围），代码块内的 # 不算标题
    in_code_block = False

    def _flush_chunk() -> None:
        """将 current_lines 中的内容保存为一个 chunk。"""
        nonlocal current_lines
        if not current_lines:
            return

        text = "\n".join(current_lines).strip()
        # 去掉纯空白和纯分隔符的尾部残留
        if not text:
            current_lines = []
            return

        # 构建完整的标题路径，如 "google_analytics_sample.ga_sessions > Schema Fields > hits"
        header_path = " > ".join([title for _, title in header_stack])
        if header_path:
            header_path = f"{source_file} > {header_path}"
        else:
            header_path = source_file

        chunks.append({
            "text": text,
            "source": source_file,
            "header_path": header_path,
        })
        current_lines = []

    for line in lines:
        stripped = line.strip()

        # 追踪代码块边界
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        # 代码块内的行不做标题解析
        if in_code_block:
            current_lines.append(line)
            continue

        # 检测标题行：# / ## / ###（至多 3 级）
        header_match = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if header_match:
            # 保存上一个 chunk
            _flush_chunk()

            level = len(header_match.group(1))
            title = header_match.group(2).strip()

            # 弹出所有 >= 当前层级的标题（同级替换，上级回退）
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, title))

            # 标题行本身也保留在 chunk 文本中（帮助模型理解结构）
            current_lines = [line]
        else:
            current_lines.append(line)

    # 处理最后一段
    _flush_chunk()

    # ── 二次切分：将过长的 chunk 在段落边界处拆分 ──
    final_chunks: List[Dict[str, object]] = []
    for chunk in chunks:
        text = str(chunk["text"])
        if len(text) <= _MAX_CHUNK_CHARS:
            if len(text) >= _MIN_CHUNK_CHARS:
                final_chunks.append(chunk)
        else:
            # 按双换行（空行）拆成段落，再贪心拼接
            paragraphs = re.split(r"\n\s*\n", text)
            sub_lines: List[str] = []
            sub_len = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if sub_len + len(para) > _MAX_CHUNK_CHARS and sub_lines:
                    sub_text = "\n\n".join(sub_lines).strip()
                    if len(sub_text) >= _MIN_CHUNK_CHARS:
                        final_chunks.append({
                            "text": sub_text,
                            "source": chunk["source"],
                            "header_path": chunk["header_path"],
                        })
                    sub_lines = [para]
                    sub_len = len(para)
                else:
                    sub_lines.append(para)
                    sub_len += len(para)

            if sub_lines:
                sub_text = "\n\n".join(sub_lines).strip()
                if len(sub_text) >= _MIN_CHUNK_CHARS:
                    final_chunks.append({
                        "text": sub_text,
                        "source": chunk["source"],
                        "header_path": chunk["header_path"],
                    })

    return final_chunks


# ──────────────────────────────────────────────────────────────────────────────
# 第二部分：向量化 & FAISS 索引构建
# ──────────────────────────────────────────────────────────────────────────────

def build_ek_index(
    doc_dir: Optional[str] = None,
    index_save_dir: Optional[str] = None,
    model_name: str = _MODEL_NAME,
) -> Tuple[int, str]:
    """遍历 Markdown 目录，切块、向量化、构建 FAISS 索引，持久化到磁盘。

    参数：
        doc_dir:         .md 文件所在目录，默认 resource/documents/
        index_save_dir:  FAISS 索引 & chunk 元数据输出目录，默认 data/ek_index/
        model_name:      sentence-transformers 模型名
    返回：
        (chunk_count, index_save_dir) — 切块总数和索引保存路径
    """
    from sentence_transformers import SentenceTransformer

    doc_path = Path(doc_dir) if doc_dir else _DEFAULT_DOC_DIR
    save_path = Path(index_save_dir) if index_save_dir else _DEFAULT_INDEX_DIR

    if not doc_path.exists():
        raise FileNotFoundError(f"文档目录不存在: {doc_path}")

    # 收集所有 .md 文件
    md_files = sorted(doc_path.glob("*.md"))
    if not md_files:
        raise ValueError(f"目录中无 .md 文件: {doc_path}")

    print(f"[RAG] 发现 {len(md_files)} 个 Markdown 文件")

    # ── 步骤 1：逐文件切块 ──
    all_chunks: List[Dict[str, object]] = []
    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"[RAG] 跳过无法读取的文件 {md_file.name}: {exc}")
            continue

        file_chunks = _split_markdown_by_headers(content, md_file.name)
        all_chunks.extend(file_chunks)
        if file_chunks:
            print(f"  {md_file.name}: {len(file_chunks)} chunks")

    print(f"[RAG] 总计切出 {len(all_chunks)} 个语义块")

    if not all_chunks:
        raise RuntimeError("未产生任何有效 chunk，请检查文档内容")

    # ── 步骤 2：加载模型并生成向量 ──
    print(f"[RAG] 加载嵌入模型: {model_name}")
    model = SentenceTransformer(model_name)

    # BGE 模型推荐前缀：提升检索质量
    texts_to_embed = [str(c["text"]) for c in all_chunks]

    print(f"[RAG] 正在向量化 {len(texts_to_embed)} 个 chunk ...")
    embeddings = model.encode(
        texts_to_embed,
        normalize_embeddings=True,   # L2 归一化，使内积等价于余弦相似度
        show_progress_bar=True,
        batch_size=32,
    )

    dim = embeddings.shape[1]
    print(f"[RAG] 向量维度: {dim}")

    # ── 步骤 3：构建 FAISS 索引 ──
    import faiss

    # IndexFlatIP = 内积索引（配合归一化向量 = 余弦相似度）
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # ── 步骤 4：持久化 ──
    save_path.mkdir(parents=True, exist_ok=True)

    faiss_path = save_path / _INDEX_FILE
    faiss.write_index(index, str(faiss_path))
    print(f"[RAG] FAISS 索引已保存: {faiss_path}")

    # 元数据：chunk 列表 + 模型名
    metadata = {
        "model_name": model_name,
        "chunks": all_chunks,
        "dim": dim,
        "chunk_count": len(all_chunks),
        "source_dir": str(doc_path),
    }
    pkl_path = save_path / _CHUNKS_FILE
    with open(pkl_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"[RAG] Chunk 元数据已保存: {pkl_path}")

    return len(all_chunks), str(save_path)


# ──────────────────────────────────────────────────────────────────────────────
# 第三部分：检索器类
# ──────────────────────────────────────────────────────────────────────────────

class EKRetriever:
    """外部知识检索器。

    用法：
        retriever = EKRetriever("data/ek_index")
        results = retriever.search("How to calculate retention rate?", top_k=3)
        for r in results:
            print(r["header_path"])
            print(r["text"][:200])
    """

    def __init__(self, index_dir: Optional[str] = None):
        """加载已保存的 FAISS 索引和 chunk 元数据。

        参数：
            index_dir: 索引目录路径，默认 data/ek_index/
        """
        import faiss
        from sentence_transformers import SentenceTransformer

        index_path = Path(index_dir) if index_dir else _DEFAULT_INDEX_DIR

        # 加载 FAISS 索引
        faiss_file = index_path / _INDEX_FILE
        if not faiss_file.exists():
            raise FileNotFoundError(
                f"FAISS 索引文件不存在: {faiss_file}\n"
                f"请先运行 build_ek_index() 构建索引。"
            )
        self._index = faiss.read_index(str(faiss_file))

        # 加载元数据
        pkl_file = index_path / _CHUNKS_FILE
        if not pkl_file.exists():
            raise FileNotFoundError(f"Chunk 元数据文件不存在: {pkl_file}")
        with open(pkl_file, "rb") as f:
            metadata = pickle.load(f)

        self._chunks: List[Dict[str, object]] = metadata["chunks"]
        self._model_name: str = metadata.get("model_name", _MODEL_NAME)

        # 加载嵌入模型
        self._model = SentenceTransformer(self._model_name)

        print(f"[EKRetriever] 已加载索引: {len(self._chunks)} 个 chunk, "
              f"模型={self._model_name}")

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, object]]:
        """根据查询文本检索最相关的 top_k 个知识片段。

        参数：
            query: 自然语言查询（通常是用户问题或问题+数据集名）
            top_k: 返回片段数，默认 3
        返回：
            [{"text": str, "source": str, "header_path": str, "score": float}, ...]
            按相似度降序排列
        """
        if top_k > len(self._chunks):
            top_k = len(self._chunks)
        if top_k <= 0:
            return []

        # 向量化查询（BGE 模型对查询和文档使用相同编码方式）
        query_vec = self._model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)

        # FAISS 检索
        scores, indices = self._index.search(query_vec, top_k)

        results: List[Dict[str, object]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "header_path": chunk.get("header_path", chunk["source"]),
                "score": float(score),
            })

        return results

    def format_context(self, query: str, top_k: int = 3,
                       max_chars: int = 3000, min_score: float = 0.600) -> str:
        """检索并格式化为可直接插入 Prompt 的上下文字符串。

        参数：
            query:     查询文本
            top_k:     检索片段数
            max_chars: 返回文本的最大字符数（防止超过 token 限制）
            min_score: 最低相似度阈值，低于此值的结果将被过滤
        返回：
            格式化的上下文字符串，形如：
            === External Knowledge ===
            [来源: xxx.md > Header Path]
            ...chunk content...
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return ""

        # 过滤低于阈值的结果
        results = [r for r in results if r["score"] >= min_score]
        if not results:
            return ""

        return self._format_results(results, max_chars)

    def search_and_format(self, query: str, top_k: int = 3,
                          max_chars: int = 3000,
                          min_score: float = 0.600) -> Tuple[str, List[Dict[str, object]]]:
        """检索 + 格式化，一次调用完成，避免重复 encode。

        返回：
            (格式化上下文字符串, 原始检索结果列表)
            原始结果可用于日志记录，无需再次调用 search()
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return "", []

        filtered = [r for r in results if r["score"] >= min_score]
        context = self._format_results(filtered, max_chars) if filtered else ""
        return context, results

    def _format_results(self, results: List[Dict[str, object]],
                        max_chars: int = 3000) -> str:
        """将检索结果格式化为 Prompt 上下文字符串。"""
        lines = ["=== External Knowledge ==="]
        total = 0
        for r in results:
            header = f"[来源: {r['header_path']}] (相似度: {r['score']:.3f})"
            lines.append(header)
            lines.append(r["text"])
            lines.append("")
            total += len(header) + len(str(r["text"])) + 2
            if total >= max_chars:
                break

        return "\n".join(lines).strip()


# ──────────────────────────────────────────────────────────────────────────────
# 命令行入口（用于独立构建索引）
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="构建 Spider 2.0-Lite 外部知识 FAISS 索引"
    )
    parser.add_argument(
        "--doc-dir", type=str, default=None,
        help="Markdown 文档目录（默认 resource/documents/）",
    )
    parser.add_argument(
        "--index-dir", type=str, default=None,
        help="索引输出目录（默认 data/ek_index/）",
    )
    args = parser.parse_args()

    count, saved = build_ek_index(
        doc_dir=args.doc_dir,
        index_save_dir=args.index_dir,
    )
    print(f"\n[RAG] 索引构建完成: {count} 个 chunk → {saved}")
