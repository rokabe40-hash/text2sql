"""
================================================================================
 Spider 2.0-Lite Text-to-SQL 推理引擎 V3 Sample — 分层抽样版
================================================================================
与 V3 的区别:
  - --skip-snowflake: 跳过全部 Snowflake 任务（节省 38% 时间/tokens）
  - --sample-bq N:    随机抽样 N 道 BigQuery 任务
  - --sample-sqlite N: 随机抽样 N 道 SQLite 任务
  - --seed N:         随机种子（默认 42）
  - --no-rag:         禁用 RAG，回退到 V2 行为（A/B 对比）
  - --no-catalog:     禁用 Schema Catalog

双通道 Prompt 结构（以 BigQuery 为例）:
  {instruction}
  === Examples ===
  ...
  === Schema Catalog (table structures) ===
  ...
  === External Knowledge (RAG) ===
  [来源: retention_rate.md > ...] (相似度: 0.809)
  ...
  === Question ===
  {question}
================================================================================
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Torch + faiss OMP 冲突修复（必须在导入 torch 前设置）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from openai import OpenAI

# Schema Catalog 支持
try:
    from schema_catalog import load_catalog, get_schema as _get_catalog_schema
    _HAS_CATALOG = True
except ImportError:
    _HAS_CATALOG = False

# RAG 支持
try:
    from core.rag_retriever import EKRetriever
    _HAS_RAG = True
except ImportError:
    _HAS_RAG = False


# API 配置 - DeepSeek V4 flash
_DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not _DEEPSEEK_API_KEY:
    print("[WARN] 环境变量 DEEPSEEK_API_KEY 未设置，API 调用将失败")

API_CONFIG = {
    "base_url": "https://api.deepseek.com",
    "api_key": _DEEPSEEK_API_KEY,
    "model": "deepseek-v4-flash",
    "temperature": 0.0,
    "max_tokens": 8192,
    "timeout": 60,
    "max_retries": 2,
}

# 路径配置
BASE_DIR = Path(__file__).resolve().parent
TASKS_FILE = BASE_DIR / "spider2-lite.jsonl"
SQLITE_DB_DIR = BASE_DIR / "resource" / "databases" / "spider2-localdb"
LOCAL_MAP_FILE = SQLITE_DB_DIR / "local-map.jsonl"
OUTPUT_DIR = BASE_DIR / "my_predicted_sqls"
FEW_SHOT_FILE = BASE_DIR / "few_shot_examples.json"
CATALOG_FILE = BASE_DIR / "schema_catalog.json"
EK_INDEX_DIR = BASE_DIR / "data" / "ek_index"


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

_LOG_FILE = None


def _get_log_file():
    global _LOG_FILE
    if _LOG_FILE is None:
        _LOG_FILE = open(BASE_DIR / "inference_log_v3_sample.txt", "w", encoding="utf-8")
    return _LOG_FILE


def log(level: str, message: str) -> None:
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    line = f"[{timestamp}] [{level}] {message}"
    print(line, flush=True)
    try:
        _get_log_file().write(line + "\n")
        _get_log_file().flush()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# 平台识别
# ──────────────────────────────────────────────────────────────────────────────

def classify_platform(instance_id: str) -> str:
    if instance_id.startswith("sf_bq") or instance_id.startswith("sf"):
        return "snowflake"
    if instance_id.startswith("bq"):
        return "bigquery"
    if instance_id.startswith("local"):
        return "sqlite"
    if instance_id.startswith("ga"):
        return "bigquery"
    return "bigquery"


# ──────────────────────────────────────────────────────────────────────────────
# 第一阶段：任务加载
# ──────────────────────────────────────────────────────────────────────────────

def load_tasks(tasks_file: Path) -> List[dict]:
    tasks: List[dict] = []
    with open(tasks_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
                tasks.append(task)
            except json.JSONDecodeError as exc:
                log("WARN", f"第 {line_num} 行 JSON 解析失败，已跳过: {exc}")
    return tasks


# ──────────────────────────────────────────────────────────────────────────────
# 第二阶段：数据库映射与 Schema 提取
# ──────────────────────────────────────────────────────────────────────────────

def build_db_mapping() -> Tuple[Dict[str, str], Dict[str, str]]:
    db_to_path: Dict[str, str] = {}
    instance_to_db: Dict[str, str] = {}

    for sqlite_file in SQLITE_DB_DIR.glob("*.sqlite"):
        db_name = sqlite_file.stem
        db_to_path[db_name] = str(sqlite_file)

    if LOCAL_MAP_FILE.exists():
        with open(LOCAL_MAP_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    mapping = json.loads(line)
                    instance_to_db.update(mapping)
                except json.JSONDecodeError:
                    pass

    return db_to_path, instance_to_db


def resolve_database_path(task: dict, db_to_path: Dict[str, str],
                          instance_to_db: Dict[str, str]) -> Optional[str]:
    db_name = task.get("db", "")
    instance_id = task.get("instance_id", "")

    if db_name in db_to_path:
        return db_to_path[db_name]

    mapped_db = instance_to_db.get(instance_id)
    if mapped_db and mapped_db in db_to_path:
        return db_to_path[mapped_db]

    return None


def extract_schema(db_path: str) -> str:
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND sql IS NOT NULL "
            "ORDER BY name;"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError(f"数据库 {db_path} 中未找到任何表定义")

    create_statements = []
    for (sql_text,) in rows:
        sql_text = sql_text.strip()
        if not sql_text.endswith(";"):
            sql_text += ";"
        create_statements.append(sql_text)

    return "\n\n".join(create_statements)


# ──────────────────────────────────────────────────────────────────────────────
# 第三阶段：Prompt 构建（三段式 Few-shot + Schema + RAG）
# ──────────────────────────────────────────────────────────────────────────────

def _load_few_shot_examples() -> Dict[str, List[dict]]:
    if FEW_SHOT_FILE.exists():
        try:
            with open(FEW_SHOT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _format_few_shot_section(examples: List[dict]) -> str:
    if not examples:
        return ""
    lines = ["=== Examples ==="]
    for i, ex in enumerate(examples, start=1):
        lines.append(f"Example {i}:")
        lines.append(f"Question: {ex['question']}")
        lines.append(f"SQL: {ex['sql']}")
        lines.append("")
    return "\n".join(lines)


def build_prompt_sqlite(schema: str, question: str,
                        ek_context: str = "",
                        few_shot_examples: Optional[List[dict]] = None) -> str:
    instruction = (
        "You are a SQLite expert. "
        "Please write SQL for the following question based on the provided database schema. "
        "Only output the SQL statement, without any explanation."
    )
    fs_section = _format_few_shot_section(few_shot_examples or [])
    ek_section = f"\n{ek_context}\n" if ek_context else ""

    return f"""{instruction}

{fs_section}
=== Database Schema ===
{schema}
{ek_section}
=== Question ===
{question}

=== SQL ===
"""


def build_prompt_bigquery(db_name: str, question: str,
                          external_knowledge: Optional[str] = None,
                          ek_context: str = "",
                          few_shot_examples: Optional[List[dict]] = None) -> str:
    instruction = (
        "You are a Google BigQuery SQL expert. "
        "Write a standard BigQuery SQL query for the question below. "
        "The dataset is `{db_name}` hosted on BigQuery public datasets. "
        "Use fully-qualified table paths from the Schema Catalog. "
        "CRITICAL BIGQUERY RULES: "
        "1. You MUST explicitly list the specific column names in the SELECT clause. "
        "Generating 'SELECT *' is strictly prohibited and will cause execution failure. "
        "2. For partitioned or clustered tables, your WHERE clause MUST include explicit "
        "filters on the partition keys to minimize scanned data. Never execute full-table "
        "scans without date/time bounds if applicable. "
        "Only output the SQL statement, without any explanation."
    ).format(db_name=db_name)

    parts = []
    if external_knowledge:
        parts.append(f"=== Schema Catalog (table structures) ===\n{external_knowledge}")
    if ek_context:
        parts.append(ek_context)
    ek_section = "\n\n".join(parts) if parts else ""
    if ek_section:
        ek_section = f"\n{ek_section}\n"

    fs_section = _format_few_shot_section(few_shot_examples or [])

    return f"""{instruction}

{fs_section}
{ek_section}
=== Question ===
{question}

=== BigQuery SQL ===
"""


def build_prompt_snowflake(db_name: str, question: str,
                           external_knowledge: Optional[str] = None,
                           ek_context: str = "",
                           few_shot_examples: Optional[List[dict]] = None) -> str:
    instruction = (
        "You are a Snowflake SQL expert. "
        "Write a standard Snowflake SQL query for the question below. "
        "The dataset is `{db_name}` hosted on Snowflake. "
        "Use Snowflake-specific syntax: LATERAL FLATTEN for nested data, TABLE(GENERATOR(...)) for series, "
        "TO_CHAR/TO_DATE for formatting, QUALIFY for window filtering. "
        "Only output the SQL statement, without any explanation."
    ).format(db_name=db_name)

    parts = []
    if external_knowledge:
        parts.append(f"=== Schema Catalog (table structures) ===\n{external_knowledge}")
    if ek_context:
        parts.append(ek_context)
    ek_section = "\n\n".join(parts) if parts else ""
    if ek_section:
        ek_section = f"\n{ek_section}\n"

    fs_section = _format_few_shot_section(few_shot_examples or [])

    return f"""{instruction}

{fs_section}
{ek_section}
=== Question ===
{question}

=== Snowflake SQL ===
"""


# ──────────────────────────────────────────────────────────────────────────────
# 第四阶段：LLM API 调用
# ──────────────────────────────────────────────────────────────────────────────

def call_llm(prompt: str, config: dict) -> str:
    client = OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
        timeout=config.get("timeout", 60),
    )

    last_error: Optional[Exception] = None
    max_retries = config.get("max_retries", 2)

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=config.get("temperature", 0.0),
                max_tokens=config.get("max_tokens", 4096),
            )
            return response.choices[0].message.content
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                wait_seconds = 2 ** attempt
                log("WARN", f"API 调用失败（第 {attempt + 1}/{max_retries + 1} 次），"
                       f"{wait_seconds} 秒后重试: {exc}")
                time.sleep(wait_seconds)

    raise RuntimeError(f"API 调用在 {max_retries + 1} 次尝试后仍然失败") from last_error


# ──────────────────────────────────────────────────────────────────────────────
# 第五阶段：结果保存与后处理
# ──────────────────────────────────────────────────────────────────────────────

def clean_sql_output(raw_text: str) -> str:
    text = raw_text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return text


def save_sql(instance_id: str, sql: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{instance_id}.sql"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(sql)
        if not sql.endswith("\n"):
            f.write("\n")
    return file_path


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def load_external_knowledge(ek_filename: Optional[str]) -> Optional[str]:
    if not ek_filename:
        return None
    for candidate in [
        BASE_DIR / "resource" / "external_knowledge" / ek_filename,
        BASE_DIR / "evaluation_suite" / "external_knowledge" / ek_filename,
    ]:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return None


def main(limit: Optional[int] = None, use_catalog: bool = True,
         use_rag: bool = True, skip_snowflake: bool = False,
         sample_bq: Optional[int] = None, sample_sqlite: Optional[int] = None,
         seed: int = 42) -> None:
    """主入口"""

    log("INFO", "=" * 60)
    catalog_str = "ON" if use_catalog else "OFF"
    rag_str = "ON" if use_rag else "OFF"
    snow_str = "SKIP" if skip_snowflake else "ALL"
    sample_info = ""
    if sample_bq or sample_sqlite:
        parts = []
        if sample_bq: parts.append(f"BQ={sample_bq}")
        if sample_sqlite: parts.append(f"SQLite={sample_sqlite}")
        sample_info = f", Sample=[{', '.join(parts)}], Seed={seed}"
    log("INFO", f"Spider 2.0-Lite 推理引擎 V3 Sample 启动 "
           f"(Catalog={catalog_str}, RAG={rag_str}, Snow={snow_str}{sample_info})")
    log("INFO", "=" * 60)

    if not TASKS_FILE.exists():
        log("ERROR", f"任务文件不存在: {TASKS_FILE}")
        sys.exit(1)

    # ---------- 0a. 加载 Few-shot 示例 ----------
    few_shot = _load_few_shot_examples()
    for plat in ["sqlite", "bigquery", "snowflake"]:
        count = len(few_shot.get(plat, []))
        log("INFO", f"Few-shot 示例 [{plat}]: {count} 条")
    if not few_shot:
        log("WARN", "未找到 few_shot_examples.json，将使用 Zero-shot 模式")

    # ---------- 0b. 加载 Schema Catalog ----------
    catalog = None
    catalog_hits = {"bigquery": 0, "snowflake": 0}
    if use_catalog and _HAS_CATALOG and CATALOG_FILE.exists():
        catalog = load_catalog()
        bq_dbs = len(catalog.get("mapping", {}).get("bigquery", {}))
        sf_dbs = len(catalog.get("mapping", {}).get("snowflake", {}))
        bq_schemas = len(catalog.get("schemas", {}).get("bigquery", {}))
        sf_schemas = len(catalog.get("schemas", {}).get("snowflake", {}))
        log("INFO", f"Schema Catalog: BigQuery={bq_dbs}dbs/{bq_schemas}schemas, "
               f"Snowflake={sf_dbs}dbs/{sf_schemas}schemas")
    else:
        if use_catalog and not _HAS_CATALOG:
            log("WARN", "schema_catalog.py 未找到，回退到 V1 模式")
        elif use_catalog and not CATALOG_FILE.exists():
            log("WARN", "schema_catalog.json 不存在")

    # ---------- 0c. 加载 RAG 检索器 ----------
    retriever = None
    rag_hits = 0
    if use_rag and _HAS_RAG:
        rag_index = EK_INDEX_DIR / "ek_index.faiss"
        if rag_index.exists():
            for retry in range(3):
                try:
                    retriever = EKRetriever(str(EK_INDEX_DIR))
                    log("INFO", f"RAG Retriever: 已加载 {retriever.chunk_count} 个知识片段")
                    break
                except Exception as exc:
                    if retry < 2:
                        log("WARN", f"RAG Retriever 加载失败（第{retry+1}/3次），5秒后重试: {exc}")
                        time.sleep(5)
                    else:
                        log("WARN", f"RAG Retriever 加载失败（已重试3次），本次运行无 RAG: {exc}")
        else:
            log("WARN", f"RAG 索引文件不存在: {rag_index}")
    elif use_rag and not _HAS_RAG:
        log("FATAL", "core.rag_retriever 未找到，RAG 不可用，拒绝运行以节省 tokens")
        sys.exit(1)

    if use_rag and retriever is None:
        log("FATAL", "RAG 已启用但加载失败，拒绝运行以节省 tokens。"
                       "请检查网络连接和索引文件后重试，或使用 --no-rag 显式跳过。")
        sys.exit(1)

    # ---------- 1. 加载任务 ----------
    log("INFO", f"正在加载任务文件: {TASKS_FILE}")
    tasks = load_tasks(TASKS_FILE)
    log("INFO", f"成功加载 {len(tasks)} 个任务")

    if limit and limit > 0:
        tasks = tasks[:limit]
        log("INFO", f"--limit 模式：仅处理前 {len(tasks)} 个任务")

    # ---------- 1b. 平台过滤与分层抽样 ----------
    if skip_snowflake:
        before = len(tasks)
        tasks = [t for t in tasks if classify_platform(t.get("instance_id", "")) != "snowflake"]
        log("INFO", f"跳过 Snowflake: {before} -> {len(tasks)} 个任务")

    if sample_bq or sample_sqlite:
        random.seed(seed)
        bq_tasks = [t for t in tasks if classify_platform(t.get("instance_id", "")) == "bigquery"]
        sqlite_tasks = [t for t in tasks if classify_platform(t.get("instance_id", "")) == "sqlite"]
        sf_tasks = [t for t in tasks if classify_platform(t.get("instance_id", "")) == "snowflake"]

        if sample_bq and len(bq_tasks) > sample_bq:
            bq_tasks = random.sample(bq_tasks, sample_bq)
            log("INFO", f"BigQuery 抽样: {len(bq_tasks)} / {sample_bq}")
        if sample_sqlite and len(sqlite_tasks) > sample_sqlite:
            sqlite_tasks = random.sample(sqlite_tasks, sample_sqlite)
            log("INFO", f"SQLite 抽样: {len(sqlite_tasks)} / {sample_sqlite}")

        tasks = bq_tasks + sqlite_tasks + sf_tasks
        log("INFO", f"抽样后总计: {len(tasks)} 个任务 (BQ={len(bq_tasks)}, SQLite={len(sqlite_tasks)}, SF={len(sf_tasks)})")

    # ---------- 2. 构建本地数据库映射 ----------
    db_to_path, instance_to_db = build_db_mapping()
    log("INFO", f"发现 {len(db_to_path)} 个本地 SQLite 数据库文件")
    if instance_to_db:
        log("INFO", f"加载 local-map.jsonl，包含 {len(instance_to_db)} 条映射记录")

    # ---------- 3. 准备输出目录 ----------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log("INFO", f"SQL 输出目录: {OUTPUT_DIR}")

    # ---------- 4. 逐任务处理 ----------
    stats = {
        "total": len(tasks),
        "success": 0,
        "skipped": 0,
        "failed": 0,
        "by_platform": {"sqlite": 0, "bigquery": 0, "snowflake": 0, "skipped": 0, "failed": 0},
        "errors": [],
    }

    for idx, task in enumerate(tasks, start=1):
        instance_id = task.get("instance_id", f"unknown_{idx}")
        question = task.get("question", "")
        db_name = task.get("db", "?")
        ek_filename = task.get("external_knowledge")
        platform = classify_platform(instance_id)

        log("INFO", f"[{idx}/{len(tasks)}] 处理任务: {instance_id} "
               f"(db={db_name}, platform={platform})")

        # ── RAG 检索（三种平台共用）──
        ek_rag_context = ""
        if retriever and question:
            # 用 question + db_name 作为检索查询（增强语义聚焦）
            query_text = f"{db_name}: {question}"
            try:
                ek_rag_context = retriever.format_context(query_text, top_k=3, max_chars=2500)
                if ek_rag_context:
                    rag_hits += 1
                    # 提取 top-1 score 用于日志
                    top1 = retriever.search(query_text, top_k=1)
                    if top1:
                        log("INFO", f"  ->RAG 命中: {top1[0]['header_path']} "
                               f"(score={top1[0]['score']:.3f})")
            except Exception as exc:
                log("WARN", f"  ->RAG 检索异常: {exc}")

        # ================================================================
        # 路径 A：SQLite 本地任务
        # ================================================================
        if platform == "sqlite":
            db_path = resolve_database_path(task, db_to_path, instance_to_db)
            if db_path is None:
                log("SKIP", f"  ->本地无对应数据库，跳过")
                stats["skipped"] += 1
                stats["by_platform"]["skipped"] += 1
                continue

            try:
                schema = extract_schema(db_path)
                table_count = schema.count("CREATE TABLE")
                log("INFO", f"  ->成功提取 Schema，共 {table_count} 张表 "
                       f"({Path(db_path).name})")
            except Exception as exc:
                log("ERROR", f"  ->Schema 提取失败: {exc}")
                stats["failed"] += 1
                stats["by_platform"]["failed"] += 1
                stats["errors"].append({
                    "instance_id": instance_id, "platform": platform,
                    "stage": "schema_extraction", "error": str(exc),
                })
                continue

            prompt = build_prompt_sqlite(schema, question, ek_rag_context,
                                         few_shot.get("sqlite"))

        # ================================================================
        # 路径 B：BigQuery 云端任务
        # ================================================================
        elif platform == "bigquery":
            ek_content = load_external_knowledge(ek_filename)
            if ek_content:
                log("INFO", f"  ->已加载外部知识文件: {ek_filename}")

            catalog_schema = None
            if catalog:
                catalog_schema = _get_catalog_schema(db_name, "bigquery")
            if catalog_schema:
                log("INFO", f"  ->Schema Catalog 命中: {db_name}")
                catalog_hits["bigquery"] += 1
                if ek_content:
                    ek_content = catalog_schema + "\n\n" + ek_content
                else:
                    ek_content = catalog_schema
            elif not ek_content:
                log("INFO", f"  ->Schema Catalog 未命中: {db_name}，依赖 LLM 先验知识")

            prompt = build_prompt_bigquery(db_name, question, ek_content,
                                           ek_rag_context, few_shot.get("bigquery"))

        # ================================================================
        # 路径 C：Snowflake 云端任务
        # ================================================================
        elif platform == "snowflake":
            ek_content = load_external_knowledge(ek_filename)
            if ek_content:
                log("INFO", f"  ->已加载外部知识文件: {ek_filename}")

            catalog_schema = None
            if catalog:
                catalog_schema = _get_catalog_schema(db_name, "snowflake")
            if catalog_schema:
                log("INFO", f"  ->Schema Catalog 命中: {db_name}")
                catalog_hits["snowflake"] += 1
                if ek_content:
                    ek_content = catalog_schema + "\n\n" + ek_content
                else:
                    ek_content = catalog_schema
            elif not ek_content:
                log("INFO", f"  ->Schema Catalog 未命中: {db_name}，依赖 LLM 先验知识")

            prompt = build_prompt_snowflake(db_name, question, ek_content,
                                            ek_rag_context, few_shot.get("snowflake"))

        else:
            log("SKIP", f"  ->未知平台 {platform}，跳过")
            stats["skipped"] += 1
            stats["by_platform"]["skipped"] += 1
            continue

        # ================================================================
        # 调用 LLM API
        # ================================================================
        try:
            raw_result = call_llm(prompt, API_CONFIG)
        except Exception as exc:
            log("ERROR", f"  ->LLM 调用失败: {exc}")
            stats["failed"] += 1
            stats["by_platform"]["failed"] += 1
            stats["errors"].append({
                "instance_id": instance_id, "platform": platform,
                "stage": "llm_call", "error": str(exc),
            })
            continue

        # ================================================================
        # 清洗并保存 SQL
        # ================================================================
        sql = clean_sql_output(raw_result)
        try:
            saved_path = save_sql(instance_id, sql, OUTPUT_DIR)
            log("OK", f"  ->SQL 已保存到 {saved_path.name}  [{platform}]")
            stats["success"] += 1
            stats["by_platform"][platform] += 1
        except Exception as exc:
            log("ERROR", f"  ->文件保存失败: {exc}")
            stats["failed"] += 1
            stats["by_platform"]["failed"] += 1
            stats["errors"].append({
                "instance_id": instance_id, "platform": platform,
                "stage": "save_file", "error": str(exc),
            })

    # ---------- 5. 输出最终统计 ----------
    log("INFO", "=" * 60)
    log("INFO", "推理完成 - 统计摘要")
    log("INFO", f"  总任务数:     {stats['total']}")
    log("INFO", f"  成功生成:     {stats['success']}")
    log("INFO", f"  跳过:         {stats['skipped']}")
    log("INFO", f"  失败:         {stats['failed']}")
    bp = stats["by_platform"]
    log("INFO", f"  按平台分布:   sqlite={bp['sqlite']}, bigquery={bp['bigquery']}, "
           f"snowflake={bp['snowflake']}, skipped={bp['skipped']}, failed={bp['failed']}")
    if use_catalog and catalog_hits["bigquery"] + catalog_hits["snowflake"] > 0:
        log("INFO", f"  Catalog 命中:  bigquery={catalog_hits['bigquery']}, "
               f"snowflake={catalog_hits['snowflake']}")
    rag_status = f"{rag_hits}/{stats['total']}" if retriever else "未启用"
    log("INFO", f"  RAG 状态:      {rag_status}")

    if stats["errors"]:
        log("WARN", f"失败详情（前 10 条）:")
        for err in stats["errors"][:10]:
            log("WARN", f"  - [{err['instance_id']}] [{err.get('platform', '?')}] "
                   f"[{err['stage']}] {err['error']}")

    stats_file = OUTPUT_DIR / "_inference_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    log("INFO", f"完整统计已保存到 {stats_file}")

    log("INFO", "全量推理引擎运行结束。")

    global _LOG_FILE
    if _LOG_FILE is not None:
        log("INFO", f"日志已保存到 {BASE_DIR / 'inference_log_v3_sample.txt'}")
        _LOG_FILE.close()


# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Spider 2.0-Lite 推理引擎 V3 Sample")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制处理的任务数（测试用）")
    parser.add_argument("--no-catalog", action="store_true",
                        help="禁用 Schema Catalog")
    parser.add_argument("--no-rag", action="store_true",
                        help="禁用 RAG（回退到纯 V2 行为）")
    parser.add_argument("--skip-snowflake", action="store_true",
                        help="跳过全部 Snowflake 任务")
    parser.add_argument("--sample-bq", type=int, default=None,
                        help="随机抽样 BigQuery 任务数")
    parser.add_argument("--sample-sqlite", type=int, default=None,
                        help="随机抽样 SQLite 任务数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认 42）")
    args = parser.parse_args()
    main(limit=args.limit, use_catalog=not args.no_catalog, use_rag=not args.no_rag,
         skip_snowflake=args.skip_snowflake,
         sample_bq=args.sample_bq, sample_sqlite=args.sample_sqlite,
         seed=args.seed)
