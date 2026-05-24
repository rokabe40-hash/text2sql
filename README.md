# Spider 2.0-Lite Text-to-SQL 评测项目

## 项目概述

Spider 2.0-Lite 是一个 Text-to-SQL 评测套件，包含 547 个任务，覆盖三个数据库平台：

- **SQLite** (localXXX) → 本地 .sqlite 文件（135 个任务）
- **BigQuery** (bqXXX, gaXXX) → Google Cloud BigQuery 公开数据集（205 个任务）
- **Snowflake** (sf_bqXXX, sfXXX) → Snowflake 云端数据库（207 个任务）

## 项目架构

```
spider2-lite/
├── run_inference_full_v5.py      # 最新推理引擎 V5（RAG 优化版）
├── schema_catalog.py             # Schema Catalog 构建 + 运行时查询模块
├── spider2-lite.jsonl            # 547 个任务数据
├── few_shot_examples.json        # Few-shot 示例（每平台 2 条）
├── core/
│   └── rag_retriever.py          # RAG 检索器（BGE + FAISS）
├── data/
│   └── ek_index/                 # RAG 向量索引
├── resource/
│   └── documents/                # 外部知识文档（69 个）
└── evaluation_suite/             # 评估套件（需从 Spider2 项目获取，见下方说明）
```

> **注意**: `evaluation_suite/` 目录（评估脚本、Gold SQL、示例提交等）来源于
> [xlang-ai/Spider2](https://github.com/xlang-ai/Spider2) 项目，请从原仓库获取。
> SQLite 数据库文件也需单独下载，详见 [Spider2 README](https://github.com/xlang-ai/Spider2)。

## 核心功能

### 1. 推理引擎

多版本演进：
- **V1**: Few-shot prompting（基线）
- **V2**: Few-shot + Schema Catalog
- **V3**: + RAG 检索增强
- **V4**: + SQLite 深度优化
- **V5**: + RAG 召回率优化（当前版本）

### 2. Schema Catalog

从 BigQuery/Snowflake 的 INFORMATION_SCHEMA 提取表结构信息，注入推理 Prompt。

### 3. RAG 检索增强

- 69 个外部知识文档 → 564 个语义块
- 使用 BAAI/bge-small-en-v1.5 模型 + FAISS 索引
- 自动检索最相关的知识片段注入 Prompt

### 4. BigQuery 配额防线

三道防线防止配额耗尽：
- P1: dry-run 预检 + 10GB 扫描量上限
- P0: Prompt 行为矫正（禁止 SELECT *）
- P2: Schema 白名单化（分区键标注）

## 评估结果

| 版本 | 总分 | SQLite | BigQuery | Snowflake |
|------|------|--------|----------|-----------|
| V1 (Few-shot) | 13.3% | 43.5% | 0% | 0% |
| V2 (+ Schema Catalog) | 20.5% | 40.7% | 27.8% | 0% |
| V3 Sample (+ RAG) | 36.2% | 33.3% | 37.0% | - |
| V4 SQLite 优化 | 待评估 | 目标 80% | - | - |
| V5 RAG 优化 | 进行中 | 进行中 | 待配额恢复 | 待权限解决 |

## 运行方法

### 环境要求

- Python 3.8+
- 依赖包：openai, google-cloud-bigquery, snowflake-connector-python
- 可选：sentence-transformers, faiss-cpu（RAG 功能）

### 获取评估套件

```bash
# 克隆 Spider2 仓库并复制 evaluation_suite
git clone https://github.com/xlang-ai/Spider2.git /tmp/spider2
cp -r /tmp/spider2/evaluation_suite ./evaluation_suite
```

### 配置凭证

1. 复制凭证模板：
   ```bash
   cp evaluation_suite/credentials.example.json evaluation_suite/credentials.json
   ```

2. 填入真实凭证：
   - BigQuery: GCP 服务账号 JSON 密钥路径
   - Snowflake: account, user, password 等

3. 设置环境变量：
   ```bash
   # Windows PowerShell
   [System.Environment]::SetEnvironmentVariable('DEEPSEEK_API_KEY', 'your-api-key', 'User')
   ```

### 运行推理

```bash
# SQLite 快速测试（30 题）
python run_inference_full_v5.py --sample-sqlite 30

# 全量 SQLite（135 题）
python run_inference_full_v5.py --sample-sqlite 135

# A/B 对照（无 RAG）
python run_inference_full_v5.py --sample-sqlite 135 --no-rag
```

### 运行评估

```bash
cd evaluation_suite
python evaluate.py --result_dir ../my_predicted_sqls_v5 --mode sql
```

### 构建 Schema Catalog

```bash
# 构建（需要 BigQuery 连接）
python schema_catalog.py --build

# 预览
python schema_catalog.py --list
```

## 平台路由

| instance_id 前缀 | 平台 |
|------------------|------|
| localXXX | SQLite |
| bqXXX | BigQuery |
| gaXXX | BigQuery |
| sf_bqXXX | Snowflake |
| sfXXX | Snowflake |

## 技术决策

### 为什么自建 RAG 切块器？

- 拒绝 LangChain：减少依赖链，精确控制边界行为
- 本地 BGE 模型：零 API 费用，数据不出域
- FAISS 内积索引：归一化后等价余弦相似度

### 为什么限制 BigQuery 扫描量？

- 1TB Sandbox 配额有限
- LLM 生成的全表扫描 SQL 会快速耗尽配额
- 三道防线：dry-run 预检 → 行为矫正 → Schema 白名单

## 已知问题

1. **Snowflake**: 数据不存在于当前账户，等待项目方授权
2. **BigQuery**: 1TB Sandbox 配额耗尽，等待每月 1 日重置
3. **RAG**: 部分任务检索召回率低，正在优化

## 致谢与许可

`evaluation_suite/` 目录（评估脚本、Gold SQL、示例提交等）来源于
[xlang-ai/Spider2](https://github.com/xlang-ai/Spider2) 项目，遵循其
[MIT 许可证](https://github.com/xlang-ai/Spider2/blob/main/LICENSE)。
本项目未包含该目录，请从原仓库获取。

如您觉得 Spider2 对您有帮助，请引用：

```bibtex
@misc{lei2024spider2,
      title={Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows},
      author={Fangyu Lei and Jixuan Chen and Yuxiao Ye and Ruisheng Cao and Dongchan Shin and Hongjin Su and Zhaoqing Suo and Hongcheng Gao and Wenjing Hu and Pengcheng Yin and Victor Zhong and Caiming Xiong and Ruoxi Sun and Qian Liu and Sida Wang and Tao Yu},
      year={2024},
      eprint={2411.07763},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.07763},
}
```

## 许可证

本项目代码采用 MIT 许可证。
`evaluation_suite/` 来源于 [xlang-ai/Spider2](https://github.com/xlang-ai/Spider2)，遵循其 MIT 许可证。
