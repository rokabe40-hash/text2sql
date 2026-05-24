""""
================================================================================
 Spider 2.0-Lite Text-to-SQL 全量推理引擎 V2 — Schema Catalog 增强版
================================================================================
与 V1 的区别:
  - BigQuery 任务: 从 schema_catalog.json 加载真实表 schema，注入 Prompt
  - Snowflake 任务: 从 schema_catalog.json 加载表名映射，注入 Prompt
  - SQLite 任务: 保持不变（动态提取 CREATE TABLE）
  - 新增 --no-catalog 参数: 禁用 catalog，回退到 V1 行为（对比测试用）

功能：
  1. 读取 spider2-lite.jsonl 中的全部 547 个任务
  2. 根据 instance_id 前缀自动识别平台
  3. 对 SQLite 任务：动态提取 CREATE TABLE 语句作为 Schema
  4. 对云任务：优先使用 Schema Catalog，无 catalog 时回退到 LLM 先验知识
  5. 调用 DeepSeek V4 flash 生成 SQL
  6. 将结果保存到 my_predicted_sqls/{instance_id}.sql
================================================================================
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

# Schema Catalog 支持
try:
    from schema_catalog import load_catalog, get_schema as _get_catalog_schema
    _HAS_CATALOG = True
except ImportError:
    _HAS_CATALOG = False


# API 配置 - DeepSeek V4 flash
# API Key 通过环境变量 DEEPSEEK_API_KEY 传入，不再硬编码
_DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not _DEEPSEEK_API_KEY:
    print("[WARN] 环境变量 DEEPSEEK_API_KEY 未设置，API 调用将失败")
    print("[WARN] 请设置: export DEEPSEEK_API_KEY=sk-xxxxx")

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
TASKS_FILE = BASE_DIR / "spider2-lite.jsonl"                      # 全量 547 任务数据文件
SQLITE_DB_DIR = BASE_DIR / "resource" / "databases" / "spider2-localdb"  # SQLite 数据库目录
LOCAL_MAP_FILE = SQLITE_DB_DIR / "local-map.jsonl"               # instance_id -> db 名映射
OUTPUT_DIR = BASE_DIR / "my_predicted_sqls"                      # 预测 SQL 输出目录
FEW_SHOT_FILE = BASE_DIR / "few_shot_examples.json"              # Few-shot 示例文件
CATALOG_FILE = BASE_DIR / "schema_catalog.json"                 # Schema Catalog 缓存文件


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数：包装日志输出，让进度更可读
# ──────────────────────────────────────────────────────────────────────────────

_LOG_FILE = None


def _get_log_file():
    global _LOG_FILE
    if _LOG_FILE is None:
        _LOG_FILE = open(BASE_DIR / "inference_log_v2.txt", "w", encoding="utf-8")
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
    """根据 instance_id 前缀返回数据库平台类型。

    规则：
      localXXX  → "sqlite"      本地 SQLite
      sf_bqXXX  → "snowflake"   Snowflake (BigQuery 适配)
      sfXXX     → "snowflake"   Snowflake 原生
      bqXXX     → "bigquery"    Google BigQuery
      gaXXX     → "bigquery"    Google Analytics on BigQuery
    """
    if instance_id.startswith("sf_bq") or instance_id.startswith("sf"):
        return "snowflake"
    if instance_id.startswith("bq"):
        return "bigquery"
    if instance_id.startswith("local"):
        return "sqlite"
    if instance_id.startswith("ga"):
        return "bigquery"
    return "bigquery"  # 兜底


# ──────────────────────────────────────────────────────────────────────────────
# 第一阶段：任务加载
# ──────────────────────────────────────────────────────────────────────────────

def load_tasks(tasks_file: Path) -> List[dict]:
    """
    从 JSONL 文件中读取所有任务
    参数：
        tasks_file: JSONL 文件路径
    返回：
        任务字典列表，每个字典包含 instance_id, db, question 等字段
    """
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
    """
    构建数据库名到 .sqlite 文件的映射表

    策略：
      1. 直接扫描 spider2-localdb 目录下的所有 .sqlite 文件，建立 {stem: path} 映射
      2. 如果存在 local-map.jsonl，则额外建立 {instance_id: db_stem} 的备选映射

    返回：
        (db_to_path, instance_to_db)
        - db_to_path: {数据库名: .sqlite 文件完整路径}
        - instance_to_db: {instance_id: 数据库名}
    """
    db_to_path: Dict[str, str] = {}
    instance_to_db: Dict[str, str] = {}

    # 扫描所有 .sqlite 文件
    for sqlite_file in SQLITE_DB_DIR.glob("*.sqlite"):
        db_name = sqlite_file.stem  # 去掉 .sqlite 后缀，如 "E_commerce"
        db_to_path[db_name] = str(sqlite_file)

    # 读取 local-map.jsonl（如果存在）
    if LOCAL_MAP_FILE.exists():
        with open(LOCAL_MAP_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    mapping = json.loads(line)
                    # mapping 格式：{"local002": "E_commerce", ...}
                    instance_to_db.update(mapping)
                except json.JSONDecodeError:
                    pass

    return db_to_path, instance_to_db


def resolve_database_path(task: dict, db_to_path: Dict[str, str],
                          instance_to_db: Dict[str, str]) -> Optional[str]:
    """
    根据任务信息解析对应的 .sqlite 文件路径

    解析优先级：
      1. task["db"] 字段值直接在 db_to_path 中能找到 → 直接返回路径
      2. task["instance_id"] 在 instance_to_db 中有映射 →
         使用映射后的 db 名再去 db_to_path 中查找
      3. 以上均失败 → 返回 None，表示本地无此数据库

    参数：
        task: 任务字典
        db_to_path: {数据库名: .sqlite 路径}
        instance_to_db: {instance_id: 数据库名}
    返回：
        .sqlite 文件路径字符串，或 None
    """
    db_name = task.get("db", "")
    instance_id = task.get("instance_id", "")

    # 策略 1：直接用 db 字段匹配
    if db_name in db_to_path:
        return db_to_path[db_name]

    # 策略 2：通过 instance_id 从 local-map.jsonl 中查找映射
    mapped_db = instance_to_db.get(instance_id)
    if mapped_db and mapped_db in db_to_path:
        return db_to_path[mapped_db]

    # 双方都匹配不上
    return None


def extract_schema(db_path: str) -> str:
    """
    连接 SQLite 数据库，提取所有表的 CREATE TABLE 语句

    参数：
        db_path: .sqlite 文件路径
    返回：
        完整的建表语句字符串（各表之间以分号和换行分隔）
    异常：
        若数据库无法打开或查询失败，抛出异常让上层处理
    """
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

    # 收集所有 CREATE TABLE 语句，确保以分号结尾
    create_statements = []
    for (sql_text,) in rows:
        sql_text = sql_text.strip()
        if not sql_text.endswith(";"):
            sql_text += ";"
        create_statements.append(sql_text)

    return "\n\n".join(create_statements)


# ──────────────────────────────────────────────────────────────────────────────
# 第三阶段：Prompt 构建（三段式 Zero-shot）
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


def build_prompt_sqlite(schema: str, question: str, few_shot_examples: Optional[List[dict]] = None) -> str:
    instruction = (
        "You are a SQLite expert. "
        "Please write SQL for the following question based on the provided database schema. "
        "Only output the SQL statement, without any explanation."
    )
    fs_section = _format_few_shot_section(few_shot_examples or [])
    return f"""{instruction}

{fs_section}
=== Database Schema ===
{schema}

=== Question ===
{question}

=== SQL ===
"""


def build_prompt_bigquery(db_name: str, question: str, external_knowledge: Optional[str] = None,
                          few_shot_examples: Optional[List[dict]] = None) -> str:
    instruction = (
        "You are a Google BigQuery SQL expert. "
        "Write a standard BigQuery SQL query for the question below. "
        "The dataset is `{db_name}` hosted on BigQuery public datasets. "
        "Use fully-qualified table paths like `project.dataset.table` where appropriate. "
        "Use BigQuery-specific functions (UNNEST, STRUCT, SAFE_CAST, PARSE_DATE, etc.) as needed. "
        "Only output the SQL statement, without any explanation."
    ).format(db_name=db_name)

    ek_section = ""
    if external_knowledge:
        ek_section = f"\n=== External Knowledge (table schemas / hints) ===\n{external_knowledge}\n"

    fs_section = _format_few_shot_section(few_shot_examples or [])

    return f"""{instruction}

{fs_section}
{ek_section}
=== Question ===
{question}

=== BigQuery SQL ===
"""


def build_prompt_snowflake(db_name: str, question: str, external_knowledge: Optional[str] = None,
                           few_shot_examples: Optional[List[dict]] = None) -> str:
    instruction = (
        "You are a Snowflake SQL expert. "
        "Write a standard Snowflake SQL query for the question below. "
        "The dataset is `{db_name}` hosted on Snowflake. "
        "Use Snowflake-specific syntax: LATERAL FLATTEN for nested data, TABLE(GENERATOR(...)) for series, "
        "TO_CHAR/TO_DATE for formatting, QUALIFY for window filtering. "
        "Only output the SQL statement, without any explanation."
    ).format(db_name=db_name)

    ek_section = ""
    if external_knowledge:
        ek_section = f"\n=== External Knowledge (table schemas / hints) ===\n{external_knowledge}\n"

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
    """
    调用 LLM API 生成 SQL

    参数：
        prompt: 完整的 prompt 字符串
        config: API 配置字典（包含 base_url, api_key, model 等）
    返回：
        模型生成的原始文本
    异常：
        网络错误、API 错误等会在重试耗尽后向上抛出
    """
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
            # 提取返回内容
            return response.choices[0].message.content
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                wait_seconds = 2 ** attempt
                log("WARN", f"API 调用失败（第 {attempt + 1}/{max_retries + 1} 次），"
                       f"{wait_seconds} 秒后重试: {exc}")
                time.sleep(wait_seconds)

    # 所有重试均失败
    raise RuntimeError(f"API 调用在 {max_retries + 1} 次尝试后仍然失败") from last_error


# ──────────────────────────────────────────────────────────────────────────────
# 第五阶段：结果保存与后处理
# ──────────────────────────────────────────────────────────────────────────────

def clean_sql_output(raw_text: str) -> str:
    """
    清洗模型输出，去掉可能的 markdown 代码块标记等杂质

    参数：
        raw_text: 模型原始输出文本
    返回：
        清洗后的纯 SQL 字符串
    """
    text = raw_text.strip()

    # 如果输出被包裹在 ```sql ... ``` 或 ``` ... ``` 中，则提取内部内容
    if text.startswith("```"):
        # 找到第一个换行符后的内容，并去掉末尾的 ```
        lines = text.split("\n")
        # 去掉开头的 ``` 行（可能带有语言标识如 ```sql）
        if lines[0].startswith("```"):
            lines = lines[1:]
        # 去掉结尾的 ``` 行
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return text


def save_sql(instance_id: str, sql: str, output_dir: Path) -> Path:
    """
    将生成的 SQL 保存为 .sql 文件

    参数：
        instance_id: 任务 ID（作为文件名前缀）
        sql: SQL 语句
        output_dir: 输出目录
    返回：
        保存的文件路径
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{instance_id}.sql"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(sql)
        if not sql.endswith("\n"):
            f.write("\n")
    return file_path


# ──────────────────────────────────────────────────────────────────────────────
# 主流程：遍历所有任务，执行推理闭环
# ──────────────────────────────────────────────────────────────────────────────

def load_external_knowledge(ek_filename: Optional[str]) -> Optional[str]:
    """尝试加载外部知识 Markdown 文件（数据库 schema 文档）。

    查找顺序：
      1. resource/external_knowledge/{filename}
      2. evaluation_suite/external_knowledge/{filename}
    """
    if not ek_filename:
        return None
    for candidate in [
        BASE_DIR / "resource" / "external_knowledge" / ek_filename,
        BASE_DIR / "evaluation_suite" / "external_knowledge" / ek_filename,
    ]:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return None


def main(limit: Optional[int] = None, use_catalog: bool = True) -> None:
    """主入口：加载全部 547 任务 → 按平台路由 → 逐个推理 → 保存结果 → 输出统计"""

    # ---------- 启动检查 ----------
    log("INFO", "=" * 60)
    log("INFO", f"Spider 2.0-Lite 全量推理引擎 V2 启动 (Schema Catalog={'ON' if use_catalog else 'OFF'})")
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
            log("WARN", "schema_catalog.json 不存在，请先运行 python schema_catalog.py --build")
        log("INFO", "Schema Catalog 不可用，云任务将依赖 LLM 先验知识")

    # ---------- 1. 加载任务 ----------
    log("INFO", f"正在加载任务文件: {TASKS_FILE}")
    tasks = load_tasks(TASKS_FILE)
    log("INFO", f"成功加载 {len(tasks)} 个任务")

    if limit and limit > 0:
        tasks = tasks[:limit]
        log("INFO", f"测试模式：仅处理前 {len(tasks)} 个任务")

    # ---------- 2. 构建本地数据库映射（仅 SQLite 任务需要） ----------
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

        # ================================================================
        # 路径 A：SQLite 本地任务（保持原有逻辑不变）
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
                    "instance_id": instance_id,
                    "platform": platform,
                    "stage": "schema_extraction",
                    "error": str(exc),
                })
                continue

            prompt = build_prompt_sqlite(schema, question, few_shot.get("sqlite"))

        # ================================================================
        # 路径 B：BigQuery 云端任务
        # ================================================================
        elif platform == "bigquery":
            ek_content = load_external_knowledge(ek_filename)
            if ek_content:
                log("INFO", f"  ->已加载外部知识: {ek_filename}")
            else:
                catalog_schema = None
                if catalog:
                    catalog_schema = _get_catalog_schema(db_name, "bigquery")
                if catalog_schema:
                    log("INFO", f"  ->Schema Catalog 命中: {db_name}")
                    catalog_hits["bigquery"] += 1
                    # 将 catalog schema 作为 external_knowledge 注入
                    if ek_content:
                        ek_content = catalog_schema + "\n\n" + ek_content
                    else:
                        ek_content = catalog_schema
                else:
                    log("INFO", f"  ->Schema Catalog 未命中: {db_name}，依赖 LLM 先验知识")
            prompt = build_prompt_bigquery(db_name, question, ek_content, few_shot.get("bigquery"))

        # ================================================================
        # 路径 C：Snowflake 云端任务
        # ================================================================
        elif platform == "snowflake":
            ek_content = load_external_knowledge(ek_filename)
            if ek_content:
                log("INFO", f"  ->已加载外部知识: {ek_filename}")
            else:
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
                else:
                    log("INFO", f"  ->Schema Catalog 未命中: {db_name}，依赖 LLM 先验知识")
            prompt = build_prompt_snowflake(db_name, question, ek_content, few_shot.get("snowflake"))

        else:
            log("SKIP", f"  ->未知平台 {platform}，跳过")
            stats["skipped"] += 1
            stats["by_platform"]["skipped"] += 1
            continue

        # ================================================================
        # 调用 LLM API（三种平台共用）
        # ================================================================
        try:
            raw_result = call_llm(prompt, API_CONFIG)
        except Exception as exc:
            log("ERROR", f"  ->LLM 调用失败: {exc}")
            stats["failed"] += 1
            stats["by_platform"]["failed"] += 1
            stats["errors"].append({
                "instance_id": instance_id,
                "platform": platform,
                "stage": "llm_call",
                "error": str(exc),
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
                "instance_id": instance_id,
                "platform": platform,
                "stage": "save_file",
                "error": str(exc),
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
        log_path = BASE_DIR / "inference_log_v2.txt"
        log("INFO", f"日志已保存到 {log_path}")
        _LOG_FILE.close()


# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Spider 2.0-Lite 全量推理引擎 V2")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制处理的任务数（测试用，默认 None = 全量）")
    parser.add_argument("--no-catalog", action="store_true",
                        help="禁用 Schema Catalog，回退到 V1 行为（对比测试用）")
    args = parser.parse_args()
    main(limit=args.limit, use_catalog=not args.no_catalog)