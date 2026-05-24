"""
================================================================================
 Spider 2.0-Lite Text-to-SQL 推理引擎 V4 — SQLite 深度优化版
================================================================================
V4 相对 V3 的改动:
  - 增强型 SQLite Schema 提取: 推断外键关系 + 样本数据 + 行数 + 值分布
  - SQLite 方言强化 Prompt: 显式禁止 PostgreSQL/MySQL 语法，指明 SQLite 特性
  - BigQuery/Snowflake 逻辑沿用 V3，未改动
  - 输出目录: my_predicted_sqls_v4/

使用:
  SQLite 快速测试:  python run_inference_full_v4.py --sample-sqlite 30
  全量 SQLite:      python run_inference_full_v4.py --sample-sqlite 135
  BigQuery 测试:     python run_inference_full_v4.py --sample-bq 30
================================================================================
"""

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from openai import OpenAI

try:
    from schema_catalog import load_catalog, get_schema as _get_catalog_schema
    _HAS_CATALOG = True
except ImportError:
    _HAS_CATALOG = False

try:
    from core.rag_retriever import EKRetriever
    _HAS_RAG = True
except ImportError:
    _HAS_RAG = False


_DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not _DEEPSEEK_API_KEY:
    print("[WARN] DEEPSEEK_API_KEY not set")

API_CONFIG = {
    "base_url": "https://api.deepseek.com",
    "api_key": _DEEPSEEK_API_KEY,
    "model": "deepseek-v4-flash",
    "temperature": 0.0,
    "max_tokens": 8192,
    "timeout": 60,
    "max_retries": 2,
}

BASE_DIR = Path(__file__).resolve().parent
TASKS_FILE = BASE_DIR / "spider2-lite.jsonl"
SQLITE_DB_DIR = BASE_DIR / "resource" / "databases" / "spider2-localdb"
LOCAL_MAP_FILE = SQLITE_DB_DIR / "local-map.jsonl"
OUTPUT_DIR = BASE_DIR / "my_predicted_sqls_v4"
FEW_SHOT_FILE = BASE_DIR / "few_shot_examples.json"
CATALOG_FILE = BASE_DIR / "schema_catalog.json"
EK_INDEX_DIR = BASE_DIR / "data" / "ek_index"


# =============================================================================
# Logging
# =============================================================================

_LOG_FILE = None

def _get_log_file():
    global _LOG_FILE
    if _LOG_FILE is None:
        _LOG_FILE = open(BASE_DIR / "inference_log_v4.txt", "w", encoding="utf-8")
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


# =============================================================================
# Platform classification
# =============================================================================

def classify_platform(instance_id: str) -> str:
    if instance_id.startswith("sf_bq") or instance_id.startswith("sf"):
        return "snowflake"
    if instance_id.startswith("bq") or instance_id.startswith("ga"):
        return "bigquery"
    if instance_id.startswith("local"):
        return "sqlite"
    return "bigquery"


# =============================================================================
# Phase 1: Task loading
# =============================================================================

def load_tasks(tasks_file: Path) -> List[dict]:
    tasks = []
    with open(tasks_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log("WARN", f"Line {line_num} JSON parse error: {exc}")
    return tasks


# =============================================================================
# Phase 2: Enhanced SQLite Schema Extraction (V4 core)
# =============================================================================

def _pluralize(name: str) -> List[str]:
    """Generate possible plural/table-name variants of a word."""
    candidates = [name, name + 's', name + 'es']
    if name.endswith('y'):
        candidates.append(name[:-1] + 'ies')
    if name.endswith('s'):
        candidates.append(name[:-1])
    return candidates


def extract_schema_v4(db_path: str) -> str:
    """
    Extract enhanced SQLite schema with:
    - Column names, types, primary key annotations
    - Row counts for each table
    - Sample data (2 rows per table)
    - Value distributions for low-cardinality columns
    - Inferred foreign key relationships
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    table_names = []
    tables = {}

    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        table_names.append(name)

    # Extract column metadata, row counts, sample data
    for tbl in table_names:
        cols = []
        for c in conn.execute(f'PRAGMA table_info("{tbl}")'):
            cols.append({
                'name': c[1], 'type': c[2], 'pk': c[5]
            })

        row_count = conn.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]

        sample_rows = []
        try:
            col_names = [c['name'] for c in cols]
            for r in conn.execute(f'SELECT * FROM "{tbl}" LIMIT 2'):
                sample_rows.append({
                    col_names[i]: str(r[i])[:80] if r[i] is not None else 'NULL'
                    for i in range(len(col_names))
                })
        except Exception:
            pass

        # Value distribution for low-cardinality columns (n_distinct <= 20)
        distributions = {}
        for c in cols:
            try:
                distinct = conn.execute(
                    f'SELECT COUNT(DISTINCT "{c["name"]}") FROM "{tbl}"'
                ).fetchone()[0]
                if distinct <= 20 and distinct > 0:
                    vals = conn.execute(
                        f'SELECT "{c["name"]}", COUNT(*) as cnt FROM "{tbl}" '
                        f'GROUP BY "{c["name"]}" ORDER BY cnt DESC LIMIT 10'
                    ).fetchall()
                    distributions[c['name']] = [(str(v[0])[:50], v[1]) for v in vals]
            except Exception:
                pass

        tables[tbl] = {
            'columns': cols,
            'row_count': row_count,
            'sample_rows': sample_rows,
            'distributions': distributions,
        }

    conn.close()

    # Infer FK relationships from column naming conventions
    fk_relations = []
    for tbl, info in tables.items():
        for col in info['columns']:
            col_name_lower = col['name'].lower()
            if not col_name_lower.endswith('_id'):
                continue
            if col_name_lower == 'id':
                continue  # skip table's own PK

            prefix = col_name_lower[:-3]  # 'customer' from 'customer_id'

            for other_tbl in table_names:
                if other_tbl.lower() == tbl.lower():
                    continue

                other_lower = other_tbl.lower().replace('_', '')
                prefix_clean = prefix.replace('_', '')

                # Match: customer_id -> customers, order_id -> orders
                if other_lower == prefix_clean or other_lower == prefix_clean + 's':
                    other_cols_lower = [c['name'].lower() for c in tables[other_tbl]['columns']]
                    # Find matching column in target table
                    for target_col in [col['name'], 'id', other_tbl.lower().rstrip('s') + '_id']:
                        if target_col.lower() in other_cols_lower:
                            fk_relations.append((tbl, col['name'], other_tbl, target_col))
                            break

    # Build formatted output
    lines = []
    lines.append(f"=== Database Schema ({len(table_names)} tables) ===\n")

    # 1. Table details
    for tbl in sorted(table_names):
        info = tables[tbl]
        pk_cols = [c['name'] for c in info['columns'] if c['pk']]
        pk_str = f"  PRIMARY KEY: {', '.join(pk_cols)}" if pk_cols else ""
        lines.append(f"## Table: {tbl}  ({info['row_count']:,} rows){pk_str}")

        for c in info['columns']:
            lines.append(f"  - {c['name']}: {c['type']}")

        # Value distributions
        if info['distributions']:
            lines.append("  Value distributions:")
            for col_name, vals in info['distributions'].items():
                val_str = ", ".join(f"{v[0]}({v[1]})" for v in vals[:6])
                lines.append(f"    {col_name}: {val_str}")

        # Sample rows
        if info['sample_rows']:
            lines.append("  Sample rows:")
            for i, row in enumerate(info['sample_rows']):
                vals = [f"{k}={v}" for k, v in row.items()]
                lines.append(f"    Row {i+1}: {', '.join(vals[:8])}")

        lines.append("")

    # 2. Join Paths
    if fk_relations:
        lines.append("## Join Paths (inferred)")
        seen = set()
        for src_tbl, src_col, tgt_tbl, tgt_col in fk_relations:
            key = (src_tbl, tgt_tbl)
            if key not in seen:
                seen.add(key)
                lines.append(f"  {src_tbl}.{src_col} -> {tgt_tbl}.{tgt_col}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Phase 3: Prompt construction
# =============================================================================

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


def build_prompt_sqlite_v4(schema_text: str, question: str,
                            ek_context: str = "",
                            few_shot_examples: Optional[List[dict]] = None) -> str:
    """
    V4 SQLite prompt with dialect-specific rules and enhanced schema.
    """
    instruction = (
        "You are a SQLite expert. Write a correct, executable SQLite query "
        "for the question below based on the provided database schema.\n\n"
        "CRITICAL SQLITE RULES - violations will cause runtime errors:\n\n"
        "1. DIALECT: SQLite does NOT support these PostgreSQL/MySQL features:\n"
        "   - NO ::type casting (use CAST(x AS type) instead)\n"
        "   - NO (composite_column).field access (split into separate columns)\n"
        "   - NO -> or ->> JSON operators (use JSON_EXTRACT if needed)\n"
        "   - NO ARRAY[], NO STORED PROCEDURES, NO PIVOT, NO CREATE TYPE\n"
        "   - NO REGEXP_CONTAINS (use LIKE or GLOB instead)\n\n"
        "2. COLUMN SCOPE: A column alias defined in a SELECT clause CANNOT be "
        "referenced by another expression in the SAME SELECT. Use a subquery "
        "or CTE to compute in steps. Example:\n"
        "   WRONG: SELECT x+1 AS a, a*2 AS b FROM t\n"
        "   RIGHT: SELECT a, a*2 AS b FROM (SELECT x+1 AS a FROM t)\n\n"
        "3. DIVISION: Always cast numerator to REAL when computing ratios:\n"
        "   CAST(x AS REAL) / y -- NOT x / y (integer division!)\n\n"
        "4. DATES: Use these SQLite functions:\n"
        "   - STRFTIME('%Y', date_col) for year extraction\n"
        "   - JULIANDAY(date1) - JULIANDAY(date2) for day differences\n"
        "   - DATE(col, '+N days') or DATE(col, '-N days') for date arithmetic\n"
        "   - DATE(col) to extract date part from datetime\n\n"
        "5. STRINGS: Use GROUP_CONCAT() for string aggregation, not STRING_AGG.\n"
        "   Use || for string concatenation, not CONCAT().\n\n"
        "6. JOINS: Use the Join Paths section to determine correct join columns. "
        "Verify that each column used in ON/USING exists in the referenced tables.\n\n"
        "7. ONLY output the SQL statement. No markdown fences, no explanations.\n"
    )

    fs_section = _format_few_shot_section(few_shot_examples or [])
    ek_section = f"\n{ek_context}\n" if ek_context else ""

    return f"""{instruction}

{fs_section}
{schema_text}
{ek_section}
=== Question ===
{question}

=== SQLite SQL ===
"""


# BigQuery and Snowflake prompts unchanged from V3
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
        "Use Snowflake-specific syntax: LATERAL FLATTEN for nested data, "
        "TABLE(GENERATOR(...)) for series, "
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


# =============================================================================
# Phase 4: LLM API call
# =============================================================================

def call_llm(prompt: str, config: dict) -> str:
    client = OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
        timeout=config.get("timeout", 60),
    )

    last_error = None
    max_retries = config.get("max_retries", 2)

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=config.get("temperature", 0.0),
                max_tokens=config.get("max_tokens", 4096),
            )
            return response.choices[0].message.content
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                wait_seconds = 2 ** attempt
                log("WARN", f"API call failed (attempt {attempt+1}/{max_retries+1}), "
                       f"retrying in {wait_seconds}s: {exc}")
                time.sleep(wait_seconds)

    raise RuntimeError(f"API call failed after {max_retries+1} attempts") from last_error


# =============================================================================
# Phase 5: Output
# =============================================================================

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


# =============================================================================
# Helpers
# =============================================================================

def build_db_mapping() -> Tuple[Dict[str, str], Dict[str, str]]:
    db_to_path = {}
    instance_to_db = {}
    for sqlite_file in SQLITE_DB_DIR.glob("*.sqlite"):
        db_to_path[sqlite_file.stem] = str(sqlite_file)
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


# =============================================================================
# Main
# =============================================================================

def main(limit: Optional[int] = None, use_catalog: bool = True,
         use_rag: bool = True, skip_snowflake: bool = False,
         sample_bq: Optional[int] = None, sample_sqlite: Optional[int] = None,
         seed: int = 42) -> None:

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
    log("INFO", f"Spider 2.0-Lite V4 (SQLite Enhanced) "
           f"(Catalog={catalog_str}, RAG={rag_str}, Snow={snow_str}{sample_info})")
    log("INFO", "=" * 60)

    if not TASKS_FILE.exists():
        log("ERROR", f"Task file not found: {TASKS_FILE}")
        sys.exit(1)

    # 0a. Few-shot
    few_shot = _load_few_shot_examples()
    for plat in ["sqlite", "bigquery", "snowflake"]:
        log("INFO", f"Few-shot [{plat}]: {len(few_shot.get(plat, []))} examples")

    # 0b. Schema Catalog
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
            log("WARN", "schema_catalog.py not found, falling back to V1 mode")
        elif use_catalog and not CATALOG_FILE.exists():
            log("WARN", "schema_catalog.json not found")

    # 0c. RAG
    retriever = None
    rag_hits = 0
    if use_rag and _HAS_RAG:
        rag_index = EK_INDEX_DIR / "ek_index.faiss"
        if rag_index.exists():
            for retry in range(3):
                try:
                    retriever = EKRetriever(str(EK_INDEX_DIR))
                    log("INFO", f"RAG Retriever: {retriever.chunk_count} chunks loaded")
                    break
                except Exception as exc:
                    if retry < 2:
                        log("WARN", f"RAG load failed (attempt {retry+1}/3), retrying: {exc}")
                        time.sleep(5)
                    else:
                        log("WARN", f"RAG load failed after 3 retries: {exc}")
        else:
            log("WARN", f"RAG index not found: {rag_index}")
    elif use_rag and not _HAS_RAG:
        log("FATAL", "core.rag_retriever not found. RAG unavailable. "
                       "Use --no-rag to skip or fix installation.")
        sys.exit(1)

    if use_rag and retriever is None:
        log("FATAL", "RAG enabled but failed to load. "
                       "Use --no-rag to skip or check network/index files.")
        sys.exit(1)

    # 1. Load tasks
    log("INFO", f"Loading tasks: {TASKS_FILE}")
    tasks = load_tasks(TASKS_FILE)
    log("INFO", f"Loaded {len(tasks)} tasks")

    if limit and limit > 0:
        tasks = tasks[:limit]
        log("INFO", f"--limit mode: first {len(tasks)} tasks")

    # 1b. Platform filtering & sampling
    if skip_snowflake:
        before = len(tasks)
        tasks = [t for t in tasks if classify_platform(t.get("instance_id", "")) != "snowflake"]
        log("INFO", f"Skip Snowflake: {before} -> {len(tasks)} tasks")

    if sample_bq or sample_sqlite:
        random.seed(seed)
        bq_tasks_all = [t for t in tasks if classify_platform(t.get("instance_id", "")) == "bigquery"]
        sqlite_tasks_all = [t for t in tasks if classify_platform(t.get("instance_id", "")) == "sqlite"]
        sf_tasks_all = [t for t in tasks if classify_platform(t.get("instance_id", "")) == "snowflake"]

        # If only SQLite sampling is requested, skip BQ and SF
        if not sample_bq and not skip_snowflake and sample_sqlite:
            bq_tasks = []
            sf_tasks = []
        else:
            bq_tasks = bq_tasks_all
            sf_tasks = sf_tasks_all

        if sample_bq and len(bq_tasks_all) > sample_bq:
            bq_tasks = random.sample(bq_tasks_all, sample_bq)
            log("INFO", f"BigQuery sample: {len(bq_tasks)}")
        if sample_sqlite and len(sqlite_tasks_all) > sample_sqlite:
            sqlite_tasks = random.sample(sqlite_tasks_all, sample_sqlite)
            log("INFO", f"SQLite sample: {len(sqlite_tasks)}")
        else:
            sqlite_tasks = sqlite_tasks_all

        tasks = bq_tasks + sqlite_tasks + sf_tasks
        log("INFO", f"Total after sampling: {len(tasks)} (BQ={len(bq_tasks)}, "
               f"SQLite={len(sqlite_tasks)}, SF={len(sf_tasks)})")

    # 2. Build local DB mapping
    db_to_path, instance_to_db = build_db_mapping()
    log("INFO", f"Found {len(db_to_path)} local SQLite databases")
    if instance_to_db:
        log("INFO", f"local-map.jsonl: {len(instance_to_db)} mappings")

    # 3. Output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log("INFO", f"Output directory: {OUTPUT_DIR}")

    # 4. Process tasks
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

        log("INFO", f"[{idx}/{len(tasks)}] {instance_id} (db={db_name}, platform={platform})")

        # RAG retrieval
        ek_rag_context = ""
        if retriever and question:
            query_text = f"{db_name}: {question}"
            try:
                ek_rag_context = retriever.format_context(query_text, top_k=3, max_chars=2500)
                if ek_rag_context:
                    rag_hits += 1
                    top1 = retriever.search(query_text, top_k=1)
                    if top1:
                        log("INFO", f"  ->RAG hit: {top1[0]['header_path']} "
                               f"(score={top1[0]['score']:.3f})")
            except Exception as exc:
                log("WARN", f"  ->RAG error: {exc}")

        # ============================================================
        # SQLite path (V4 enhanced)
        # ============================================================
        if platform == "sqlite":
            db_path = resolve_database_path(task, db_to_path, instance_to_db)
            if db_path is None:
                log("SKIP", f"  ->No local DB found, skipping")
                stats["skipped"] += 1
                stats["by_platform"]["skipped"] += 1
                continue

            try:
                schema_text = extract_schema_v4(db_path)
                table_count = schema_text.count("## Table:")
                log("INFO", f"  ->V4 schema extracted: {table_count} tables "
                       f"({Path(db_path).name})")
            except Exception as exc:
                log("ERROR", f"  ->Schema extraction failed: {exc}")
                stats["failed"] += 1
                stats["by_platform"]["failed"] += 1
                stats["errors"].append({
                    "instance_id": instance_id, "platform": platform,
                    "stage": "schema_extraction", "error": str(exc),
                })
                continue

            prompt = build_prompt_sqlite_v4(schema_text, question, ek_rag_context,
                                             few_shot.get("sqlite"))

        # ============================================================
        # BigQuery path (unchanged from V3)
        # ============================================================
        elif platform == "bigquery":
            ek_content = load_external_knowledge(ek_filename)
            if ek_content:
                log("INFO", f"  ->External knowledge loaded: {ek_filename}")

            catalog_schema = None
            if catalog:
                catalog_schema = _get_catalog_schema(db_name, "bigquery")
            if catalog_schema:
                log("INFO", f"  ->Catalog hit: {db_name}")
                catalog_hits["bigquery"] += 1
                if ek_content:
                    ek_content = catalog_schema + "\n\n" + ek_content
                else:
                    ek_content = catalog_schema
            elif not ek_content:
                log("INFO", f"  ->Catalog miss: {db_name}, relying on LLM prior knowledge")

            prompt = build_prompt_bigquery(db_name, question, ek_content,
                                           ek_rag_context, few_shot.get("bigquery"))

        # ============================================================
        # Snowflake path (unchanged from V3)
        # ============================================================
        elif platform == "snowflake":
            ek_content = load_external_knowledge(ek_filename)
            if ek_content:
                log("INFO", f"  ->External knowledge loaded: {ek_filename}")

            catalog_schema = None
            if catalog:
                catalog_schema = _get_catalog_schema(db_name, "snowflake")
            if catalog_schema:
                log("INFO", f"  ->Catalog hit: {db_name}")
                catalog_hits["snowflake"] += 1
                if ek_content:
                    ek_content = catalog_schema + "\n\n" + ek_content
                else:
                    ek_content = catalog_schema
            elif not ek_content:
                log("INFO", f"  ->Catalog miss: {db_name}, relying on LLM prior knowledge")

            prompt = build_prompt_snowflake(db_name, question, ek_content,
                                            ek_rag_context, few_shot.get("snowflake"))
        else:
            log("SKIP", f"  ->Unknown platform {platform}, skipping")
            stats["skipped"] += 1
            stats["by_platform"]["skipped"] += 1
            continue

        # API call
        try:
            raw_result = call_llm(prompt, API_CONFIG)
        except Exception as exc:
            log("ERROR", f"  ->LLM call failed: {exc}")
            stats["failed"] += 1
            stats["by_platform"]["failed"] += 1
            stats["errors"].append({
                "instance_id": instance_id, "platform": platform,
                "stage": "llm_call", "error": str(exc),
            })
            continue

        # Save
        sql = clean_sql_output(raw_result)
        try:
            saved_path = save_sql(instance_id, sql, OUTPUT_DIR)
            log("OK", f"  ->Saved: {saved_path.name} [{platform}]")
            stats["success"] += 1
            stats["by_platform"][platform] += 1
        except Exception as exc:
            log("ERROR", f"  ->Save failed: {exc}")
            stats["failed"] += 1
            stats["by_platform"]["failed"] += 1
            stats["errors"].append({
                "instance_id": instance_id, "platform": platform,
                "stage": "save_file", "error": str(exc),
            })

    # 5. Summary
    log("INFO", "=" * 60)
    log("INFO", "Inference Complete - Summary")
    log("INFO", f"  Total:    {stats['total']}")
    log("INFO", f"  Success:  {stats['success']}")
    log("INFO", f"  Skipped:  {stats['skipped']}")
    log("INFO", f"  Failed:   {stats['failed']}")
    bp = stats["by_platform"]
    log("INFO", f"  By platform: sqlite={bp['sqlite']}, bigquery={bp['bigquery']}, "
           f"snowflake={bp['snowflake']}, skipped={bp['skipped']}, failed={bp['failed']}")
    if use_catalog and catalog_hits["bigquery"] + catalog_hits["snowflake"] > 0:
        log("INFO", f"  Catalog hits: bigquery={catalog_hits['bigquery']}, "
               f"snowflake={catalog_hits['snowflake']}")
    rag_status = f"{rag_hits}/{stats['total']}" if retriever else "disabled"
    log("INFO", f"  RAG:        {rag_status}")

    if stats["errors"]:
        log("WARN", f"Failure details (first 10):")
        for err in stats["errors"][:10]:
            log("WARN", f"  - [{err['instance_id']}] [{err.get('platform','?')}] "
                   f"[{err['stage']}] {err['error']}")

    stats_file = OUTPUT_DIR / "_inference_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    log("INFO", f"Stats saved to {stats_file}")

    global _LOG_FILE
    if _LOG_FILE is not None:
        log("INFO", f"Log saved to {BASE_DIR / 'inference_log_v4.txt'}")
        _LOG_FILE.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Spider 2.0-Lite V4 (SQLite Enhanced)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of tasks (testing)")
    parser.add_argument("--no-catalog", action="store_true",
                        help="Disable Schema Catalog")
    parser.add_argument("--no-rag", action="store_true",
                        help="Disable RAG")
    parser.add_argument("--skip-snowflake", action="store_true",
                        help="Skip all Snowflake tasks")
    parser.add_argument("--sample-bq", type=int, default=None,
                        help="Random sample N BigQuery tasks")
    parser.add_argument("--sample-sqlite", type=int, default=None,
                        help="Random sample N SQLite tasks")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42)")
    args = parser.parse_args()
    main(limit=args.limit, use_catalog=not args.no_catalog, use_rag=not args.no_rag,
         skip_snowflake=args.skip_snowflake,
         sample_bq=args.sample_bq, sample_sqlite=args.sample_sqlite,
         seed=args.seed)
