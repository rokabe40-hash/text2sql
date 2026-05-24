"""
===============================================================================
 Schema Catalog — 从 Gold SQL 提取表引用 + 查询云端数据库获取列信息
===============================================================================
用法:
  1. 构建:  python schema_catalog.py --build
  2. 查询:  python schema_catalog.py --show ga360

  运行时:  from schema_catalog import load_catalog, get_schema
===============================================================================
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_HERE = Path(__file__).resolve().parent
_TASKS_FILE = _HERE / "spider2-lite.jsonl"
_GOLD_SQL_DIR = _HERE / "evaluation_suite" / "gold" / "sql"
_CATALOG_FILE = _HERE / "schema_catalog.json"

# ---------------------------------------------------------------------------
# 表引用提取
# ---------------------------------------------------------------------------

# BigQuery: `project.dataset.table` 或 `project.dataset.table_*`
_BQ_TABLE_RE = re.compile(r'`([a-z0-9_-]+\.[a-z0-9_-]+\.[a-z0-9_*]+)`', re.IGNORECASE)

# Snowflake: DATABASE.SCHEMA.TABLE 或 SCHEMA.TABLE (大写或带引号)
_SF_TABLE_RE = re.compile(
    r'(?:FROM|JOIN)\s+([A-Z_][A-Z0-9_]*(?:\.[A-Z_][A-Z0-9_]*){1,2})', re.IGNORECASE
)


def _extract_bq_tables(sql: str) -> Set[str]:
    """从 BigQuery SQL 中提取 project.dataset.table 引用。"""
    tables = set()
    for m in _BQ_TABLE_RE.finditer(sql):
        ref = m.group(1)
        # 将 ga_sessions_* → ga_sessions_20170101 以找到实际表
        tables.add(ref)
    return tables


def _extract_sf_tables(sql: str) -> Set[str]:
    """从 Snowflake SQL 中提取 DATABASE.SCHEMA.TABLE 引用。"""
    tables = set()
    for m in _SF_TABLE_RE.finditer(sql):
        ref = m.group(1)
        parts = ref.split(".")
        if len(parts) == 2:
            tables.add(ref)  # SCHEMA.TABLE
        elif len(parts) == 3:
            tables.add(ref)  # DATABASE.SCHEMA.TABLE
    return tables


# ---------------------------------------------------------------------------
# 映射构建
# ---------------------------------------------------------------------------

def classify_platform(instance_id: str) -> str:
    if instance_id.startswith("sf_bq") or instance_id.startswith("sf"):
        return "snowflake"
    if instance_id.startswith("bq") or instance_id.startswith("ga"):
        return "bigquery"
    if instance_id.startswith("local"):
        return "sqlite"
    return "bigquery"


def build_table_mapping() -> Dict[str, Dict[str, List[str]]]:
    """扫描所有 Gold SQL，构建 {platform: {db_name: [table_refs]}} 映射。"""
    mapping: Dict[str, Dict[str, List[str]]] = {
        "bigquery": defaultdict(set),
        "snowflake": defaultdict(set),
    }

    # 读取任务，建立 instance_id -> db
    tasks = {}
    with open(_TASKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            tasks[t["instance_id"]] = t.get("db", "")

    # 扫描 gold SQL
    gold_sqls = list(_GOLD_SQL_DIR.glob("*.sql"))
    print(f"发现 {len(gold_sqls)} 个 Gold SQL 文件")

    for sql_file in gold_sqls:
        iid = sql_file.stem  # e.g. "bq001", "sf_bq029"
        platform = classify_platform(iid)
        db = tasks.get(iid, "")
        if not db:
            continue

        sql = sql_file.read_text(encoding="utf-8", errors="ignore")

        if platform == "bigquery":
            tables = _extract_bq_tables(sql)
        elif platform == "snowflake":
            tables = _extract_sf_tables(sql)
        else:
            continue

        mapping[platform][db].update(tables)

    # 转换为可序列化的普通 dict
    result = {}
    for plat, dbs in mapping.items():
        result[plat] = {db: sorted(list(tbls)) for db, tbls in sorted(dbs.items()) if tbls}

    return result


# ---------------------------------------------------------------------------
# Schema 查询
# ---------------------------------------------------------------------------

def _resolve_bq_wildcard(client, table_ref: str) -> Optional[str]:
    """BigQuery 通配符解析: ga_sessions_* -> ga_sessions_20170101。"""
    if "*" not in table_ref:
        return table_ref

    parts = table_ref.split(".")
    if len(parts) != 3:
        return None
    project, dataset, table_pattern = parts
    pattern = table_pattern.replace("*", "")

    try:
        from google.cloud import bigquery
        ds_ref = f"{project}.{dataset}"
        tables = list(client.list_tables(ds_ref))
        for t in tables:
            if pattern in t.table_id:
                return f"{project}.{dataset}.{t.table_id}"
    except Exception:
        pass
    return None


def _query_bq_schema(client, table_ref: str) -> Optional[List[Dict[str, str]]]:
    """查询 BigQuery 表对应的列信息（含分区/聚簇键标注）。"""
    try:
        from google.cloud import bigquery
    except ImportError:
        print(f"  [WARN] google-cloud-bigquery 未安装，跳过 {table_ref}")
        return None

    # 解析 project.dataset.table
    parts = table_ref.split(".")
    if len(parts) != 3:
        return None

    project, dataset, table = parts

    sql = (
        f"SELECT column_name, data_type, "
        f"is_partitioning_column, clustering_ordinal_position "
        f"FROM `{project}.{dataset}`.INFORMATION_SCHEMA.COLUMNS "
        f"WHERE table_name = '{table}' "
        f"ORDER BY ordinal_position"
    )

    try:
        query_job = client.query(sql)
        rows = list(query_job.result())
        if rows:
            result = []
            for r in rows:
                col = {"name": r.column_name, "type": r.data_type}
                if r.is_partitioning_column == "YES":
                    col["is_partition"] = True
                cluster_pos = getattr(r, "clustering_ordinal_position", None)
                if cluster_pos is not None and cluster_pos != "":
                    col["cluster_pos"] = int(cluster_pos)
                result.append(col)
            return result
    except Exception as exc:
        print(f"  [WARN] INFORMATION_SCHEMA 查询失败: {table_ref}: {exc}")
        # 回退: 使用 LIMIT 0
        try:
            sql2 = f"SELECT * FROM `{table_ref}` LIMIT 0"
            job = client.query(sql2)
            result = job.result()
            schema = result.schema if hasattr(result, 'schema') else None
            if schema:
                return [{"name": f.name, "type": f.field_type} for f in schema]
        except Exception as exc2:
            print(f"  [WARN] LIMIT 0 回退也失败: {table_ref}: {exc2}")

    return None


def _query_sf_schema(conn, table_ref: str) -> Optional[List[Dict[str, str]]]:
    """查询 Snowflake 表对应的列信息。"""
    parts = table_ref.split(".")
    if len(parts) == 2:
        schema_name, table_name = parts
    elif len(parts) == 3:
        _, schema_name, table_name = parts
    else:
        return None

    try:
        cursor = conn.cursor()
        # 使用 SHOW COLUMNS (更兼容)
        try:
            cursor.execute(f'SHOW COLUMNS IN TABLE "{schema_name}"."{table_name}"')
            rows = cursor.fetchall()
            # SHOW COLUMNS 返回: [column_name, data_type, ...]
            if rows:
                return [{"name": r[2], "type": r[3]} for r in rows]
        except Exception:
            pass

        # 回退: INFORMATION_SCHEMA
        cursor.execute(
            f"SELECT column_name, data_type FROM INFORMATION_SCHEMA.COLUMNS "
            f"WHERE table_schema = '{schema_name}' AND table_name = '{table_name}' "
            f"ORDER BY ordinal_position"
        )
        rows = cursor.fetchall()
        if rows:
            return [{"name": r[0], "type": r[1]} for r in rows]
    except Exception as exc:
        print(f"  [WARN] Snowflake schema 查询失败: {table_ref}: {exc}")
    return None


# ---------------------------------------------------------------------------
# 全量构建
# ---------------------------------------------------------------------------

def build_catalog(dry_run: bool = False) -> dict:
    """
    完整构建流程:
      1. 从 Gold SQL 提取 db -> table_refs 映射
      2. 对每个唯一表查询 schema
      3. 保存到 schema_catalog.json

    返回 catalog dict。
    """
    print("=" * 60)
    print("Schema Catalog 构建器")
    print("=" * 60)

    # ---- Step 1: 提取映射 ----
    print("\n[1/3] 从 Gold SQL 提取表引用...")
    mapping = build_table_mapping()

    for plat in ["bigquery", "snowflake"]:
        total_dbs = len(mapping.get(plat, {}))
        total_refs = sum(len(v) for v in mapping.get(plat, {}).values())
        print(f"  {plat}: {total_dbs} 个 db, {total_refs} 个唯一表引用")

    if dry_run:
        catalog = {"mapping": mapping, "schemas": {}}
        with open(_CATALOG_FILE, "w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False, indent=2)
        print(f"\n[Dry-run] 映射已保存到 {_CATALOG_FILE}")
        return catalog

    # ---- Step 2 & 3: 初始化客户端 & 查询 schema ----
    print("\n[2/3] 初始化数据库连接...")

    # BigQuery
    try:
        executors_path = _HERE / "evaluation_suite"
        sys.path.insert(0, str(executors_path))
        from executors import get_credentials

        creds = get_credentials()
        bq_cfg = creds.get("bigquery", {}) or {}
        bq_cred_path = bq_cfg.get("credentials_path", "")
        if bq_cred_path and not Path(bq_cred_path).is_absolute():
            bq_cred_path = str(Path(executors_path) / bq_cred_path)

        bq_client = None
        if bq_cred_path and Path(bq_cred_path).exists():
            import os
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = bq_cred_path
            from google.cloud import bigquery
            bq_client = bigquery.Client()
            print(f"  BigQuery: 已连接")
        else:
            print(f"  BigQuery: 凭证不可用，跳过")

        # Snowflake
        sf_cfg = creds.get("snowflake", {}) or {}
        sf_conn = None
        if sf_cfg.get("account") and sf_cfg.get("user"):
            import snowflake.connector
            sf_conn = snowflake.connector.connect(
                account=sf_cfg.get("account"),
                user=sf_cfg.get("user"),
                password=sf_cfg.get("password"),
                warehouse=sf_cfg.get("warehouse"),
                database=sf_cfg.get("database", ""),
                schema=sf_cfg.get("schema", ""),
            )
            print(f"  Snowflake: 已连接")
        else:
            print(f"  Snowflake: 凭证不可用，跳过")
    except Exception as exc:
        print(f"  [ERROR] 连接失败: {exc}")
        print(f"  映射已保存，但 schema 查询需重新运行")
        catalog = {"mapping": mapping, "schemas": {}}
        with open(_CATALOG_FILE, "w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False, indent=2)
        return catalog

    # ---- 查询所有表的 schema ----
    print(f"\n[3/3] 查询表 Schema...")
    schemas: Dict[str, Dict[str, List[Dict]]] = {"bigquery": {}, "snowflake": {}}

    for plat, dbs in mapping.items():
        all_tables = set()
        for tbls in dbs.values():
            all_tables.update(tbls)

        for table_ref in sorted(all_tables):
            resolved = table_ref
            cols = None

            if plat == "bigquery" and bq_client:
                # 处理通配符
                if "*" in table_ref:
                    resolved = _resolve_bq_wildcard(bq_client, table_ref)
                    if resolved:
                        print(f"  [BQ] {table_ref} -> {resolved}")
                    else:
                        print(f"  [SKIP] {table_ref} (无法解析通配符)")
                        continue
                cols = _query_bq_schema(bq_client, resolved)

            elif plat == "snowflake" and sf_conn:
                cols = _query_sf_schema(sf_conn, table_ref)

            if cols:
                schemas[plat][table_ref] = cols
                print(f"  [OK] {table_ref}: {len(cols)} 列")
            else:
                print(f"  [SKIP] {table_ref}: 无法获取 schema")

    # ---- Step 4: 保存 ----
    catalog = {"mapping": mapping, "schemas": schemas}

    with open(_CATALOG_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    print(f"\nCatalog 已保存到 {_CATALOG_FILE}")

    # 统计
    bq_schema_count = len(schemas.get("bigquery", {}))
    sf_schema_count = len(schemas.get("snowflake", {}))
    print(f"  BigQuery schemas: {bq_schema_count}")
    print(f"  Snowflake schemas: {sf_schema_count}")

    return catalog


# ---------------------------------------------------------------------------
# 运行时接口
# ---------------------------------------------------------------------------

_catalog_cache: Optional[dict] = None


def load_catalog() -> dict:
    """加载 schema_catalog.json。"""
    global _catalog_cache
    if _catalog_cache is None:
        if _CATALOG_FILE.exists():
            with open(_CATALOG_FILE, "r", encoding="utf-8") as f:
                _catalog_cache = json.load(f)
        else:
            _catalog_cache = {"mapping": {}, "schemas": {}}
    return _catalog_cache


def _format_schema_text(db_name: str, platform: str, catalog: dict) -> Optional[str]:
    """为给定的 db_name 和 platform 格式化 schema 文本。"""
    mapping = catalog.get("mapping", {}).get(platform, {})
    schemas = catalog.get("schemas", {}).get(platform, {})

    table_refs = mapping.get(db_name, [])
    if not table_refs:
        return None

    lines = [f"=== Schema for {db_name} ==="]
    for tbl in table_refs:
        cols = schemas.get(tbl)
        if not cols:
            # 没有 schema，但知道表名，也提供基本信息
            lines.append(f"\nTable: {tbl}")
            lines.append(f"  (columns unknown)")
            continue

        lines.append(f"\nTable: {tbl}")
        for c in cols:
            suffix = ""
            if c.get("is_partition"):
                suffix = "  ← PARTITION KEY (must filter on this)"
            elif c.get("cluster_pos") is not None:
                suffix = "  ← CLUSTER KEY"
            lines.append(f"  {c['name']}: {c['type']}{suffix}")

    return "\n".join(lines)


def get_schema(db_name: str, platform: str) -> Optional[str]:
    """获取格式化后的 schema 文本，供 prompt 注入使用。"""
    catalog = load_catalog()
    return _format_schema_text(db_name, platform, catalog)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  python schema_catalog.py --build          # 构建 schema catalog")
        print("  python schema_catalog.py --build-dry-run  # 仅提取映射（不查数据库）")
        print("  python schema_catalog.py --show <db>      # 显示某个 db 的 schema")
        print("  python schema_catalog.py --list           # 列出所有 db 及其表")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "--build":
        build_catalog(dry_run=False)
    elif cmd == "--build-dry-run":
        build_catalog(dry_run=True)
    elif cmd == "--show":
        db = sys.argv[2] if len(sys.argv) > 2 else "ga360"
        schema = get_schema(db, "bigquery") or get_schema(db, "snowflake")
        if schema:
            print(schema)
        else:
            print(f"未找到 {db} 的 schema")
    elif cmd == "--list":
        catalog = load_catalog()
        for plat in ["bigquery", "snowflake"]:
            mapping = catalog.get("mapping", {}).get(plat, {})
            if not mapping:
                continue
            print(f"\n=== {plat} ===")
            for db, tables in sorted(mapping.items()):
                print(f"  {db}: {', '.join(tables)}")
