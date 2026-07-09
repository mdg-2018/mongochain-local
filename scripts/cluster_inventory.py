#!/usr/bin/env python3
"""
Map all databases and collections in a MongoDB cluster: generate descriptions,
infer schema from samples, list indexes, and store the inventory as an array of
JSON objects in the same cluster.

Usage:
    Set MONGODB_URI (or pass --uri), then:
    python scripts/cluster_inventory.py [--output-file cluster_inventory.json] [--target-db cluster_metadata] [--target-collection cluster_inventory]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

try:
    from pymongo import MongoClient
    from pymongo.database import Database
    from pymongo.collection import Collection
except ImportError:
    print("Run: pip install pymongo", file=sys.stderr)
    sys.exit(1)


# System databases to skip (optional; set to empty to include all)
SKIP_DATABASES = {"local"}  # often huge; include admin/config if you want them
# Collection name prefixes to skip (e.g. system collections)
SKIP_COLLECTION_PREFIXES = ("system.", "enxcol_.", "enxcol_.*.esc", "enxcol_.*.ecoc")


def _should_skip_collection(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in ("system.", "enxcol_."))


def _infer_schema_summary(doc: dict, max_depth: int = 2) -> dict:
    """Build a shallow schema summary from a single document."""
    if not doc:
        return {}
    out = {}
    for k, v in doc.items():
        if k.startswith("_"):
            continue
        if max_depth <= 0:
            out[k] = type(v).__name__
            continue
        if isinstance(v, dict):
            out[k] = _infer_schema_summary(v, max_depth - 1)
        elif isinstance(v, list):
            if v and isinstance(v[0], dict):
                out[k] = ["array", _infer_schema_summary(v[0], max_depth - 1)]
            else:
                out[k] = "array"
        else:
            out[k] = type(v).__name__
    return out


def _describe_collection(db_name: str, coll_name: str, sample: dict | None) -> str:
    """Generate a short human-readable description from db/coll name and optional sample."""
    name = f"{db_name}.{coll_name}"
    if sample:
        keys = [k for k in sample.keys() if not k.startswith("_")]
        if keys:
            return f"Collection {name} with top-level fields: {', '.join(keys[:15])}{'...' if len(keys) > 15 else ''}"
    return f"Collection {name}"


def get_collection_indexes(coll: Collection) -> list[dict]:
    """Return list of index specs (name + key) for a collection. Views have no indexes."""
    try:
        indexes = list(coll.list_indexes())
    except Exception:
        return []
    return [
        {"name": idx.get("name"), "key": idx.get("key", {})}
        for idx in indexes
    ]


def build_inventory(client: MongoClient) -> list[dict]:
    """Build full cluster inventory: databases, collections, descriptions, schema, indexes."""
    inventory = []
    db_names = [d for d in client.list_database_names() if d not in SKIP_DATABASES]

    for db_name in db_names:
        db: Database = client[db_name]
        try:
            coll_infos = list(db.list_collections())
        except Exception as e:
            inventory.append({
                "database": db_name,
                "collection": None,
                "description": f"Error listing collections: {e}",
                "schemaSummary": None,
                "indexes": [],
                "isView": False,
                "error": str(e),
            })
            continue

        for info in coll_infos:
            coll_name = info.get("name", "")
            if _should_skip_collection(coll_name):
                continue
            is_view = info.get("type") == "view"
            coll = db[coll_name]

            indexes: list[dict] = []
            if not is_view:
                indexes = get_collection_indexes(coll)

            sample: dict | None = None
            try:
                sample = coll.find_one()
            except Exception:
                pass

            schema_summary = _infer_schema_summary(sample) if sample else None
            description = _describe_collection(db_name, coll_name, sample)

            inventory.append({
                "database": db_name,
                "collection": coll_name,
                "description": description,
                "schemaSummary": schema_summary,
                "indexes": indexes,
                "isView": is_view,
                "inventoryGeneratedAt": datetime.utcnow().isoformat() + "Z",
            })

    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Map MongoDB cluster and store inventory")
    parser.add_argument("--uri", default=None, help="MongoDB connection URI (default: MONGODB_URI env)")
    parser.add_argument("--output-file", "-o", default=None, help="Write inventory array to this JSON file")
    parser.add_argument("--target-db", default="cluster_metadata", help="Database to store inventory in")
    parser.add_argument("--target-collection", default="cluster_inventory", help="Collection name for inventory docs")
    parser.add_argument("--no-insert", action="store_true", help="Do not insert into cluster; only write file if -o given")
    parser.add_argument("--one-doc-per-collection", action="store_true", help="Insert one document per collection (default: one doc with 'inventory' array)")
    args = parser.parse_args()

    uri = args.uri or __import__("os").environ.get("MONGODB_URI")
    if not uri:
        print("Set MONGODB_URI or pass --uri", file=sys.stderr)
        sys.exit(1)

    client = MongoClient(uri)
    try:
        client.admin.command("ping")
    except Exception as e:
        print(f"Cannot connect: {e}", file=sys.stderr)
        sys.exit(1)

    inventory = build_inventory(client)
    print(f"Collected {len(inventory)} collection entries", file=sys.stderr)

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(inventory, f, indent=2, default=str)
        print(f"Wrote {args.output_file}", file=sys.stderr)

    if not args.no_insert and args.target_db and args.target_collection:
        target = client[args.target_db][args.target_collection]
        target.delete_many({})
        if args.one_doc_per_collection:
            target.insert_many(inventory)
            print(f"Inserted {len(inventory)} documents into {args.target_db}.{args.target_collection}", file=sys.stderr)
        else:
            target.insert_one({
                "inventory": inventory,
                "generatedAt": datetime.utcnow(),
                "count": len(inventory),
            })
            print(f"Inserted 1 document into {args.target_db}.{args.target_collection}", file=sys.stderr)

    # Print array to stdout for piping
    print(json.dumps(inventory, default=str))


if __name__ == "__main__":
    main()
