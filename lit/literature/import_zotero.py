#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


CORE_TYPES = {"attachment", "note", "annotation"}
FIELDS = ["title", "abstractNote", "date", "DOI", "url", "citationKey", "extra", "publicationTitle", "conferenceName", "proceedingsTitle", "repository"]


def main() -> int:
    args = parse_args()
    db = Path(args.db).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(f"file:{quote(str(db))}?mode=ro&nolock=1", uri=True)
    con.row_factory = sqlite3.Row
    records = export_records(con, include_deleted=args.include_deleted)
    standalone_notes = export_standalone_notes(con, include_deleted=args.include_deleted)

    write_jsonl(out / "items.jsonl", records)
    write_jsonl(out / "standalone_notes.jsonl", standalone_notes)
    write_markdown(out / "notes.md", records, standalone_notes)
    write_index(out / "index.md", db, records, standalone_notes)

    print(json.dumps({"items": len(records), "standalone_notes": len(standalone_notes), "out": str(out)}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Zotero items, notes, and annotations into literature/zotero.")
    parser.add_argument("--db", default="~/Zotero/zotero.sqlite", help="Path to Zotero sqlite database.")
    parser.add_argument("--out", default="literature/zotero", help="Output directory.")
    parser.add_argument("--include-deleted", action="store_true", help="Include items present in Zotero's deletedItems table.")
    return parser.parse_args()


def export_records(con: sqlite3.Connection, include_deleted: bool = False) -> list[dict]:
    deleted = deleted_ids(con) if not include_deleted else set()
    item_types = {row["itemID"]: row["typeName"] for row in con.execute("select i.itemID, t.typeName from items i join itemTypes t on t.itemTypeID=i.itemTypeID")}
    fields = item_fields(con)
    creators = item_creators(con)
    tags = item_tags(con)
    collections = item_collections(con)
    notes = child_notes(con)
    attachments = child_attachments(con)
    annotations = attachment_annotations(con)

    records = []
    for item_id, item_type in sorted(item_types.items()):
        if item_id in deleted or item_type in CORE_TYPES:
            continue
        item_attachments = attachments.get(item_id, [])
        for att in item_attachments:
            att["annotations"] = annotations.get(att["itemID"], [])
        title = fields.get(item_id, {}).get("title") or f"Untitled {item_type} {item_id}"
        records.append(
            {
                "item_id": item_id,
                "key": key_for(con, item_id),
                "type": item_type,
                "title": title,
                "fields": {k: v for k, v in fields.get(item_id, {}).items() if k in FIELDS or k not in {"title"}},
                "creators": creators.get(item_id, []),
                "tags": tags.get(item_id, []),
                "collections": collections.get(item_id, []),
                "notes": notes.get(item_id, []),
                "attachments": item_attachments,
            }
        )
    return records


def export_standalone_notes(con: sqlite3.Connection, include_deleted: bool = False) -> list[dict]:
    deleted = deleted_ids(con) if not include_deleted else set()
    rows = con.execute(
        """
        select n.itemID, i.key, n.title, n.note
        from itemNotes n
        join items i on i.itemID=n.itemID
        where n.parentItemID is null
        order by n.itemID
        """
    )
    return [{"item_id": r["itemID"], "key": r["key"], "title": r["title"], "note_html": r["note"], "note_text": html_to_text(r["note"])} for r in rows if r["itemID"] not in deleted]


def item_fields(con: sqlite3.Connection) -> dict[int, dict[str, str]]:
    rows = con.execute(
        """
        select d.itemID, f.fieldName, v.value
        from itemData d
        join fields f on f.fieldID=d.fieldID
        join itemDataValues v on v.valueID=d.valueID
        """
    )
    out: dict[int, dict[str, str]] = defaultdict(dict)
    for row in rows:
        out[row["itemID"]][row["fieldName"]] = row["value"]
    return out


def item_creators(con: sqlite3.Connection) -> dict[int, list[dict]]:
    rows = con.execute(
        """
        select ic.itemID, ct.creatorType, c.firstName, c.lastName, c.fieldMode, ic.orderIndex
        from itemCreators ic
        join creators c on c.creatorID=ic.creatorID
        join creatorTypes ct on ct.creatorTypeID=ic.creatorTypeID
        order by ic.itemID, ic.orderIndex
        """
    )
    out: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        name = r["lastName"] if r["fieldMode"] else " ".join(x for x in [r["firstName"], r["lastName"]] if x)
        out[r["itemID"]].append({"type": r["creatorType"], "name": name})
    return out


def item_tags(con: sqlite3.Connection) -> dict[int, list[str]]:
    rows = con.execute("select it.itemID, t.name from itemTags it join tags t on t.tagID=it.tagID order by t.name")
    out: dict[int, list[str]] = defaultdict(list)
    for r in rows:
        out[r["itemID"]].append(r["name"])
    return out


def item_collections(con: sqlite3.Connection) -> dict[int, list[str]]:
    rows = con.execute("select ci.itemID, c.collectionName from collectionItems ci join collections c on c.collectionID=ci.collectionID order by c.collectionName")
    out: dict[int, list[str]] = defaultdict(list)
    for r in rows:
        out[r["itemID"]].append(r["collectionName"])
    return out


def child_notes(con: sqlite3.Connection) -> dict[int, list[dict]]:
    rows = con.execute("select itemID, parentItemID, title, note from itemNotes where parentItemID is not null order by itemID")
    out: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["parentItemID"]].append({"item_id": r["itemID"], "title": r["title"], "note_html": r["note"], "note_text": html_to_text(r["note"])})
    return out


def child_attachments(con: sqlite3.Connection) -> dict[int, list[dict]]:
    rows = con.execute(
        """
        select a.itemID, a.parentItemID, a.contentType, a.path, i.key
        from itemAttachments a
        join items i on i.itemID=a.itemID
        where a.parentItemID is not null
        order by a.itemID
        """
    )
    out: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["parentItemID"]].append({"itemID": r["itemID"], "key": r["key"], "content_type": r["contentType"], "path": r["path"]})
    return out


def attachment_annotations(con: sqlite3.Connection) -> dict[int, list[dict]]:
    rows = con.execute("select itemID, parentItemID, type, text, comment, color, pageLabel from itemAnnotations order by parentItemID, sortIndex")
    out: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["parentItemID"]].append({"item_id": r["itemID"], "type": r["type"], "text": r["text"], "comment": r["comment"], "color": r["color"], "page": r["pageLabel"]})
    return out


def deleted_ids(con: sqlite3.Connection) -> set[int]:
    return {r["itemID"] for r in con.execute("select itemID from deletedItems")}


def key_for(con: sqlite3.Connection, item_id: int) -> str:
    row = con.execute("select key from items where itemID=?", (item_id,)).fetchone()
    return row["key"] if row else ""


def html_to_text(value: str | None) -> str:
    if not value:
        return ""
    text = html.unescape(value)
    text = re.sub(r"<br\\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(p|div|li|h[1-6]|ol|ul)>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(path: Path, records: list[dict], standalone_notes: list[dict]) -> None:
    lines = ["# Zotero Notes", ""]
    for r in records:
        lines.extend([f"## {r['title']}", "", f"- Type: `{r['type']}`", f"- Key: `{r['key']}`"])
        if r["creators"]:
            lines.append("- Creators: " + ", ".join(c["name"] for c in r["creators"]))
        if r["fields"].get("url"):
            lines.append(f"- URL: {r['fields']['url']}")
        lines.append("")
        for note in r["notes"]:
            lines.extend([f"### Note: {note.get('title') or note['item_id']}", "", note["note_text"], ""])
        for attachment in r["attachments"]:
            for annotation in attachment.get("annotations", []):
                text = annotation.get("text") or ""
                comment = annotation.get("comment") or ""
                if text or comment:
                    lines.extend([f"> {text}", "", comment, ""])
    if standalone_notes:
        lines.extend(["# Standalone Notes", ""])
        for note in standalone_notes:
            lines.extend([f"## {note.get('title') or note['key']}", "", note["note_text"], ""])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_index(path: Path, db: Path, records: list[dict], standalone_notes: list[dict]) -> None:
    lines = [
        "# Zotero Literature Export",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"Source DB: `{db}`",
        f"Items: {len(records)}",
        f"Standalone notes: {len(standalone_notes)}",
        "",
        "Files:",
        "- `items.jsonl`: structured items with fields, creators, tags, collections, child notes, attachments, and annotations.",
        "- `standalone_notes.jsonl`: Zotero notes without a parent item.",
        "- `notes.md`: readable note and annotation dump.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
