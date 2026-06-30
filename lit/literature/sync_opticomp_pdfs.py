#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


@dataclass(frozen=True)
class ZoteroPdf:
    key: str
    filename: str
    source: Path
    paper_key: str


def main() -> int:
    args = parse_args()
    db = Path(args.db).expanduser()
    zotero_dir = Path(args.zotero_dir).expanduser()
    literature_dir = Path(args.literature_dir)
    literature_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(f"file:{quote(str(db))}?mode=ro&nolock=1", uri=True)
    con.row_factory = sqlite3.Row
    try:
        zotero_pdfs = collection_pdfs(con, zotero_dir, args.collection)
    finally:
        con.close()

    preferred = samsung_overrides(literature_dir)
    selected: dict[str, Path] = {}
    for pdf in zotero_pdfs:
        source = preferred.get(pdf.paper_key, pdf.source)
        filename = source.name if source.parent == literature_dir else pdf.filename
        selected[filename] = source

    for path in literature_dir.glob("*.pdf"):
        if path.name not in selected:
            path.unlink()

    for filename, source in selected.items():
        dest = literature_dir / filename
        if source.resolve() == dest.resolve():
            continue
        shutil.copy2(source, dest)

    print(f"synced {len(selected)} PDFs from {args.collection}")
    for filename, source in selected.items():
        kind = "preferred-local" if source.parent == literature_dir else "zotero"
        print(f"{kind}\t{filename}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync only the opticomp Zotero PDFs into literature/ with local annotated-PDF preference.")
    parser.add_argument("--db", default="~/Zotero/zotero.sqlite")
    parser.add_argument("--zotero-dir", default="~/Zotero")
    parser.add_argument("--literature-dir", default="literature")
    parser.add_argument("--collection", default="reasoning_opticomp")
    return parser.parse_args()


def collection_pdfs(con: sqlite3.Connection, zotero_dir: Path, collection: str) -> list[ZoteroPdf]:
    rows = con.execute(
        """
        with recursive selected_collections(collectionID) as (
          select collectionID from collections where collectionName=?
          union all
          select c.collectionID from collections c join selected_collections s on c.parentCollectionID=s.collectionID
        ), selected_items(itemID) as (
          select itemID from collectionItems where collectionID in (select collectionID from selected_collections)
        )
        select i.key as attachmentKey, substr(ia.path, 9) as filename
        from itemAttachments ia
        join items i on i.itemID=ia.itemID
        where ia.path like 'storage:%'
          and lower(ia.path) like '%.pdf'
          and (
            ia.parentItemID in (select itemID from selected_items)
            or ia.itemID in (select itemID from selected_items)
          )
        order by filename
        """,
        (collection,),
    )
    out: list[ZoteroPdf] = []
    for row in rows:
        source = zotero_dir / "storage" / row["attachmentKey"] / row["filename"]
        if source.is_file():
            out.append(ZoteroPdf(row["attachmentKey"], row["filename"], source, paper_key(row["filename"])))
    return out


def samsung_overrides(literature_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in literature_dir.glob("*.pdf"):
        if pdf_producer(path).casefold().find("samsung") == -1:
            continue
        out.setdefault(paper_key(path.name), path)
    return out


def pdf_producer(path: Path) -> str:
    try:
        result = subprocess.run(["pdfinfo", str(path)], check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return ""
    text = result.stdout + result.stderr
    return "\n".join(line for line in text.splitlines() if line.startswith(("Creator:", "Producer:")))


def paper_key(filename: str) -> str:
    stem = Path(filename).stem.casefold()
    stem = re.sub(r"\b(19|20)\d{2}\b", " ", stem)
    stem = re.sub(r"[^a-z0-9]+", " ", stem)
    return " ".join(stem.split())


if __name__ == "__main__":
    raise SystemExit(main())
