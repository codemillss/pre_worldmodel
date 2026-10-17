#!/usr/bin/env python3
"""Read-only P5 standalone PDF delivery audit; never create or render a PDF.

Use bundled document Python with pypdf and -B. Six current Markdown chapters,
their immutable export manifest, and every already rendered page PNG are
explicit inputs. This automatic audit does not substitute for direct visual QA.
Only directly bound files are SHA-streamed; linked evidence graphs are not read.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import unicodedata
from urllib.parse import unquote

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "p5-pdf-delivery-audit-v2"
EXPORT_SCHEMA = "p5-standalone-final-pdf-export-v1"
EXPORT_STATUS = "PASS_TEXT_AND_INTERNAL_LINK_CHECKS_VISUAL_REVIEW_REQUIRED"
MAX_JSON_BYTES = 16 * 1024**2
MAX_OUTPUT_BYTES = 4 * 1024**2
SHA = re.compile(r"[0-9a-f]{64}\Z")
PENDING = re.compile(r"FINAL_PENDING\w*|\b(?:TODO|TBD|TBC|FIXME)\b|\[\s*작성\s*예정\s*\]", re.I)
UNFINISHED_TITLE = re.compile(r"초안|작성\s*중|미완성")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
RENDER_EQUIVALENCE_SCHEMA = "p5-final-PDF-fresh-render-exact-reuse-v1"
RENDER_EQUIVALENCE_STATUS = "ALL35_FRESHLY_RENDERED_REUSED_ONLY_IF_EXACT"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def normal_chars(text):
    return str(text).replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", " - ")


def normal(text):
    return unicodedata.normalize("NFC", normal_chars(text))


def compact(text):
    return re.sub(r"\s+", "", normal(text))


def heading_text(text):
    # The exporter hashes the original Unicode heading bytes, without NFC.
    return normal_chars(re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text).replace("**", "").replace("`", ""))


def read_json(path):
    need(path.stat().st_size <= MAX_JSON_BYTES, "Export manifest exceeds16MiB")
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "Duplicate manifest key: " + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("Nonfinite manifest value: " + value)
    result = json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=nonfinite)
    def finite(value, depth=0):
        need(depth <= 64, "Manifest nesting limit exceeded")
        if type(value) is float:
            need(math.isfinite(value), "Nonfinite manifest float")
        elif isinstance(value, dict):
            for child in value.values(): finite(child, depth + 1)
        elif isinstance(value, list):
            for child in value: finite(child, depth + 1)
    finite(result)
    return result


class Inputs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.bindings = {}

    def path(self, value, *, external=False, directory=False):
        raw = Path(value)
        need(not raw.is_symlink() and ".." not in raw.parts, "Input symlink/traversal forbidden: " + str(value))
        path = (raw if raw.is_absolute() else self.root / raw).resolve()
        need(external or path.is_relative_to(self.root), "Input escapes project: " + str(value))
        need(path.is_dir() if directory else path.is_file(), "Missing input: " + str(value))
        return path

    def bind(self, value, expected=None, *, external=False):
        path = self.path(value, external=external)
        if expected is not None:
            need(type(expected) is str and SHA.fullmatch(expected), "Explicit SHA256 required: " + str(path))
        current = digest(path)
        need(expected is None or current == expected, "Input SHA256 differs: " + str(path))
        need(str(path) not in self.bindings or self.bindings[str(path)] == current,
             "Input changed during audit: " + str(path))
        self.bindings[str(path)] = current
        return path

    def recheck(self):
        for path, expected in self.bindings.items():
            need(digest(path) == expected, "Input changed during audit: " + path)


def manuscript(path):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    need(bool(lines) and lines[0].startswith("# "), "Markdown first-line title required: " + str(path))
    need(not PENDING.search(text), "Unfinished marker in Markdown: " + str(path))
    title = lines[0].lstrip("# ").strip()
    need(title and not UNFINISHED_TITLE.search(title), "Unfinished/empty chapter title: " + str(path))
    sections, images, code = [], [], False
    for index, line in enumerate(lines[1:], 2):
        if line.startswith("```"):
            code = not code
            continue
        if code:
            continue
        match = re.match(r"^##\s+(.+)", line)
        if match:
            label = heading_text(match[1])
            need(not UNFINISHED_TITLE.search(label), "Unfinished section title: " + str(path))
            sections.append((index, label))
        image = re.fullmatch(r"!\[[^\]]*\]\(([^)]+)\)", line.strip())
        if image:
            images.append((path.parent / unquote(image[1].split("#", 1)[0])).resolve())
    need(not code, "Unclosed Markdown code fence")
    return {"text": text, "title": title, "sections": sections, "images": images}


def png_info(path):
    need(path.suffix.lower() == ".png", "Rendered page must bePNG")
    with path.open("rb") as stream:
        header = stream.read(33)
        stream.seek(-12, 2)
        tail = stream.read(12)
    need(len(header) == 33 and header[:8] == PNG_SIGNATURE and header[12:16] == b"IHDR",
         "Invalid PNG header: " + str(path))
    width, height = struct.unpack(">II", header[16:24])
    need(width > 0 and height > 0 and tail[4:8] == b"IEND", "Invalid/incomplete PNG: " + str(path))
    match = re.fullmatch(r"(.*?)([0-9]+)", path.stem)
    need(match is not None, "PNG filename must end in its one-based page number")
    return {"path": str(path), "page": int(match[2]), "prefix": match[1],
            "width_px": width, "height_px": height, "bytes": path.stat().st_size}


def destination_page(reader, destination, page_ids):
    if isinstance(destination, str):
        named = reader.named_destinations.get(destination)
        return None if named is None else reader.get_destination_page_number(named) + 1
    if isinstance(destination, (list, tuple)) and destination:
        return page_ids.get(getattr(destination[0], "idnum", None))
    return None


def inspect_pdf(reader, record, chapters, documents, brief):
    pages = list(reader.pages)
    need(bool(pages) and len(pages) == record["page_count"], "PDF page count differs")
    texts = [page.extract_text() or "" for page in pages]
    need(not PENDING.search("\n".join(texts)), "Unfinished marker in extracted PDF text")
    page_ids = {page.indirect_reference.idnum: index for index, page in enumerate(pages, 1)}
    need(len(page_ids) == len(pages), "Duplicate PDF page identity")
    outline, outline_levels = [], []
    def walk(items, depth=0):
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
            else:
                number = reader.get_destination_page_number(item)
                need(type(number) is int and 0 <= number < len(pages), "Invalid bookmark destination")
                outline.append((normal(item.title), number + 1))
                outline_levels.append((normal(item.title), number + 1, depth))
    walk(reader.outline)
    toc = [page for title, page in outline if title == "전체 목차"]
    catalog = [page for title, page in outline if title == "별도 원본 근거 경로"]
    need(len(toc) == len(catalog) == 1, "Unique TOC/catalog bookmarks required")
    starts = [chapter["page"] for chapter in chapters]
    need(all(type(page) is int and 1 <= page <= len(pages) for page in starts)
         and starts == sorted(set(starts)) and toc[0] < starts[0] and starts[-1] < catalog[0],
         "Chapter/TOC/catalog page order differs")
    targets, expected_sections, required_levels = [], {}, []
    for index, (chapter, document) in enumerate(zip(chapters, documents)):
        targets.append((normal(chapter["title"]), chapter["page"]))
        required_levels.append((normal(chapter["title"]), chapter["page"], 1))
        chapter_end = starts[index + 1] if index + 1 < len(starts) else catalog[0]
        for line, title in document["sections"]:
            key = "section_" + hashlib.sha256(f"{chapter['path']}:{line}:{title}".encode()).hexdigest()[:20]
            need(key in record["section_pages"], "Missing exported section anchor: " + key)
            page = record["section_pages"][key]
            need(type(page) is int and chapter["page"] <= page < chapter_end, "Section page outside its chapter")
            expected_sections[key] = page
            targets.append((normal(title), page))
            required_levels.append((normal(title), page, 2))
    need(set(expected_sections) == set(record["section_pages"]), "Unexpected/missing section anchors")
    need(all(Counter(outline_levels)[entry] >= count for entry, count in Counter(required_levels).items()),
         "Missing/misdirected H1/H2 bookmark hierarchy")
    counts = Counter(outline)
    for target, quantity in Counter(targets).items():
        need(counts[target] >= quantity, "Missing/misdirected heading bookmark: " + repr(target))
        need(compact(target[0]) in compact(texts[target[1] - 1]), "Heading absent from bookmark destination: " + target[0])
    toc_text = compact("\n".join(texts[toc[0] - 1:starts[0] - 1]))
    for title, _ in targets:
        need(compact(title) in toc_text, "Heading absent from TOC text: " + title)
    link_counts, toc_destinations = {"internal": 0, "external": 0}, set()
    for index, page in enumerate(pages, 1):
        for reference in page.get("/Annots", []):
            annotation = reference.get_object()
            if annotation.get("/Subtype") != "/Link":
                continue
            destination, action = annotation.get("/Dest"), annotation.get("/A")
            if destination is None and action and action.get("/S") == "/GoTo":
                destination = action.get("/D")
            if destination is not None:
                number = destination_page(reader, destination, page_ids)
                need(number is not None and 1 <= number <= len(pages), "Broken PDF internal link")
                link_counts["internal"] += 1
                if toc[0] <= index < starts[0]: toc_destinations.add(number)
            elif action and action.get("/S") == "/URI":
                need(bool(action.get("/URI")), "Empty external link")
                link_counts["external"] += 1
            else:
                raise ValueError("Unresolved/unsupported PDF link action")
    need(link_counts == record["automatic_checks"]["pdf_link_counts"], "PDF link counts differ from export manifest")
    need({page for _, page in targets} | {catalog[0]} <= toc_destinations,
         "TOC links do not cover every declared heading/catalog destination page")
    brief_index = next(index for index, ch in enumerate(chapters) if ch["path"] == brief)
    first = starts[brief_index]
    last = (starts[brief_index + 1] if brief_index + 1 < len(starts) else catalog[0]) - 1
    need(first == last, "Problem brief body occupies more than one PDF page")
    brief_text = compact(re.sub(r"\[\d{3}\]", "", texts[first - 1]))
    paragraphs = re.split(r"\n\s*\n", documents[brief_index]["text"])
    probes = [heading_text(p.strip()) for p in paragraphs if p.strip() and not p.lstrip().startswith("#")]
    need(bool(probes), "Problem brief has no body")
    for probe in probes:
        need(compact(probe) in brief_text, "Problem brief paragraph absent from its single page")
    return {"page_count": len(pages), "heading_bookmarks_checked": len(targets),
            "toc_pages": [toc[0], starts[0] - 1], "pdf_link_counts": link_counts,
            "brief_page": first, "brief_body_page_count": 1, "brief_paragraphs_checked": len(probes),
            "characters_per_page": [len(text) for text in texts],
            "toc_check_scope": "Declared heading titles/bookmark pages and TOC destination-page coverage; no geometric label-to-link or visual layout claim."}


def verify_render_equivalence(inputs, proof_path, pdf, png_paths, rendered, page_count):
    """Verify a saved fresh-render attestation; never run or replace its renderer."""
    proof_path = inputs.bind(proof_path)
    proof_sha = inputs.bindings[str(proof_path)]
    proof = read_json(proof_path)
    need(type(proof) is dict and proof.get("schema") == RENDER_EQUIVALENCE_SCHEMA
         and proof.get("status") == RENDER_EQUIVALENCE_STATUS,
         "Fresh-render equivalence schema/status differs")
    need(inputs.path(proof.get("pdf", "")) == pdf
         and proof.get("pdf_sha256") == inputs.bindings[str(pdf)]
         and type(proof.get("page_count")) is int and proof["page_count"] == page_count,
         "Fresh-render proof PDF identity/SHA/page count differs")
    pages = proof.get("pages")
    need(type(pages) is list and len(pages) == page_count,
         "Fresh-render proof must contain every PDF page")
    reused, new_pages, new_bytes = 0, [], 0
    for number, (entry, path, image) in enumerate(zip(pages, png_paths, rendered), 1):
        need(type(entry) is dict and type(entry.get("page")) is int
             and entry["page"] == number,
             "Fresh-render proof page numbers must be unique ordered1..N")
        need(inputs.path(entry.get("selected_png", "")) == path,
             "Fresh-render proof selected PNG order/path differs from CLI")
        selected, fresh = entry.get("selected_png_sha256"), entry.get("fresh_render_png_sha256")
        need(type(selected) is str and SHA.fullmatch(selected) and selected == fresh
             and selected == inputs.bindings[str(path)],
             "Fresh-render proof selected/fresh/current PNG SHA differs")
        need(type(entry.get("fresh_render_bytes")) is int
             and entry["fresh_render_bytes"] == image["bytes"],
             "Fresh-render proof PNG byte count differs")
        need(type(entry.get("reuse_exact")) is bool,
             "Fresh-render proof requires explicit reuse flag")
        if entry["reuse_exact"]:
            need(entry.get("reuse_reason") == "EXACT_PNG_BYTES",
                 "Only exact PNG byte reuse is supported")
            reused += 1
        else:
            need(entry.get("reuse_reason") is None,
                 "Fresh-render proof new PNG has unsupported reuse reason")
            new_pages.append(number)
            new_bytes += image["bytes"]
    need(type(proof.get("reused_page_count")) is int and proof["reused_page_count"] == reused
         and proof.get("new_page_numbers") == new_pages
         and type(proof.get("new_png_bytes")) is int and proof["new_png_bytes"] == new_bytes,
         "Fresh-render proof reuse/new-page summary differs")
    return {"path": str(proof_path), "sha256_before": proof_sha, "sha256_after": None,
            "schema": proof["schema"], "status": proof["status"], "verified_pages": page_count,
            "reused_page_count": reused, "new_page_numbers": new_pages,
            "comparison": "EXACT_PNG_BYTES_FOR_EVERY_SELECTED_PAGE",
            "scope": "Saved renderer attestation and selected-file exact hashes verified; no fresh rendering or direct visual review by this auditor."}


def audit_delivery(pdf, manifest, sources, render_pngs, *, brief, project_root=None,
                   render_equivalence=None):
    result = {"schema": SCHEMA, "status": "FAIL_PDF_DELIVERY_AUDIT", "accepted": False,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "checks": {}, "errors": [],
        "direct_visual_review": "NOT_ASSESSED_BY_THIS_TOOL", "pdf_files_created": 0, "render_calls": 0}
    inputs = Inputs(ROOT if project_root is None else project_root)
    try:
        generator = Path(__file__).resolve()
        result["generator"] = {"path": str(generator), "sha256_before": digest(generator), "sha256_after": None}
        pdf = inputs.bind(pdf)
        manifest = inputs.bind(manifest)
        need(pdf.suffix.lower() == ".pdf" and manifest == pdf.with_suffix(".manifest.json"), "PDF/export manifest pairing differs")
        record = read_json(manifest)
        need(record.get("schema") == EXPORT_SCHEMA and record.get("status") == EXPORT_STATUS
             and record.get("all_file_sources_unchanged") is True, "Export did not pass its declared automatic checks")
        need(record.get("output") == pdf.relative_to(inputs.root).as_posix()
             and record.get("pdf_sha256") == inputs.bindings[str(pdf)]
             and record.get("pdf_bytes") == pdf.stat().st_size, "PDF identity/bytes/SHA differs")
        before, after = record.get("source_sha256_before"), record.get("source_sha256_after")
        need(type(before) is dict and before and before == after, "Export before/after source identities differ")
        need(len(before) <= 4096, "Too many directly bound source files")
        for path, expected in before.items(): inputs.bind(path, expected)
        sources = [inputs.bind(path) for path in sources]
        need(len(sources) == len(set(sources)) == 6, "Exactly six distinct current Markdown chapters required")
        need(all(path.suffix.lower() == ".md" for path in sources), "Markdown sources required")
        brief_path = inputs.path(brief)
        need(brief_path in sources, "Explicit problem brief must be one of six sources")
        brief_relative = brief_path.relative_to(inputs.root).as_posix()
        chapters = record.get("chapters")
        need(type(chapters) is list and len(chapters) == record.get("chapter_count") == 6, "Six exported chapters required")
        documents = [manuscript(path) for path in sources]
        for path, chapter, document in zip(sources, chapters, documents):
            relative = path.relative_to(inputs.root).as_posix()
            anchor = "doc_" + hashlib.sha256(relative.encode()).hexdigest()[:16]
            need(chapter.get("path") == relative and chapter.get("title") == document["title"]
                 and chapter.get("anchor") == anchor and chapter.get("sha256") == before.get(relative)
                 == inputs.bindings[str(path)], "Latest Markdown chapter/order/SHA differs: " + relative)
        result["checks"]["six_current_markdown_sources"] = True
        references = record.get("references")
        need(type(references) is list, "Explicit reference catalog required")
        for ref in references:
            if ref.get("kind") == "directory":
                need(ref.get("sha256") is None, "Directory reference cannot claim file SHA")
                inputs.path(ref["path"], directory=True)
            else:
                need(ref.get("kind") == "file" and before.get(ref["path"]) == ref.get("sha256"),
                     "Reference not bound to exported source identity")
                inputs.bind(ref["path"], ref["sha256"])
        for document in documents:
            for image in document["images"]:
                relative = image.relative_to(inputs.root).as_posix()
                need(relative in before, "Markdown image lacks export source binding")
                inputs.bind(image, before[relative])
        font = Path(record["font"])
        need(font.is_absolute() and record.get("font_sha256_before") == record.get("font_sha256_after"),
             "Selected font before/after SHA differs")
        inputs.bind(font, record["font_sha256_before"], external=True)
        result["checks"]["direct_file_reference_image_font_hashes"] = True
        pdf_checks = inspect_pdf(PdfReader(str(pdf)), record, chapters, documents, brief_relative)
        result["pdf_checks"] = pdf_checks
        result["checks"]["pdf_text_bookmarks_toc_links_and_brief_page"] = True
        png_paths = [inputs.bind(path) for path in render_pngs]
        need(len(png_paths) == len(set(png_paths)) == pdf_checks["page_count"], "One unique PNG per PDF page required")
        rendered = [png_info(path) for path in png_paths]
        need([item["page"] for item in rendered] == list(range(1, len(rendered) + 1))
             and len({item["prefix"] for item in rendered}) == 1
             and (render_equivalence is not None or len({path.parent for path in png_paths}) == 1),
             "Rendered PNG list is not one continuous ordered page1..N series")
        if render_equivalence is not None:
            result["render_equivalence"] = verify_render_equivalence(
                inputs, render_equivalence, pdf, png_paths, rendered, pdf_checks["page_count"])
            result["checks"]["saved_fresh_render_exact_equivalence_proof"] = True
        for item in rendered: item["sha256"] = inputs.bindings[item["path"]]
        result["rendered_pages"] = rendered
        result["checks"]["all_pages_continuous_png_inventory"] = True
        inputs.recheck()
        if "render_equivalence" in result:
            result["render_equivalence"]["sha256_after"] = digest(result["render_equivalence"]["path"])
            need(result["render_equivalence"]["sha256_after"] == result["render_equivalence"]["sha256_before"],
                 "Fresh-render equivalence proof changed during audit")
        result.update(status="PASS_AUTOMATIC_PDF_DELIVERY_CHECKS_VISUAL_REVIEW_REQUIRED", accepted=True,
            pdf_sha256=inputs.bindings[str(pdf)], export_manifest_sha256=inputs.bindings[str(manifest)],
            page_count=pdf_checks["page_count"], chapter_count=6)
    except Exception as error:
        result["errors"].append({"type": type(error).__name__, "message": str(error)})
    if "generator" in result:
        try:
            result["generator"]["sha256_after"] = digest(result["generator"]["path"])
            need(result["generator"]["sha256_before"] == result["generator"]["sha256_after"],
                 "Delivery auditor source changed during audit")
        except Exception as error:
            result.update(status="FAIL_PDF_DELIVERY_AUDIT", accepted=False)
            result["errors"].append({"type": type(error).__name__, "message": str(error)})
    result["input_bindings_sha256"] = inputs.bindings
    result["scope"] = ["Read-only pypdf text/link/bookmark and direct input hash audit; no PDF editing, generation or rendering.",
        "PNG signature/dimensions/hash inventory is not a check that its pixels render this PDF or that layout is readable.",
        "Direct visual review and the shared report/artifact budget remain separate requirements.",
        "Source/reference files are hashed directly; linked raw evidence graphs are not recursively opened.",
        "NOT_ASSESSED/NOT_ESTABLISHED and honest scientific limitations are not unfinished placeholders."]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", action="append", nargs="+", type=Path, required=True)
    parser.add_argument("--brief", type=Path, required=True)
    parser.add_argument("--render-png", action="append", nargs="+", type=Path, required=True)
    parser.add_argument("--render-equivalence", type=Path,
                        help="Optional saved fresh-render exact-PNG proof required for mixed PNG directories")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.out.resolve()
    need(not args.out.is_symlink() and not output.exists() and output.suffix == ".json"
         and output.is_relative_to(ROOT.resolve())
         and output.relative_to(ROOT.resolve()).parts[0] in {"reports", "tmp"}, "New report/tmp JSON output required")
    result = audit_delivery(args.pdf, args.manifest, [p for group in args.source for p in group],
        [p for group in args.render_png for p in group], brief=args.brief,
        render_equivalence=args.render_equivalence)
    data = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n"
    need(len(data) <= MAX_OUTPUT_BYTES, "Audit output exceeds4MiB")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": result["status"], "accepted": result["accepted"], "out": str(output)}, ensure_ascii=False))
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
