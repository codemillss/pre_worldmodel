#!/usr/bin/env python3
"""Export explicitly supplied final Markdown files to a new standalone PDF.

Layout v2 preserves the11pt/18pt body and original manuscripts, reducing only
body paragraph trailing space10pt to8pt after a two-line spill page in v1.
Cover scope prose is shorter; source evidence and scientific text are unchanged.

Use the bundled document Python. This imports only the existing report layout
helpers, never its board discovery or make_pdf. No physics/model code is used.
PDF and adjacent manifest must both be new files. Rendering/direct visual QA
remain separate, required work after this tool's automatic checks.
Chapter/H2 headings do not accept links; put evidence links in the following
body paragraph. Body links to included H1/H2 fragments are supported.

Example (run the PDF skill's operation marker immediately before creation):
  python tools/export_p5_final_pdf_v1.py --source reports/p5_final.v1.md \
    reports/p5_03_final.v1.md --output output/pdf/p5_final.v1.pdf --as-of 2026-10-17
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import html
import importlib
import json
import os
from pathlib import Path
import re
import sys
import unicodedata
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
layout = importlib.import_module("export_report_pdf")

TITLE = "부품 인계 자세 정렬 - P5 최종 비교 보고서"
GROUP = "최종 비교와 단계별 증거"
SCHEMA = "p5-standalone-final-pdf-export-v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside_root(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("Project-local source/output required: " + str(path))
    return resolved


def slug(text):
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"\s+", "-", re.sub(r"[^\w\s-]", "", text).strip())


def compact_text(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", layout.normal_chars(text)))


def heading_text(text):
    """Read a heading label without resolving links before the anchor map exists."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "").replace("`", "")
    return layout.normal_chars(text)


def validate_inputs(sources, output, as_of):
    """Fail before opening any output if the explicit final manuscript is absent."""
    datetime.strptime(as_of, "%Y-%m-%d")
    output_arg = Path(output)
    if output_arg.is_symlink():
        raise ValueError("Output symlink is forbidden")
    output = inside_root(output_arg)
    if output.suffix.lower() != ".pdf":
        raise ValueError("A new .pdf output path is required")
    manifest = output.with_suffix(".manifest.json")
    if output.exists() or manifest.exists() or manifest.is_symlink():
        raise FileExistsError("Keep previous PDF/manifest; choose a new version: " + str(output))
    if not sources:
        raise ValueError("Explicit final Markdown --source files are required; no historical fallback")
    chapters = []
    for value in sources:
        path_arg = Path(value)
        if path_arg.is_symlink():
            raise ValueError("Source symlink is forbidden: " + str(value))
        path = inside_root(path_arg)
        if not path.is_file():
            raise FileNotFoundError("Final Markdown source does not exist: " + str(path))
        if path.suffix.lower() != ".md":
            raise ValueError("Markdown final source required: " + str(path))
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        body = [line.strip() for line in lines[1:]
                if line.strip() and not re.match(r"^#{1,6}\s", line.strip())
                and not line.strip().startswith(("```", "<!--"))
                and line.strip() not in ("---", "***", "___")]
        if not lines or not lines[0].startswith("# ") or not lines[0][2:].strip() or not body:
            raise ValueError("Final source requires first-line '# title' and nonempty body: " + str(path))
        if any(re.match(r"^#{1,2}\s", line) and re.search(r"\[[^\]]+\]\([^)]+\)", line) for line in lines):
            raise ValueError("Move links out of chapter/H2 headings into the following paragraph: " + str(path))
        if re.search(r"^\s*```\s*mermaid\b", text, re.M):
            raise ValueError("Mermaid is not rendered by this exporter; supply an explicit static image: " + str(path))
        if any(line.lstrip().startswith("```") and not line.startswith("```") for line in lines):
            raise ValueError("Code fences must start at column zero for this renderer: " + str(path))
        if len(re.findall(r"^\s*```", text, re.M)) % 2:
            raise ValueError("Unclosed code fence: " + str(path))
        if path in [chapter.path for chapter in chapters]:
            raise ValueError("Duplicate Markdown source: " + str(path))
        chapters.append(layout.Chapter(path, GROUP))
    if output in [chapter.path for chapter in chapters] or manifest in [chapter.path for chapter in chapters]:
        raise ValueError("Output cannot alias an input")
    return chapters, output, manifest


class FinalRenderer(layout.Renderer):
    """Reuse Markdown/table layout; add exact references and section anchors."""

    def __init__(self, chapters, output):
        super().__init__(chapters, output)
        self.file_sources = {}
        self.directory_references = set()
        self.section_map = {}
        self.fragment_map = {}
        self.text_probes = []
        for chapter in chapters:
            self.bind_file(chapter.path)
            aliases, headings, counts = {}, [], {}
            # ReportDoc supplies the file's first title, Renderer skips that line.
            title = layout.normal_chars(chapter.title)
            aliases[slug(title)] = chapter.anchor
            self.text_probes.append({"source": chapter.relative, "kind": "chapter_title", "text": title})
            in_code = False
            for index, line in enumerate(chapter.path.read_text().splitlines()[1:], 2):
                if line.strip().startswith("```"):
                    in_code = not in_code
                if in_code:
                    continue
                match = re.match(r"^##\s+(.+)", line)
                if match:
                    title = heading_text(match[1])
                    base = slug(title)
                    occurrence = counts.get(base, 0)
                    counts[base] = occurrence + 1
                    key = base + ("-" + str(occurrence) if occurrence else "")
                    anchor = "section_" + hashlib.sha256(f"{chapter.relative}:{index}:{title}".encode()).hexdigest()[:20]
                    aliases[key] = anchor
                    headings.append((title, anchor))
                    self.text_probes.append({"source": chapter.relative, "kind": "section_title", "text": title})
            self.fragment_map[chapter.path] = aliases
            self.section_map[chapter.path] = headings

    def bind_file(self, path):
        path = inside_root(path)
        if not path.is_file():
            raise FileNotFoundError("Referenced source file is absent: " + str(path))
        digest = sha(path)
        relative = path.relative_to(ROOT).as_posix()
        if relative in self.file_sources and self.file_sources[relative] != digest:
            raise ValueError("Source changed while constructing report: " + relative)
        self.file_sources[relative] = digest
        return path, digest

    def link(self, label, target, source):
        if target.startswith(("https://", "http://", "mailto:")):
            self.link_intents.append({"kind": "web", "from": str(source.relative_to(ROOT)), "target": target})
            return f'<link href="{html.escape(target, quote=True)}" color="#166A8F">{self.inline(label, source)}</link>'
        raw_path, _, fragment = target.partition("#")
        source_line = None
        candidate = re.fullmatch(r"(.+):([1-9]\d*(?:-[1-9]\d*)?)", unquote(raw_path))
        if candidate and (source.parent / candidate[1]).is_file():
            raw_path, source_line = candidate[1], candidate[2]
        path = inside_root((source.parent / unquote(raw_path)) if raw_path else source)
        if path == self.output and not fragment:
            destination = "toc"
        elif not raw_path and fragment == "toc":
            destination = "toc"
        elif path in self.by_path:
            destination = self.by_path[path].anchor
            if fragment:
                destination = self.fragment_map.get(path, {}).get(unicodedata.normalize("NFC", unquote(fragment)))
                if destination is None:
                    raise ValueError("Unsupported/missing included Markdown fragment (H1/H2 only): " + target)
        else:
            if not path.exists():
                raise FileNotFoundError(f"{source}: {target}")
            relative = path.relative_to(ROOT).as_posix()
            if path.is_file():
                _, digest = self.bind_file(path)
            elif path.is_dir():
                digest = None
                self.directory_references.add(relative)
            else:
                raise ValueError("Regular file/directory evidence required: " + str(path))
            if relative not in self.references:
                self.references[relative] = {"id": len(self.references) + 1, "path": relative,
                    "sha256": digest, "kind": "file" if digest else "directory", "source_lines": []}
            if source_line and source_line not in self.references[relative]["source_lines"]:
                self.references[relative]["source_lines"].append(source_line)
            reference_id = self.references[relative]["id"]
            destination = f"ref_{reference_id:03d}"
            self.link_intents.append({"kind": "reference", "from": str(source.relative_to(ROOT)),
                                      "to": relative, "destination": destination, "source_fragment": fragment or None,
                                      "requested_source_line": source_line})
            return (f'<link href="#{destination}" color="#166A8F">{self.inline(label, source)}'
                    f'<super><font size="7.7"> [{reference_id:03d}]</font></super></link>')
        self.link_intents.append({"kind": "included", "from": str(source.relative_to(ROOT)),
                                  "to": str(path.relative_to(ROOT)), "destination": destination,
                                  "requested_source_line": source_line})
        return f'<link href="#{destination}" color="#166A8F">{self.inline(label, source)}</link>'

    def markdown(self, source):
        # The base image renderer does not register images in its evidence map.
        in_code = False
        for line in source.read_text().splitlines():
            if line.startswith("```"):
                in_code = not in_code
                continue
            match = None if in_code else re.fullmatch(r"!\[[^\]]*\]\(([^)]+)\)", line.strip())
            if match:
                self.bind_file(self.local_target(match[1], source))
        flow = super().markdown(source)
        headings = list(self.section_map[source])
        def mark(items):
            for item in items:
                if (headings and isinstance(item, layout.Paragraph)
                        and item.style.name in ("h2", "joined_heading")
                        and item.getPlainText() == headings[0][0]):
                    item._section_title, item._anchor = headings.pop(0)
                elif isinstance(item, layout.KeepTogether):
                    mark(item._content)
        mark(flow)
        if headings:
            raise ValueError("Could not attach section bookmarks: " + str(source))
        # A body phrase proves extraction includes more than the title/TOC.
        for item in flow:
            if isinstance(item, layout.Paragraph) and item.style.name in ("body", "list", "quote"):
                plain = item.getPlainText().strip()
                if plain:
                    self.text_probes.append({"source": str(source.relative_to(ROOT)), "kind": "body", "text": plain[:120]})
                    break
        return flow

    def verify_sources(self):
        actual = {path: sha(ROOT / path) if (ROOT / path).is_file() else None for path in self.file_sources}
        changed = [path for path, digest in self.file_sources.items() if actual[path] != digest]
        missing_directories = [path for path in self.directory_references if not (ROOT / path).is_dir()]
        if changed or missing_directories:
            raise ValueError("Report source changed: " + str(changed + missing_directories))
        return actual


class FinalDoc(layout.ReportDoc):
    def __init__(self, output, as_of):
        self.as_of = as_of
        self.section_pages = {}
        super().__init__(output)

    def beforeDocument(self):
        super().beforeDocument()
        self.section_pages = {}
        self.canv.setTitle(TITLE)
        self.canv.setAuthor("프로젝트 최종 비교 기록")

    def page_chrome(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(layout.RULE)
        canvas.line(layout.MARGIN, layout.PAGE_H - 35, layout.PAGE_W - layout.MARGIN, layout.PAGE_H - 35)
        canvas.setFont("ReportKR", 8.2)
        canvas.setFillColor(layout.MUTED)
        canvas.drawString(layout.MARGIN, layout.PAGE_H - 27, "부품 인계 자세 정렬 | P5 최종 비교")
        if doc.page > 1:
            canvas.setFillColor(layout.BLUE)
            canvas.drawRightString(layout.PAGE_W - layout.MARGIN, layout.PAGE_H - 27, "전체 목차")
            canvas.linkRect("", "toc", (layout.PAGE_W - layout.MARGIN - 46, layout.PAGE_H - 31,
                                       layout.PAGE_W - layout.MARGIN, layout.PAGE_H - 20), relative=0)
        canvas.setFillColor(layout.MUTED)
        canvas.setFont("ReportKR", 8)
        canvas.drawString(layout.MARGIN, 25, self.as_of + " 기준")
        canvas.drawRightString(layout.PAGE_W - layout.MARGIN, 25, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if hasattr(flowable, "_section_title"):
            self.canv.bookmarkPage(flowable._anchor)
            self.canv.addOutlineEntry(flowable._section_title, flowable._anchor, 2, False)
            self.notify("TOCEntry", (2, flowable._section_title, self.page, flowable._anchor))
            self.section_pages[flowable._anchor] = self.page
        else:
            super().afterFlowable(flowable)


def story_for(chapters, renderer):
    paragraph, styles = layout.Paragraph, layout.STYLES
    story = [layout.Spacer(1, 45), paragraph("부품 인계 자세 정렬", styles["h1"]),
             paragraph("P5 최종 비교 보고서", styles["h2"]),
             paragraph("문제·시스템·최종 비교·재현·주장 근거", styles["body"]),
             paragraph("최종 원고를 묶었다. 실험 성능·개인 이해·현업 효과는 각 원문의 판정 범위를 따른다.", styles["body"]),
             paragraph('파란 목차·근거 번호로 이동할 수 있다. PDF에 원시 데이터나 모델 파일을 첨부한 것은 아니다.', styles["body"]),
             paragraph('<link href="#toc" color="#166A8F">전체 목차 열기</link>', styles["nav"]),
             layout.PageBreak(), layout.annotated_title("전체 목차", "toc")]
    toc = layout.TableOfContents()
    toc.levelStyles = [layout.style("final_toc0", 12, 19, spaceBefore=12),
                       layout.style("final_toc1", 10.3, 16, leftIndent=12, spaceBefore=3, textColor=layout.BLUE),
                       layout.style("final_toc2", 9.5, 15, leftIndent=26, spaceBefore=0)]
    story.extend([toc, layout.PageBreak()])
    for chapter in chapters:
        story.append(layout.annotated_title(chapter.title, chapter.anchor, chapter))
        story.append(paragraph('<link href="#toc" color="#166A8F">전체 목차로 돌아가기</link>', styles["nav"]))
        story.extend(renderer.markdown(chapter.path))
        story.append(layout.PageBreak())
    story.extend([layout.annotated_title("별도 원본 근거 경로", "evidence_catalog"),
                  paragraph("아래 경로는 프로젝트 폴더 기준이다. 포함하지 않은 원문은 근거 번호로 연결하며 파일 SHA는 작성 전후 확인한다. 디렉터리 링크는 경로만 검증하고 내부 파일 전체를 해시하지 않는다.", styles["body"])])
    if not renderer.references:
        story.append(paragraph("별도 로컬 근거 링크 없음.", styles["body"]))
    for reference in renderer.references.values():
        heading = paragraph(f'근거 {reference["id"]:03d}', styles["h3"])
        heading._anchor = f'ref_{reference["id"]:03d}'
        story.extend([heading, paragraph(layout.safe_text(reference["path"]), styles["small"]),
                      paragraph("SHA256: " + (reference["sha256"] or "디렉터리 경로 - 내부 파일 미해시"), styles["small"])])
        if reference["source_lines"]:
            story.append(paragraph("원문 참조 줄: " + ", ".join(reference["source_lines"]), styles["small"]))
    return story


def check_pdf(path, renderer, chapters, doc):
    reader = layout.PdfReader(path)
    if not reader.pages:
        raise ValueError("Generated PDF has no pages")
    texts = [page.extract_text() or "" for page in reader.pages]
    text = compact_text("\n".join(texts))
    missing = [probe for probe in renderer.text_probes if compact_text(probe["text"]) not in text]
    if missing:
        raise ValueError("PDF text extraction lost source titles/body: " + str(missing[:5]))
    unexpected = [character for character in ("\ufffd", "\u25a0") if character in "\n".join(texts)
                  and not any(character in chapter.path.read_text() for chapter in chapters)]
    if unexpected:
        raise ValueError("Unexpected replacement/square characters in PDF text: " + repr(unexpected))
    page_ids = {page.indirect_reference.idnum: index + 1 for index, page in enumerate(reader.pages)}
    counts, targets, bad = {"internal": 0, "external": 0}, [], []
    for index, page in enumerate(reader.pages, 1):
        for indirect in page.get("/Annots", []):
            annotation = indirect.get_object()
            if annotation.get("/Subtype") != "/Link":
                continue
            destination, action = annotation.get("/Dest"), annotation.get("/A")
            if destination is None and action and action.get("/S") == "/GoTo":
                destination = action.get("/D")
            if destination is not None:
                counts["internal"] += 1
                try:
                    if isinstance(destination, str):
                        named = reader.named_destinations.get(destination)
                        target = reader.get_destination_page_number(named) + 1 if named else None
                    else:
                        target = page_ids.get(destination[0].idnum) if hasattr(destination[0], "idnum") else None
                except (IndexError, KeyError, TypeError, AttributeError):
                    target = None
                if target is None or not 1 <= target <= len(reader.pages):
                    bad.append({"from_page": index, "destination": str(destination)})
                else:
                    targets.append({"from_page": index, "to_page": target})
            elif action and action.get("/S") == "/URI":
                counts["external"] += 1
    if bad or not counts["internal"]:
        raise ValueError("Missing/broken internal PDF links: " + str(bad[:5]))
    outline_titles = []
    def outlines(items):
        for item in items:
            if isinstance(item, list):
                outlines(item)
            else:
                number = reader.get_destination_page_number(item)
                if number is None or not 0 <= number < len(reader.pages):
                    raise ValueError("Outline points outside PDF")
                outline_titles.append(str(item.title))
    outlines(reader.outline)
    required_titles = ["전체 목차", "별도 원본 근거 경로"] + [chapter.title for chapter in chapters]
    if any(title not in outline_titles for title in required_titles) or len(doc.chapter_pages) != len(chapters):
        raise ValueError("Missing chapter/TOC/catalog bookmarks")
    return {"status": "PASS_TEXT_AND_INTERNAL_LINK_CHECKS_VISUAL_REVIEW_REQUIRED",
            "page_count": len(reader.pages), "characters_per_page": [len(value) for value in texts],
            "text_probes": renderer.text_probes, "missing_text_probes": [],
            "unexpected_replacement_characters": [], "pdf_link_counts": counts,
            "resolved_internal_links": targets, "broken_internal_links": [],
            "outline_titles": outline_titles, "direct_visual_review": "NOT_YET_RECORDED"}


def export(sources, output, as_of):
    chapters, output, manifest_path = validate_inputs(sources, output, as_of)
    layout.STYLES["body"].spaceAfter = 8
    font_path, glyphs = layout.font_setup()
    font_before = sha(font_path)
    renderer = FinalRenderer(chapters, output)
    story = story_for(chapters, renderer)
    # Reused code and selected font are provenance too; no legacy file is edited.
    renderer.bind_file(__file__)
    renderer.bind_file(layout.__file__)
    missing_glyphs = sorted({character for chapter in chapters
                            for character in layout.normal_chars(chapter.path.read_text())
                            if not character.isspace() and ord(character) not in glyphs})
    if missing_glyphs:
        raise ValueError("Font lacks manuscript glyphs: " + repr(missing_glyphs))
    renderer.verify_sources()
    before = dict(renderer.file_sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    # ReportDoc normally coerces its filename to str. Replacing only the instance
    # filename with the exclusively opened stream prevents ReportLab's wb reopen.
    with ExitStack() as stack:
        pdf_stream = stack.enter_context(output.open("xb"))
        manifest_stream = stack.enter_context(manifest_path.open("x", encoding="utf-8"))
        try:
            doc = FinalDoc(output, as_of)
            doc.filename = pdf_stream
            passes = doc.multiBuild(story)
            pdf_stream.flush()
            os.fsync(pdf_stream.fileno())
            renderer.verify_sources()
            automatic = check_pdf(output, renderer, chapters, doc)
            after = renderer.verify_sources()
            if sha(font_path) != font_before:
                raise ValueError("Selected font changed during export")
            record = {"schema": SCHEMA, "status": automatic["status"],
                "exported_at_utc": datetime.now(timezone.utc).isoformat(), "report_as_of_date": as_of,
                "output": output.relative_to(ROOT).as_posix(), "pdf_sha256": sha(output),
                "pdf_bytes": output.stat().st_size, "page_count": automatic["page_count"],
                "chapter_count": len(chapters), "build_passes": passes,
                "source_sha256_before": before, "source_sha256_after": after,
                "all_file_sources_unchanged": before == after,
                "directory_references_not_recursively_hashed": sorted(renderer.directory_references),
                "font": str(font_path), "font_sha256_before": font_before, "font_sha256_after": sha(font_path),
                "chapters": [{"path": chapter.relative, "title": chapter.title,
                              "page": doc.chapter_pages[chapter.relative], "anchor": chapter.anchor,
                              "sha256": before[chapter.relative]} for chapter in chapters],
                "section_pages": doc.section_pages, "references": list(renderer.references.values()),
                "link_intents": renderer.link_intents, "layout_stats": renderer.layout_stats,
                "automatic_checks": automatic, "direct_visual_review": "NOT_YET_RECORDED",
                "limitations": ["No new robot experiment, performance result or personal competency assessment.",
                                "Raw/model evidence is linked by path and SHA, not embedded.",
                                "Automatic text/link checks do not establish readable page layout.",
                                "Shared64MiB report/PDF/visual/QA aggregate budget is checked by the caller."]}
        except Exception as error:
            record = {"schema": SCHEMA, "status": "FAILED_PARTIAL_OUTPUT_PRESERVED",
                "error_type": type(error).__name__, "error": str(error),
                "output": str(output.relative_to(ROOT)), "source_sha256_before": before,
                "direct_visual_review": "NOT_RUN"}
            json.dump(record, manifest_stream, ensure_ascii=False, indent=2, allow_nan=False)
            manifest_stream.write("\n")
            manifest_stream.flush()
            os.fsync(manifest_stream.fileno())
            raise
        json.dump(record, manifest_stream, ensure_ascii=False, indent=2, allow_nan=False)
        manifest_stream.write("\n")
        manifest_stream.flush()
        os.fsync(manifest_stream.fileno())
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", nargs="+", type=Path, required=True,
                        help="Explicit existing final Markdown files, in chapter order")
    parser.add_argument("--output", type=Path, required=True, help="New standalone .pdf path")
    parser.add_argument("--as-of", required=True, help="Client-facing YYYY-MM-DD date")
    args = parser.parse_args()
    sources = [path for group in args.source for path in group]
    record = export(sources, args.output, args.as_of)
    print(json.dumps({"status": record["status"], "output": record["output"],
                      "pages": record["page_count"], "manifest": str(args.output.with_suffix(".manifest.json")),
                      "direct_visual_review": record["direct_visual_review"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
