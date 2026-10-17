"""Synthetic reader tests only: no valid PDF is created, parsed or rendered."""
from contextlib import redirect_stdout
from copy import deepcopy
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unicodedata
import unittest
from unittest.mock import patch


TOOL = Path(__file__).resolve().parents[1] / "tools/p5_pdf_delivery_audit_v2.py"
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a54sAAAAASUVORK5CYII=")


class Annotation(dict):
    def get_object(self):
        return self


class Page:
    def __init__(self, number, text):
        self.indirect_reference = SimpleNamespace(idnum=100 + number)
        self.text, self.annotations = text, []

    def extract_text(self):
        return self.text

    def get(self, key, default=None):
        return self.annotations if key == "/Annots" else default


class Reader:
    def __init__(self, pages, outline):
        self.pages, self.outline = pages, outline
        self.named_destinations = {}

    def get_destination_page_number(self, value):
        return value.number - 1


def destination(title, page):
    return SimpleNamespace(title=title, number=page)


class Fixture:
    def __init__(self, base, tool):
        self.tool = tool
        self.root = Path(base).resolve() / "project"
        self.root.mkdir()
        self.pdf = self.root / "output/final.pdf"
        self.pdf.parent.mkdir()
        self.pdf.write_bytes(b"OPAQUE-SYNTHETIC-INPUT-NOT-A-PDF")
        self.manifest = self.pdf.with_suffix(".manifest.json")
        self.sources = []
        for index in range(6):
            path = self.root / "reports" / f"chapter-{index}.md"
            path.parent.mkdir(exist_ok=True)
            body = "부품의 위치와 방향을 맞춘다." if index == 0 else "NOT_ASSESSED와 NOT_ESTABLISHED는 정직한 한계이다."
            path.write_text(f"# 문서{index}\n\n## 설명{index}\n\n{body}\n")
            self.sources.append(path)
        self.brief = self.sources[0]
        self.image = self.root / "output/scientific-image.png"
        self.image.write_bytes(PNG)
        self.sources[1].write_text(self.sources[1].read_text() + "\n![그림](../output/scientific-image.png)\n")
        self.evidence = self.root / "reports/evidence.json"
        self.evidence.write_text('{"scope":"synthetic direct reference"}\n')
        # The system-font provenance may legitimately be outside the project.
        self.font = Path(base).resolve() / "synthetic-external-font.bin"
        self.font.write_bytes(b"synthetic-font-identity")
        before = {p.relative_to(self.root).as_posix(): tool.digest(p)
                  for p in self.sources + [self.image, self.evidence]}
        self.record = {"schema": tool.EXPORT_SCHEMA, "status": tool.EXPORT_STATUS,
            "all_file_sources_unchanged": True, "output": self.pdf.relative_to(self.root).as_posix(),
            "pdf_sha256": tool.digest(self.pdf), "pdf_bytes": self.pdf.stat().st_size,
            "chapter_count": 6, "source_sha256_before": before, "source_sha256_after": deepcopy(before),
            "references": [{"id": 1, "path": self.evidence.relative_to(self.root).as_posix(),
                "kind": "file", "sha256": tool.digest(self.evidence)}],
            "font": str(self.font), "font_sha256_before": tool.digest(self.font),
            "font_sha256_after": tool.digest(self.font)}
        self.build_layout()

    def save(self):
        self.manifest.write_text(json.dumps(self.record, ensure_ascii=False, sort_keys=True))

    def reseal_source(self, path):
        relative = path.relative_to(self.root).as_posix()
        current = self.tool.digest(path)
        self.record["source_sha256_before"][relative] = current
        self.record["source_sha256_after"][relative] = current
        for chapter in self.record["chapters"]:
            if chapter["path"] == relative:
                chapter["sha256"] = current
                chapter["title"] = path.read_text().splitlines()[0].lstrip("# ").strip()
        self.save()

    def build_layout(self, *, extra_brief_page=False):
        starts = [3] + [n + int(extra_brief_page) for n in range(4, 9)]
        catalog = 9 + int(extra_brief_page)
        pages = [Page(n, "") for n in range(1, catalog + 1)]
        chapters, sections, title_list, child_outline = [], {}, [], []
        for path, page in zip(self.sources, starts):
            relative = path.relative_to(self.root).as_posix()
            document = self.tool.manuscript(path)
            anchor = "doc_" + hashlib.sha256(relative.encode()).hexdigest()[:16]
            chapters.append({"path": relative, "title": document["title"], "page": page,
                "anchor": anchor, "sha256": self.tool.digest(path)})
            child_outline.append(destination(document["title"], page))
            section_outline = []
            title_list.append(document["title"])
            for line, title in document["sections"]:
                key = "section_" + hashlib.sha256(f"{relative}:{line}:{title}".encode()).hexdigest()[:20]
                sections[key] = page
                section_outline.append(destination(title, page))
                title_list.append(title)
            child_outline.append(section_outline)
            pages[page - 1].text = document["text"]
        pages[1].text = "전체 목차\n" + "\n".join(title_list) + "\n별도 원본 근거 경로"
        destinations = starts + list(sections.values()) + [catalog]
        pages[1].annotations = [Annotation({"/Subtype": "/Link", "/Dest": [pages[p - 1].indirect_reference]})
                                for p in destinations]
        pages[catalog - 1].text = "별도 원본 근거 경로"
        outline = [destination("전체 목차", 2), destination("최종 비교와 단계별 증거", starts[0]),
                   child_outline, destination("별도 원본 근거 경로", catalog)]
        self.reader = Reader(pages, outline)
        self.record.update(page_count=catalog, chapters=chapters, section_pages=sections,
            automatic_checks={"pdf_link_counts": {"internal": len(destinations), "external": 0}})
        self.pngs = []
        for number in range(1, catalog + 1):
            p = self.root / "tmp/render" / f"page-{number}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(PNG)
            self.pngs.append(p)
        self.save()


class DeliveryAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("p5_delivery_audit_v2_tested", TOOL)
        cls.tool = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.tool
        spec.loader.exec_module(cls.tool)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="p5-pdf-audit-synthetic-")
        self.addCleanup(temporary.cleanup)
        self.fixture = Fixture(temporary.name, self.tool)
        self.render_equivalence = None

    def audit(self):
        f = self.fixture
        with patch.object(self.tool, "PdfReader", return_value=f.reader):
            return self.tool.audit_delivery(f.pdf, f.manifest, f.sources, f.pngs,
                                           brief=f.brief, project_root=f.root,
                                           render_equivalence=self.render_equivalence)

    def proof(self, *, mixed=True):
        f = self.fixture
        if mixed:
            original = f.pngs[1]
            selected = f.root / "tmp/render-new" / original.name
            selected.parent.mkdir()
            selected.write_bytes(original.read_bytes())
            f.pngs[1] = selected
        pages = []
        for number, path in enumerate(f.pngs, 1):
            reused = not mixed or number != 2
            pages.append({"page": number, "selected_png": path.relative_to(f.root).as_posix(),
                "fresh_render_png_sha256": self.tool.digest(path),
                "selected_png_sha256": self.tool.digest(path), "fresh_render_bytes": path.stat().st_size,
                "reuse_exact": reused, "reuse_reason": "EXACT_PNG_BYTES" if reused else None})
        self.proof_record = {"schema": self.tool.RENDER_EQUIVALENCE_SCHEMA,
            "status": self.tool.RENDER_EQUIVALENCE_STATUS,
            "pdf": f.pdf.relative_to(f.root).as_posix(), "pdf_sha256": self.tool.digest(f.pdf),
            "page_count": len(f.pngs), "pages": pages,
            "reused_page_count": sum(p["reuse_exact"] for p in pages),
            "new_page_numbers": [p["page"] for p in pages if not p["reuse_exact"]],
            "new_png_bytes": sum(p["fresh_render_bytes"] for p in pages if not p["reuse_exact"])}
        self.render_equivalence = f.root / "tmp/render-new-equivalence.json"
        self.save_proof()
        return self.proof_record

    def save_proof(self):
        self.render_equivalence.write_text(json.dumps(self.proof_record, ensure_ascii=False))

    def test_valid_mixed_parents_require_complete_exact_proof(self):
        f = self.fixture
        self.proof()
        before = {str(p): self.tool.digest(p) for p in f.root.rglob("*") if p.is_file()}
        result = self.audit()
        self.assertTrue(result["accepted"], result["errors"])
        proof = result["render_equivalence"]
        self.assertEqual(proof["sha256_before"], proof["sha256_after"])
        self.assertEqual(proof["sha256_after"], self.tool.digest(self.render_equivalence))
        self.assertIn(str(self.render_equivalence), result["input_bindings_sha256"])
        self.assertEqual(proof["verified_pages"], 9)
        self.assertEqual(proof["new_page_numbers"], [2])
        self.assertEqual(result["direct_visual_review"], "NOT_ASSESSED_BY_THIS_TOOL")
        self.assertEqual(before, {str(p): self.tool.digest(p) for p in f.root.rglob("*") if p.is_file()})

    def test_no_proof_still_rejects_mixed_parents(self):
        self.proof()
        self.render_equivalence = None
        self.rejected("continuous ordered")

    def test_proof_rejects_current_png_sha_change(self):
        self.proof()
        path = self.fixture.pngs[1]
        data = bytearray(path.read_bytes())
        data[33] ^= 1
        path.write_bytes(data)
        self.rejected("selected/fresh/current PNG SHA")

    def test_proof_rejects_stale_pdf_sha(self):
        self.proof()["pdf_sha256"] = "0" * 64
        self.save_proof()
        self.rejected("PDF identity/SHA/page count")

    def test_proof_rejects_wrong_pdf_path_even_same_bytes(self):
        self.proof()
        other = self.fixture.pdf.with_name("other.pdf")
        other.write_bytes(self.fixture.pdf.read_bytes())
        self.proof_record["pdf"] = other.relative_to(self.fixture.root).as_posix()
        self.save_proof()
        self.rejected("PDF identity/SHA/page count")

    def test_proof_rejects_wrong_page_count(self):
        self.proof()["page_count"] -= 1
        self.save_proof()
        self.rejected("PDF identity/SHA/page count")

    def test_proof_rejects_missing_page_record(self):
        self.proof()["pages"].pop()
        self.save_proof()
        self.rejected("every PDF page")

    def test_proof_rejects_duplicate_page_number(self):
        self.proof()["pages"][1]["page"] = 1
        self.save_proof()
        self.rejected("unique ordered1..N")

    def test_proof_rejects_different_selected_order_despite_same_sha(self):
        pages = self.proof()["pages"]
        pages[0]["selected_png"], pages[1]["selected_png"] = pages[1]["selected_png"], pages[0]["selected_png"]
        self.save_proof()
        self.rejected("order/path differs")

    def test_proof_rejects_fresh_png_sha_difference(self):
        self.proof()["pages"][0]["fresh_render_png_sha256"] = "f" * 64
        self.save_proof()
        self.rejected("selected/fresh/current PNG SHA")

    def test_proof_rejects_rgb_only_reuse(self):
        self.proof()["pages"][0]["reuse_reason"] = "EXACT_RGB_PIXELS"
        self.save_proof()
        self.rejected("Only exact PNG byte reuse")

    def test_proof_rejects_inconsistent_new_page_summary(self):
        self.proof()["new_page_numbers"] = []
        self.save_proof()
        self.rejected("reuse/new-page summary")

    def test_proof_rejects_wrong_status(self):
        self.proof()["status"] = "PARTIAL"
        self.save_proof()
        self.rejected("schema/status")

    def test_proof_cannot_exempt_different_png_prefixes(self):
        self.proof()
        path = self.fixture.pngs[1]
        renamed = path.with_name("different-2.png")
        path.rename(renamed)
        self.fixture.pngs[1] = renamed
        self.proof_record["pages"][1]["selected_png"] = renamed.relative_to(self.fixture.root).as_posix()
        self.save_proof()
        self.rejected("continuous ordered")

    def test_proof_after_sha_rechecked(self):
        self.proof()
        original = self.tool.Inputs.recheck
        proof_path = self.render_equivalence
        def mutate_then_recheck(inputs):
            proof_path.write_bytes(proof_path.read_bytes() + b"\n")
            original(inputs)
        with patch.object(self.tool.Inputs, "recheck", mutate_then_recheck):
            self.rejected("Input changed during audit")

    def test_cli_proof_and_exclusive_fail_receipt_preservation(self):
        f = self.fixture
        self.proof()
        out = f.root / "reports/new-delivery-audit.json"
        arguments = ["--pdf", str(f.pdf), "--manifest", str(f.manifest), "--source",
            *map(str, f.sources), "--brief", str(f.brief), "--render-png", *map(str, f.pngs),
            "--render-equivalence", str(self.render_equivalence), "--out", str(out)]
        with patch.object(self.tool, "ROOT", f.root), patch.object(self.tool, "PdfReader", return_value=f.reader), redirect_stdout(io.StringIO()):
            self.assertEqual(self.tool.main(arguments), 0)
            self.assertTrue(json.loads(out.read_bytes())["accepted"])
            original = b'{"status":"PRESERVED_ORIGINAL_FAIL"}\n'
            out.write_bytes(original)
            with self.assertRaises(ValueError):
                self.tool.main(arguments)
            self.assertEqual(out.read_bytes(), original)

    def rejected(self, reason=None):
        result = self.audit()
        self.assertEqual(result["status"], "FAIL_PDF_DELIVERY_AUDIT", result.get("errors"))
        self.assertIs(result["accepted"], False)
        self.assertTrue(result["errors"])
        if reason: self.assertIn(reason, result["errors"][0]["message"])

    def test_complete_synthetic_inventory_passes_without_writes(self):
        f = self.fixture
        before = {str(p): self.tool.digest(p) for p in f.root.rglob("*") if p.is_file()}
        result = self.audit()
        self.assertIs(result["accepted"], True, result["errors"])
        self.assertEqual(result["pdf_checks"]["brief_body_page_count"], 1)
        self.assertEqual(len(result["rendered_pages"]), 9)
        self.assertEqual(result["direct_visual_review"], "NOT_ASSESSED_BY_THIS_TOOL")
        after = {str(p): self.tool.digest(p) for p in f.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_stale_markdown_is_rejected(self):
        self.fixture.sources[3].write_text(self.fixture.sources[3].read_text() + "\n새 내용\n")
        self.rejected("SHA256")

    def test_pdf_bytes_changed(self):
        self.fixture.pdf.write_bytes(b"changed opaque synthetic bytes")
        self.rejected("PDF identity")

    def test_source_order_changed(self):
        self.fixture.sources[1], self.fixture.sources[2] = self.fixture.sources[2], self.fixture.sources[1]
        self.rejected("chapter/order/SHA")

    def test_image_sha_changed(self):
        self.fixture.image.write_bytes(b"changed image")
        self.rejected("SHA256")

    def test_external_font_sha_changed(self):
        self.fixture.font.write_bytes(b"changed font")
        self.rejected("SHA256")

    def test_reference_sha_changed(self):
        self.fixture.evidence.write_text('{"changed":true}')
        self.rejected("SHA256")

    def test_bookmark_wrong_destination(self):
        self.fixture.reader.outline[2][0].number = 4
        self.rejected("bookmark")

    def test_missing_section_bookmark(self):
        self.fixture.reader.outline[2][1].clear()
        self.rejected("bookmark")

    def test_section_bookmark_must_have_h2_depth(self):
        outline = self.fixture.reader.outline[2]
        outline[1] = outline[1][0]
        self.rejected("hierarchy")

    def test_section_cannot_point_into_the_next_chapter(self):
        f = self.fixture
        key = next(iter(f.record["section_pages"]))
        f.record["section_pages"][key] = 4
        f.reader.outline[2][1][0].number = 4
        f.reader.pages[3].text += "\n설명0\n"
        f.save()
        self.rejected("outside its chapter")

    def test_exporter_raw_unicode_section_anchor_is_preserved(self):
        f = self.fixture
        p = f.sources[0]
        p.write_text(p.read_text().replace("설명0", unicodedata.normalize("NFD", "설명0")))
        f.reseal_source(p)
        f.build_layout()
        result = self.audit()
        self.assertIs(result["accepted"], True, result["errors"])

    def test_toc_destination_missing_despite_correct_link_count(self):
        f = self.fixture
        # Remove the only catalog link and update declared counts as an adversary.
        f.reader.pages[1].annotations.pop()
        f.record["automatic_checks"]["pdf_link_counts"]["internal"] -= 1
        f.save()
        self.rejected("TOC links")

    def test_pdf_pending_marker(self):
        self.fixture.reader.pages[3].text += "\nFINAL_PENDING_APPLICATION"
        self.rejected("Unfinished marker")

    def test_resealed_markdown_pending_marker(self):
        p = self.fixture.sources[2]
        p.write_text(p.read_text() + "\n<!-- FINAL_PENDING_RESULT -->\n")
        self.fixture.reseal_source(p)
        self.rejected("Unfinished marker")

    def test_resealed_draft_title(self):
        p = self.fixture.sources[2]
        p.write_text(p.read_text().replace("# 문서2", "# 결과 초안"))
        self.fixture.reseal_source(p)
        self.rejected("Unfinished/empty")

    def test_missing_png_page(self):
        self.fixture.pngs.pop()
        self.rejected("One unique PNG")

    def test_duplicate_png_page(self):
        self.fixture.pngs[-1] = self.fixture.pngs[-2]
        self.rejected("One unique PNG")

    def test_noncontinuous_png_page_numbers(self):
        p = self.fixture.pngs[-1]
        q = p.with_name("page-10.png")
        p.rename(q)
        self.fixture.pngs[-1] = q
        self.rejected("continuous ordered")

    def test_invalid_png_signature(self):
        self.fixture.pngs[0].write_bytes(b"not png but enough bytes for a header" + b"x" * 40)
        self.rejected("PNG header")

    def test_brief_must_occupy_one_body_page(self):
        self.fixture.build_layout(extra_brief_page=True)
        self.rejected("more than one PDF page")

    def test_brief_paragraph_cannot_be_dropped(self):
        self.fixture.reader.pages[2].text = "문서0\n설명0\n"
        self.rejected("brief paragraph")

    def test_export_before_after_source_hash_difference(self):
        key = next(iter(self.fixture.record["source_sha256_after"]))
        self.fixture.record["source_sha256_after"][key] = "f" * 64
        self.fixture.save()
        self.rejected("before/after")

    def test_cli_new_output_and_overwrite_rejection(self):
        f = self.fixture
        out = f.root / "reports/delivery-audit.json"
        arguments = ["--pdf", str(f.pdf), "--manifest", str(f.manifest), "--source",
            *map(str, f.sources), "--brief", str(f.brief), "--render-png", *map(str, f.pngs), "--out", str(out)]
        with patch.object(self.tool, "ROOT", f.root), patch.object(self.tool, "PdfReader", return_value=f.reader), redirect_stdout(io.StringIO()):
            self.assertEqual(self.tool.main(arguments), 0)
            original = out.read_bytes()
            with self.assertRaises(ValueError): self.tool.main(arguments)
            self.assertEqual(out.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
