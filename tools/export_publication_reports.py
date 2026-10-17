#!/usr/bin/env python3
"""Build the two public reports from Markdown, with bookmarks and web evidence links.

Run with the document runtime (reportlab, pypdf, Pillow). No physics/training is
performed. Core evidence is in the manuscripts, rather than a path-only appendix.
Set --font to a Korean-capable TTF on another operating system.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import quote, unquote

from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, PageBreak, Spacer, KeepTogether
from reportlab.platypus.tableofcontents import TableOfContents
from pypdf import PdfReader
import export_report_pdf as layout

ROOT = Path(__file__).resolve().parents[1]
WEB = 'https://github.com/codemillss/pre_worldmodel/blob/main/'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def slug(value):
    value = unicodedata.normalize('NFC', value).lower()
    return re.sub(r'\s+', '-', re.sub(r'[^\w\s-]', '', value).strip())

class Renderer(layout.Renderer):
    def __init__(self, source, output):
        super().__init__([layout.Chapter(source, '공개 보고서')], output)
        self.source = source
        self.anchors = {}
        self.headings = []
        self.input_hashes = {source.relative_to(ROOT).as_posix(): digest(source)}
        counts = {}
        for line in source.read_text().splitlines():
            m = re.match(r'^(#{2,3}) (.+)', line)
            if m:
                label = m[2]
                key = slug(label)
                occurrence = counts.get(key, 0)
                counts[key] = occurrence + 1
                key += '-' + str(occurrence) if occurrence else ''
                anchor = 's' + hashlib.sha256(key.encode()).hexdigest()[:16]
                self.anchors[key] = anchor
                self.headings.append((label, len(m[1])-2, anchor))

    def link(self, label, target, source):
        if target.startswith(('https://', 'http://')):
            url = target
        elif target.startswith('#'):
            anchor = self.anchors.get(unquote(target[1:]))
            if not anchor:
                raise ValueError('Unknown internal anchor: ' + target)
            return '<link href="#' + anchor + '" color="#166A8F">' + self.inline(label, source) + '</link>'
        else:
            raw, _, fragment = target.partition('#')
            path = (source.parent / unquote(raw)).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                raise ValueError('Missing/nonportable report link: ' + target)
            relative = path.relative_to(ROOT).as_posix()
            self.input_hashes[relative] = digest(path)
            url = WEB + quote(relative) + ('#' + quote(fragment) if fragment else '')
        self.link_intents.append(url)
        return '<link href="' + html.escape(url, quote=True) + '" color="#166A8F">' + self.inline(label, source) + '</link>'

    def table(self, lines, source):
        flow = super().table(lines, source)
        # Small scientific comparisons are one reading unit: never isolate the
        # decisive final row (e.g. SK-S1) on the following page.
        if len(flow) == 1 and isinstance(flow[0], layout.LongTable) and len(lines) <= 12:
            return [KeepTogether(flow)]
        return flow

    def build_body(self):
        flow = self.markdown(self.source)
        headings = iter(self.headings)
        def mark(items):
            for item in items:
                if isinstance(item, Paragraph) and item.style.name in ('h2', 'h3', 'joined_heading'):
                    label, level, anchor = next(headings)
                    if item.getPlainText() != label:
                        raise ValueError('Heading mismatch: ' + label)
                    item._heading = (label, level, anchor)
                elif isinstance(item, KeepTogether):
                    mark(item._content)
        mark(flow)
        if next(headings, None) is not None:
            raise ValueError('Lost report heading')
        for match in re.finditer(r'!\[[^\]]*\]\(([^)]+)\)', self.source.read_text()):
            path = self.local_target(match[1], self.source)
            self.input_hashes[path.relative_to(ROOT).as_posix()] = digest(path)
        return flow

class Document(BaseDocTemplate):
    def __init__(self, output, title, kind):
        super().__init__(str(output), pagesize=layout.A4, leftMargin=48, rightMargin=48,
                         topMargin=56, bottomMargin=49)
        self.title, self.kind = title, kind
        self.addPageTemplates(PageTemplate('body', [Frame(48,49,layout.BODY_W,layout.PAGE_H-105,
            leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)], onPage=self.chrome))

    def beforeDocument(self):
        self.canv.setTitle(self.title)
        self.canv.setAuthor('pre_worldmodel project')
        self.canv.setSubject('Fixed-scope planar pushing: prediction, selection and handoff evaluation')

    def chrome(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(layout.RULE)
        canvas.line(48,layout.PAGE_H-34,layout.PAGE_W-48,layout.PAGE_H-34)
        canvas.setFont('ReportKR',8)
        canvas.setFillColor(layout.MUTED)
        canvas.drawString(48,layout.PAGE_H-26,'PRE_WORLDMODEL | ' + self.kind)
        canvas.drawString(48,25,'2026-10-18 | 제한된 시뮬레이션의 기술 검증')
        canvas.drawRightString(layout.PAGE_W-48,25,str(doc.page))
        if doc.page > 1:
            canvas.drawRightString(layout.PAGE_W-48,layout.PAGE_H-26,'목차로 이동')
            canvas.linkRect('', 'contents', (layout.PAGE_W-105,layout.PAGE_H-31,layout.PAGE_W-48,layout.PAGE_H-20),relative=0)
        canvas.restoreState()

    def afterFlowable(self, item):
        if getattr(item,'_is_contents',False):
            self.canv.bookmarkPage('contents')
            self.canv.addOutlineEntry('목차','contents',0,False)
        if hasattr(item,'_heading'):
            label, level, anchor = item._heading
            self.canv.bookmarkPage(anchor)
            self.canv.addOutlineEntry(label,anchor,level,False)
            self.notify('TOCEntry',(level,label,self.page,anchor))

def make(source, output, kind, subtitle):
    title = source.read_text().splitlines()[0][2:]
    renderer = Renderer(source,output)
    # Make short reports easy to read; do not compress the font to meet a page target.
    layout.STYLES['body'].spaceAfter = 8
    layout.STYLES['h2'].spaceBefore = 16
    body = renderer.build_body()
    toc = TableOfContents()
    toc.levelStyles = [layout.style('toc0',10,15,spaceBefore=3),
                       layout.style('toc1',9,14,leftIndent=12,spaceBefore=1)]
    heading = Paragraph('목차',layout.STYLES['h2'])
    heading._is_contents = True
    cover_title = layout.safe_text(title).replace(': ', '<br/>', 1)
    story = [Paragraph(cover_title,layout.STYLES['h1']),
             Paragraph(layout.safe_text(subtitle),layout.STYLES['body']),
             Paragraph('핵심 근거는 본문에 수록했다. 세부 코드·수치는 공개 저장소로 연결한다. GitHub PDF 미리보기에서는 링크가 동작하지 않으므로 내려받아 사용한다.',layout.STYLES['small']),
             heading,toc,PageBreak()] + body
    output.parent.mkdir(parents=True,exist_ok=True)
    Document(output,title,kind).multiBuild(story)
    reader = PdfReader(output)
    texts = [page.extract_text() or '' for page in reader.pages]
    if any(c in '\n'.join(texts) for c in ('\ufffd','\u25a0')):
        raise ValueError('Unexpected replacement glyph')
    joined = re.sub(r'\s+','',unicodedata.normalize('NFC','\n'.join(texts)))
    for label, _, _ in renderer.headings:
        if re.sub(r'\s+','',layout.normal_chars(label)) not in joined:
            raise ValueError('Missing heading: ' + label)
    ids = {page.indirect_reference.idnum for page in reader.pages}
    links = {'internal':0,'web':0}
    for page in reader.pages:
        for ref in page.get('/Annots',[]):
            annotation = ref.get_object()
            dest = annotation.get('/Dest')
            if dest is not None:
                if dest[0].idnum not in ids:
                    raise ValueError('Invalid internal destination')
                links['internal'] += 1
            action = annotation.get('/A',{})
            if action.get('/S') == '/URI':
                links['web'] += 1
    for relative, value in renderer.input_hashes.items():
        if digest(ROOT/relative) != value:
            raise ValueError('Input changed during export: ' + relative)
    record = {'schema':'public-report-export-v1','source':source.relative_to(ROOT).as_posix(),
        'output':output.relative_to(ROOT).as_posix(),'sha256':digest(output),'pages':len(texts),
        'input_sha256':renderer.input_hashes,'links':links,
        'visual_review':'REQUIRED_SEPARATE_RECORD',
        'body_characters_per_page':[len(t) for t in texts]}
    output.with_suffix('.manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('output','sha256','pages','links')},ensure_ascii=False))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--font',type=Path)
    parser.add_argument('--only',choices=['all','business','technical'],default='all')
    args = parser.parse_args()
    candidates = [args.font] if args.font else [Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf'),
        Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')]
    font = next((p for p in candidates if p and p.is_file()),None)
    if font is None:
        raise SystemExit('Supply a Korean-capable TrueType font with --font')
    pdfmetrics.registerFont(TTFont('ReportKR',str(font)))
    pdfmetrics.registerFontFamily('ReportKR',normal='ReportKR',bold='ReportKR',italic='ReportKR',boldItalic='ReportKR')
    if args.only in ('all','business'):
        make(ROOT/'docs/business_report.ko.md',ROOT/'reports/Business_Report.ko.pdf','실무 의사결정 요약',
             '목표 달성 · 도입 판단 · 확인된 사실과 남은 가설')
    if args.only in ('all','technical'):
        make(ROOT/'docs/technical_report.ko.md',ROOT/'reports/Technical_Report.ko.pdf','논문 형식 기술 보고서',
             '문제 정의 · 학습/계획 설계 · 통제 비교 · 실패 분석 · 재현 범위')

if __name__ == '__main__':
    main()
