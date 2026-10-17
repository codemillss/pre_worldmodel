#!/usr/bin/env python3
"""Build one navigable living report from the project's Markdown sources.

Use the bundled document runtime, leaving the robotics .venv unchanged.
Completed task chapters are discovered from the existing task board.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import unquote, urlsplit

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, PageBreak,
    LongTable, TableStyle, Image, KeepTogether, Flowable,
)
from reportlab.platypus.tableofcontents import TableOfContents
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
INK = colors.HexColor('#172B3A')
BLUE = colors.HexColor('#166A8F')
MUTED = colors.HexColor('#617383')
RULE = colors.HexColor('#D5DFE5')
PALE = colors.HexColor('#F1F6F8')
PAGE_W, PAGE_H = A4
MARGIN = 48
BODY_W = PAGE_W - 2 * MARGIN
TODAY = datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d')


@dataclass
class Chapter:
    path: Path
    group: str
    historical: bool = False

    @property
    def relative(self):
        return self.path.relative_to(ROOT).as_posix()

    @property
    def title(self):
        return self.path.read_text().splitlines()[0].lstrip('# ').strip()

    @property
    def anchor(self):
        return 'doc_' + hashlib.sha256(self.relative.encode()).hexdigest()[:16]


def chapters_from_board():
    board = (ROOT / 'reports/task_status.md').read_text()
    rows = re.findall(r'^\| (P[0-5]-\d{2}) \| ([^|]+) \| (DONE|TODO) \| (.+) \|$', board, re.M)
    if not rows or len({r[0] for r in rows}) != len(rows):
        raise ValueError('Task board rows missing or duplicate')
    result = [Chapter(ROOT / 'docs/00_프로젝트_목표와_전체진행안내.md', '목표와 현재 상태')]
    result.append(Chapter(ROOT / 'reports/00_전체보고서_색인.md', '목표와 현재 상태'))
    for stage in range(6):
        for suffix in ['stage_summary', 'progress_summary']:
            p = ROOT / f'reports/p{stage}_{suffix}.md'
            if p.exists():
                result.append(Chapter(p, '단계별 결론'))
                break
    for task, _, state, evidence in rows:
        candidates = [ROOT / 'reports' / unquote(x) for x in re.findall(r'\[[^]]+\]\(([^)]+\.md)\)', evidence)]
        selected = next((p for p in candidates if p.exists() and p.read_text().splitlines()[0].startswith('# ' + task)), None)
        if selected is None:
            if state == 'DONE':
                raise ValueError(f'Completed task {task} has no identified report')
            continue
        result.append(Chapter(selected, '작업별 상세 보고' if state == 'DONE' else '진행 중 작업 보고'))
    for rel, group, old in [
        ('reports/p1_solver_contact_review.md', '진단과 결정 근거', False),
        ('reports/p3_02_main_data_gate.md', '진단과 결정 근거', True),
        ('reports/p4_handoff_cost_comparison.md', '진단과 결정 근거', False),
        ('reports/p4_local_prediction_diagnostic.md', '진단과 결정 근거', False),
        ('reports/p4_data_intervention_design.md', '진단과 결정 근거', False),
        ('reports/p4_data_preflight.md', '진단과 결정 근거', False),
        ('reports/p4_data_main_collection.md', '진단과 결정 근거', False),
        ('reports/p4_data_retraining_comparison.md', '진단과 결정 근거', False),
        ('reports/p4_error_mechanism_diagnostic.md', '진단과 결정 근거', False),
        ('reports/p4_relative_loss_comparison.md', '진단과 결정 근거', False),
        ('reports/p4_gradient_probe_diagnostic.md', '진단과 결정 근거', False),
        ('reports/p4_adam_direction_diagnostic.md', '진단과 결정 근거', False),
        ('reports/p4_finite_step_comparison.md', '진단과 결정 근거', False),
        ('reports/p4_cross_batch_comparison.md', '진단과 결정 근거', False),
        ('reports/p4_first_moment_comparison.md', '진단과 결정 근거', False),
        ('reports/p5_execution_preflight.md', '진단과 결정 근거', False),
        ('reports/p5_g2_resource_review.md', '진단과 결정 근거', False),
        ('reports/decision_register.md', '진단과 결정 근거', False),
        ('docs/06_단계별_완료판정.md', '작성·판정 기준', False),
        ('docs/07_단계별_보고서_작성기준.md', '작성·판정 기준', False),
        ('reports/task_status.md', '작성·판정 기준', False),
        ('reports/p1_01_to_03_implementation.md', '과거 기록', True),
        ('reports/p1_04_to_05_results.md', '과거 기록', True),
        ('reports/p1_physics_pilot.md', '과거 기록', True),
    ]:
        result.append(Chapter(ROOT / rel, group, old))
    if len({c.relative for c in result}) != len(result):
        raise ValueError('Duplicate chapter')
    return result, rows


def font_setup():
    candidates = [Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf'),
                  Path('/System/Library/Fonts/Supplemental/AppleGothic.ttf')]
    font_path = next((p for p in candidates if p.exists()), None)
    if font_path is None:
        raise FileNotFoundError('A Korean-capable TrueType font is required')
    font = TTFont('ReportKR', str(font_path))
    pdfmetrics.registerFont(font)
    pdfmetrics.registerFontFamily('ReportKR', normal='ReportKR', bold='ReportKR', italic='ReportKR', boldItalic='ReportKR')
    return font_path, set(font.face.charToGlyph)


def style(name, size=11, leading=18, **kw):
    options = dict(fontName='ReportKR', fontSize=size, leading=leading,
                   textColor=INK, wordWrap='CJK', alignment=TA_LEFT, spaceAfter=10,
                   allowWidows=0, allowOrphans=0)
    options.update(kw)
    return ParagraphStyle(name, **options)


STYLES = {
    'body': style('body'),
    'h1': style('h1', 23, 33, spaceBefore=5, spaceAfter=13, keepWithNext=True),
    'h2': style('h2', 15, 22, spaceBefore=19, spaceAfter=10, keepWithNext=True, textColor=BLUE),
    'h3': style('h3', 12.2, 19, spaceBefore=13, keepWithNext=True),
    'cell': style('cell', 9.5, 15, spaceAfter=0),
    'headcell': style('headcell', 9.5, 15, spaceAfter=0, textColor=BLUE),
    'card': style('card', 10.3, 16.5, spaceAfter=5),
    'cardlabel': style('cardlabel', 9.6, 15.5, textColor=MUTED, spaceAfter=0, keepWithNext=True),
    'cardtitle': style('cardtitle', 11.5, 17.5, spaceAfter=0, textColor=BLUE),
    'small': style('small', 8.5, 13, textColor=MUTED),
    'nav': style('nav', 9.5, 15, spaceAfter=14, textColor=BLUE),
    'quote': style('quote', 10.5, 17, leftIndent=10, rightIndent=8, backColor=PALE, borderPadding=7),
    'list': style('list', 11, 18, leftIndent=15, firstLineIndent=-12),
}


def normal_chars(text):
    return text.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', ' - ')


def safe_text(text):
    return html.escape(normal_chars(text), quote=False)


def split_cells(line):
    """Markdown pipes inside inline code are content, not column boundaries."""
    cells, current, code, escaped = [], [], False, False
    for ch in line.strip():
        if escaped:
            current.append(ch)
            escaped = False
        elif ch == '\\':
            escaped = True
        elif ch == '`':
            code = not code
            current.append(ch)
        elif ch == '|' and not code:
            cells.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    cells.append(''.join(current).strip())
    if cells and not cells[0]:
        cells.pop(0)
    if cells and not cells[-1]:
        cells.pop()
    return cells


class Renderer:
    def __init__(self, chapters, output):
        self.chapters = chapters
        self.output = output.resolve()
        self.by_path = {c.path.resolve(): c for c in chapters}
        self.references = {}
        self.link_intents = []
        self.layout_stats = {'criteria_blocks': 0, 'wide_records': 0, 'comparison_tables': 0}
        self.unsupported = []

    def local_target(self, target, source):
        target = unquote(target.split('#', 1)[0])
        return (source.parent / target).resolve()

    def link(self, label, target, source):
        if target.startswith(('https://', 'http://', 'mailto:')):
            self.link_intents.append({'kind': 'web', 'from': str(source.relative_to(ROOT)), 'target': target})
            return f'<link href="{html.escape(target, quote=True)}" color="#166A8F">{self.inline(label, source)}</link>'
        p = self.local_target(target, source)
        if p == self.output:
            dest = 'toc'
            self.link_intents.append({'kind': 'self', 'from': str(source.relative_to(ROOT)), 'destination': dest})
        elif p in self.by_path:
            dest = self.by_path[p].anchor
            self.link_intents.append({'kind': 'chapter', 'from': str(source.relative_to(ROOT)), 'to': self.by_path[p].relative, 'destination': dest})
        else:
            if not p.exists():
                raise FileNotFoundError(f'{source}: {target}')
            try:
                rel = p.relative_to(ROOT).as_posix()
            except ValueError:
                rel = str(p)
            if rel not in self.references:
                self.references[rel] = {'id': len(self.references) + 1, 'path': rel, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None}
            dest = f'ref_{self.references[rel]["id"]:03d}'
            reference_id = self.references[rel]['id']
            self.link_intents.append({'kind': 'reference', 'from': str(source.relative_to(ROOT)), 'to': rel, 'destination': dest})
            return (f'<link href="#{dest}" color="#166A8F">{self.inline(label, source)}'
                    f'<super><font size="7.7"> [{reference_id:03d}]</font></super></link>')
        return f'<link href="#{dest}" color="#166A8F">{self.inline(label, source)}</link>'

    def inline(self, text, source):
        pattern = r'\[[^\]]+\]\([^)]+\)|`[^`]+`|\*\*.+?\*\*'
        chunks, last = [], 0
        for match in re.finditer(pattern, text):
            chunks.append(safe_text(text[last:match.start()]))
            token = match.group()
            if token.startswith('['):
                m = re.match(r'\[([^\]]+)\]\(([^)]+)\)', token)
                chunks.append(self.link(m.group(1), m.group(2), source))
            elif token.startswith('`'):
                chunks.append(f'<font color="#324E62">{safe_text(token[1:-1])}</font>')
            else:
                chunks.append(f'<b>{self.inline(token[2:-2], source)}</b>')
            last = match.end()
        chunks.append(safe_text(text[last:]))
        return ''.join(chunks)

    def table(self, lines, source):
        rows = [split_cells(line) for line in lines]
        if len(rows) >= 2 and all(re.fullmatch(r':?-{2,}:?', x) for x in rows[1]):
            rows.pop(1)
        n = len(rows[0])
        if any(len(row) != n for row in rows):
            raise ValueError(f'Unequal table column count in {source}')
        if n == 5 and rows[0][0] == '기준 ID':
            self.layout_stats['criteria_blocks'] += len(rows)-1
            return self.record_blocks(rows, source, criteria=True)
        if n >= 4 and (rows[0][0] in ['ID', '작업 ID'] or
                       sum(max(len(c) for c in row) for row in rows[1:])/max(len(rows)-1, 1) > 72):
            self.layout_stats['wide_records'] += len(rows)-1
            return self.record_blocks(rows, source)
        self.layout_stats['comparison_tables'] += 1
        if n == 2:
            fractions = [.26, .74]
        elif n == 3:
            fractions = [.23, .39, .38]
        elif (n == 4 and source.name == 'p4_gradient_probe_diagnostic.md'
              and rows[0] == ['배치', '보조/기존 크기 비율', '코사인 범위', '전체 반대']):
            fractions = [.18, .30, .32, .20]
        elif (n == 4 and source.name == 'p4_gradient_probe_diagnostic.md'
              and rows[0][0] == '배치' and rows[0][1].startswith('Lbase:')):
            fractions = [.19, .27, .27, .27]
        elif (n == 4 and source.name == 'p4_gradient_probe_diagnostic.md'
              and rows[0] == ['확인 항목', '실제 관측', '현재 판정', '근거']):
            fractions = [.24, .30, .25, .21]
        elif n == 4:
            fractions = [.19, .26, .17, .38]
        elif n == 5 and source.name == 'p4_relative_loss_comparison.md' and rows[0][0] == '정지 끝점 지표':
            fractions = [.24, .16, .16, .16, .28]
        elif n == 5:
            fractions = [.12, .20, .26, .22, .20]
        else:
            fractions = [1/n] * n
        data = [[Paragraph(self.inline(cell, source), STYLES['headcell' if i == 0 else 'cell']) for cell in row] for i, row in enumerate(rows)]
        t = LongTable(data, colWidths=[BODY_W*x for x in fractions], repeatRows=1,
                      hAlign='LEFT', splitByRow=1, splitInRow=1, spaceBefore=7, spaceAfter=16)
        t.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BACKGROUND', (0, 0), (-1, 0), PALE),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#FAFCFD')]),
            ('LINEBELOW', (0, 0), (-1, 0), .7, BLUE),
            ('LINEBELOW', (0, 1), (-1, -1), .25, RULE),
            ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 8), ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        return [t]

    def record_blocks(self, rows, source, criteria=False):
        """Keep each source field but give long prose the full reading width."""
        header, flow = rows[0], []
        for row in rows[1:]:
            if criteria:
                title, condition, observed, verdict, evidence = row
                status_style = style('status', 10.2, 16, textColor=colors.HexColor('#256450') if verdict.startswith('PASS') else MUTED, spaceAfter=0)
                top = LongTable([[Paragraph(self.inline(title, source), STYLES['cardtitle']),
                                  Paragraph(self.inline(verdict, source), status_style)]],
                                colWidths=[BODY_W*.70, BODY_W*.30], spaceBefore=10, spaceAfter=7)
                top.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE),('VALIGN',(0,0),(-1,-1),'TOP'),
                                         ('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),
                                         ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
                                         ('LINEABOVE',(0,0),(-1,0),.6,RULE)]))
                fields = [('사전 조건과 출처', condition),('실제 관측값과 범위', observed),('증거', evidence)]
            else:
                top = Paragraph(self.inline(f'{header[0]}: {row[0]}', source), STYLES['h3'])
                fields = list(zip(header[1:], row[1:]))
            details = []
            for label, value in fields:
                details.extend([Paragraph(safe_text(label), STYLES['cardlabel']),
                                Paragraph(self.inline(value, source), STYLES['card'])])
            # A criterion is one reading unit; exceptionally large records may split.
            flow.append(KeepTogether([top]+details+[Spacer(1, 9)]))
        return flow

    def markdown(self, source):
        lines, flow, i = source.read_text().splitlines(), [], 1
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if line.startswith('|'):
                block = []
                while i < len(lines) and lines[i].strip().startswith('|'):
                    block.append(lines[i].strip())
                    i += 1
                table_flow = self.table(block, source)
                if (flow and isinstance(flow[-1], Paragraph) and
                        flow[-1].style.name in ['h2', 'h3'] and
                        table_flow and isinstance(table_flow[0], KeepTogether)):
                    heading = flow.pop()
                    heading.style = ParagraphStyle('joined_heading', parent=heading.style, keepWithNext=False)
                    first = table_flow.pop(0)
                    flow.append(KeepTogether([heading]+first._content))
                flow.extend(table_flow)
                continue
            if line.startswith('```'):
                language = line[3:]
                block = []
                i += 1
                while i < len(lines) and not lines[i].startswith('```'):
                    block.append(lines[i]); i += 1
                i += 1
                if language == 'mermaid':
                    flow.append(PlanningDiagram())
                else:
                    flow.append(Paragraph('<br/>'.join(safe_text(x) for x in block), STYLES['small']))
                continue
            m = re.match(r'^(#{2,6})\s+(.+)', line)
            if m:
                flow.append(Paragraph(self.inline(m.group(2), source), STYLES['h2' if len(m.group(1)) == 2 else 'h3']))
                i += 1
                continue
            m = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
            if m:
                p = self.local_target(m.group(2), source)
                picture = Image(str(p))
                # Portrait charts need enough height to preserve readable labels.
                # Wide overview figures keep the established compact layout.
                # Near-square two-panel scientific figures need full column width
                # to retain readable labels; reserve the shorter box for wide figures.
                height_limit = 480 if picture.imageWidth / picture.imageHeight < 1.4 else 300
                ratio = min(BODY_W / picture.imageWidth, height_limit / picture.imageHeight)
                picture.drawWidth, picture.drawHeight = picture.imageWidth*ratio, picture.imageHeight*ratio
                flow.append(KeepTogether([picture, Paragraph(safe_text(m.group(1)), STYLES['small'])]))
                i += 1
                continue
            if line.startswith('>'):
                flow.append(Paragraph(self.inline(line.lstrip('> '), source), STYLES['quote']))
                i += 1
                continue
            m = re.match(r'^([-*]|\d+\.)\s+(.+)', line)
            if m:
                prefix = '• ' if m.group(1) in ['-', '*'] else m.group(1) + ' '
                flow.append(Paragraph(safe_text(prefix) + self.inline(m.group(2), source), STYLES['list']))
                i += 1
                continue
            if line == '---':
                flow.append(Spacer(1, 6)); i += 1
                continue
            block = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r'^(#|\||>|!\[|```|[-*] |\d+\. )', lines[i].strip()):
                block.append(lines[i].strip()); i += 1
            flow.append(Paragraph(self.inline(' '.join(block), source), STYLES['body']))
        return flow


class PlanningDiagram(Flowable):
    """The existing overview's Mermaid flow, drawn as a compact PDF figure."""

    def __init__(self):
        super().__init__()
        self.width, self.height = BODY_W, 144

    def draw(self):
        c = self.canv
        labels = ['현재 상태 관측', '후보 행동열 생성', '학습 예측과 비용', '첫 명령 실행', '속도 제어·접촉 물리', '다음 상태 관측']
        w, h, gap = (BODY_W - 30) / 3, 34, 15
        for i, label in enumerate(labels):
            row, slot = divmod(i, 3)
            col = slot if row == 0 else 2-slot
            x, y = col*(w+gap), 88-row*57
            c.setFillColor(PALE); c.setStrokeColor(RULE)
            c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
            c.setFillColor(INK); c.setFont('ReportKR', 9.2)
            c.drawCentredString(x+w/2, y+13, label)
            if slot < 2:
                c.setFillColor(BLUE)
                c.drawString(x+w+4 if row == 0 else x-11, y+13, '→' if row == 0 else '←')
        c.setFillColor(BLUE)
        c.drawCentredString(2*(w+gap)+w/2, 75, '↓')
        c.setFont('ReportKR', 8.5); c.setFillColor(MUTED)
        c.drawString(0, 1, '관측으로 인계 조건·실패를 검사하고, 실행 결과를 다음 계획에 반영한다.')


class ReportDoc(BaseDocTemplate):
    def __init__(self, output, **kw):
        super().__init__(str(output), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=61, bottomMargin=51, **kw)
        self.chapter_pages = {}
        self.current_title = ''
        self.addPageTemplates(PageTemplate(id='report', frames=[Frame(MARGIN, 51, BODY_W, PAGE_H-112,
                                                                     leftPadding=0, rightPadding=0,
                                                                     topPadding=0, bottomPadding=0)], onPage=self.page_chrome))

    def beforeDocument(self):
        self.chapter_pages = {}
        self.current_title = ''
        self.current_source = None
        self.toc_group = None
        self.canv.setTitle('부품 인계 자세 정렬 - 프로젝트 통합 보고서')
        self.canv.setAuthor('프로젝트 기록')

    def page_chrome(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(RULE); canvas.line(MARGIN, PAGE_H-35, PAGE_W-MARGIN, PAGE_H-35)
        canvas.setFont('ReportKR', 8.2); canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, PAGE_H-27, '부품 인계 자세 정렬 | 프로젝트 통합 보고서')
        if doc.page > 1:
            canvas.setFillColor(BLUE); canvas.drawRightString(PAGE_W-MARGIN, PAGE_H-27, '전체 목차')
            canvas.linkRect('', 'toc', (PAGE_W-MARGIN-46, PAGE_H-31, PAGE_W-MARGIN, PAGE_H-20), relative=0)
        canvas.setFillColor(MUTED); canvas.setFont('ReportKR', 8)
        canvas.drawString(MARGIN, 25, TODAY + ' 갱신')
        canvas.drawRightString(PAGE_W-MARGIN, 25, str(doc.page))
        canvas.restoreState()

    def afterPage(self):
        if self.current_source:
            self.canv.saveState()
            self.canv.setFont('ReportKR', 8)
            self.canv.setFillColor(MUTED)
            self.canv.drawString(MARGIN, 38, '원문: ' + self.current_source)
            self.canv.restoreState()

    def afterFlowable(self, flowable):
        if hasattr(flowable, '_anchor'):
            key = flowable._anchor
            self.canv.bookmarkPage(key)
            if key == 'toc':
                self.canv.addOutlineEntry('전체 목차', key, 0, False)
            elif hasattr(flowable, '_chapter'):
                ch = flowable._chapter
                self.chapter_pages[ch.relative] = self.page
                self.current_title = ch.title
                self.current_source = ch.relative
                group = ch.group
                if group == '작업별 상세 보고':
                    stage = re.match(r'P[0-5]', ch.title).group()
                    group = stage + ' 작업별 상세 보고'
                if group != self.toc_group:
                    self.toc_group = group
                    group_key = 'group_' + hashlib.sha256(group.encode()).hexdigest()[:10]
                    self.canv.bookmarkPage(group_key)
                    self.canv.addOutlineEntry(group, group_key, 0, False)
                    self.notify('TOCEntry',(0, group, self.page, group_key))
                self.canv.addOutlineEntry(ch.title, key, 1, False)
                self.notify('TOCEntry', (1, ch.title, self.page, key))
            elif key == 'evidence_catalog':
                self.current_source = None
                self.canv.addOutlineEntry('별도 원본 근거 경로', key, 0, False)
                self.notify('TOCEntry', (0, '별도 원본 근거 경로', self.page, key))


def annotated_title(title, anchor, chapter=None):
    marked = safe_text(title)
    prefix = re.match(r'^(P[0-5](?:-\d{2}(?:~\d{2})?)?)\s+(.+)', normal_chars(title))
    if prefix:
        marked = '<font size="12" color="#166A8F">'+safe_text(prefix.group(1))+'</font><br/>'+safe_text(prefix.group(2))
    p = Paragraph(marked, STYLES['h1'])
    p._anchor = anchor
    if chapter is not None:
        p._chapter = chapter
    return p


def make_pdf(output, chapters, rows):
    font_path, glyphs = font_setup()
    renderer = Renderer(chapters, output)
    doc = ReportDoc(output)
    done = sum(row[2] == 'DONE' for row in rows)
    cover = [Spacer(1, 70), Paragraph('부품 인계 자세 정렬', style('cover_title', 28, 39)),
             Paragraph('프로젝트 통합 보고서', style('cover_subtitle', 21, 31, textColor=BLUE)),
             Spacer(1, 20), Paragraph('학습 동역학 기반 밀기 제어', style('cover_topic', 13, 20)),
             Paragraph('목표와 현재 상태 · 단계별 결론 · 작업별 근거 · 과거 기록', STYLES['body']),
             Spacer(1, 27),
             Paragraph(f'완료 작업 {done}/{len(rows)}개. 최신 상태와 다음 작업은 본문의 작업 보드를 따른다.', STYLES['body']),
             Paragraph('작업 검사 통과, 정렬 성과, 사용자 이해를 별도로 기록한다.', STYLES['body']),
             Paragraph('파란 목차와 보고서명을 누르면 해당 장으로 이동한다. 각 페이지 위의 전체 목차로 돌아올 수 있다.', STYLES['body']),
             Paragraph('PDF에 포함하지 않은 원본 파일 링크는 별도 근거 경로 목록으로 이동한다. 원본 데이터·코드를 PDF에 포함한 재현 묶음은 아니다.', STYLES['small']),
             Spacer(1, 16), Paragraph('<link href="#toc" color="#166A8F">목차 열기 →</link>', style('cover_link', 13, 19)),
             Spacer(1, 28), Paragraph('갱신: ' + TODAY + ' | 기존 보고서를 모아 읽는 통합본', STYLES['small']), PageBreak(),
             annotated_title('전체 목차', 'toc'),
             Paragraph('항목 이름이나 쪽수를 누르면 해당 보고서로 이동한다. PDF 뷰어의 책갈피에서도 같은 장을 찾을 수 있다.', STYLES['body'])]
    summaries = [ch for ch in chapters if ch.group == '단계별 결론']
    summary_links = ' · '.join(f'<link href="#{ch.anchor}" color="#166A8F">{ch.title.split()[0]} 결론</link>' for ch in summaries)
    cover.insert(-3, Paragraph('빠르게 읽기: ' + summary_links, STYLES['nav']))
    cover.insert(-3, Paragraph('단계 결론을 먼저 읽고, 필요한 작업의 상세와 근거로 들어간다.', STYLES['small']))
    toc = TableOfContents()
    toc.levelStyles = [style('toc0', 12.2, 19, leftIndent=0, firstLineIndent=0, spaceBefore=16, textColor=INK),
                       style('toc1', 10.2, 16, leftIndent=14, firstLineIndent=0, spaceBefore=0, textColor=BLUE)]
    story = cover + [toc, PageBreak()]
    latest_p1 = next(c for c in chapters if c.relative == 'reports/p1_stage_summary.md')
    latest_main = next(c for c in chapters if c.relative == 'reports/p3_02_main_collection.md')
    for ch in chapters:
        story.append(Paragraph(safe_text(ch.group), STYLES['small']))
        story.append(annotated_title(ch.title, ch.anchor, ch))
        story.append(Paragraph('<link href="#toc" color="#166A8F">전체 목차로 돌아가기</link>', STYLES['nav']))
        if ch.historical:
            latest = latest_main if ch.relative == 'reports/p3_02_main_data_gate.md' else latest_p1
            label = '최신 P3-02 완료 보고로 이동' if latest is latest_main else '최신 P1 종합으로 이동'
            story.append(Paragraph('과거 기록 / 현재 판정이 아니다. 원문에 남은 날짜·판정은 당시의 결과다. '
                                   f'<link href="#{latest.anchor}" color="#166A8F">{label}</link>', STYLES['quote']))
        story.extend(renderer.markdown(ch.path))
        story.append(PageBreak())
    story.append(annotated_title('별도 원본 근거 경로', 'evidence_catalog'))
    story.append(Paragraph('보고서에서 근거 번호를 누르면 아래의 해당 항목으로 이동한다. 파일 경로는 프로젝트 폴더 기준이며, 이 PDF에 원본 파일을 첨부한 것은 아니다. 실행 당시 source 안의 문서는 현재 문서와 구분해 보존한다.', STYLES['body']))
    story.append(Paragraph('<link href="#toc" color="#166A8F">전체 목차로 돌아가기</link>', STYLES['nav']))
    for ref in renderer.references.values():
        p = Paragraph(f'근거 {ref["id"]:03d}', STYLES['h3'])
        p._anchor = f'ref_{ref["id"]:03d}'
        story.append(p)
        story.append(Paragraph(safe_text(ref['path']), STYLES['small']))
    missing_glyphs = sorted({c for ch in chapters for c in normal_chars(ch.path.read_text()) if ord(c) not in glyphs and not c.isspace()})
    if missing_glyphs:
        raise ValueError(f'Font missing glyphs: {missing_glyphs}')
    doc.multiBuild(story)
    reader = PdfReader(output)
    page_ids = {page.indirect_reference.idnum: idx+1 for idx, page in enumerate(reader.pages)}
    link_counts = {'internal': 0, 'external': 0}
    bad = []
    targets = []
    for idx, page in enumerate(reader.pages):
        for a in page.get('/Annots', []):
            annotation = a.get_object()
            if annotation.get('/Subtype') != '/Link':
                continue
            destination = annotation.get('/Dest')
            action = annotation.get('/A')
            if destination is None and action and action.get('/S') == '/GoTo':
                destination = action.get('/D')
            if destination is not None:
                link_counts['internal'] += 1
                if isinstance(destination, str):
                    named = reader.named_destinations.get(destination)
                    destpage = reader.get_destination_page_number(named)+1 if named else None
                else:
                    destpage = page_ids.get(destination[0].idnum) if hasattr(destination[0], 'idnum') else None
                if destpage is None:
                    bad.append({'page': idx+1, 'destination': str(destination)})
                else:
                    targets.append({'from_page':idx+1,'to_page':destpage})
            elif action and action.get('/S') == '/URI':
                link_counts['external'] += 1
    if bad:
        raise ValueError(f'Broken PDF internal links: {bad[:10]}')
    manifest = {
        'record_kind': 'living_report_pdf_export',
        'exported_at': datetime.now(timezone(timedelta(hours=9))).isoformat(),
        'report_as_of_date': TODAY,
        'output': str(output.relative_to(ROOT)), 'pdf_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'page_count':len(reader.pages), 'chapter_count':len(chapters), 'completed_tasks':done, 'total_tasks':len(rows),
        'font': str(font_path),
        'layout_version':'readable-v2',
        'typography':{'body_pt':11,'body_leading_pt':18,'comparison_table_pt':9.5,'criterion_body_pt':10.3},
        'layout_stats':renderer.layout_stats,
        'chapters':[{'path':ch.relative,'title':ch.title,'group':ch.group,'historical':ch.historical,
                     'page':doc.chapter_pages[ch.relative],'anchor':ch.anchor,
                     'sha256':hashlib.sha256(ch.path.read_bytes()).hexdigest()} for ch in chapters],
        'references':list(renderer.references.values()), 'link_intents':renderer.link_intents,
        'pdf_link_counts':link_counts, 'resolved_internal_link_count':len(targets), 'broken_internal_links':bad,
        'limitations':['This export preserves report content; it is not a new robot experiment or personal competency assessment.',
                      'Nonincluded original files are listed by their exact path and hash; their bytes are not embedded in the PDF.',
                      'Visual review is recorded separately after rendering.'],
    }
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'output':str(output),'pages':len(reader.pages),'chapters':len(chapters),
                      'completed_tasks':done,'references':len(renderer.references), 'links':link_counts,
                      'broken_internal_links':len(bad)},ensure_ascii=False))


def main():
    global TODAY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'output/pdf/프로젝트_통합보고서.pdf')
    parser.add_argument('--as-of', default=TODAY, help='Client-facing report date, YYYY-MM-DD')
    args = parser.parse_args()
    datetime.strptime(args.as_of, '%Y-%m-%d')
    TODAY = args.as_of
    chapters, rows = chapters_from_board()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_pdf(args.output.resolve(), chapters, rows)


if __name__ == '__main__':
    main()
