# 문서와 근거를 관리하는 방법

## 학술 논문은 LaTeX로 관리

2026-10-19부터 학술 논문의 원본은 [paper/main.tex](../paper/main.tex)와 [참고문헌](../paper/references.bib)입니다. IEEEtran conference 클래스의 2단 조판, 번호 수식·표·그림, 본문 인용 및 BibTeX를 사용합니다. [논문 생성 안내](../paper/README.md)를 따릅니다. 아래의 Markdown→PDF 절차는 기존 실무 보고서와 설명용 기술 보고서에 적용합니다. 논문 PDF는 이 Markdown 변환 도구로 만들지 않습니다.

## 세 문서의 독자와 원본

| 문서 | 독자와 역할 | 원본과 생성 |
|---|---|---|
| 학술 논문 | 정식화·방법·실험·결과·논의를 심사하는 독자 | paper/main.tex 및 references.bib → LaTeX PDF |
| 설명용 기술 보고서 | 개발 경과·입출력·국소 진단·한계를 파악하는 독자 | docs/technical_report.ko.md → 보고서 PDF |
| 사업 관점 보고서 | 문제 가치·목표 달성·도입 판단을 검토하는 독자 | docs/business_report.ko.md → 보고서 PDF |

두 보고서는 Markdown을, 학술 논문은 TeX를 원본으로 유지합니다. 각 PDF는 해당 원고에서 생성하며 본문을 수작업으로 고치지 않습니다. 원고와 PDF의 출처 SHA는 각 PDF 옆 manifest에 기록합니다. 같은 실험을 다루므로 수치와 해석 범위는 세 문서에서 맞춰 검토합니다.

## 첨부 문서가 없어도 읽히는 PDF

기존 보고서의 근거 경로만 눌러서는 별도 MD 내용을 PDF 안에서 볼 수 없었습니다. 이번 두 보고서는 핵심 목표 달성표, 성공 분모, 실패 사례, 사실·가설의 구분을 본문에 직접 넣었습니다. 기술 보고서의 부록에는 핵심 증거와 재현 범위도 들어 있습니다.

세부 코드·JSON·자료·가중치는 GitHub에 별도 보관하고 PDF에는 공개 웹 링크를 넣습니다. 모든 수백 개 MD·원시를 합쳐 거대한 PDF를 만드는 대신, 결론 이해에 필요한 근거는 본문에 두고 세부 검산은 저장소로 연결합니다. 링크가 불가능한 오프라인 환경에서도 본문과 부록의 판단은 읽을 수 있습니다. 원시 전체를 PDF에 첨부한 것은 아닙니다.

GitHub는 Markdown과 PDF를 표시합니다. 다만 PDF 미리보기에서는 링크를 무시하므로 GitHub 안에서 탐색할 때는 MD를, 제출·오프라인 독서·PDF 목차 탐색에는 다운로드한 PDF를 사용합니다. [GitHub 공식 안내](https://docs.github.com/en/repositories/working-with-files/using-files/working-with-non-code-files).

## 갱신과 재생성

새 관찰은 기존 정식 결과를 바꾸지 않고 별도 후속 실험으로 기록합니다. 같은 수치를 고칠 때에는 원래 집계와 변경 이유를 보존하고 무엇을 검산했는지 적습니다. 공개 파일의 내용이 바뀌면 inventory와 공개본 검사 기록도 다시 생성해야 합니다. 마지막으로 확인한 검사 결과를 이후의 모든 커밋에 자동 적용하지 않습니다.

PDF는 `reportlab`, `pypdf`, `Pillow`가 있는 문서용 Python 환경과 한글 TTF를 사용해 프로젝트 루트에서 생성합니다. 실험용 환경의 패키지를 문서 제작 때문에 바꾸지 않아도 됩니다.

```bash
python tools/export_publication_reports.py --font /path/to/Korean-capable.ttf
```

`/path/to/Korean-capable.ttf`는 실제 설치된 글꼴 파일로 바꿉니다. macOS 기본 Arial Unicode 또는 Linux NanumGothic이 발견되면 `--font`를 생략할 수 있습니다. 글꼴 파일 자체는 저장소에 배포하지 않습니다.

생성 후 전체 페이지를 렌더링해 표·줄바꿈·한글·목차를 확인합니다. 텍스트 추출 검사는 화면 가독성의 대체 증거가 아닙니다. PDF만 고친 상태로 커밋하지 않고 원고·생성 도구·manifest·검사 기록을 함께 갱신합니다.

보고서 PDF의 날짜는 원고 상단의 날짜에서 읽습니다. 생성 도구와 레이아웃 코드의 SHA도 manifest에 기록합니다. 이전 검토 기록은 해당 날짜·SHA의 이력이며 현재 문서에 자동 적용하지 않습니다. 2026-10-22의 내용 판단과 미해결 실험 항목은 [리뷰어 관점 평가](reviewer_assessment.ko.md)에 정리했습니다.

2026-10-22 후속 갱신에서는 그림·표에 G/R/S1/SK의 방법 정의와 평가 분모를 추가했습니다. [목표 근처 보완 진단](near_goal_diagnostic_design.ko.md)은 Markdown 설계와 별도 JSON으로 등록하며 기존 동결 결과·논문 결과표와 합치지 않습니다. 설계와 실행 상태를 구별하고 첫 복원·후보 해시 검사를 통과하기 전에 새로운 미래 물리 결과를 기록하지 않습니다.
