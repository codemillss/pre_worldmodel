# 학술 논문 원고

이 폴더의 [main.tex](main.tex)가 논문 원본이며 [references.bib](references.bib)와 [figures](figures/)를 함께 관리합니다. IEEEtran conference LaTeX 클래스(1.8b)의 A4·10pt·2단 구성을 사용한 한국어 학술 원고입니다.

- 제목과 초록·주요어
- I. 서론
- II. 관련 연구
- III. 문제 정식화
- IV. 방법
- V. 실험 설정
- VI. 결과
- VII. 논의
- VIII. 결론
- 재현성·도구 사용과 IEEE 번호형 참고문헌

수식은 LaTeX의 실제 수학 조판을 사용하며 그림·표·수식·문헌 번호를 본문에서 상호 참조합니다. 프로젝트 단계 완료표·개인 역량 판정·면접 활용·문서 운영 설명은 논문 본문에 넣지 않았습니다.

투고 학회·학술지가 지정되지 않은 원고입니다. 해당 학회에서 심사·게재되었거나 공식 제출 규정 전체를 충족했다고 표시하지 않습니다. 저자란은 저장소 식별자 codemillss를 사용하며 소속·직책은 만들지 않았습니다.

## 읽기

[조판 PDF](../reports/Research_Paper.ko.pdf)를 사용합니다. 기존 [기술 보고서](../docs/technical_report.ko.md)는 프로젝트 전반의 설명용 보조 문서로 보존합니다.

## 생성

XeLaTeX와 BibTeX를 사용할 수 있는 TeX Live 환경 또는 Tectonic에서 컴파일합니다. IEEEtran, kotex, amsmath, amssymb, bm, graphicx, booktabs, array, cite, hyperref, url과 TeX Gyre·ko.TeX 기본 한글 글꼴을 사용합니다.

    cd paper
    tectonic main.tex

원본 작업에서는 Tectonic 0.17.0을 사용하여 TeX와 BibTeX를 실제 실행했습니다. 공식 배포 파일의 공개 SHA256을 대조했으며 빌드 도구·TeX 캐시·컴파일 중간 파일은 Git에 포함하지 않습니다. 시스템 전용 글꼴은 사용하지 않습니다.

그림 생성기는 이미 저장된 결과로부터 그림을 생성합니다. Matplotlib·NumPy가 필요하며 새 학습이나 물리 실험을 실행하지 않습니다. 개념도와 측정 결과를 구분하고 예측 그림은 공통 유효 창에서 비교 가능한 seed 23·47만 사용합니다.

실험 데이터·모델·학습·계획 설정은 변경하지 않았습니다. 결과와 검증 가능한 범위는 [실험 근거 색인](../artifacts/index.md)에 기록합니다.

## 내용 재검토

2026-10-21에 수식·코드·표·그림 및 문제–방법–실험–결론의 연결을 다시 검토했습니다. 기준선과 선정 절차, 비용과 유지 성공의 차이, E1의 방향 제한 및 위치 예측 개선의 범위를 보완했습니다. 현재 PDF는 6쪽이며 수치와 원시 실험 결과를 교체하지 않았습니다.

- [발견 사항·수정 및 남은 연구 한계](content_review.ko.md)
- [현재 내용·빌드·전체 페이지 검토 기록](../artifacts/validation/paper_content_review.v2.json)
- [수치 재계산 기록](../artifacts/validation/paper_results_review.v2.json)
- [이전 버전의 형식·빌드 검토 기록](../artifacts/validation/paper_review.json)

저장된 E1/E2 통계, 예측 표 및 그림의 family 평균을 확인하는 공개 검산은 다음과 같습니다. NumPy가 필요하며 모델 호출·물리 실행·재학습은 하지 않습니다.

    python -B scripts/verify_paper_results.py

위 명령은 저장소 루트에서 실행합니다. 포함하지 않은 원시 자료 전체나 고정 예측 창의 재추론을 검증하는 명령은 아닙니다.
