# 공개 실험 자료와 재현 범위

이 폴더는 최종 실험의 **선별 공개본**입니다. 로컬의 동결 원본, 중단 실행, 전체 원시 자료는 변경하지 않았습니다. 공개 파일로 가능한 검산과 원시 자료가 필요한 재현을 구분합니다.

## 먼저 확인할 자료

| 자료 | 내용 | 공개본에서 확인할 수 있는 범위 |
|---|---|---|
| [E2 원행](results/E2/rows.json), [요약](results/E2/summary.json), [bootstrap](results/E2/bootstrap.npz) | 12개 초기조건 가족 × 8개 제어기, 96회 | 종료 유형·성공 수·전체 통계와 163개 bootstrap 배열 재생성 |
| [E1 요약](results/E1/summary.json), [snapshot 자료](results/E1/snapshots/) | 12개 상태, 16개 후보, 후보별 실제 반복 3회, 7개 모델 | 저장된 후보·예측·실제 미래 배열과 84개 모델 진단, 선택 결과 검산 |
| [고정 경로 예측 요약](results/fixed_prediction/summary.json) | 같은 G 경로에서 K1·K5·K10 예측 | 분모·무효 예측·조건부 오차와 가족별 수치 읽기. 전체 window trace는 미포함 |
| [실제 경계 궤적](results/trajectories/compact_visual_source.json) | E2 96회 실제 행동 경계 상태·종료·지표 | 궤적과 최종 상태 확인. 물리 substep 원시 기록을 대신하지 않음 |
| [실패 목록](audits/p5_final_failure_inventory.v1.json) | 63개 실패의 종료·조건 미충족 분류 | 실패를 삭제하지 않은 전체 분포 확인 |
| [선택 일치 분석](audits/p5_final_E1_choice_agreement.v1.json) | S1·SK 36쌍 중 35쌍 같은 후보 선택 | 주어진 E1 후보집합의 선택 일치 확인. 전체 온라인 정책의 동일성은 아님 |
| [별도 적용 결과](audits/p5_application_completion.v1.json) | 같은 초기조건 범위의 예약된 13번째 가족, 8개 제어기 | G 1회 성공·나머지 7회 timeout. OOD나 새 태스크 일반화 결과가 아님 |
| [학습 자료](learning/) | 4,800개 준비 전이, train/validation 분할과 학습창, 정규화 | 입력·출력·분할과 최종 모델 추론 확인. 전체 원시 자료 재수집 검증은 아님 |
| [최종 모델](models/) | R 1개, S1·SK 각각 seed 11·23·47 | 가중치·엄격한 메타데이터·Adam/RNG 상태를 포함한 기존 loader 사용 |
| [파일 출처 목록](provenance/publication_files.json) | 원본 상대 경로·원본 SHA·공개 경로·공개 SHA·변환 종류 | byte 동일 복사와 개인 경로 제거 파생본을 구분 |

E1의 독립 가족은 6개, E2의 독립 가족은 12개입니다. 후보·물리 반복·학습 seed·겹치는 예측 창을 독립 표본 수로 늘리지 않습니다.

## 실행

프로젝트 루트의 source checkout에서 실행합니다. Python 3.14.7 / arm64 Darwin에서 실제 검증했습니다. [정확한 설치 목록](../requirements-reproduction.txt)은 당시 장치의 패키지 snapshot이며, 해시 잠금이나 다른 운영체제의 설치 보장은 아닙니다. `pyproject.toml`을 통한 wheel 설치보다 소스와 `configs/`, `assets/`를 함께 둔 실행이 현재 지원 범위입니다.

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-reproduction.txt

# 표준 라이브러리: 공개 파일 SHA와 결과 개수 확인
.venv/bin/python -B scripts/verify_public_results.py

# 저장된 96행에서 E2 통계·bootstrap 전체 재생성, E1 저장 선택 검산
.venv/bin/python -B scripts/verify_public_results.py --recompute

# 최소 물리 1구간과 기존 모델 7개의 5단계 추론; 학습하지 않음
.venv/bin/python -B scripts/smoke_public.py
```

확인한 단위 검사는 다음의 **7개 모듈·149개**입니다. 전체 `unittest discover`는 이전 개발 실행을 fixture로 읽는 항목이 포함되어 공개 선별본만으로 통과를 약속하지 않습니다.

```sh
PYTHONPATH=src:scripts:tests .venv/bin/python -B -m unittest \
  test_p2_representation test_p2_dynamics test_p2_planning test_p2_task \
  test_p4_sequence_training test_p4_fork_checkpoint test_p5_final_statistics_v2
```

## 공개 과정의 변환

- 최종 모델 7개, 정규화, 준비 학습 자료와 두 통계/배열은 원본 byte를 유지했습니다.
- 개인 절대 경로가 있는 JSON·과거 진단 설정은 `${LOCAL_PROJECT_ROOT}`로 바꾼 **명시적 파생본**입니다. 내부의 원본 SHA는 원본 식별자이며, 변환된 파일이 그 SHA와 같다고 주장하지 않습니다. 공개본의 실제 SHA는 출처 목록의 `public_sha256`입니다.
- 오래된 그림 생성기의 개인 Python 기본 경로는 현재 실행 중인 Python으로 바꿨습니다. 일부 과거 분석 코드의 `${LOCAL_TEMP}`는 사용자가 정할 로컬 임시 경로 표시입니다. 이 과거 운영 스크립트를 그대로 실행하는 것은 위 공개 검산 절차가 아닙니다.
- E2 `rows.json`은 96개 원본 worker의 `row` 필드만 모았습니다. 값과 실패 유형을 바꾸지 않았고, 원본 worker SHA를 출처 목록에 연결했습니다.
- 공개용 두 검사 스크립트와 이 안내는 최종 동결 이후의 전달 도구입니다. 동결 과학 소스의 일부였다고 표시하지 않습니다.

## 포함하지 않은 자료

전체 `runs/` 약 16.3GB, 물리 substep 원시 기록, 모든 과거 실행·중단 결과·재현 witness, 전체 fixed-prediction trace, 로컬 가상환경과 PDF 시각 검수용 캐시는 Git에 올리지 않습니다. 이 자료는 로컬 원본에 보존되어 있습니다. 원본 경로가 적힌 audit는 **실제로 수행한 과거 검사에 대한 기록**이며, 이 선별 저장소에서 해당 검사를 전부 재실행할 수 있다는 뜻이 아닙니다.

따라서 공개본의 `PASS`는 파일 검산·저장 통계·엄격한 모델 추론·선정 단위 검사 범위입니다. 실제 제어 재현 33회나 전체 원시 물리 검증을 이번 공개 검사에서 새로 수행한 것으로 세지 않습니다. 학습 재개에는 저장된 Adam/RNG 외에도 공통 출발 checkpoint와 당시 샘플러·실행 계약 등이 필요하므로, 전체 P4/P5 운영 파이프라인의 즉시 재실행을 약속하지 않습니다.

## 과거 개발 경과의 원고

기술 보고서 §8의 수치와 진단 경과는 [P4 개발 원고 7개와 SHA 색인](development/index.md)에서 당시 기록으로 추적할 수 있습니다. 이 수치는 공개 준비 중 새로 검산한 측정값이 아닙니다. 과거 원고의 `HOLD`는 당시 판정이며, 진단 설정이나 모델이 최종 checkpoint로 승격됐다는 뜻이 아닙니다. 원문 내부의 상대 링크는 당시 원본 자료 식별자이고, 해당 자료 전체가 공개 저장소에 포함된 것은 아닙니다.
