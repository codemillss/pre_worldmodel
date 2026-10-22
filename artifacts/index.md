# 공개 실험 자료와 재현 범위

이 폴더는 최종 실험의 **선별 공개본**입니다. 로컬의 동결 원본, 중단 실행, 전체 원시 자료는 변경하지 않았습니다. 공개 파일로 가능한 검산과 원시 자료가 필요한 재현을 구분합니다.

## 먼저 확인할 자료

| 자료 | 내용 | 공개본에서 확인할 수 있는 범위 |
|---|---|---|
| [E2 원행](results/E2/rows.json), [요약](results/E2/summary.json), [bootstrap](results/E2/bootstrap.npz) | 12개 초기조건 가족 × 8개 제어기, 96회 | 종료 유형·성공 수·전체 통계와 163개 bootstrap 배열 재생성 |
| [E1 요약](results/E1/summary.json), [snapshot 자료](results/E1/snapshots/) | 초기·2.9초의 12개 접근 상태, 16개 후보, 실제 반복 3회, 7개 모델 | 저장된 84개 진단과 379개 bootstrap 배열 재생성. 종단 정렬·유지 상태를 표집한 진단은 아님 |
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

# 논문 수치·그림, E1 379/E2 163 bootstrap 배열 및 사후 범위/짝분석 검산
.venv/bin/python -B scripts/verify_paper_results.py

# 최소 물리 1구간과 기존 모델 7개의 5단계 추론; 학습하지 않음
.venv/bin/python -B scripts/smoke_public.py
```

최초 공개 과정에서 확인한 단위 검사는 다음의 **7개 모듈·149개**입니다. 전체 `unittest discover`는 이전 개발 실행을 fixture로 읽는 항목이 포함되어 공개 선별본만으로 통과를 약속하지 않습니다. 이후 문서 변경마다 이 검사를 모두 새로 실행했다는 뜻은 아닙니다.

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
- 공개용 검산·smoke 스크립트와 이 안내는 최종 동결 이후의 전달 도구입니다. 동결 과학 소스의 일부였다고 표시하지 않습니다.

## 포함하지 않은 자료

전체 `runs/` 약 16.3GB, 물리 substep 원시 기록, 모든 과거 실행·중단 결과·재현 witness, 전체 fixed-prediction trace, 로컬 가상환경과 PDF 시각 검수용 캐시는 Git에 올리지 않습니다. 이 자료는 로컬 원본에 보존되어 있습니다. 원본 경로가 적힌 audit는 **실제로 수행한 과거 검사에 대한 기록**이며, 이 선별 저장소에서 해당 검사를 전부 재실행할 수 있다는 뜻이 아닙니다.

따라서 공개본의 `PASS`는 파일 검산·저장 통계·엄격한 모델 추론·선정 단위 검사 범위입니다. 실제 제어 재현 33회나 전체 원시 물리 검증을 이번 공개 검사에서 새로 수행한 것으로 세지 않습니다. 학습 재개에는 저장된 Adam/RNG 외에도 공통 출발 checkpoint와 당시 샘플러·실행 계약 등이 필요하므로, 전체 P4/P5 운영 파이프라인의 즉시 재실행을 약속하지 않습니다.

## 과거 개발 경과의 원고

기술 보고서 §8의 수치와 진단 경과는 [P4 개발 원고 7개와 SHA 색인](development/index.md)에서 당시 기록으로 추적할 수 있습니다. 이 수치는 공개 준비 중 새로 검산한 측정값이 아닙니다. 과거 원고의 `HOLD`는 당시 판정이며, 진단 설정이나 모델이 최종 checkpoint로 승격됐다는 뜻이 아닙니다. 원문 내부의 상대 링크는 당시 원본 자료 식별자이고, 해당 자료 전체가 공개 저장소에 포함된 것은 아닙니다.

## 최신 문서 검토

2026-10-22의 [리뷰어 관점 평가](../docs/reviewer_assessment.ko.md)는 E1의 시간·목표 거리 범위와 E2의 짝별 성공 차이를 저장 배열에서 추가 검산했습니다. 이는 사후 기술 분석이며 새 학습·모델 추론·물리 실행이 아닙니다. 현재 문서·PDF 시각 검토는 [v4 검토 기록](validation/document_review.v4.json), 현재 원고 수치 검산은 [v4 수치 기록](validation/paper_results_review.v4.json)에 연결합니다. 이전 검토 기록은 당시 SHA의 이력입니다.

[목표 근처·실패 직전 진단 설계 v1](../docs/near_goal_diagnostic_design.ko.md)은 별도 후속 개발 계획입니다. [6상태 선정 근거](validation/followup_state_selection.v1.json)는 기존 저장 자료·원본 witness만 읽은 기록이며 snapshot 복원·모델 추론·실제 미래 실행의 증거가 아닙니다. 원본 전체 raw가 공개되지 않았으므로 선정 출처의 local-original 경로는 공개본에서 직접 열리지 않을 수 있습니다.

## 후속 개발: 복원·작은 사전 실행

2026-10-22 D1/D2의 [실행 결과](../docs/followup_restore_and_pilot.ko.md)와 [선별 공개 자료](followup/20261022-d1-d2/manifest.json)를 추가했습니다. 전체 192개 raw는 로컬 원본에 보존하고, 공개본에는 복원 보고·일곱 모델 예측·첫 반복 경계/유지 배열과 raw 해시를 포함합니다. [독립 raw 검토](validation/followup_execution_review.v1.json)와 [공개 산식 검산](validation/followup_public_results.v1.json)의 범위를 구별합니다. D3 전체128후보 진단과 조건부 확장은 미실행입니다.

이후 [저장·재현 비용 분리 측정](../docs/followup_resource_profile.ko.md)을 추가했습니다. 같은 9조합의 계산 3반복과 기록 3회 측정, [해시·요약](followup/20261022-resource-profile/manifest.json), [2시간 자원 계산서](followup/20261022-resource-profile/resource_summary.json), [독립 검토](validation/followup_profile_execution_review.v1.json)를 연결합니다. 원래 자료와 새 측정 raw는 로컬에 보존했고, N-T 전체128후보 부분 진단은 실행기 사전 검사 권고 상태입니다.

그 뒤 [N-T 전체128 사전 검사](../docs/nt_fullbank_preflight.ko.md)를 완료했습니다. [118파일 선별 자료](followup/20261022-nt-preflight/manifest.json)는 전체128 예측·18건 사전 실행 영수증·기존80행/반복640행 평가 기록입니다. 전체 실제128후보 평가가 완료된 자료는 아닙니다. [적용 범위별 예산 v2](validation/nt_fullbank_resource_budget.v2.json)는 기존 일괄 계산을 보존하면서 원본 D2 대조96건·완전 결과256그룹을 분리했습니다. [독립 실행 검토](validation/followup_nt_preflight_execution_review.v2.json), [공개 검산](validation/nt_preflight_public_checks.v2.json), [초기24파일 가정 정정](validation/nt_preflight_binding_correction.v1.json)을 함께 확인할 수 있습니다. 첫 실패 실행과 당시 코드·검사는 `development/nt-preflight-attempt-01`에 보존했습니다.

v1 설계의 미실행 표시는 당시 기록이며 현재 진행 상태는 위 실행 결과가 기준입니다. 기존 학술 논문·보고서와 v4 검토는 원래 동결 비교의 문서 이력으로 유지합니다.


## N-T 전체 진입·자원 상한 실행

[전체 실행 보고](../docs/nt_fullbank_execution.ko.md): 128후보×7모델의 예측 봉인과 진입 검사는 통과했지만, 새 실제 미래는 **70/768회**에서 `STOP_RESOURCE`로 중단됐습니다. 관측 최대 전체 결과 저장·복원 단가가 증가해 남은 검증·최종 정리를 포함한 예상이 121.73분이 됐기 때문입니다. 실제 누적 실행은 15.88분입니다. 698회는 미실행이며 전체 후보 순위나 인계 개선으로 해석하지 않습니다.

[선별 공개 자료](followup/20261022-nt-full/manifest.json)에는 전체128 예측, 완료 행의 요약·기록, 첫 반복 경계 배열과 로컬 원시·전체 결과의 해시 목록을 제공합니다. [저장 자료 분석](followup/20261022-nt-full/analysis.json)은 부분집합임을 명시하며, [독립 사후 감사](validation/nt_full_execution_review.v1.json)는 전수 해시·판정과 제한된 독립 복원 표본의 범위를 구분합니다. 새 teacher-forced 평가는 전체 실제 실행 이후 단계여서 미실행입니다. 기존 논문·정식 결과와 raw는 보존했습니다.


## 저장·복원 묶음의 비용 분리 측정

[계측 보고](../docs/nt_storage_profile.ko.md)는 저장된6사례×3회와 원래 함수3회 대조를 다룹니다. 전체 exact 검증을 유지했고21개 복제 파일 쌍이 원래 사례와 바이트 단위로 같았습니다. 새 물리·모델 호출은 없습니다. 후보11/B의 평균6.171초 중71.9%가 네 번의 exact 비교였으며, 단순한 디스크 쓰기 병목으로 설명하지 않습니다.

[선별 공개 자료](followup/20261022-nt-storage-profile/manifest.json)에 단계별 타이밍·중첩 합산 검증·집계·잔여 예산을 제공합니다. [독립 감사](validation/nt_storage_profile_review.v1.json)는 파일 해시와 기록 검산 범위이며 전체 트리를 다시 디코딩한 의미 검증이 아닙니다. 원래6.159초와 새 원래 함수 관측 최대6.429초를 보존하고, 잔여698회는 누적2시간 예상 초과로 보류했습니다. 프로파일링 반복은 새 실제 태스크 표본이 아닙니다.
