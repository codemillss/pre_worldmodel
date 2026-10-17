# P4 개발 경과 원고와 출처

기술 보고서 §8의 개발 경과를 추적하기 위한 당시 원고 7개입니다. `.source.txt`는 Markdown 원문을 byte 그대로 보존한 일반 텍스트 파일입니다. 이 문서 묶음의 수치는 **당시 수행한 진단·관찰의 기록**이며, 이번 공개 준비에서 원시 자료로 새로 재검산한 결과가 아닙니다.

원고 안의 상대 경로·링크는 **당시 원본 자료를 식별하는 표기**입니다. 대상 파일이 현재 선별 공개 저장소에 포함되거나 링크가 동작함을 약속하지 않습니다. 이 폴더의 색인에서 연결한 7개 TXT와 [SHA 목록](provenance.json)은 공개되어 있습니다.

| 당시 기록 | 공개 원문 | byte 수 |
|---|---|---:|
| 인계 비용 대조 | [p4_handoff_cost_comparison.source.txt](p4_handoff_cost_comparison.source.txt) | 17,501 |
| 후보11·77 국소 예측 진단 | [p4_local_prediction_diagnostic.source.txt](p4_local_prediction_diagnostic.source.txt) | 16,959 |
| 자료 보완 재학습 대조 | [p4_data_retraining_comparison.source.txt](p4_data_retraining_comparison.source.txt) | 18,108 |
| 접촉·정지 오차 기전 진단 | [p4_error_mechanism_diagnostic.source.txt](p4_error_mechanism_diagnostic.source.txt) | 16,299 |
| 상대위치 학습항 대조 | [p4_relative_loss_comparison.source.txt](p4_relative_loss_comparison.source.txt) | 17,702 |
| 다른 학습 배치의 갱신 방향 교차 평가 | [p4_cross_batch_comparison.source.txt](p4_cross_batch_comparison.source.txt) | 11,503 |
| 비접촉 방향의 Adam 첫 모멘트 대조 | [p4_first_moment_comparison.source.txt](p4_first_moment_comparison.source.txt) | 6,112 |

[원본 SHA와 공개본 SHA의 전체 목록](provenance.json)에서 각 파일의 byte 동일성을 확인할 수 있습니다. 이번 7개 원문에는 개인 절대 경로가 없어 변환하지 않았으며, 원본 MD와 공개 TXT의 SHA는 각각 같습니다.

과거 `HOLD`, `NOT_EVALUATED`, `NOT_DEMONSTRATED`는 해당 원고가 작성된 시점의 판정입니다. 이후 P5의 유효한 최종 비교 완료와 구분합니다. 비용·자료 보완·상대위치 항·Adam 진단에서 사용한 설정이나 모델을 최종 모델로 승격했다는 뜻은 아닙니다. 최종 비교 모델은 P4-02의 S1 및 SK λ0.1, update1000을 따릅니다.

이전 관찰이 제시한 원인 가설은 최종 결과의 단일 원인으로 확정하지 않습니다. 전체 원시 자료·모든 개발 모델·과거 실행을 이 폴더에 추가 복제하지 않았습니다.
