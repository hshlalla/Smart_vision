# 8. Appendices

이 부록 장은 본문에 모두 넣기에는 너무 길거나 기술적이지만, 프로젝트의 반복적 발전 과정을 보여주는 데 중요한 자료를 보존하기 위해 포함한다. 특히 본 프로젝트는 초기 draft 이후 설계와 구현이 크게 바뀌었기 때문에, 부록은 단순한 잔여 자료 보관함이 아니라 **초기 가설, 설계 변화, 사용자 근거, 확장 실험 자료**를 정리해 주는 보조 장으로 기능해야 한다. 초기 자료는 최종 구현과 다를 경우 반드시 `초기 설계안` 또는 `superseded design`으로 명확히 표기하는 것이 바람직하다.

## Appendix A. User Requirements and Survey Evidence

이 부록에는 1장에서 요약한 초기 요구사항 조사(`n = 6`)와 이후 usability pilot의 보조 자료를 함께 정리한다. survey 질문지, 참여 안내 문구, 정리된 응답 요약, requirement implication 메모, usability questionnaire summary, 그리고 `userbillaty.xlsx`를 바탕으로 정리한 cleaned response table을 포함할 수 있다. 다만 raw spreadsheet를 그대로 붙이기보다 해석 가능한 표나 요약 캡처 형태로 재구성하는 편이 더 적절하다.

## Appendix B. Evolution of System Architecture (Draft vs. Final)

이 부록은 초기 draft 이후 시스템 구조가 어떻게 바뀌었는지를 보여주기 위해 포함한다. 초기 architecture diagram, OCR-heavy pipeline sketch, Gradio-first 흐름, broader orchestration concept 등은 여기서 유지할 수 있다. 이러한 자료는 현재 설계와의 불일치가 아니라 evidence-driven iteration을 보여주는 근거로 사용해야 하며, 반드시 `초기 설계안` 또는 `superseded design`으로 명확히 표기해야 한다.

## Appendix C. User Interface Iterations

이 부록에는 본문과 영상에서 충분히 보여주지 못한 UI 자료를 정리한다. 로그인 페이지, 검색 페이지, 인덱싱 페이지, Catalog 페이지, Agent 페이지, 언어 전환 화면, metadata preview/edit/confirm 흐름, evidence-section close-up 등을 포함할 수 있다. 필요하다면 draft 시절의 Gradio 기반 화면도 `early prototype UI`로 라벨링해 함께 두어, 초기 프로토타입에서 현재 사용자 워크플로우로 발전한 과정을 보여줄 수 있다. 실제 캡션 초안은 `appendix_C_D_caption_drafts_ko.md`를 함께 참고한다.

## Appendix D. Qualitative Error Analysis (OCR Failure Cases)

이 부록은 5장에서 일부만 제시한 OCR 실패 사례와 retrieval failure 사례를 더 넉넉하게 정리하는 공간이다. 복잡한 배경, irrelevant specification noise, mixed vertical/horizontal text, engraved label, glare/blur/occlusion, wrong Top-1 but correct Top-5 사례 등을 짧은 캡션과 함께 제시하면 좋다. 이 부록의 목적은 실패가 우연이 아니라 도메인 특성에 의해 반복적으로 발생하는 구조적 현상임을 보여주는 데 있다. 실제 캡션 초안은 `appendix_C_D_caption_drafts_ko.md`를 함께 참고한다.

## Appendix E. Project Management and Risk Assessment

이 부록은 프로젝트 관리 자료와 risk assessment 자료를 함께 정리하는 공간이다. 기존 draft의 risk table, impact/mitigation 표, work breakdown structure, milestone plan, sprint summary, planning screenshot 등을 여기에 둘 수 있다. 이 자료는 scope와 구현 변경이 임의적이 아니라 기술적 제약, runtime limitation, evaluation evidence에 대한 대응이었다는 점을 보여주는 데 유용하다.

## Appendix F. Evaluation Data and Extended Results

이 부록은 4장과 5장의 확장판으로 사용한다. 본문에는 핵심 retrieval 결과, latency 결과, OCR benchmark 요약, usability pilot 요약만 남기고, 더 긴 세부표와 프로토콜 설명, query/index split note, dataset schema, API schema, QC evidence, extended result table 등은 이곳에 정리하는 것이 적절하다. 이렇게 하면 최종 운영 권고가 단일 지표가 아니라 여러 근거의 조합에서 도출되었다는 점을 더 투명하게 보여줄 수 있다.
