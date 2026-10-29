# Workshop: Decision Layer for a Financial Support Agent / 금융 상담 Agent의 Decision Layer 실습

**EN** — Build a Korean banking-support agent with LangGraph, then swap its decision layer across several backends and measure speed, cost and quality. Notebooks come in two languages with a filename prefix: **`[EN]_`** (English) and **`[KR]_`** (Korean). Each notebook also ships an executed reference copy `*_solution.ipynb`. An SGLang server (:8001, Qwen3-VL-8B) must be running; `[EN]_07`/`[KR]_07` additionally use strands-decider on :8002. Customer utterances stay Korean in both language versions on purpose — Korean handling is part of what the workshop demonstrates.

**KR** — 한국어 금융 상담 agent를 LangGraph로 직접 조립하고, 판단(decision) 계층을 여러 backend로 바꿔가며 속도·비용·품질을 실측하는 실습입니다. 노트북은 파일명 prefix로 두 언어를 제공합니다: **`[KR]_`**(한국어), **`[EN]_`**(영어). 각 노트북에는 실행 완료본 `*_solution.ipynb`가 함께 있습니다. SGLang 서버(:8001, Qwen3-VL-8B)가 떠 있어야 하며, 07은 strands-decider(:8002)도 사용합니다.

## Contents / 대목차

| Folder | Notebooks (EN / KR) | Topic |
|---|---|---|
| `10_agent_foundation/` | `[EN]_01_build_agent.ipynb` / `[KR]_01_build_agent.ipynb` | Build the agent step by step with LangGraph; staged graph visualization; one standalone decision on Haiku vs SGLang / LangGraph로 agent를 한 조각씩 조립, 단계별 그래프 시각화 |
| `20_text_scenarios/` | `[EN]_02_scenario1_clarify` / `[KR]_02...` | Vague inquiry -> clarification loop / 모호한 문의 -> 되묻기 loop |
| | `[EN]_03_scenario2_retry` / `[KR]_03...` | Tool system error -> retry loop / tool 오류 -> 대체 tool 재시도 loop |
| | `[EN]_04_scenario3_risk_kyc` / `[KR]_04...` | Fraud flag -> KYC sub-flow + guardrail over-blocking on both backends / 이상거래 flag -> KYC + guardrail 과차단 체험 |
| `30_image_scenarios/` | `[EN]_05_scenario4_image` / `[KR]_05...` | The model reads approval-alert / phishing-SMS screenshots (EN uses `[EN]_*.png` fixtures, KR uses `[KR]_*.png`) / 스크린샷 판독 |
| `40_other_deciders/` | `[EN]_06_bedrock_intelligent_routing` / `[KR]_06...` | Bedrock Intelligent Prompt Routing (Nova live lab, custom router, $1/1,000 req) |
| | `[EN]_07_strands_decider` / `[KR]_07...` | Strands team's dedicated 2B decision model (:8002, same `/v1/systemone` wire format) |
| | `[EN]_08_laya` / `[KR]_08...` | Laya — open-source multilingual System 1 engine (322M), native `Router` + 20-line adapter |

Scenario notebooks follow the same shape: show the scenario's path on the full graph -> run with Haiku (STEP-by-STEP output) -> run with SGLang -> markdown comparison table. / 시나리오 노트북 공통 구성: 전체 그래프에서 경로 표시 -> Haiku 실행 -> SGLang 실행 -> 비교표.

> Perspective: the **decision layer** (SGLang in these labs) *removes* the decision LLM call; **Intelligent Prompt Routing** picks *which model* a generation call goes to — complements, not substitutes. / **decision layer**는 판단용 LLM 호출 자체를 제거하고, **IPR**은 생성 호출의 모델을 골라줍니다 — 대체재가 아니라 보완재.

## Measured results / 핵심 실측

**Korean decision quality** (160 conversations, D1/D2, `scripts/95_decider_quality_bench.py`):
SGLang Qwen3-VL-8B **0.913/0.629** > strands-decider 2B 0.613/0.500 > Laya multilingual 0.419/0.357 — model capacity dominated the quality ceiling over language specialization; small dedicated models presume fine-tuning (Laya official: 0.362 -> 0.766 tuned). Image decisions: Qwen3-VL only. / 언어 특화보다 **모델 용량이 품질 상한을 지배**, 소형 전용 모델은 fine-tuning 전제. 이미지 판단은 Qwen3-VL만 가능.

## Shared code / 공용 코드

- `ws_common.py` : graph building/drawing (`draw_graph`, `draw_build_stage`), streamed STEP output (`run_verbose`), comparison tables (`compare`), image-decision helpers. Language of helper output and graph captions follows **`DMWS_LANG`** (`ko` default, `en`) — the `[EN]_` notebooks set it in their first cell. / helper 출력·그래프 캡션 언어는 `DMWS_LANG` 환경변수(`ko` 기본, `en`)를 따르며 `[EN]_` 노트북 첫 셀에서 설정됩니다.

## References / 참고 문서

- Guide / 가이드: `../report/guide/202610271814_workshop_guide.md` (+pdf)
- 160-conversation A/B report / 실측 보고서: `../report/research/202610271445_agent_decision_ab_report.md`
