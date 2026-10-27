# Workshop: 금융 상담 Agent의 Decision Layer 실습

한국어 금융 상담 agent를 직접 조립하고, 판단(decision) 계층을 여러 방식으로 바꿔가며 속도·비용·품질을 체험하는 실습입니다. Jupyter kernel은 **Python (agent venv)** 를 선택하세요. SGLang 서버(:8001, Qwen3-VL-8B)가 떠 있어야 합니다.

## 대목차

### 10_agent_foundation — Agent 기초
- `01_build_agent.ipynb` : LangGraph로 상담 agent를 한 조각씩 조립. 단계마다 그래프가 자라나는 과정을 시각화. 판단 1회를 Haiku vs SGLang으로 단건 체험.

### 20_text_scenarios — 한국어 상담 시나리오 (Haiku vs SGLang decision)
- `02_scenario1_clarify.ipynb` : 모호한 문의 → 되묻기(clarification) loop
- `03_scenario2_retry.ipynb` : tool 시스템 오류 → 대체 tool 재시도 loop
- `04_scenario3_risk_kyc.ipynb` : 이상거래 flag → KYC 재검증 sub-flow + guardrail 과차단을 양쪽 backend로 체험

각 노트북: 전체 그래프에서 이 시나리오가 지나는 경로를 먼저 표시 → Haiku 실행(STEP별 출력) → SGLang 실행 → 수행시간 비교표(markdown).

### 30_image_scenarios — 이미지 상담 시나리오 (Qwen3-VL)
- `05_scenario4_image.ipynb` : 승인 알림·피싱/정상 문자 스크린샷을 모델이 직접 읽고 판정 (증빙 분류, 사건 분류, 피싱 판별)

### 40_other_deciders — 다른 decision/routing 제품 체험
- `06_bedrock_intelligent_routing.ipynb` : Amazon Bedrock Intelligent Prompt Routing — 요청 난이도를 예측해 모델 family 안에서 생성 모델을 골라주는 관리형 라우터 (Nova 라이브 실습, custom router 생성, $1/1,000건 과금)
- `07_strands_decider.ipynb` : Strands 팀의 전용 decision model(2B, `/v1/systemone` 서빙, :8002). 기존 client에 port만 바꿔 **우리 LangGraph에 그대로 꽂아** 상담 예제 실행
- `08_laya.ipynb` : Laya — 오픈소스 다국어 System 1 decision engine(322M). Python native `Router` + 20줄 adapter로 graph에 연결

> 방식 비교 관점: **decision layer(본 실습의 SGLang)** 는 "판단용 LLM 호출 자체를 제거"하고, **Intelligent Prompt Routing**은 "생성 호출이 어느 모델로 갈지"를 골라줍니다 — 서로 대체재가 아니라 보완재입니다.

**한국어 판단 품질 실측** (160건 D1/D2, `scripts/95_decider_quality_bench.py`): SGLang Qwen3-VL-8B **0.913/0.629** > strands-decider 2B 0.613/0.500 > Laya multilingual 0.419/0.357 — 언어 특화보다 **모델 용량이 품질 상한을 지배**했고, 소형 전용 모델은 fine-tuning이 전제(Laya 공식: 튜닝 시 0.362→0.766). 이미지 판단은 Qwen3-VL만 가능.

## 공용 코드
- `ws_common.py` : 그래프 생성/시각화(`draw_graph`, `draw_build_stage`), 단계별 실행 출력(`run_verbose`), 비교표(`compare`), 이미지 판단 helper

## 참고 문서
- 가이드: `../report/guide/202610271814_workshop_guide.md` (+pdf)
- 160건 A/B 실측 보고서: `../report/research/202610271445_agent_decision_ab_report.md`
