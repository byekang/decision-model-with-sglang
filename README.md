# Decision Model with SGLang — 한국어 금융 상담 Agent Workshop

LangGraph로 만든 한국어 금융 상담 agent의 **판단(decision) 계층**을 여러 backend로 바꿔가며
속도·비용·품질을 실측 비교하는 hands-on workshop입니다.

- 판단 backend: Amazon Bedrock Haiku 4.5 / **SGLang `/v1/systemone` + Qwen3-VL-8B** /
  strands-decider 2B / Laya 322M (+ 참고: Bedrock Intelligent Prompt Routing)
- 응답 생성: Amazon Bedrock Sonnet 4.5 (모든 arm 공통)
- workflow: 결정 노드 8종(D1-D8) + recursive loop 2개(되묻기, tool 재시도) + KYC sub-flow

## 구성 (workshop/)

| 폴더 | 내용 |
|---|---|
| `10_agent_foundation/` | 01 — LangGraph step-by-step 조립 (단계별 그래프 시각화, 금융 업무 관점 해설) |
| `20_text_scenarios/` | 02-04 — 한국어 상담 3종: 되묻기 loop / tool 재시도 loop / KYC sub-flow. Haiku vs SGLang 비교 |
| `30_image_scenarios/` | 05 — 승인 알림·피싱 문자 스크린샷 판독 (Qwen3-VL) |
| `40_other_deciders/` | 06 Bedrock Intelligent Prompt Routing, 07 strands-decider, 08 Laya |

각 노트북에는 실행 완료본 `*_solution.ipynb`(기대 출력 포함)이 함께 있습니다.
대목차와 핵심 실측 수치는 `workshop/README.md`, 진행 가이드는 `report/guide/202610271814_workshop_guide.md`(+pdf) 참조.

## 핵심 실측 결과 요약

- 결정 latency 합계(상담 1건): Haiku 약 5.2-8.5초 -> SGLang 약 0.28-0.33초 (**약 19-26배**)
- 이미지 판단(한국어 스크린샷): Qwen3-VL 94-117ms vs Haiku 0.85-1.7초
- 한국어 판단 품질(160건 D1/D2): SGLang 8B 0.913/0.629 > strands-decider 2B 0.613/0.500 > Laya 322M 0.419/0.357
- guardrail(D1) 주의: 소형 decision model은 한국어 피해 신고 표현을 과차단(conf 1.0 포화) — 안전 판단은 상위 모델 유지 권고

## 실행 환경 요구사항

이 저장소는 아래 환경을 전제로 작성되었습니다 (경로가 코드에 포함되어 있음):

- 프로젝트 루트: clone 위치 무관 (`workshop/ws_common.py`가 자기 위치 기준으로 자동 탐지, 필요 시 `DMWS_ROOT` 환경변수로 지정)
- 모델 서버 주소: 기본 `localhost`, 원격 서버 사용 시 `SGLANG_HOST` 환경변수에 서버 IP 지정 (SGLang :8001, strands-decider :8002 공통)
- GPU: NVIDIA H100/A100 등 (Qwen3-VL-8B 서빙)
- SGLang v0.5.21 서버: `Qwen/Qwen3-VL-8B-Instruct`, port 8001, `--attention-backend triton --enable-multimodal`
- (07) strands-decider 서버: `strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8002 --device cuda`
- (08) `pip install laya` (Jupyter kernel venv에)
- AWS 자격 증명: us-west-2 Bedrock(Haiku 4.5/Sonnet 4.5 `us.` inference profile) 호출 권한
- Python 3.12 venv에 `anthropic`, `langgraph`, `boto3`, `graphviz`(+ 시스템 `dot`), `ipykernel`
- 전체 노트북 1회 실행 비용: 약 $1-2 (Bedrock)

```bash
# 서버 기동 예 (경로/venv는 환경에 맞게)
python -m sglang.launch_server --model-path Qwen/Qwen3-VL-8B-Instruct \
  --port 8001 --mem-fraction-static 0.75 --attention-backend triton --enable-multimodal

jupyter lab workshop/
```

## 저장소 내용물

- `agent/` — LangGraph workflow(graph_v2.py), 결정 backend 클라이언트(decisions.py), prompt/tool 정의
- `workshop/` — 노트북 8종(+solution), 공용 helper `ws_common.py`(단계별 STEP 출력, 한글 그래프 시각화, 비교표)
- `scripts/` — 시연 이미지 생성(92), 판단 반복 측정(93), workflow 데모(94), backend 품질 벤치(95), 공용 유틸(common.py)
- `data/` — 합성 스크린샷 3종, 한국어 160건 평가 데이터(agent_convs_v2ko.jsonl, Banking77 기반 번역)
- `results/summary/` — 실측 결과 JSON 3종
- `report/` — workflow 다이어그램, 진행 가이드(md/pdf)

모든 은행·카드·금액·스크린샷은 합성 데이터입니다(가상의 "KB한빛").
Banking77 데이터셋(원 출처: PolyAI, CC-BY-4.0)을 번역·가공해 평가에 사용했습니다.
