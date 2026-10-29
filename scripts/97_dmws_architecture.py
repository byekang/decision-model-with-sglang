"""Workshop architecture diagram (KR/EN) for the Workshop Studio content.

Reflects the current stack: SageMaker Notebook (ml.t3.xlarge) -> GPU EC2 g5.xlarge
(SGLang + Qwen3-VL-8B, :8001) / CPU EC2 c7i.2xlarge (strands-decider 2B, :8002),
all in a private subnet (no public IP), Bedrock via NAT.

Output: ws-studio static/images/dmws_architecture.png (KR),
        dmws_architecture_en.png (EN)
Usage: /data/venvs/agent/bin/python scripts/97_dmws_architecture.py
"""

import sys
from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.compute import EC2
from diagrams.aws.ml import SagemakerNotebook
from diagrams.onprem.client import Users

try:
    from diagrams.aws.ml import Bedrock
except ImportError:  # 구버전 diagrams fallback
    from diagrams.aws.ml import Sagemaker as Bedrock

sys.path.insert(0, str(Path(__file__).parent))
from common import log

PHASE = "dmws_architecture"
OUT_DIR = Path("/home/ubuntu/LLMRouter/ws-studio/decision-model-with-sglang/static/images")

TEXT = {
    "ko": {
        "file": "dmws_architecture",
        "cloud": "AWS 클라우드 (us-east-1)",
        "vpc": "VPC 10.1.0.0/16",
        "subnet": "프라이빗 서브넷 (public IP 없음, use1-az5)",
        "users": "워크샵 참가자",
        "nb": "SageMaker Notebook\nml.t3.xlarge, JupyterLab\n실습 노트북 + decision-ws kernel",
        "gpu": "GPU EC2 g5.xlarge\nSGLang + Qwen3-VL-8B\nDecision Model (:8001)",
        "cpu": "CPU EC2 c7i.2xlarge\nstrands-decider 2B\n(:8002, CPU 서빙)",
        "bedrock": "Amazon Bedrock\nClaude Sonnet (응답 생성)\nClaude Haiku (Arm A 판단)\nIntelligent Prompt Routing",
        "e1": "1. JupyterLab 접속",
        "e2": "2. 판단: /v1/systemone\n(prefill-only, VPC 내부)",
        "e3": "3. 판단: strands-decider\n(VPC 내부)",
        "e4": "4. 생성·판단 API 호출 (NAT 경유)",
    },
    "en": {
        "file": "dmws_architecture_en",
        "cloud": "AWS Cloud (us-east-1)",
        "vpc": "VPC 10.1.0.0/16",
        "subnet": "Private subnet (no public IP, use1-az5)",
        "users": "Workshop participants",
        "nb": "SageMaker Notebook\nml.t3.xlarge, JupyterLab\nhands-on notebooks + decision-ws kernel",
        "gpu": "GPU EC2 g5.xlarge\nSGLang + Qwen3-VL-8B\nDecision Model (:8001)",
        "cpu": "CPU EC2 c7i.2xlarge\nstrands-decider 2B\n(:8002, CPU serving)",
        "bedrock": "Amazon Bedrock\nClaude Sonnet (reply generation)\nClaude Haiku (Arm A decisions)\nIntelligent Prompt Routing",
        "e1": "1. open JupyterLab",
        "e2": "2. decisions: /v1/systemone\n(prefill-only, intra-VPC)",
        "e3": "3. decisions: strands-decider\n(intra-VPC)",
        "e4": "4. generation/decision API calls (via NAT)",
    },
}


def render(lang):
    t = TEXT[lang]
    attrs = {"fontname": "Noto Sans CJK KR", "fontsize": "13", "dpi": "140",
             "splines": "spline", "nodesep": "0.7", "ranksep": "0.9"}
    with Diagram("", filename=str(OUT_DIR / t["file"]), outformat="png", show=False,
                 direction="LR", graph_attr=attrs,
                 node_attr={"fontname": "Noto Sans CJK KR", "fontsize": "12"},
                 edge_attr={"fontname": "Noto Sans CJK KR", "fontsize": "11"}):
        users = Users(t["users"])
        with Cluster(t["cloud"], graph_attr={"fontname": "Noto Sans CJK KR"}):
            bedrock = Bedrock(t["bedrock"])
            with Cluster(t["vpc"], graph_attr={"fontname": "Noto Sans CJK KR"}):
                with Cluster(t["subnet"], graph_attr={"fontname": "Noto Sans CJK KR"}):
                    nb = SagemakerNotebook(t["nb"])
                    gpu = EC2(t["gpu"])
                    cpu = EC2(t["cpu"])
        users >> Edge(label=t["e1"]) >> nb
        nb >> Edge(label=t["e2"]) >> gpu
        nb >> Edge(label=t["e3"]) >> cpu
        nb >> Edge(label=t["e4"], style="dashed") >> bedrock
    log(PHASE, f"wrote {OUT_DIR / t['file']}.png")


if __name__ == "__main__":
    for lang in ("ko", "en"):
        render(lang)
