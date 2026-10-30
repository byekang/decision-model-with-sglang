#!/bin/bash
# Per-start setup for the workshop notebook instance. Run as root
# (nohup'd from the notebook lifecycle-config OnStart).
# STRANDS_HOST must be exported by the caller (the lifecycle config passes the
# strands-decider EC2 private IP); SGLang serves on this instance's GPU.
# Everything outside /home/ec2-user/SageMaker is wiped on stop/start, so the
# Jupyter kernel and its packages must be rebuilt here each time; the venv,
# model cache and repo live on the persistent volume (built by bootstrap.sh).
set -x
WORK=/home/ec2-user/SageMaker
PY=/home/ec2-user/anaconda3/envs/python3/bin
LOGS=$WORK/logs
STRANDS_HOST=${STRANDS_HOST:-localhost}
mkdir -p $LOGS
chown ec2-user:ec2-user $LOGS
exec >> $LOGS/onstart.log 2>&1

echo "===== onstart $(date -u +%FT%TZ) ====="

# graphviz + Korean fonts (graph captions; prevents jamo-decomposed rendering)
yum install -y -q graphviz || true
yum install -y -q google-noto-sans-cjk-ttc-fonts || yum install -y -q google-noto-sans-cjk-fonts || true
fc-cache -f || true

# workshop kernel. anthropic is version-pinned: newer SDKs send an API version
# header that Bedrock rejects ("Invalid API version" 400).
su ec2-user -c "$PY/pip install -q ipykernel 'anthropic==1.8.0' 'langgraph==1.2.18' boto3 graphviz pillow"
su ec2-user -c "$PY/python -m ipykernel install --user --name decision-ws \
  --display-name 'Python (decision-ws)' \
  --env SGLANG_HOST localhost --env STRANDS_HOST $STRANDS_HOST"
echo "export SGLANG_HOST=localhost"           > /etc/profile.d/dmws.sh
echo "export STRANDS_HOST=$STRANDS_HOST"      >> /etc/profile.d/dmws.sh

# laya (notebook 08): large CPU-torch install, run in background
nohup su ec2-user -c "$PY/pip install -q torch --index-url https://download.pytorch.org/whl/cpu && $PY/pip install -q laya" > $LOGS/laya-install.log 2>&1 &

# wait for the one-time bootstrap (first boot: venv/model still downloading)
for i in $(seq 1 240); do [ -f $WORK/.dmws_bootstrap_done ] && break; sleep 15; done
[ -f $WORK/.dmws_bootstrap_done ] || { echo ONSTART_BOOTSTRAP_TIMEOUT; exit 1; }

# SGLang on the GPU (:8001). PATH must include the venv bin so the JIT kernel
# compiler finds ninja — without it the server crashes during CUDA graph capture.
su ec2-user -c "cd $WORK && export PATH=$WORK/venvs/sglang/bin:\$PATH && HF_HOME=$WORK/hf-cache \
  nohup $WORK/venvs/sglang/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct --host 0.0.0.0 --port 8001 \
  --mem-fraction-static 0.87 --context-length 8192 \
  --attention-backend triton --enable-multimodal > $LOGS/sglang.log 2>&1 &"

for i in $(seq 1 120); do curl -sf http://localhost:8001/health >/dev/null && break; sleep 10; done
curl -sf http://localhost:8001/health >/dev/null && echo ONSTART_SGLANG_OK || echo ONSTART_SGLANG_FAIL
echo "===== onstart done $(date -u +%FT%TZ) ====="
