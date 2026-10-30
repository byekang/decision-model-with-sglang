#!/bin/bash
# One-time environment build for the workshop notebook instance
# (ml.g5.2xlarge: SGLang serves Qwen3-VL-8B on the local GPU; strands-decider
# runs on its own CPU EC2 — see the CloudFormation stack).
# Run as root (nohup'd from the notebook lifecycle-config OnCreate). Takes ~15 min.
# Everything lands under /home/ec2-user/SageMaker so it survives stop/start.
set -x
WORK=/home/ec2-user/SageMaker
PY=/home/ec2-user/anaconda3/envs/python3/bin
LOGS=$WORK/logs
mkdir -p $LOGS
chown ec2-user:ec2-user $LOGS
exec >> $LOGS/bootstrap.log 2>&1

echo "===== bootstrap start $(date -u +%FT%TZ) ====="

# venv: SGLang (GPU). ninja is required by SGLang's JIT kernel compiler.
su ec2-user -c "$PY/python -m venv $WORK/venvs/sglang"
su ec2-user -c "$WORK/venvs/sglang/bin/pip install -q --upgrade pip"
su ec2-user -c "$WORK/venvs/sglang/bin/pip install -q sglang ninja" || { echo BOOTSTRAP_SGLANG_PIP_FAIL; exit 1; }

# model weights into the persistent volume
su ec2-user -c "HF_HOME=$WORK/hf-cache $WORK/venvs/sglang/bin/python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-8B-Instruct')\"" || { echo BOOTSTRAP_HF_QWEN_FAIL; exit 1; }

su ec2-user -c "touch $WORK/.dmws_bootstrap_done"
echo "===== bootstrap done $(date -u +%FT%TZ) ====="
