#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash download_catlora_and_e2e_assets.sh
# 可选：
#   HF_TOKEN=xxx bash download_catlora_and_e2e_assets.sh
#   HUGGINGFACE_HUB_TOKEN=xxx bash download_catlora_and_e2e_assets.sh
#   DOWNLOAD_RETRIES=12 DOWNLOAD_RETRY_SLEEP=20 HF_DOWNLOAD_MAX_WORKERS=1 bash download_catlora_and_e2e_assets.sh
#
# 说明：
# - 下载到 Hugging Face / datasets 默认缓存路径。
# - 仅下载时临时移除代理环境变量，不改系统永久配置。
# - 所有 Python 命令固定在 bitvae conda 环境执行。
# - 弱网场景默认启用：断点续传 + 自动重试 + 低并发。

run_no_proxy() {
  env \
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u ALL_PROXY -u no_proxy -u NO_PROXY \
    "$@"
}

run_python_no_proxy() {
  local tmp_py rc
  tmp_py="$(mktemp /tmp/bitvae_download_XXXXXX.py)"
  cat > "${tmp_py}"
  if run_no_proxy \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_XET_HIGH_PERFORMANCE=0 \
    HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT}" \
    HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
    NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
    TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM}" \
    python "${tmp_py}"; then
    rc=0
  else
    rc=$?
  fi
  rm -f "${tmp_py}"
  return "${rc}"
}

if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "错误：当前环境不是 hif4。请先执行: conda activate hif4" >&2
  exit 1
fi

CPU_CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"

: "${DOWNLOAD_RETRIES:=8}"
: "${DOWNLOAD_RETRY_SLEEP:=15}"
: "${HF_HUB_DOWNLOAD_TIMEOUT:=120}"
: "${HF_HUB_ETAG_TIMEOUT:=30}"
: "${OMP_NUM_THREADS:=1}"
: "${MKL_NUM_THREADS:=1}"
: "${OPENBLAS_NUM_THREADS:=1}"
: "${NUMEXPR_NUM_THREADS:=1}"
: "${TOKENIZERS_PARALLELISM:=false}"

if [[ -z "${HF_DOWNLOAD_MAX_WORKERS+x}" ]]; then
  if [[ "${CPU_CORES}" =~ ^[0-9]+$ ]] && [[ "${CPU_CORES}" -ge 16 ]]; then
    HF_DOWNLOAD_MAX_WORKERS=3
  elif [[ "${CPU_CORES}" =~ ^[0-9]+$ ]] && [[ "${CPU_CORES}" -ge 8 ]]; then
    HF_DOWNLOAD_MAX_WORKERS=2
  else
    HF_DOWNLOAD_MAX_WORKERS=1
  fi
fi

if [[ "${CPU_CORES}" =~ ^[0-9]+$ ]] && [[ "${CPU_CORES}" -le 4 ]] && [[ "${HF_DOWNLOAD_MAX_WORKERS}" -gt 1 ]]; then
  echo "检测到 CPU 核心数=${CPU_CORES}，将 HF_DOWNLOAD_MAX_WORKERS 自动下调为 1。"
  HF_DOWNLOAD_MAX_WORKERS=1
fi

if [[ "${HF_DOWNLOAD_MAX_WORKERS}" -gt 4 ]]; then
  echo "检测到 HF_DOWNLOAD_MAX_WORKERS=${HF_DOWNLOAD_MAX_WORKERS} 过高，限制为 4 以保证稳定。"
  HF_DOWNLOAD_MAX_WORKERS=4
fi

echo "下载配置: CPU_CORES=${CPU_CORES}, HF_DOWNLOAD_MAX_WORKERS=${HF_DOWNLOAD_MAX_WORKERS}, DOWNLOAD_RETRIES=${DOWNLOAD_RETRIES}, DOWNLOAD_RETRY_SLEEP=${DOWNLOAD_RETRY_SLEEP}, HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT}, HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT}"

echo "[1/1] 预下载模型: Qwen/Qwen3.5-27B"
run_python_no_proxy <<'PY'
import os
import time
from huggingface_hub import snapshot_download

repo_id = "Qwen/Qwen3.5-27B"
retries = int(os.environ.get("DOWNLOAD_RETRIES", "8"))
sleep_s = int(os.environ.get("DOWNLOAD_RETRY_SLEEP", "15"))
max_workers = int(os.environ.get("HF_DOWNLOAD_MAX_WORKERS", "1"))
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

for attempt in range(1, retries + 1):
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            resume_download=True,
            max_workers=max_workers,
        )
        print(f"模型下载完成: {repo_id}")
        break
    except Exception as exc:
        if attempt >= retries:
            raise
        print(f"模型下载失败，将重试 ({attempt}/{retries})：{exc}")
        time.sleep(sleep_s)
PY

echo "全部下载完成。"
