#!/usr/bin/env bash

set -euo pipefail

NODES_FILE="${NODES_FILE:-/Users/jianzhengnie/work_dir/Kimi2-PCL/infer/nodel_liist.txt}"
IMAGE_NAME="${IMAGE_NAME:-quay.io/ascend/vllm-ascend:main-a3}"
IMAGE_TAR="${IMAGE_TAR:-/llm_workspace_1P/robin/hfhub/docker/image/vllm-ascend.main-a3.tar}"
RUN_CONTAINER_SCRIPT="${RUN_CONTAINER_SCRIPT:-/Users/jianzhengnie/work_dir/Kimi2-PCL/infer/ascend_infer_docker_run.sh}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-ascend-env-a3}"
VLLM_START_SCRIPT="${VLLM_START_SCRIPT:-/Users/jianzhengnie/work_dir/Kimi2-PCL/infer/vllm_start.sh}"

MASTER_NODE="${MASTER_NODE:-}"

SSH_USER_HOST_PREFIX="${SSH_USER_HOST_PREFIX:-}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
PARALLELISM="${PARALLELISM:-8}"

RAY_PORT="${RAY_PORT:-6379}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES:-1}"

VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"

usage() {
  cat <<'USAGE'
Usage:
  bash infer/cluster_deploy_ray_vllm.sh [--prepare-only|--ray-only|--serve-only]

Environment:
  NODES_FILE
  MASTER_NODE
  SSH_USER_HOST_PREFIX
  SSH_OPTS
  PARALLELISM

  IMAGE_NAME
  IMAGE_TAR
  RUN_CONTAINER_SCRIPT
  CONTAINER_NAME

  RAY_PORT
  ASCEND_RT_VISIBLE_DEVICES
  RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES

  VLLM_START_SCRIPT
  VLLM_HOST
  VLLM_PORT

Notes:
  - 节点名会按 NODES_FILE 每行一个读取，空行会被忽略
  - MASTER_NODE 为空时默认取节点列表第一行
USAGE
}

MODE="all"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
elif [[ "${1:-}" == "--prepare-only" ]]; then
  MODE="prepare"
elif [[ "${1:-}" == "--ray-only" ]]; then
  MODE="ray"
elif [[ "${1:-}" == "--serve-only" ]]; then
  MODE="serve"
elif [[ -n "${1:-}" ]]; then
  echo "Unknown option: $1" >&2
  usage >&2
  exit 2
fi

read_nodes() {
  awk 'NF {print $1}' "$NODES_FILE"
}

ensure_nodes_file() {
  if [[ ! -f "$NODES_FILE" ]]; then
    echo "NODES_FILE not found: $NODES_FILE" >&2
    exit 2
  fi
}

ssh_target() {
  local node="$1"
  printf "%s%s" "$SSH_USER_HOST_PREFIX" "$node"
}

ssh_run() {
  local node="$1"
  shift
  ssh ${SSH_OPTS} "$(ssh_target "$node")" "$@"
}

limit_jobs() {
  local max="$1"
  while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$max" ]]; do
    wait -n
  done
}

prepare_node() {
  local node="$1"
  ssh_run "$node" bash -lc "set -euo pipefail
if ! command -v docker >/dev/null 2>&1; then
  echo '[${node}] docker not found' >&2
  exit 127
fi
if docker image inspect '${IMAGE_NAME}' >/dev/null 2>&1; then
  :
else
  if [[ ! -f '${IMAGE_TAR}' ]]; then
    echo '[${node}] image tar not found: ${IMAGE_TAR}' >&2
    exit 2
  fi
  docker load -i '${IMAGE_TAR}'
fi
if [[ ! -f '${RUN_CONTAINER_SCRIPT}' ]]; then
  echo '[${node}] run script not found: ${RUN_CONTAINER_SCRIPT}' >&2
  exit 2
fi
bash '${RUN_CONTAINER_SCRIPT}'
docker ps --format '{{.Names}}' | grep -Fx '${CONTAINER_NAME}' >/dev/null
echo '[${node}] container ready: ${CONTAINER_NAME}'
"
}

detect_master() {
  if [[ -n "$MASTER_NODE" ]]; then
    echo "$MASTER_NODE"
    return 0
  fi
  read_nodes | head -n 1
}

ray_in_container() {
  local node="$1"
  shift
  local cmd="$*"
  ssh_run "$node" bash -lc "set -euo pipefail
docker exec '${CONTAINER_NAME}' bash -lc \"${cmd}\"
"
}

start_ray_head() {
  local node="$1"
  ray_in_container "$node" "
set -euo pipefail
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES='${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES}'
export ASCEND_RT_VISIBLE_DEVICES='${ASCEND_RT_VISIBLE_DEVICES}'
local_ip=\"\$(hostname -I 2>/dev/null | awk '{print \$1}')\"
if [[ -z \"\$local_ip\" ]]; then
  local_ip=\"\$(hostname -i 2>/dev/null | awk '{print \$1}')\"
fi
nic_name=\"\$(ip route show default 2>/dev/null | awk '{print \$5; exit}')\"
if [[ -z \"\$nic_name\" ]]; then
  nic_name=\"eth0\"
fi
export HCCL_IF_IP=\"\$local_ip\"
export GLOO_SOCKET_IFNAME=\"\$nic_name\"
export TP_SOCKET_IFNAME=\"\$nic_name\"
export HCCL_SOCKET_IFNAME=\"\$nic_name\"
ray stop -f || true
ray start --head --port='${RAY_PORT}' --node-ip-address=\"\$local_ip\"
echo \"\$local_ip\"
"
}

start_ray_worker() {
  local node="$1"
  local head_ip="$2"
  ray_in_container "$node" "
set -euo pipefail
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES='${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES}'
export ASCEND_RT_VISIBLE_DEVICES='${ASCEND_RT_VISIBLE_DEVICES}'
local_ip=\"\$(hostname -I 2>/dev/null | awk '{print \$1}')\"
if [[ -z \"\$local_ip\" ]]; then
  local_ip=\"\$(hostname -i 2>/dev/null | awk '{print \$1}')\"
fi
nic_name=\"\$(ip route get '${head_ip}' 2>/dev/null | awk '{for(i=1;i<=NF;i++) if(\$i==\\\"dev\\\"){print \$(i+1); exit}}')\"
if [[ -z \"\$nic_name\" ]]; then
  nic_name=\"\$(ip route show default 2>/dev/null | awk '{print \$5; exit}')\"
fi
if [[ -z \"\$nic_name\" ]]; then
  nic_name=\"eth0\"
fi
export HCCL_IF_IP=\"\$local_ip\"
export GLOO_SOCKET_IFNAME=\"\$nic_name\"
export TP_SOCKET_IFNAME=\"\$nic_name\"
export HCCL_SOCKET_IFNAME=\"\$nic_name\"
ray stop -f || true
ray start --address='${head_ip}:${RAY_PORT}' --node-ip-address=\"\$local_ip\"
"
}

serve_vllm_on_master() {
  local node="$1"
  ray_in_container "$node" "
set -euo pipefail
if [[ ! -f '${VLLM_START_SCRIPT}' ]]; then
  echo 'vllm start script not found: ${VLLM_START_SCRIPT}' >&2
  exit 2
fi
export HOST='${VLLM_HOST}'
export PORT='${VLLM_PORT}'
nohup bash '${VLLM_START_SCRIPT}' >/tmp/vllm_serve.log 2>&1 &
echo \$! >/tmp/vllm_serve.pid
"
  echo "vLLM log: ${node}:/tmp/vllm_serve.log"
}

ensure_nodes_file

nodes="$(read_nodes)"
if [[ -z "$nodes" ]]; then
  echo "No nodes found in NODES_FILE: $NODES_FILE" >&2
  exit 2
fi

master="$(detect_master)"
if [[ -z "$master" ]]; then
  echo "MASTER_NODE not set and cannot infer master from NODES_FILE" >&2
  exit 2
fi

if [[ "$MODE" == "prepare" || "$MODE" == "all" ]]; then
  for node in $nodes; do
    limit_jobs "$PARALLELISM"
    (prepare_node "$node") &
  done
  wait
fi

if [[ "$MODE" == "ray" || "$MODE" == "all" ]]; then
  head_ip="$(start_ray_head "$master" | tail -n 1)"
  if [[ -z "$head_ip" ]]; then
    echo "Failed to detect head ip on master: $master" >&2
    exit 1
  fi
  for node in $nodes; do
    if [[ "$node" == "$master" ]]; then
      continue
    fi
    limit_jobs "$PARALLELISM"
    (start_ray_worker "$node" "$head_ip") &
  done
  wait
  ssh_run "$master" bash -lc "docker exec '${CONTAINER_NAME}' ray status || true"
fi

if [[ "$MODE" == "serve" || "$MODE" == "all" ]]; then
  serve_vllm_on_master "$master"
fi
