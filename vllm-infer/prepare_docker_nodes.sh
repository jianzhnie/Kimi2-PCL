#!/usr/bin/env bash
# ==========================================
# Docker 容器准备脚本 (prepare_docker_nodes.sh)
# 用于在集群节点上准备 Docker 容器环境
# ==========================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/set_env.sh"

# ------------------------------------------
# 引入环境变量
# ------------------------------------------
if [[ -f "${ENV_FILE}" ]]; then
  source "${ENV_FILE}"
else
  echo "[ERROR] 环境配置文件未找到: ${ENV_FILE}" >&2
  exit 1
fi

# ------------------------------------------
# 日志函数
# ------------------------------------------
log_info() { echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $*"; }
log_err()  { echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2; }

# ------------------------------------------
# 帮助信息
# ------------------------------------------
usage() {
  cat <<'USAGE'
Usage:
  bash prepare_docker_nodes.sh [start|stop] [OPTIONS]

Description:
  该脚本用于在集群节点上准备 Docker 容器环境。
  环境变量请在同目录下的 set_env.sh 中配置。

Options:
  -h, --help           显示帮助信息
  -a, --action <start|stop>
                       start: 优雅停止并清理旧容器，加载镜像并启动新容器（默认）
                       stop:  仅优雅停止并清理旧容器，不启动新容器
USAGE
}

# ------------------------------------------
# 参数解析
# ------------------------------------------
ACTION="start"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -a|--action)
      if [[ -n "${2:-}" && "${2:-}" != -* ]]; then
        ACTION="$2"
        shift 2
      else
        log_err "选项 $1 需要一个参数: start 或 stop"
        usage
        exit 1
      fi
      ;;
    start|stop)
      ACTION="$1"
      shift
      ;;
    *)
      log_err "未知参数: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "$ACTION" != "start" && "$ACTION" != "stop" ]]; then
  log_err "动作参数必须是 start 或 stop，当前值: $ACTION"
  usage
  exit 1
fi

# ------------------------------------------
# 辅助函数
# ------------------------------------------

# 获取非空节点列表
read_nodes() {
  if [[ ! -f "$NODES_FILE" ]]; then
    log_err "节点列表文件未找到: $NODES_FILE"
    exit 2
  fi
  awk 'NF {print $1}' "$NODES_FILE"
}

# 拼接 SSH 目标地址
ssh_target() {
  local node="$1"
  printf "%s%s" "$SSH_USER_HOST_PREFIX" "$node"
}

# 执行 SSH 命令
ssh_run() {
  local node="$1"
  shift
  ssh ${SSH_OPTS} "$(ssh_target "$node")" "$@"
}

# 并发数控制
limit_jobs() {
  local max="$1"
  while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$max" ]]; do
    wait -n
  done
}

# ------------------------------------------
# 容器准备远程执行逻辑定义
# 注意: 下列带 _remote 前缀的函数将通过 declare -f 被序列化发送至远程执行
# ------------------------------------------

_remote_prepare_node() {
  local image_name="$1"
  local image_tar="$2"
  local run_container_script="$3"
  local container_name="$4"
  local action="${5:-start}"

  set -euo pipefail
  if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] docker not found" >&2
    exit 127
  fi

  # 确保 Docker 服务已启动
  echo "[INFO] Starting Docker service..."
  systemctl daemon-reload && systemctl start docker || {
    echo "[WARN] Failed to start Docker via systemctl, trying to check Docker status..." >&2
  }

  # 优雅 stop & kill 并删除所有已存在的容器
  echo "[INFO] Stopping and removing all existing containers..."
  docker ps -aq 2>/dev/null | xargs -r docker stop 2>/dev/null || true
  docker ps -aq 2>/dev/null | xargs -r docker kill 2>/dev/null || true
  docker ps -aq 2>/dev/null | xargs -r docker rm -f 2>/dev/null || true

  if [[ "$action" == "start" ]]; then
    if docker image inspect "${image_name}" >/dev/null 2>&1; then
      : # 镜像已存在
    else
      if [ ! -f "${image_tar}" ]; then
        echo "[ERROR] image tar not found: ${image_tar}" >&2
        exit 2
      fi
      echo "[INFO] Loading image from ${image_tar}..."
      docker load -i "${image_tar}"
    fi

    if [ ! -f "${run_container_script}" ]; then
      echo "[ERROR] run script not found: ${run_container_script}" >&2
      exit 2
    fi

    export IMAGE_NAME="${image_name}"
    export CONTAINER_NAME="${container_name}"
    bash "${run_container_script}"

    if docker ps --format '{{.Names}}' | grep -Fx "${container_name}" >/dev/null; then
      echo "[INFO] Container ready: ${container_name}"
    else
      echo "[ERROR] Failed to start container: ${container_name}" >&2
      exit 1
    fi
  else
    echo "[INFO] Action is 'stop', skipping image load and container start."
  fi
}

# ------------------------------------------
# 主控调度逻辑
# ------------------------------------------

prepare_node() {
  local node="$1"
  log_info "[${node}] 开始环境准备..."

  local func_code call_code
  func_code="$(declare -f _remote_prepare_node)"
  call_code="_remote_prepare_node '${IMAGE_NAME}' '${IMAGE_TAR}' '${RUN_CONTAINER_SCRIPT}' '${CONTAINER_NAME}' '${ACTION}'"

  # prepare_node 在宿主机执行，不需要进容器
  if ! echo "${func_code}; ${call_code}" | ssh_run "$node" bash -lc "bash -s"; then
     log_err "[${node}] 环境准备失败"
     return 1
  fi
  log_info "[${node}] 环境准备完成"
}

# ------------------------------------------
# 主流程入口
# ------------------------------------------

nodes="$(read_nodes)"
if [[ -z "$nodes" ]]; then
  log_err "NODES_FILE 中未找到任何节点信息"
  exit 2
fi

log_info "目标节点列表: $(echo $nodes | tr '\n' ' ')"
log_info "动作模式: ${ACTION}"
log_info "=== 开始准备节点 ==="

for node in $nodes; do
  limit_jobs "$PARALLELISM"
  (prepare_node "$node") &
done
wait

log_info "=== 节点准备完成 ==="
