#!/usr/bin/env bash
# ==========================================
# 容器停止脚本 (stop_containers.sh)
# 用于停止所有节点上的 Docker 容器，并在停止前清理 Ray 进程
# ==========================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/set_env.sh"

# ------------------------------------------
# 引入环境变量
# ------------------------------------------
if [[ -f "${ENV_FILE}" ]]; then
  source "${ENV_FILE}"
fi

# ------------------------------------------
# 默认配置
# ------------------------------------------
PARALLELISM="${PARALLELISM:-8}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-ascend-env-a3}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"

# ------------------------------------------
# 日志函数
# ------------------------------------------
log_info() { echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $*"; }
log_err()  { echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2; }
log_warn() { echo "[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $*"; }

# ------------------------------------------
# 帮助信息
# ------------------------------------------
usage() {
  cat <<'USAGE'
Usage:
  bash stop_containers.sh [OPTIONS]

Description:
  停止集群所有节点上的 Docker 容器，并在停止前清理容器内的 Ray 进程。
  这样可以避免 Ray 残余进程导致的端口冲突问题。
  环境变量请在同目录下的 set_env.sh 中配置。

Options:
  --skip-ray-stop    跳过停止 Ray 进程（直接停止容器）
  -f, --force        强制停止容器（docker kill）
  -y, --yes          跳过确认步骤
  -h, --help         显示帮助信息
USAGE
}

# ------------------------------------------
# 参数解析
# ------------------------------------------
SKIP_RAY_STOP=false
FORCE=false
SKIP_CONFIRM=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-ray-stop)
      SKIP_RAY_STOP=true
      shift
      ;;
    -f|--force)
      FORCE=true
      shift
      ;;
    -y|--yes)
      SKIP_CONFIRM=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      log_err "未知选项: $1"
      usage >&2
      exit 2
      ;;
  esac
done

# ------------------------------------------
# 辅助函数
# ------------------------------------------

# 获取非空节点列表
read_nodes() {
  if [[ ! -f "${NODES_FILE:-}" ]]; then
    log_err "节点列表文件未找到: ${NODES_FILE:-}"
    exit 2
  fi
  awk 'NF && !/^#/ {print $1}' "$NODES_FILE"
}

# 拼接 SSH 目标地址
ssh_target() {
  local node="$1"
  printf "%s%s" "${SSH_USER_HOST_PREFIX:-}" "$node"
}

# 执行 SSH 命令
ssh_run() {
  local node="$1"
  shift
  ssh ${SSH_OPTS:-} "$(ssh_target "$node")" "$@"
}

# 并发数控制
limit_jobs() {
  local max="$1"
  while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$max" ]]; do
    wait -n 2>/dev/null || sleep 0.1
  done
}

# ------------------------------------------
# 远程函数（用于 declare -f 传递）
# ------------------------------------------

_remote_stop_ray_in_container() {
  local container="$1"

  set -euo pipefail

  # 检查容器是否运行
  if ! docker ps --format '{{.Names}}' | grep -Fx "$container" >/dev/null 2>&1; then
    echo "容器未运行: $container"
    return 0
  fi

  # 在容器内停止 Ray
  echo "停止容器内的 Ray 进程..."
  docker exec "$container" bash -c '
    # 尝试 ray stop
    ray stop -f --grace-period 0 2>/dev/null || true

    # 清理残余进程
    pids=$(ps aux | grep -E "raylet|plasma_store|gcs_server|ray::|ray.worker|python.*ray|dashboard_agent|runtime_env_agent" | grep -v grep | awk "{print \$2}" | sort -u || true)
    if [ -n "$pids" ]; then
      echo "清理残余进程: $pids"
      for pid in $pids; do
        kill -9 "$pid" 2>/dev/null || true
      done
    fi
  ' 2>/dev/null || true

  echo "Ray 进程已清理"
}

_remote_stop_container() {
  local container="$1"
  local force="$2"
  local timeout="$3"

  set -euo pipefail

  # 检查容器是否存在
  if ! docker ps -a --format '{{.Names}}' | grep -Fx "$container" >/dev/null 2>&1; then
    echo "容器不存在: $container"
    return 0
  fi

  # 检查容器是否运行
  if ! docker ps --format '{{.Names}}' | grep -Fx "$container" >/dev/null 2>&1; then
    echo "容器已停止: $container"
    return 0
  fi

  # 停止容器
  if [[ "$force" == "true" ]]; then
    echo "强制停止容器: $container"
    docker kill "$container" 2>/dev/null || true
  else
    echo "优雅停止容器: $container (超时: ${timeout}s)"
    docker stop -t "$timeout" "$container" 2>/dev/null || {
      echo "优雅停止失败，强制停止..."
      docker kill "$container" 2>/dev/null || true
    }
  fi

  echo "容器已停止: $container"
}

# ------------------------------------------
# 主控调度逻辑
# ------------------------------------------

stop_node() {
  local node="$1"

  if ! $SKIP_RAY_STOP; then
    log_info "[${node}] 停止容器内的 Ray 进程..."
    local ray_func_code ray_call_code
    ray_func_code="$(declare -f _remote_stop_ray_in_container)"
    ray_call_code="_remote_stop_ray_in_container '${CONTAINER_NAME}'"

    if ! echo "${ray_func_code}; ${ray_call_code}" | ssh_run "$node" bash -lc "bash -s" 2>/dev/null; then
      log_warn "[${node}] 停止 Ray 进程可能失败，继续停止容器..."
    fi
  fi

  log_info "[${node}] 停止容器..."
  local stop_func_code stop_call_code
  stop_func_code="$(declare -f _remote_stop_container)"
  stop_call_code="_remote_stop_container '${CONTAINER_NAME}' '${FORCE}' '${STOP_TIMEOUT}'"

  if ! echo "${stop_func_code}; ${stop_call_code}" | ssh_run "$node" bash -lc "bash -s" 2>/dev/null; then
    log_err "[${node}] 停止容器失败"
    return 1
  fi

  log_info "[${node}] 容器已停止"
}

# ------------------------------------------
# 主流程入口
# ------------------------------------------

nodes="$(read_nodes)"
if [[ -z "$nodes" ]]; then
  log_err "NODES_FILE 中未找到任何节点信息"
  exit 2
fi

# 用户确认（除非跳过）
if ! $SKIP_CONFIRM; then
  echo "================================================================"
  echo "⚠️  警告: 此操作将停止所有节点上的 Docker 容器"
  echo "   目标节点: $(echo $nodes | tr '\n' ' ')"
  echo "   容器名称: $CONTAINER_NAME"
  echo "   停止 Ray: $([[ $SKIP_RAY_STOP == true ]] && echo '否' || echo '是')"
  echo "   强制模式: $([[ $FORCE == true ]] && echo '是' || echo '否')"
  echo "================================================================"
  read -p "输入 'yes' 继续，或其他内容取消: " user_confirm
  if [[ "$user_confirm" != "yes" ]]; then
    log_info "已取消操作"
    exit 0
  fi
fi

log_info "目标节点列表: $(echo $nodes | tr '\n' ' ')"
log_info "容器名称: $CONTAINER_NAME"
log_info "=== 开始停止容器 ==="

# 并发停止所有节点
for node in $nodes; do
  limit_jobs "$PARALLELISM"
  (stop_node "$node") &
done
wait

log_info "=== 容器停止完成 ==="
