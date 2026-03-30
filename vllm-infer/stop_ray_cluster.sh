#!/usr/bin/env bash
# ==========================================
# Ray 集群停止脚本 (stop_ray_cluster.sh)
# 用于停止所有节点上的 Ray 进程
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
log_warn() { echo "[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $*"; }

# ------------------------------------------
# 帮助信息
# ------------------------------------------
usage() {
  cat <<'USAGE'
Usage:
  bash stop_ray_cluster.sh [OPTIONS]

Description:
  该脚本用于停止集群所有节点上的 Ray 进程。
  环境变量请在同目录下的 set_env.sh 中配置。

Options:
  --on-host        在宿主机上停止 Ray（不进容器）
  -h, --help       显示帮助信息
USAGE
}

# ------------------------------------------
# 参数解析
# ------------------------------------------
ON_HOST=false
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
elif [[ "${1:-}" == "--on-host" ]]; then
  ON_HOST=true
elif [[ -n "${1:-}" ]]; then
  log_err "未知选项: $1"
  usage >&2
  exit 2
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
  awk 'NF && !/^#/ {print $1}' "$NODES_FILE"
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
  ssh ${SSH_OPTS:-} "$(ssh_target "$node")" "$@"
}

# 并发数控制
limit_jobs() {
  local max="$1"
  while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$max" ]]; do
    wait -n
  done
}

# ------------------------------------------
# 远程停止 Ray 函数（用于 declare -f 传递）
# ------------------------------------------

_remote_stop_ray() {
  set -euo pipefail
  ray stop -f >/dev/null 2>&1 || true
}

# ------------------------------------------
# 主控调度逻辑
# ------------------------------------------

stop_ray_node() {
  local node="$1"
  log_info "[${node}] 正在停止 Ray 进程..."

  local func_code call_code
  func_code="$(declare -f _remote_stop_ray)"
  call_code="_remote_stop_ray"

  if $ON_HOST; then
    # 在宿主机上执行（不进容器）
    if ! echo "${func_code}; ${call_code}" | ssh_run "$node" bash -lc "bash -s" 2>/dev/null; then
      log_warn "[${node}] 宿主机上停止 Ray 可能失败或 Ray 未运行"
    else
      log_info "[${node}] Ray 进程已停止（宿主机）"
    fi
  else
    # 在容器内执行
    local ssh_cmd="cd '${SCRIPT_DIR}' && source set_env.sh && \
        docker exec -i \"\${CONTAINER_NAME:-vllm-ascend-env-a3}\" bash -s"

    if ! echo "${func_code}; ${call_code}" | ssh_run "$node" "$ssh_cmd" 2>/dev/null; then
      log_warn "[${node}] 容器内停止 Ray 可能失败或 Ray 未运行"
    else
      log_info "[${node}] Ray 进程已停止（容器内）"
    fi
  fi
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
log_info "执行模式: $([[ $ON_HOST == true ]] && echo '宿主机' || echo '容器内')"
log_info "=== 开始停止 Ray 集群 ==="

# 并发停止所有节点
PARALLELISM="${PARALLELISM:-8}"
for node in $nodes; do
  limit_jobs "$PARALLELISM"
  (stop_ray_node "$node") &
done
wait

log_info "=== Ray 集群停止完成 ==="
