#!/usr/bin/env bash
# ==========================================
# Ray 集群停止脚本 (stop_ray_cluster.sh)
# 用于停止所有节点上的 Ray 进程并清理残余
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
KILL_TIMEOUT="${KILL_TIMEOUT:-3}"
SSH_TIMEOUT="${SSH_TIMEOUT:-10}"
PARALLELISM="${PARALLELISM:-16}"

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
  该脚本用于停止集群所有节点上的 Ray 进程，并彻底清理残余进程。
  包括 raylet、plasma_store、gcs_server、ray::IDLE 等工作进程。
  环境变量请在同目录下的 set_env.sh 中配置。

Options:
  --on-host        在宿主机上停止 Ray（不进容器）
  -f, --force      强制模式：立即停止并清理所有残余（跳过温和终止）
  -y, --yes        跳过确认步骤
  -h, --help       显示帮助信息
USAGE
}

# ------------------------------------------
# 参数解析
# ------------------------------------------
ON_HOST=false
FORCE=false
SKIP_CONFIRM=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --on-host)
      ON_HOST=true
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
# 远程停止 Ray 函数（用于 declare -f 传递）
# ------------------------------------------

_remote_stop_and_cleanup_ray() {
  local force="$1"
  local kill_timeout="$2"

  set -euo pipefail

  # 定义要清理的 Ray 相关进程关键词
  local ray_keywords="raylet|plasma_store|gcs_server|ray::|ray.worker|python.*ray|dashboard_agent|runtime_env_agent"

  # 函数：温和终止进程
  graceful_kill() {
    local pids="$1"
    if [[ -n "$pids" ]]; then
      for pid in $pids; do
        kill -15 "$pid" 2>/dev/null || true
      done
    fi
  }

  # 函数：强制终止进程
  force_kill() {
    local pids="$1"
    if [[ -n "$pids" ]]; then
      for pid in $pids; do
        kill -9 "$pid" 2>/dev/null || true
      done
    fi
  }

  # 函数：获取 Ray 相关进程
  get_ray_pids() {
    ps aux | grep -E "$ray_keywords" | grep -v grep | awk '{print $2}' | sort -u | tr '\n' ' ' || true
  }

  # 步骤 1: 尝试使用 ray stop 命令（如果可用）
  if command -v ray >/dev/null 2>&1; then
    if [[ "$force" == "true" ]]; then
      ray stop -f --grace-period 0 >/dev/null 2>&1 || true
    else
      ray stop -f >/dev/null 2>&1 || true
    fi
  fi

  # 步骤 2: 查找并终止所有 Ray 相关进程
  local all_pids
  all_pids=$(get_ray_pids)

  if [[ -n "$all_pids" ]]; then
    echo "找到 Ray 相关进程: $all_pids"

    if [[ "$force" == "true" ]]; then
      echo "强制终止所有 Ray 进程..."
      force_kill "$all_pids"
    else
      # 温和终止
      echo "尝试温和终止 Ray 进程 (SIGTERM)..."
      graceful_kill "$all_pids"

      # 等待进程退出
      sleep "$kill_timeout"

      # 检查残余进程
      local remaining_pids
      remaining_pids=$(get_ray_pids)

      if [[ -n "$remaining_pids" ]]; then
        echo "进程仍在运行: $remaining_pids，强制终止 (SIGKILL)..."
        force_kill "$remaining_pids"
        sleep 1
      else
        echo "所有 Ray 进程已正常终止"
      fi
    fi

    # 最终检查
    local final_pids
    final_pids=$(get_ray_pids)
    if [[ -n "$final_pids" ]]; then
      echo "警告: 仍有残余进程: $final_pids"
      return 1
    else
      echo "Ray 进程清理完成"
    fi
  else
    echo "未找到 Ray 相关进程"
  fi

  return 0
}

# ------------------------------------------
# 主控调度逻辑
# ------------------------------------------

stop_ray_node() {
  local node="$1"
  log_info "[${node}] 正在停止 Ray 进程..."

  local func_code call_code
  func_code="$(declare -f _remote_stop_and_cleanup_ray)"
  call_code="_remote_stop_and_cleanup_ray '${FORCE}' '${KILL_TIMEOUT}'"

  if $ON_HOST; then
    # 在宿主机上执行（不进容器）
    if ! echo "${func_code}; ${call_code}" | ssh_run "$node" bash -lc "bash -s" 2>/dev/null; then
      log_warn "[${node}] 宿主机上停止 Ray 可能失败"
    else
      log_info "[${node}] Ray 进程已停止（宿主机）"
    fi
  else
    # 在容器内执行
    local ssh_cmd="cd '${SCRIPT_DIR}' && source set_env.sh && \
        docker exec -i \"\${CONTAINER_NAME:-vllm-ascend-env-a3}\" bash -s"

    if ! echo "${func_code}; ${call_code}" | ssh_run "$node" "$ssh_cmd" 2>/dev/null; then
      log_warn "[${node}] 容器内停止 Ray 可能失败"
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

# 用户确认（除非跳过）
if ! $SKIP_CONFIRM; then
  echo "================================================================"
  echo "⚠️  警告: 此操作将停止所有节点上的 Ray 集群"
  echo "   目标节点: $(echo $nodes | tr '\n' ' ')"
  echo "   执行模式: $([[ $ON_HOST == true ]] && echo '宿主机' || echo '容器内')"
  echo "   强制模式: $([[ $FORCE == true ]] && echo '是' || echo '否')"
  echo "================================================================"
  read -p "输入 'yes' 继续，或其他内容取消: " user_confirm
  if [[ "$user_confirm" != "yes" ]]; then
    log_info "已取消操作"
    exit 0
  fi
fi

log_info "目标节点列表: $(echo $nodes | tr '\n' ' ')"
log_info "执行模式: $([[ $ON_HOST == true ]] && echo '宿主机' || echo '容器内')"
log_info "=== 开始停止 Ray 集群 ==="

# 并发停止所有节点
for node in $nodes; do
  limit_jobs "$PARALLELISM"
  (stop_ray_node "$node") &
done
wait

log_info "=== Ray 集群停止完成 ==="
