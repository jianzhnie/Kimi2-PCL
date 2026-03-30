#!/usr/bin/env bash
# ==============================================================================
# 多节点进程清理脚本 (kill_multi_nodes.sh)
#
# 该脚本通过 SSH 并发连接到多个节点，根据关键字终止指定的进程。
# 脚本会首先尝试温和地终止进程（SIGTERM），超时后若进程仍存活，则强制终止（SIGKILL）。
# ==============================================================================

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
# 默认配置（可被环境变量或命令行参数覆盖）
# ------------------------------------------
NODE_LIST_FILE="${NODES_FILE:-${SCRIPT_DIR}/node_list.txt}"
MAX_JOBS="${MAX_JOBS:-16}"
SSH_TIMEOUT="${SSH_TIMEOUT:-10}"
KILL_TIMEOUT="${KILL_TIMEOUT:-3}"

# 定义要 kill 的关键词（支持正则，可通过环境变量扩展）
# 包含 Ray 各组件：raylet, plasma_store, gcs_server, ray::IDLE, dashboard_agent, runtime_env_agent
DEFAULT_KEYWORDS=("llmtuner" "mindspeed" "ray" "vllm" "verl" "python" "raylet" "plasma_store" "gcs_server" "dashboard_agent" "runtime_env_agent")
if [[ -n "${EXTRA_KEYWORDS:-}" ]]; then
  IFS=',' read -ra EXTRA_KEYWORDS_ARRAY <<< "$EXTRA_KEYWORDS"
  KEYWORDS=("${DEFAULT_KEYWORDS[@]}" "${EXTRA_KEYWORDS_ARRAY[@]}")
else
  KEYWORDS=("${DEFAULT_KEYWORDS[@]}")
fi

# SSH 选项
SSH_OPTS="${SSH_OPTS:-}"
SSH_USER_HOST_PREFIX="${SSH_USER_HOST_PREFIX:-}"

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
    cat <<EOF
Usage: $0 [OPTIONS] [node_list_file]

Description:
  通过 SSH 并发连接到多个节点，根据关键字终止指定的进程。
  首先尝试 SIGTERM 温和终止，超时后若进程仍存活则使用 SIGKILL 强制终止。

Arguments:
  node_list_file    节点列表文件路径 (默认: $NODE_LIST_FILE)
                    环境变量 NODES_FILE 也可指定此路径

Options:
  -y, --yes              跳过确认步骤，直接执行
  -k, --keywords LIST    自定义关键词列表，逗号分隔 (默认: llmtuner,mindspeed,ray,vllm,verl,python)
  -t, --timeout SEC      终止进程超时时间，秒 (默认: $KILL_TIMEOUT)
  -j, --jobs NUM         最大并发任务数 (默认: $MAX_JOBS)
  --ssh-timeout SEC      SSH 连接超时时间，秒 (默认: $SSH_TIMEOUT)
  -h, --help             显示此帮助信息

Environment Variables:
  NODES_FILE             节点列表文件路径
  SSH_OPTS               SSH 连接选项
  SSH_USER_HOST_PREFIX   SSH 目标前缀 (如: user@)
  EXTRA_KEYWORDS         额外的关键词，逗号分隔
  MAX_JOBS               最大并发任务数
  KILL_TIMEOUT           终止进程超时时间
  SSH_TIMEOUT            SSH 连接超时时间

Examples:
  $0                                    # 使用默认配置
  $0 /path/to/nodes.txt                 # 指定节点列表文件
  $0 -y                                 # 跳过确认
  $0 -k "myapp,worker"                  # 自定义关键词
  $0 -y -k "ray" -t 5                   # 强制模式，只杀 ray 进程，超时 5 秒

EOF
}

# ------------------------------------------
# 参数解析
# ------------------------------------------
SKIP_CONFIRM=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--yes)
            SKIP_CONFIRM=true
            shift
            ;;
        -k|--keywords)
            IFS=',' read -ra KEYWORDS <<< "$2"
            shift 2
            ;;
        -t|--timeout)
            KILL_TIMEOUT="$2"
            shift 2
            ;;
        -j|--jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --ssh-timeout)
            SSH_TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            log_err "未知选项: $1"
            usage >&2
            exit 1
            ;;
        *)
            # 非选项参数，视为节点列表文件
            NODE_LIST_FILE="$1"
            shift
            ;;
    esac
done

# ------------------------------------------
# 辅助函数
# ------------------------------------------

# 拼接 SSH 目标地址
ssh_target() {
    local node="$1"
    printf "%s%s" "$SSH_USER_HOST_PREFIX" "$node"
}

# 执行带超时的 SSH 命令
ssh_run_with_timeout() {
    local node="$1"
    shift
    # 使用 perl 实现超时控制，避免依赖外部 timeout 命令
    perl -e '
        use strict;
        use warnings;
        my $timeout = shift @ARGV;
        my @cmd = @ARGV;
        eval {
            local $SIG{ALRM} = sub { die "TIMEOUT\n" };
            alarm $timeout;
            system(@cmd);
            alarm 0;
        };
        if ($@ eq "TIMEOUT\n") {
            print STDERR "[ERROR] Command timed out after ${timeout}s\n";
            exit 124;
        }
        exit $? >> 8;
    ' "$SSH_TIMEOUT" ssh ${SSH_OPTS} "$(ssh_target "$node")" "$@"
}

# 并发数控制
limit_jobs() {
    local max="$1"
    while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$max" ]]; do
        wait -n 2>/dev/null || sleep 0.1
    done
}

# ------------------------------------------
# 核心函数：终止指定节点上的进程
# ------------------------------------------
kill_processes_on_node() {
    local node=$1
    log_info "🔎 [Node: $node] 开始检查进程..."

    # 构建正则表达式
    local pattern
    pattern=$(IFS='|'; echo "${KEYWORDS[*]}")

    # 构建远程执行命令
    local remote_cmd='
        set -euo pipefail

        # 函数：获取匹配关键词的进程 PID
        get_matching_pids() {
            ps aux | grep -E "'"$pattern"'" | grep -v grep | grep -v vscode-server | grep -v '"'"'extension'"'"' | grep -v agent | awk '"'"'{print $1}'"'"' | sort -u | tr "\n" " " || true
        }

        # 函数：温和终止进程
        graceful_kill() {
            local pids="$1"
            if [ -n "$pids" ]; then
                for pid in $pids; do
                    kill -15 "$pid" 2>/dev/null || true
                done
            fi
        }

        # 函数：强制终止进程
        force_kill() {
            local pids="$1"
            if [ -n "$pids" ]; then
                for pid in $pids; do
                    kill -9 "$pid" 2>/dev/null || true
                done
            fi
        }

        # 函数：检查进程是否存活
        get_alive_pids() {
            local pids="$1"
            local alive=""
            for pid in $pids; do
                if kill -0 "$pid" 2>/dev/null; then
                    alive="$alive $pid"
                fi
            done
            echo "${alive# }"
        }

        # 主逻辑
        all_pids=$(get_matching_pids)

        if [ -n "$all_pids" ]; then
            echo "找到匹配的进程 PID: $all_pids"
            echo "进程详情:"
            for pid in $all_pids; do
                ps -p "$pid" -o pid,ppid,user,args 2>/dev/null || true
            done

            # 1. 尝试温和终止 (SIGTERM)
            echo "尝试温和终止进程 (SIGTERM)..."
            graceful_kill "$all_pids"

            # 等待进程退出
            sleep '"$KILL_TIMEOUT"'

            # 2. 检查剩余进程并强制终止
            remaining_pids=$(get_alive_pids "$all_pids")

            if [ -n "$remaining_pids" ]; then
                echo "进程仍在运行: $remaining_pids，强制终止 (SIGKILL)..."
                force_kill "$remaining_pids"
                sleep 1

                # 3. 最终检查
                final_pids=$(get_alive_pids "$remaining_pids")
                if [ -n "$final_pids" ]; then
                    echo "警告: 仍有进程存活: $final_pids"
                    # 再次强制终止
                    force_kill "$final_pids"
                    sleep 1
                    # 最后一次检查
                    last_check=$(get_alive_pids "$final_pids")
                    if [ -n "$last_check" ]; then
                        echo "错误: 无法终止的进程: $last_check"
                        return 1
                    fi
                fi
                echo "已强制终止剩余进程"
            else
                echo "所有进程已正常终止"
            fi
        else
            echo "未找到匹配的进程"
        fi
    '

    # 执行远程命令
    if ssh_run_with_timeout "$node" "$remote_cmd"; then
        log_info "✅ [Node: $node] 处理完成"
    else
        local exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            log_err "❌ [Node: $node] SSH 连接超时 (${SSH_TIMEOUT}s)"
        else
            log_err "❌ [Node: $node] 处理失败 (退出码: $exit_code)"
        fi
    fi
}

# ------------------------------------------
# 主逻辑
# ------------------------------------------

# 检查节点列表文件
if [[ ! -f "$NODE_LIST_FILE" ]]; then
    log_err "节点列表文件未找到: $NODE_LIST_FILE"
    exit 1
fi

# 读取节点列表
mapfile -t NODES < <(awk 'NF && !/^#/ {print $1}' "$NODE_LIST_FILE")

if [[ ${#NODES[@]} -eq 0 ]]; then
    log_err "节点列表为空: $NODE_LIST_FILE"
    exit 1
fi

# 输出配置信息
log_info "🚀 开始多节点进程清理..."
log_info "目标关键词: ${KEYWORDS[*]}"
log_info "节点列表文件: $NODE_LIST_FILE"
log_info "节点数量: ${#NODES[@]}"
log_info "最大并发数: $MAX_JOBS"
log_info "终止超时: ${KILL_TIMEOUT}s"
log_info "SSH 超时: ${SSH_TIMEOUT}s"

# 用户确认（除非使用 -y 跳过）
if ! $SKIP_CONFIRM; then
    echo "================================================================"
    echo "⚠️  警告: 此脚本将终止以下节点上的指定进程"
    echo "   目标关键词: ${KEYWORDS[*]}"
    echo "   此操作不可恢复，可能会中断正在运行的任务"
    echo "----------------------------------------------------------------"
    echo "待处理节点:"
    for node in "${NODES[@]}"; do
        echo "  - $node"
    done
    echo "----------------------------------------------------------------"
    read -p "输入 'yes' 继续，或其他内容取消: " user_confirm

    if [[ "$user_confirm" != "yes" ]]; then
        log_info "已取消操作，未做任何更改"
        exit 0
    fi
    echo "================================================================"
    log_info "确认继续，开始清理..."
else
    log_info "跳过确认步骤 (-y 模式)"
fi

# 并发处理所有节点
for node in "${NODES[@]}"; do
    [[ -z "$node" ]] && continue
    limit_jobs "$MAX_JOBS"
    (kill_processes_on_node "$node") &
done

# 等待所有后台任务完成
wait

log_info "🎉 所有节点处理完成"
