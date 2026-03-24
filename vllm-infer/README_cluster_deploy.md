# Ray 集群与 vLLM 自动部署指南

本文档介绍了如何使用 `cluster_deploy_ray_vllm.sh` 脚本和 `set_env.sh` 配置文件在多节点集群上快速部署 Ray 和 vLLM 模型服务。

## 1. 目录结构与文件说明

在 `vllm-infer` 目录下，部署相关的核心文件如下：
- **`cluster_deploy_ray_vllm.sh`**: 自动化部署主脚本。负责拉起远程容器、组建 Ray 集群、以及在主节点启动 vLLM 模型服务。
- **`set_env.sh`**: 环境变量集中配置文件。所有的部署参数（节点、镜像、端口、Ascend NPU 等）都在此文件中统一管理。
- **`node_list.txt`**: 集群节点列表文件（默认为脚本同目录下的 `node_list.txt`）。按行存放集群的机器 IP 或主机名。

## 2. 环境配置 (`set_env.sh`)

在运行部署脚本前，必须配置好 `set_env.sh`。核心配置项说明：

### 部署与节点配置
- `NODES_FILE`: 节点列表文件路径。
- `MASTER_NODE`: 指定作为 Ray Head 节点及运行 vLLM 服务的主节点。如果不设置，则默认取 `node_list.txt` 中的第一行。
- `SSH_USER_HOST_PREFIX`: SSH 登录前缀，例如 `root@`。如果配置了免密 `~/.ssh/config` 可以留空。
- `PARALLELISM`: 节点并发准备的进程数，默认为 8。

### 容器与镜像配置
- `IMAGE_NAME`: Docker 镜像名称。
- `IMAGE_TAR`: 如果目标节点本地没有镜像，将从该路径加载 tar 包。
- `RUN_CONTAINER_SCRIPT`: 容器启动脚本路径（例如 `ascend_infer_docker_run.sh`）。
- `CONTAINER_NAME`: 容器启动后的名称，默认为 `vllm-ascend-env-a3`。
- `VLLM_START_SCRIPT`: vLLM 服务的启动脚本路径（例如 `vllm_model_server.sh`）。

### Ray 与 vLLM 配置
- `RAY_PORT`: Ray head 节点监听的端口，默认 `6379`。
- `VLLM_HOST` & `VLLM_PORT`: vLLM 服务的监听地址和端口，默认 `0.0.0.0:8000`。

### Ascend NPU 配置
- `ASCEND_RT_VISIBLE_DEVICES`: 容器内可见的 NPU 设备号，例如 `0,1,2,3,4,5,6,7`。
- 其他网络参数如 `GLOO_SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME` 根据实际网卡名称配置。

## 3. 使用方法

确保已为各节点配置好 SSH 免密登录，然后在宿主机（部署机）执行脚本。

### 3.1 一键完整部署

默认会依次执行节点环境准备、Ray 集群启动、vLLM 服务启动：
```bash
bash vllm-infer/cluster_deploy_ray_vllm.sh
```

### 3.2 分步独立执行

脚本支持通过参数指定仅执行特定阶段：

- **仅准备节点环境**（拉取镜像、启动容器）：
  ```bash
  bash vllm-infer/cluster_deploy_ray_vllm.sh --prepare-only
  ```

- **仅启动 Ray 集群**（在 Master 启动 Head，其他节点启动 Worker 并加入集群）：
  ```bash
  bash vllm-infer/cluster_deploy_ray_vllm.sh --ray-only
  ```

- **仅启动 vLLM 服务**（在 Master 节点启动模型服务）：
  ```bash
  bash vllm-infer/cluster_deploy_ray_vllm.sh --serve-only
  ```

- **查看帮助信息**：
  ```bash
  bash vllm-infer/cluster_deploy_ray_vllm.sh --help
  ```

## 4. 常见问题与注意事项

1. **SSH 免密登录**：
   部署机必须能够免密 SSH 到所有目标节点（包含本机，如果本机也在集群列表中）。
2. **容器启动脚本限制**：
   在执行 `--prepare-only` 时，远端会调用 `ascend_infer_docker_run.sh`。请确保该脚本在远端路径确实存在，或者部署目录已挂载到所有节点相同的路径（例如 NFS / 共享存储）。
3. **网卡名称匹配**：
   Ray 和 HCCL 依赖正确的网卡名称（`GLOO_SOCKET_IFNAME` 和 `HCCL_SOCKET_IFNAME`）。如果在不同节点网卡名称不同，请确保默认推断逻辑能够获取到正确网卡，或根据集群实际情况修改 `set_env.sh` 的默认值。
4. **日志查看**：
   脚本自身提供了结构化的日志输出。vLLM 服务由于在后台运行，其服务日志保存在 Master 节点容器内的 `/tmp/vllm_serve.log` 中。
5. **异常排查**：
   脚本使用了 `set -euo pipefail` 并在失败时返回非 0 退出码，方便与 CI/CD 流水线集成。如果报错退出，请检查对应的控制台 `[ERROR]` 日志。
