# vLLM 推理集群使用指南

本文档介绍如何在 Ascend NPU 集群上部署和运行 vLLM 推理服务。

## 环境准备

### 1. 配置环境变量

编辑 `set_env.sh` 文件，根据实际环境配置以下参数：

```bash
# 节点列表文件路径
export NODES_FILE="${SCRIPT_DIR}/node_list.txt"

# SSH 配置（如需用户名前缀）
export SSH_USER_HOST_PREFIX=""  # 例如: "user@"

# Docker 镜像与容器配置
export IMAGE_NAME="quay.io/ascend/vllm-ascend:main-a3"
export IMAGE_TAR="/llm_workspace_1P/robin/hfhub/docker/image/vllm-ascend.main-a3.tar"
export CONTAINER_NAME="vllm-ascend-env-a3"

# 端口配置
export RAY_PORT=6379
export RAY_DASHBOARD_PORT=8266
export VLLM_PORT=8000
```

### 2. 配置节点列表

编辑 `node_list.txt` 文件，列出所有集群节点：

```
bms1425
bms1426
bms1427
bms1428
bms1429
bms1430
bms1431
bms1432
bms1433
bms1434
bms1435
bms1436
bms1437
bms1438
bms1439
bms1440
```

---

## 部署流程

### Step 1: 准备 Docker 环境

在所有节点上准备 Docker 容器环境：

```bash
bash vllm-infer/prepare_docker_nodes.sh
```

此脚本会：
- 在每台节点上加载 Docker 镜像（如未加载）
- 启动指定名称的容器
- 确保容器运行正常

### Step 2: 复制文件到容器

将代码或配置文件复制到各节点的容器内。

#### 2.1 单文件复制（远程文件模式）

如果文件已存在于远程节点的 `/llm_workspace_1P/wf/` 目录：

```bash
cd /llm_workspace_1P/robin/Kimi2-PCL/vllm-infer

# 复制 DeepSeek v2 模型文件
bash copy_to_docker.sh -p 16 -r \
    /llm_workspace_1P/wf/deepseek_v2_real.py \
    /vllm-workspace/vllm/vllm/model_executor/models/deepseek_v2.py

# 复制默认加载器
bash copy_to_docker.sh -p 16 -r \
    /llm_workspace_1P/wf/default_loader_real.py \
    /vllm-workspace/vllm/vllm/model_executor/model_loader/default_loader.py

# 复制 Model Runner
bash copy_to_docker.sh -p 16 -r \
    /llm_workspace_1P/wf/model_runner_v1_real.py \
    /vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py

# 复制 vLLM 配置
bash copy_to_docker.sh -p 16 -r \
    /llm_workspace_1P/wf/vllm_config_real.py \
    /vllm-workspace/vllm/vllm/config/vllm.py
```

#### 2.2 批量复制（推荐）

使用配置文件批量复制多个文件：

```bash
bash copy_to_docker.sh -p 16 -r -c copy_files.conf.example
```

配置文件格式（`copy_files.conf.example`）：

```
# 格式: 源文件路径|容器内目标路径
/llm_workspace_1P/wf/deepseek_v2_real.py|/vllm-workspace/vllm/vllm/model_executor/models/deepseek_v2.py
/llm_workspace_1P/wf/default_loader_real.py|/vllm-workspace/vllm/vllm/model_executor/model_loader/default_loader.py
```


#### 参数说明

| 参数          | 说明                               |
| ------------- | ---------------------------------- |
| `-p 16`       | 并发数（默认 8）                   |
| `-r`          | 远程文件模式（文件已在远程节点上） |
| `-n <node>`   | 仅复制到指定节点（可多次使用）     |
| `-c <config>` | 使用配置文件批量复制               |

### Step 3: 启动 Ray 集群

在容器内启动 Ray 集群：

```bash
bash vllm-infer/start_ray_cluster.sh
```

启动成功后会显示：
- Ray Dashboard URL: `http://<master_node>:8266`
- 集群节点状态

### Step 4: 进入容器（可选）

如需进入容器调试：

```bash
docker exec -it vllm-ascend-env-a3 /bin/bash
```

### Step 5: 运行 vLLM 推理服务

在容器内启动 vLLM 推理服务：

```bash
cd /llm_workspace_1P/robin/Kimi2-PCL/vllm-infer
bash run_vllm_test.sh
```

---

## 常用操作

### 停止 Ray 集群

```bash
bash vllm-infer/stop_ray_cluster.sh
```

### 清理集群进程（谨慎使用）

```bash
# 基本使用（交互式确认）
bash vllm-infer/kill_multi_nodes.sh

# 跳过确认直接执行
bash vllm-infer/kill_multi_nodes.sh -y

# 只终止 ray 相关进程
bash vllm-infer/kill_multi_nodes.sh -y -k "ray"
```
