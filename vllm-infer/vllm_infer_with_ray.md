# Ray 分布式推理（Kimi-K2-Base）

多节点推理适用于模型无法在单机上部署的场景。此时，可以通过张量并行或流水线并行实现模型的分布式部署。具体的并行策略将在后续章节中详细介绍。要成功部署多节点推理，需要完成以下三个步骤：

- **验证多节点通信环境**
- **设置并启动 Ray 集群**
- **在多节点上启动在线推理服务**

## 验证多节点通信环境

### 物理层要求

- 物理机必须位于同一局域网内，并具备网络连通性。
- 所有 NPU 需通过光模块连接，且连接状态必须正常。

### 验证流程

依次在每个节点上执行以下命令。所有结果必须显示为 `success` 且状态为 `UP`：

```bash
 # Check the remote switch ports
 for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done 
 # Get the link status of the Ethernet ports (UP or DOWN)
 for i in {0..7}; do hccn_tool -i $i -link -g ; done
 # Check the network health status
 for i in {0..7}; do hccn_tool -i $i -net_health -g ; done
 # View the network detected IP configuration
 for i in {0..7}; do hccn_tool -i $i -netdetect -g ; done
 # View gateway configuration
 for i in {0..7}; do hccn_tool -i $i -gateway -g ; done
 # View NPU network configuration
 cat /etc/hccn.conf
```



### NPU 互连验证

#### 1. 获取 NPU IP 地址

```bash
for i in {0..7}; do hccn_tool -i $i -ip -g | grep ipaddr; done
```



#### 2. 跨节点 PING 测试

```bash
# Execute on the target node (replace with actual IP)
hccn_tool -i 0 -ping -g address 10.20.0.20
```



## 设置并启动 Ray 集群

### 配置基础容器

为确保所有节点（包括模型路径和 Python 环境）拥有统一的执行环境，建议使用 Docker 镜像。

在使用 Ray 搭建多节点推理集群时，**容器化部署**是首选方案。需在主节点和从节点上同时启动容器，并添加 `--net=host` 参数以确保网络连通性正常。

以下是容器设置命令示例，应在**所有节点**上执行：

```bash
# Update the vllm-ascend image
export IMAGE=quay.nju.edu.cn/ascend/vllm-ascend:v0.17.0rc1
export NAME=vllm-ascend

# Run the container using the defined variables
# Note if you are running bridge network with docker, please expose available ports for multiple nodes communication in advance.
# IMPORTANT: /path/to/shared/cache 必须是所有节点都可访问的共享目录（例如 NFS），用于统一缓存与代码下载目录。
docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /path/to/shared/cache:/root/.cache \
-it $IMAGE bash
```



### 启动 Ray 集群

在各节点完成容器设置并安装 vllm-ascend 后，请按以下步骤启动 Ray 集群并执行推理任务。

选择一台机器作为主节点，其余作为从节点。操作前请使用 `ip addr` 命令查看本机 `nic_name`（网络接口名称）。

设置 `ASCEND_RT_VISIBLE_DEVICES` 环境变量以指定使用的 NPU 设备。对于 Ray 2.1 以上版本，还需设置 `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES` 变量以避免设备识别问题。

主从节点启动命令如下：

**主节点** ：

注意：启动多节点推理的 Ray 集群时，各节点的环境变量必须在**启动 Ray 前**完成设置才能生效。更新环境变量后需要重启 Ray 集群。

```bash
# Head node
export HCCL_IF_IP={local_ip}
export GLOO_SOCKET_IFNAME={nic_name}
export TP_SOCKET_IFNAME={nic_name}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ray start --head --node-ip-address={local_ip}
```



**从节点** ：

注意：启动多节点推理的 Ray 集群时，各节点的环境变量必须在**启动 Ray 前**完成设置才能生效。更新环境变量后需要重启 Ray 集群。

```bash
# Worker node
export HCCL_IF_IP={local_ip}
export GLOO_SOCKET_IFNAME={nic_name}
export TP_SOCKET_IFNAME={nic_name}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ray start --address='{head_node_ip}:6379' --node-ip-address={local_ip}
```



多节点集群启动后，执行 `ray status` 和 `ray list nodes` 命令验证集群状态，应显示正确的节点数与 NPU 数量。

Ray 成功启动后通常会显示以下信息：

- 本地 Ray 实例已成功启动
- Ray Dashboard 地址（默认 [http://localhost:8265](http://localhost:8265/)）
- 节点与资源状态（CPU/内存/健康节点数）
- 集群连接地址（用于添加多节点）

### 常见排错要点

- 环境变量必须在 `ray start` 之前设置；变更后需 `ray stop -f` 再重新 `ray start`
- `HCCL_IF_IP`、`GLOO_SOCKET_IFNAME`、`TP_SOCKET_IFNAME` 要与实际通信网卡匹配（`ip addr` 可查看）
- 使用非 `--net=host` 时需要额外开放 Ray 相关端口（建议优先使用 host 网络简化通信）

## 多节点场景启动在线推理服务

在容器中可像所有 NPU 位于单节点般使用 vLLM。vLLM 将自动利用 Ray 集群所有节点的 NPU 资源。

**仅需在任一节点运行 vllm 命令即可。**

设置并行化时，常规做法是将 `tensor-parallel-size` 设为每节点 NPU 数量，`pipeline-parallel-size` 设为节点总数。

如需从其他机器访问推理服务，请在 `vllm serve` 中额外添加 `--host 0.0.0.0`（默认仅本机可访问）。

例如在 2 节点 16 NPU（每节点 8 NPU）场景中，设置张量并行大小为 8，流水线并行大小为 2：

```bash
vllm serve Kimi-K2-Base \
  --distributed-executor-backend ray \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --seed 1024 \
  --max-model-len 8192  \
  --max-num-seqs 25 \
  --served-model-name kimi \
  --trust-remote-code \
  --gpu-memory-utilization 0.9
```



若仅需使用张量并行，可将张量并行大小设为集群 NPU 总数。例如 2 节点 16 NPU 场景中设置张量并行大小为 16：

```bash
vllm serve Kimi-K2-Base \
  --distributed-executor-backend ray \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --seed 1024 \
  --max-model-len 8192  \
  --max-num-seqs 25 \
  --served-model-name kimi \
  --trust-remote-code \
  --gpu-memory-utilization 0.9
```



服务器启动后，您可以通过输入提示词查询模型：

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "kimi",
        "prompt": "tell me how to sleep well",
        "max_completion_tokens": 100,
        "temperature": 0
    }'
```
