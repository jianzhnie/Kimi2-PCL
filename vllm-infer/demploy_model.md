### 方式二：手动进入 Master 节点容器启动（适合调试）

如果你想实时看到 vLLM 启动时的详细日志（比如权重加载进度、Ray Actor 分配情况），你可以手动进入 Master 节点的容器来启动：

**1. 确认 Ray 集群正常运行**
```bash
bash vllm-infer/start_ray_cluster.sh
# 确保输出显示 128 个 NPU (16 个节点) 都已就绪
```

**2. 登录到 Master 节点并进入容器**
```bash
# 假设你的 Master 节点 IP 是 192.168.1.100
ssh 192.168.1.100

# 进入部署的容器
docker exec -it vllm-ascend-env-a3 bash
```

**3. 在容器内加载环境并启动**
```bash
# 进入工作目录
cd /llm_workspace_1P/robin/Kimi2-PCL/vllm-infer

# 必须先 source 环境，加载 NPU 驱动和 Ray 网络变量
source set_env.sh

# 直接运行优化后的脚本
bash vllm_model_server.sh
```

### 💡 启动过程中的注意事项
1. **加载时间长**：600B+ 的模型分布到 16 个节点，权重的分发和 Ray Actor 的创建（特别是 Pipeline Parallelism 的初始化）会消耗大量时间（可能需要 10 分钟甚至更长），请耐心等待。
2. **日志监控**：如果你使用的是方式一（后台启动），记得去 Master 节点查看日志：
   ```bash
   docker exec -it vllm-ascend-env-a3 tail -f /tmp/vllm_serve.log
   ```
3. **参数覆盖**：如果在临时测试时你想调整某些参数（比如把 TP 降到 4），可以直接在命令前加环境变量：
   ```bash
   TENSOR_PARALLEL_SIZE=4 PIPELINE_PARALLEL_SIZE=8 bash vllm_model_server.sh
   ```
