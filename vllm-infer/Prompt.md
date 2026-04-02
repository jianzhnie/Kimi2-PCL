## 优化 Prepare Docker Nodes 脚本逻辑

优化逻辑 /Users/jianzhengnie/work_dir/Kimi2-PCL/vllm-infer/prepare_docker_nodes.sh, 

```bash
1. 默认的情况
  - 确保 Docker 命令可用
    - 如果Docker命令不可用，则执行 systemctl daemon-reload && systemctl start docker 
    - 可用则继续执行后续步骤
  - 不进行  stop & kill Docker 容器的操作

2. 使用 restart 参数
  - 确保 Docker 命令可用
    - 如果Docker命令不可用，则执行 systemctl daemon-reload && systemctl start docker 
    - 可用则继续执行后续步骤
  - 执行 stop & kill Docker 容器的操作
    - 停止所有正在运行的 Docker 容器
    - 杀死所有正在运行的 Docker 容器
  - 继续执行后续步骤
```