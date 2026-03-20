## 挂载 dtfs 文件系统
mount -t dtfs  /llm_workspace_1P  /llm_workspace_1P

## 卸载 dtfs 文件系统
umount /llm_workspace_1P

# 文件上传
## 同步本地文件到服务器
rsync -avz K8s-Cluster  root@10.42.29.130:/llm_workspace_1P/robin/

rsync -avz Kimi2-PCL  k8s-130:/llm_workspace_1P/robin/

rsync -avz kimi_pcl_eval k8s-130:/llm_workspace_1P/robin/

rsync -avz vllm  k8s-130:/llm_workspace_1P/robin/

rsync -avz vllm-ascend  k8s-130:/llm_workspace_1P/robin/

## 进入容器
docker exec -it mindspeed-llm-env /bin/bash

