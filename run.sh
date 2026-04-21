## 挂载 dtfs 文件系统
mount -t dtfs  /llm_workspace_1P  /llm_workspace_1P

## 卸载 dtfs 文件系统
umount /llm_workspace_1P


# docker tag && push
docker tag quay.io/ascend/vllm-ascend:main-a3 cis-pengcheng.cmecloud.cn/ascendhub/quay.io/ascend/vllm-ascend:main-a3

docker push cis-pengcheng.cmecloud.cn/ascendhub/quay.io/ascend/vllm-ascend:main-a3

# 文件上传
## 同步本地文件到服务器
rsync -avz K8s-Cluster  root@10.42.29.130:/llm_workspace_1P/robin/
rsync -avz Kimi2-PCL  k8s-130:/llm_workspace_1P/robin/
rsync -avz Kimi2-PCL  k8s-131:/llm_workspace_1P/robin/
rsync -avz kimi_pcl_eval k8s-130:/llm_workspace_1P/robin/
rsync -avz vllm  k8s-130:/llm_workspace_1P/robin/
rsync -avz vllm-pcl  k8s-130:/llm_workspace_1P/robin/
rsync -avz vllm-ascend  k8s-130:/llm_workspace_1P/robin/

rsync -avz ./utils k8s-130:/llm_workspace_1P/robin/Kimi2-PCL
rsync -avz ./scripts k8s-130:/llm_workspace_1P/robin/Kimi2-PCL
rsync -avz ./models k8s-130:/llm_workspace_1P/robin/Kimi2-PCL


## 远程文件下载到本地
rsync -avz  k8s-130:/llm_workspace_1P/robin/Kimi2-PCL.tar  /Users/robin/work_dir/
rsync -avz  k8s-130:/llm_workspace_1P/robin/Kimi2-PCL/model_param_mapping.json /Users/robin/work_dir/Kimi2-PCL
rsync -avz  k8s-130:/llm_workspace_1P/robin/Kimi2-PCL/model_param_mapping_2.json /Users/robin/work_dir/Kimi2-PCL
rsync -avz  k8s-130:/llm_workspace_1P/robin/Kimi2-PCL/model_param_hf.json /Users/robin/work_dir/Kimi2-PCL


## 进入容器
docker exec -it mindspeed-llm-env /bin/bash
docker exec -it vllm-ascend-env-a3  /bin/bash

# 转换模型
nohup bash scripts/ckpt_convert_hf2mcore.sh > hf2mcore.log  2>&1 &
nohup bash scripts/ckpt_convert_kimi2_hf2mcore.sh > ckpt_convert_kimi2_hf2mcore.log 2>&1 &
nohup bash scripts/ckpt_convert_kimi2_mcore2hf.sh >  ckpt_convert_kimi2_mcore2hf.log 2>&1 &
nohup bash scripts/ckpt_convert_mcore2hf.sh > mcore2hf.log  2>&1 &
systemctl daemon-reload && systemctl start docker


# 复制文件到镜像
# 使用示例
# 1. 远程文件模式（类似原脚本用法）
cd /llm_workspace_1P/robin/Kimi2-PCL/vllm-infer
bash copy_to_docker.sh -p 16 -r \
    /llm_workspace_1P/wf/deepseek_v2_real.py \
    /vllm-workspace/vllm/vllm/model_executor/models/deepseek_v2.py

# file1.py 是本地文件，file2.py 是容器内路径
bash copy_to_docker.sh -p 16 -r \
    /llm_workspace_1P/wf/default_loader_real.py \
    /vllm-workspace/vllm/vllm/model_executor/model_loader/default_loader.py

# file1.py 是本地文件，file2.py 是容器内路径
bash copy_to_docker.sh -p 16 -r \
    /llm_workspace_1P/wf/model_runner_v1_real.py \
    /vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py

# file1.py 是本地文件，file2.py 是容器内路径
bash copy_to_docker.sh -p 16 -r \
    /llm_workspace_1P/wf/vllm_config_real.py \
    /vllm-workspace/vllm/vllm/config/vllm.py


# 2. 批量复制（推荐！替代4条原命令）
bash copy_to_docker.sh -p 16 -r -c copy_files.conf.example

# 3. 仅复制到指定节点
bash copy_to_docker.sh -p 16 -r -n bms1905 -n bms1906 \
    /path/to/file.py /container/path/file.py

# 4. 本地文件复制到容器
bash copy_to_docker.sh -p 16 ./local_file.py /container/path/file.py

# 5. 调整并发数
bash copy_to_docker.sh -p 16 -r /remote/file.py /container/file.py


# 运行 vllm 推理
bash /llm_workspace_1P/robin/Kimi2-PCL/vllm-infer/run_vllm_test.sh


model_path=/llm_workspace_1P/fdd/workspace/MindSpeed-LLM-0227/MindSpeed-LLM/TrainResults/kimi2_L32_exp_4096_dies/b834d725-34df-47ca-ab07-4b93a36b9e87
python /llm_workspace_1P/robin/Kimi2-PCL/utils/get_mcore_weights.py $model_path \
  --tp 2 \
  --pp 8 \
  --ep 64 \
  --schedules-method dualpipev \
  --vpp-stage 2 \
  --num-layers 32 \
  --first-k-dense-replace 2 \
  --num-attention-heads 64 \
  --num-key-value-heads 32 \
  --hidden-size 7168 \
  --vocab-size 163840 \
  --num-experts 128 \
  --output /llm_workspace_1P/robin/Kimi2-PCL/mcore_weights_info.json
