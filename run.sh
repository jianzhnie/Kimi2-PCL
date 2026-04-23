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
rsync -avz Kimi2-PCL  bms1889:/llm_workspace_1P/robin/
rsync -avz Kimi2-PCL  k8s-131:/llm_workspace_1P/robin/
rsync -avz kimi_pcl_eval bms1889:/llm_workspace_1P/robin/
rsync -avz vllm  bms1889:/llm_workspace_1P/robin/
rsync -avz vllm-pcl  bms1889:/llm_workspace_1P/robin/
rsync -avz vllm-ascend  bms1889:/llm_workspace_1P/robin/

rsync -avz ./utils bms1889:/llm_workspace_1P/robin/Kimi2-PCL
rsync -avz ./scripts bms1889:/llm_workspace_1P/robin/Kimi2-PCL
rsync -avz ./models bms1889:/llm_workspace_1P/robin/Kimi2-PCL
rsync -avz ~/hfhub/datasets/openai/gsm8k bms1889:/llm_workspace_1P/robin/hfhub/datasets/openai

## 远程文件下载到本地
rsync -avz  bms1889:/llm_workspace_1P/robin/Kimi2-PCL.tar  /Users/robin/work_dir/
rsync -avz  bms1889:/llm_workspace_1P/robin/Kimi2-PCL/model_param_mapping.json /Users/robin/work_dir/Kimi2-PCL
rsync -avz  bms1889:/llm_workspace_1P/robin/Kimi2-PCL/model_param_mapping_2.json /Users/robin/work_dir/Kimi2-PCL
rsync -avz  bms1889:/llm_workspace_1P/robin/Kimi2-PCL/model_param_hf.json /Users/robin/work_dir/Kimi2-PCL

## 进入容器
docker exec -it mindspeed-llm-env /bin/bash
docker exec -it vllm-ascend-env-a3  /bin/bash
docker exec -it 2cd3a9664398 /bin/bash

# 转换模型
nohup bash scripts/ckpt_convert_hf2mcore.sh > hf2mcore.log  2>&1 &
nohup bash scripts/ckpt_convert_kimi2_hf2mcore.sh > ckpt_convert_kimi2_hf2mcore_step900_v2.log 2>&1 &
nohup bash scripts/ckpt_convert_kimi2_mcore2hf.sh >  ckpt_convert_kimi2_mcore2hf_step900_v2.log 2>&1 &
nohup bash scripts/ckpt_convert_mcore2hf.sh > mcore2hf.log  2>&1 &
systemctl daemon-reload && systemctl start docker

# 运行 vllm 推理
bash /llm_workspace_1P/robin/Kimi2-PCL/vllm-infer/run_vllm_test.sh
