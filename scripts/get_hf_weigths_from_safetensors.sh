ckpt_path="/llm_workspace_1P/robin/hfhub/pcl-kimi2/kimi2-mcore2hf"
LOAD_DIR="${LOAD_DIR:-$ckpt_path}"
SAVE_DIR="${SAVE_DIR:-/llm_workspace_1P/robin/Kimi2-PCL}"

# --schedules-method dualpipev --vpp-stage 2 \
python /llm_workspace_1P/robin/Kimi2-PCL/utils/get_hf_weights_from_safetensors.py \
    $ckpt_path \
    --pretty \
    --output $SAVE_DIR/model_param_hf.json