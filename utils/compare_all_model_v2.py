import torch
import sys
import os
from pathlib import Path

# ===================== 【只改这里】 =====================
WEIGHT_ROOT = "/llm_workspace_1P/robin/hfhub/pcl-kimi2/kimi2-hf2mcore_step10000_v2_tp2_pp8_ep32_dualpipev_moe_tp_extend_ep1/iter_0000001"
WEIGHT_ORI_ROOT = "/llm_workspace_1P/fdd/workspace/MindSpeed-LLM-0227/MindSpeed-LLM/TrainResults/kimi2_L32_exp_4096_dies/da943594-a2aa-4999-8729-4d935c0bfbfc/iter_0010000"
# ========================================================

CHECKPOINT_FILE = "model_optim_rng.pt"
sys.path.append(os.path.abspath("/MindSpeed-LLM/Megatron-LM"))

def safe_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except:
        return None

def tensors_eq(a, b, atol=1e-6):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    try:
        if a.is_floating_point():
            return torch.allclose(a, b, atol=atol)
        else:
            return torch.all(a == b).item()
    except:
        return False

def get_model_state(ckpt):
    for k in ["model", "module", "model_state"]:
        if k in ckpt:
            return ckpt[k]
    return ckpt

def compare_one(new_path, ori_path):
    ck_new = safe_load(new_path)
    ck_ori = safe_load(ori_path)
    if not ck_new or not ck_ori:
        return False, "加载失败"

    s_new = get_model_state(ck_new)
    s_ori = get_model_state(ck_ori)

    total_ok = True

    # 对比 model0
    m0_new = s_new.get("model0", {})
    m0_ori = s_ori.get("model0", {})
    for k in set(m0_new.keys()) & set(m0_ori.keys()):
        t0, t1 = m0_new[k], m0_ori[k]
        if isinstance(t0, torch.Tensor) and isinstance(t1, torch.Tensor):
            if not tensors_eq(t0, t1):
                total_ok = False

    # 对比 model1
    m1_new = s_new.get("model1", {})
    m1_ori = s_ori.get("model1", {})
    for k in set(m1_new.keys()) & set(m1_ori.keys()):
        t0, t1 = m1_new[k], m1_ori[k]
        if isinstance(t0, torch.Tensor) and isinstance(t1, torch.Tensor):
            if not tensors_eq(t0, t1):
                total_ok = False

    return total_ok, "全部一致" if total_ok else "存在不一致"

def get_rank_dirs(root):
    return sorted([d for d in Path(root).iterdir() if d.is_dir() and d.name.startswith("mp_rank_")])

if __name__ == "__main__":
    dirs_new = get_rank_dirs(WEIGHT_ROOT)
    dirs_ori = get_rank_dirs(WEIGHT_ORI_ROOT)

    map_new = {d.name: d for d in dirs_new}
    map_ori = {d.name: d for d in dirs_ori}

    common_ranks = sorted(set(map_new.keys()) & set(map_ori.keys()))

    print("=" * 60)
    print("公共分片目录：", common_ranks)
    print("=" * 60)

    final_all_pass = True

    for rank in common_ranks:
        pt_new = map_new[rank] / CHECKPOINT_FILE
        pt_ori = map_ori[rank] / CHECKPOINT_FILE
        ok, msg = compare_one(pt_new, pt_ori)
        print(f"{rank} : {msg}")
        if not ok:
            final_all_pass = False

    print("=" * 60)
    if final_all_pass:
        print("✅ 最终总结果：所有公共目录、公共层的值完全一致")
    else:
        print("❌ 最终总结果：部分公共层存在不一致")
    print("=" * 60)
