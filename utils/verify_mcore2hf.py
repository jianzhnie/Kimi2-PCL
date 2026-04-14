#!/usr/bin/env python3
"""
验证 Mcore -> Huggingface 转换结果是否与 model_param_mapping.json 对齐。

用法:
    cd /Users/jianzhengnie/work_dir/Kimi2-PCL
    python verify_mcore2hf.py
"""

import json
import sys

MCORE_JSON = "model_param_mapping.json"
HF_JSON = "model_param_hf.json"

NUM_LAYERS = 32
NUM_EXPERTS = 128
DENSE_FFN = 18432
MOE_FFN = 12288
HIDDEN = 7168


def load_json(path):
    with open(path) as f:
        return json.load(f)


def check_attention(mcore, hf, layer):
    """专门验证 Attention 子模块的所有参数映射与形状。"""
    errors = []
    prefix_m = f"module.decoder.layers.{layer}"
    prefix_h = f"model.layers.{layer}"

    # 1. LayerNorms
    in_ln_m = mcore[f"{prefix_m}.input_layernorm.weight"]["shape"]
    in_ln_h = hf[f"{prefix_h}.input_layernorm.weight"]["shape"]
    if in_ln_m != in_ln_h:
        errors.append(
            f"Layer {layer} input_layernorm mismatch: mcore={in_ln_m}, hf={in_ln_h}"
        )

    pre_mlp_m = mcore[f"{prefix_m}.pre_mlp_layernorm.weight"]["shape"]
    post_attn_h = hf[f"{prefix_h}.post_attention_layernorm.weight"]["shape"]
    if pre_mlp_m != post_attn_h:
        errors.append(
            f"Layer {layer} pre_mlp_layernorm mismatch: mcore={pre_mlp_m}, hf={post_attn_h}"
        )

    # 2. Q/K LayerNorm (qk_layernorm 启用时必须存在)
    q_ln_m = mcore[f"{prefix_m}.self_attention.q_layernorm.weight"]["shape"]
    q_ln_h = hf[f"{prefix_h}.self_attn.q_layernorm.weight"]["shape"]
    if q_ln_m != q_ln_h:
        errors.append(
            f"Layer {layer} q_layernorm mismatch: mcore={q_ln_m}, hf={q_ln_h}"
        )

    k_ln_m = mcore[f"{prefix_m}.self_attention.k_layernorm.weight"]["shape"]
    k_ln_h = hf[f"{prefix_h}.self_attn.k_layernorm.weight"]["shape"]
    if k_ln_m != k_ln_h:
        errors.append(
            f"Layer {layer} k_layernorm mismatch: mcore={k_ln_m}, hf={k_ln_h}"
        )

    # 3. QKV 拆分验证
    qkv_shape = mcore[f"{prefix_m}.self_attention.linear_qkv.weight"]["shape"]
    q_shape = hf[f"{prefix_h}.self_attn.q_proj.weight"]["shape"]
    k_shape = hf[f"{prefix_h}.self_attn.k_proj.weight"]["shape"]
    v_shape = hf[f"{prefix_h}.self_attn.v_proj.weight"]["shape"]
    if qkv_shape[0] != q_shape[0] + k_shape[0] + v_shape[0]:
        errors.append(
            f"Layer {layer} QKV split mismatch: "
            f"mcore={qkv_shape}, q={q_shape}, k={k_shape}, v={v_shape}"
        )

    # 4. Output projection
    proj_shape = mcore[f"{prefix_m}.self_attention.linear_proj.weight"]["shape"]
    o_shape = hf[f"{prefix_h}.self_attn.o_proj.weight"]["shape"]
    if proj_shape != o_shape:
        errors.append(
            f"Layer {layer} proj/o mismatch: mcore={proj_shape}, hf={o_shape}"
        )

    # 5. Rotary embedding inv_freq (GQA: qk_head_dim=128 -> 64)
    rotary_shape = hf[f"{prefix_h}.self_attn.rotary_emb.inv_freq"]["shape"]
    expected_rotary = [64]
    if rotary_shape != expected_rotary:
        errors.append(
            f"Layer {layer} rotary_emb.inv_freq mismatch: hf={rotary_shape}, expected {expected_rotary}"
        )

    return errors


def check_dense_layer(mcore, hf, layer):
    errors = []
    prefix_m = f"module.decoder.layers.{layer}"
    prefix_h = f"model.layers.{layer}"

    # Attention 子模块完整验证
    errors.extend(check_attention(mcore, hf, layer))

    # MLP: linear_fc1 -> gate_proj + up_proj
    fc1_shape = mcore[f"{prefix_m}.mlp.linear_fc1.weight"]["shape"]
    gate_shape = hf[f"{prefix_h}.mlp.gate_proj.weight"]["shape"]
    up_shape = hf[f"{prefix_h}.mlp.up_proj.weight"]["shape"]
    if fc1_shape[0] != gate_shape[0] * 2 or gate_shape != up_shape:
        errors.append(
            f"Layer {layer} dense MLP fc1 split mismatch: "
            f"mcore_fc1={fc1_shape}, hf_gate={gate_shape}, hf_up={up_shape}"
        )

    # MLP: linear_fc2 -> down_proj
    fc2_shape = mcore[f"{prefix_m}.mlp.linear_fc2.weight"]["shape"]
    down_shape = hf[f"{prefix_h}.mlp.down_proj.weight"]["shape"]
    if fc2_shape != down_shape:
        errors.append(
            f"Layer {layer} dense MLP fc2/down mismatch: "
            f"mcore={fc2_shape}, hf={down_shape}"
        )

    return errors


def check_moe_layer(mcore, hf, layer):
    errors = []
    prefix_m = f"module.decoder.layers.{layer}"
    prefix_h = f"model.layers.{layer}"

    # Attention 子模块完整验证
    errors.extend(check_attention(mcore, hf, layer))

    # 1. Grouped GEMM weight1 / weight2 shapes
    w1_shape = mcore[f"{prefix_m}.mlp.experts.weight1"]["shape"]
    w2_shape = mcore[f"{prefix_m}.mlp.experts.weight2"]["shape"]
    expected_w1 = [HIDDEN, NUM_EXPERTS * 2 * MOE_FFN]
    expected_w2 = [NUM_EXPERTS * MOE_FFN, HIDDEN]
    if w1_shape != expected_w1:
        errors.append(
            f"Layer {layer} experts.weight1 shape unexpected: {w1_shape}, expected {expected_w1}"
        )
    if w2_shape != expected_w2:
        errors.append(
            f"Layer {layer} experts.weight2 shape unexpected: {w2_shape}, expected {expected_w2}"
        )

    # 2. Each expert gate/up/down shapes
    for e in range(NUM_EXPERTS):
        gate = hf[f"{prefix_h}.mlp.experts.{e}.gate_proj.weight"]["shape"]
        up = hf[f"{prefix_h}.mlp.experts.{e}.up_proj.weight"]["shape"]
        down = hf[f"{prefix_h}.mlp.experts.{e}.down_proj.weight"]["shape"]
        if gate != [MOE_FFN, HIDDEN] or up != [MOE_FFN, HIDDEN] or down != [HIDDEN, MOE_FFN]:
            errors.append(
                f"Layer {layer} expert {e} shape mismatch: "
                f"gate={gate}, up={up}, down={down}"
            )
            break

    # 3. Router weight
    router_shape = mcore[f"{prefix_m}.mlp.router.weight"]["shape"]
    gate_shape = hf[f"{prefix_h}.mlp.gate.weight"]["shape"]
    if router_shape != gate_shape:
        errors.append(
            f"Layer {layer} router/gate mismatch: mcore={router_shape}, hf={gate_shape}"
        )

    # 4. Shared experts
    shared_fc1 = mcore[f"{prefix_m}.mlp.shared_experts.linear_fc1.weight"]["shape"]
    shared_fc2 = mcore[f"{prefix_m}.mlp.shared_experts.linear_fc2.weight"]["shape"]
    shared_gate = hf[f"{prefix_h}.mlp.shared_experts.gate_proj.weight"]["shape"]
    shared_up = hf[f"{prefix_h}.mlp.shared_experts.up_proj.weight"]["shape"]
    shared_down = hf[f"{prefix_h}.mlp.shared_experts.down_proj.weight"]["shape"]
    if shared_fc1[0] != shared_gate[0] * 2:
        errors.append(
            f"Layer {layer} shared expert fc1 split mismatch: mcore={shared_fc1}, gate={shared_gate}"
        )
    if shared_gate != shared_up:
        errors.append(
            f"Layer {layer} shared expert gate/up mismatch: gate={shared_gate}, up={shared_up}"
        )
    if shared_fc2 != shared_down:
        errors.append(
            f"Layer {layer} shared expert fc2/down mismatch: mcore={shared_fc2}, hf={shared_down}"
        )

    return errors


def check_nonlayer(mcore, hf):
    errors = []
    if mcore["module.embedding.word_embeddings.weight"]["shape"] != hf["model.embed_tokens.weight"]["shape"]:
        errors.append("Embedding shape mismatch")
    if mcore["module.decoder.final_layernorm.weight"]["shape"] != hf["model.norm.weight"]["shape"]:
        errors.append("Final layernorm shape mismatch")
    if mcore["module.output_layer.weight"]["shape"] != hf["lm_head.weight"]["shape"]:
        errors.append("LM head shape mismatch")
    return errors


def check_rotary_and_bias(hf):
    errors = []
    for i in range(NUM_LAYERS):
        key = f"model.layers.{i}.self_attn.rotary_emb.inv_freq"
        if key not in hf:
            errors.append(f"Missing rotary_emb.inv_freq in layer {i}")
    for i in range(2, NUM_LAYERS):
        key = f"model.layers.{i}.mlp.gate.e_score_correction_bias"
        if key not in hf:
            errors.append(f"Missing gate.e_score_correction_bias in MoE layer {i}")
    return errors


def main():
    print(f"Loading {MCORE_JSON} ...")
    mcore_data = load_json(MCORE_JSON)
    mcore = mcore_data["megatron_params"]

    print(f"Loading {HF_JSON} ...")
    hf_data = load_json(HF_JSON)
    hf = hf_data["weight_map"]

    print(f"\nMcore params: {len(mcore)}")
    print(f"HF params:    {len(hf)}")

    all_errors = []

    # Dense layers
    for layer in range(2):
        all_errors.extend(check_dense_layer(mcore, hf, layer))

    # MoE layers
    for layer in range(2, NUM_LAYERS):
        all_errors.extend(check_moe_layer(mcore, hf, layer))

    # Non-layer
    all_errors.extend(check_nonlayer(mcore, hf))

    # Rotary & bias presence
    all_errors.extend(check_rotary_and_bias(hf))

    # Expert count sanity check
    experts_count = {}
    for k in hf:
        if ".mlp.experts." in k:
            layer = int(k.split("model.layers.")[1].split(".")[0])
            experts_count.setdefault(layer, set())
            expert_id = int(k.split(".mlp.experts.")[1].split(".")[0])
            experts_count[layer].add(expert_id)

    moe_layers_with_128 = sum(1 for ids in experts_count.values() if len(ids) == NUM_EXPERTS)
    print(f"MoE layers with {NUM_EXPERTS} experts: {moe_layers_with_128} (expected {NUM_LAYERS - 2})")
    if moe_layers_with_128 != NUM_LAYERS - 2:
        all_errors.append(
            f"Expert count mismatch: {moe_layers_with_128} layers have {NUM_EXPERTS} experts"
        )

    print()
    if all_errors:
        print(f"FAILED — {len(all_errors)} error(s) found:")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("PASSED — All shape and key checks aligned correctly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
