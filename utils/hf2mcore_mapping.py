"""
Huggingface (HF) 到 Megatron-Core (MCore) 权重映射定义

基于 Kimi2-1T 模型架构，支持：
- GQA (Grouped Query Attention)
- MoE (Mixture of Experts)
- TP/PP/EP 并行
- DualPipeV 调度

参考配置：
- scripts/pretrain_kimi2_1t_4k.sh (训练脚本)
- megatron_model.py (MCore 模型定义)
- model_param_mapping.json (参数映射)
"""

from typing import Callable, Dict, List, Tuple

# =============================================================================
# 模型架构常量定义 (与 models/config.json 和训练脚本保持一致)
# =============================================================================

class ModelConfig:
    """Kimi2-1T 模型配置"""
    
    # 基础架构参数
    HIDDEN_SIZE = 7168
    NUM_LAYERS = 32
    NUM_ATTENTION_HEADS = 64
    NUM_KEY_VALUE_HEADS = 32  # GQA: 64 Q heads / 2 groups = 32 KV heads
    
    # 注意力维度
    QK_HEAD_DIM = 128          # qk_nope_head_dim
    QK_POS_EMB_HEAD_DIM = 64   # qk_rope_head_dim
    V_HEAD_DIM = 128           # v_head_dim
    
    # MoE 配置
    NUM_EXPERTS = 128
    FIRST_K_DENSE_REPLACE = 2  # 前 2 层使用 Dense MLP
    MOE_FFN_HIDDEN_SIZE = 12288
    SHARED_EXPERTS = 1
    
    # 标准 FFN 配置
    FFN_HIDDEN_SIZE = 18432
    
    # 词汇表
    VOCAB_SIZE = 163840
    
    # 位置编码
    MAX_POSITION_EMBEDDINGS = 131072
    ROTARY_BASE = 50000.0
    ROPE_SCALING_FACTOR = 32.0
    ROPE_SCALING_TYPE = "yarn"
    
    # 归一化
    NORM_EPS = 1e-6
    
    # 数据类型
    DTYPE = "bfloat16"


# =============================================================================
# 权重映射定义
# =============================================================================

class WeightMapping:
    """HF 到 MCore 的权重名称映射"""
    
    # -------------------------------------------------------------------------
    # 嵌入层
    # -------------------------------------------------------------------------
    EMBEDDING = {
        "hf": "model.embed_tokens.weight",
        "mcore": "embedding.word_embeddings.weight",
        "tp_dim": 0,  # 在 vocab 维度切分
    }
    
    # -------------------------------------------------------------------------
    # 输出层
    # -------------------------------------------------------------------------
    FINAL_NORM = {
        "hf": "model.norm.weight",
        "mcore": "decoder.final_layernorm.weight",
    }
    
    LM_HEAD = {
        "hf": "lm_head.weight",
        "mcore": "output_layer.weight",
        "tp_dim": 0,  # 在 vocab 维度切分
    }
    
    # -------------------------------------------------------------------------
    # 层内映射模板
    # -------------------------------------------------------------------------
    @staticmethod
    def get_layer_mappings(layer_idx: int) -> List[Dict[str, str]]:
        """获取指定层的权重映射列表"""
        hf_prefix = f"model.layers.{layer_idx}"
        mcore_prefix = f"decoder.layers.{layer_idx}"
        
        return [
            # LayerNorm
            {
                "hf": f"{hf_prefix}.input_layernorm.weight",
                "mcore": f"{mcore_prefix}.input_layernorm.weight",
            },
            {
                "hf": f"{hf_prefix}.post_attention_layernorm.weight",
                "mcore": f"{mcore_prefix}.pre_mlp_layernorm.weight",
            },
            
            # Attention - QKV (HF 分离, MCore 融合)
            {
                "hf": f"{hf_prefix}.self_attn.q_proj.weight",
                "mcore": f"{mcore_prefix}.self_attention.linear_qkv.weight",
                "is_fused": True,
                "fuse_order": ["q", "k", "v"],
                "tp_dim": 0,
            },
            {
                "hf": f"{hf_prefix}.self_attn.k_proj.weight",
                "mcore": f"{mcore_prefix}.self_attention.linear_qkv.weight",
                "is_fused": True,
                "fuse_part": "k",
            },
            {
                "hf": f"{hf_prefix}.self_attn.v_proj.weight",
                "mcore": f"{mcore_prefix}.self_attention.linear_qkv.weight",
                "is_fused": True,
                "fuse_part": "v",
            },
            
            # Attention - Output
            {
                "hf": f"{hf_prefix}.self_attn.o_proj.weight",
                "mcore": f"{mcore_prefix}.self_attention.linear_proj.weight",
                "tp_dim": 1,  # 在 hidden_size 维度切分
            },
            
            # Attention - QK LayerNorm (可选)
            {
                "hf": f"{hf_prefix}.self_attn.q_layernorm.weight",
                "mcore": f"{mcore_prefix}.self_attention.q_layernorm.weight",
                "optional": True,
            },
            {
                "hf": f"{hf_prefix}.self_attn.k_layernorm.weight",
                "mcore": f"{mcore_prefix}.self_attention.k_layernorm.weight",
                "optional": True,
            },
        ]
    
    @staticmethod
    def get_dense_mlp_mappings(layer_idx: int) -> List[Dict[str, str]]:
        """获取 Dense MLP 层的权重映射 (前 FIRST_K_DENSE_REPLACE 层)"""
        hf_prefix = f"model.layers.{layer_idx}"
        mcore_prefix = f"decoder.layers.{layer_idx}"
        
        return [
            # MLP - gate_up_proj (HF 分离, MCore 融合)
            {
                "hf": f"{hf_prefix}.mlp.gate_proj.weight",
                "mcore": f"{mcore_prefix}.mlp.linear_fc1.weight",
                "is_fused": True,
                "fuse_order": ["gate", "up"],
                "tp_dim": 0,
            },
            {
                "hf": f"{hf_prefix}.mlp.up_proj.weight",
                "mcore": f"{mcore_prefix}.mlp.linear_fc1.weight",
                "is_fused": True,
                "fuse_part": "up",
            },
            # MLP - down_proj
            {
                "hf": f"{hf_prefix}.mlp.down_proj.weight",
                "mcore": f"{mcore_prefix}.mlp.linear_fc2.weight",
                "tp_dim": 1,
            },
        ]
    
    @staticmethod
    def get_moe_mappings(layer_idx: int, num_experts: int = 128) -> List[Dict[str, str]]:
        """获取 MoE 层的权重映射"""
        hf_prefix = f"model.layers.{layer_idx}"
        mcore_prefix = f"decoder.layers.{layer_idx}"
        
        mappings = [
            # Router
            {
                "hf": f"{hf_prefix}.mlp.gate.weight",
                "mcore": f"{mcore_prefix}.mlp.router.weight",
            },
            {
                "hf": f"{hf_prefix}.mlp.gate.e_score_correction_bias",
                "mcore": f"{mcore_prefix}.mlp.router.expert_bias",
                "optional": True,
            },
            
            # Shared Experts
            {
                "hf": f"{hf_prefix}.mlp.shared_experts.gate_proj.weight",
                "mcore": f"{mcore_prefix}.mlp.shared_experts.linear_fc1.weight",
                "is_fused": True,
                "fuse_order": ["gate", "up"],
                "tp_dim": 0,
            },
            {
                "hf": f"{hf_prefix}.mlp.shared_experts.up_proj.weight",
                "mcore": f"{mcore_prefix}.mlp.shared_experts.linear_fc1.weight",
                "is_fused": True,
                "fuse_part": "up",
            },
            {
                "hf": f"{hf_prefix}.mlp.shared_experts.down_proj.weight",
                "mcore": f"{mcore_prefix}.mlp.shared_experts.linear_fc2.weight",
                "tp_dim": 1,
            },
        ]
        
        # 专家权重 (每个专家独立的 gate_proj, up_proj, down_proj)
        for expert_idx in range(num_experts):
            mappings.extend([
                {
                    "hf": f"{hf_prefix}.mlp.experts.{expert_idx}.gate_proj.weight",
                    "mcore": f"{mcore_prefix}.mlp.experts.local_experts.{expert_idx}.linear_fc1.weight",
                    "is_fused": True,
                    "fuse_order": ["gate", "up"],
                    "expert_idx": expert_idx,
                },
                {
                    "hf": f"{hf_prefix}.mlp.experts.{expert_idx}.up_proj.weight",
                    "mcore": f"{mcore_prefix}.mlp.experts.local_experts.{expert_idx}.linear_fc1.weight",
                    "is_fused": True,
                    "fuse_part": "up",
                    "expert_idx": expert_idx,
                },
                {
                    "hf": f"{hf_prefix}.mlp.experts.{expert_idx}.down_proj.weight",
                    "mcore": f"{mcore_prefix}.mlp.experts.local_experts.{expert_idx}.linear_fc2.weight",
                    "expert_idx": expert_idx,
                    "tp_dim": 1,
                },
            ])
        
        return mappings


# =============================================================================
# 维度计算工具
# =============================================================================

class DimensionCalculator:
    """计算各权重张量的维度"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
    
    @property
    def q_head_dim(self) -> int:
        """Q head 的总维度 (包含位置编码)"""
        return self.config.QK_HEAD_DIM + self.config.QK_POS_EMB_HEAD_DIM
    
    @property
    def q_proj_rows(self) -> int:
        """Q projection 的行数"""
        return self.config.NUM_ATTENTION_HEADS * self.q_head_dim
    
    @property
    def k_proj_rows(self) -> int:
        """K projection 的行数"""
        return self.config.NUM_KEY_VALUE_HEADS * self.q_head_dim
    
    @property
    def v_proj_rows(self) -> int:
        """V projection 的行数"""
        return self.config.NUM_KEY_VALUE_HEADS * self.config.V_HEAD_DIM
    
    @property
    def qkv_proj_rows(self) -> int:
        """融合 QKV projection 的总行数"""
        return self.q_proj_rows + self.k_proj_rows + self.v_proj_rows
    
    def get_tp_sharded_shape(self, shape: Tuple[int, ...], tp_dim: int, tp_size: int) -> Tuple[int, ...]:
        """计算 TP 切分后的形状"""
        if tp_dim is None:
            return shape
        
        sharded_shape = list(shape)
        if tp_dim < len(shape):
            sharded_shape[tp_dim] = shape[tp_dim] // tp_size
        
        return tuple(sharded_shape)


# =============================================================================
# 转换辅助函数
# =============================================================================

def get_weight_transform_func(mapping: Dict) -> Callable:
    """
    根据映射定义返回权重转换函数
    
    Args:
        mapping: 权重映射定义字典
        
    Returns:
        转换函数，输入 HF 权重，输出 MCore 格式权重
    """
    if mapping.get("is_fused"):
        # 融合权重的转换由专门的融合函数处理
        return lambda x: x
    
    # 标准转换：仅确保 contiguous
    return lambda x: x.contiguous() if not x.is_contiguous() else x


def validate_shapes(hf_shape: Tuple[int, ...], expected_shape: Tuple[int, ...], 
                    name: str, tolerance: float = 0.01) -> bool:
    """
    验证 HF 权重形状是否符合预期
    
    Args:
        hf_shape: HF 权重的实际形状
        expected_shape: 期望的形状
        name: 权重名称 (用于错误信息)
        tolerance: 形状不匹配时的容差比例
        
    Returns:
        验证是否通过
        
    Raises:
        ValueError: 形状不匹配且超出容差
    """
    if hf_shape == expected_shape:
        return True
    
    # 检查是否在第一维有轻微不匹配 (可能是 vocab size 的 padding)
    if len(hf_shape) == len(expected_shape):
        mismatches = sum(1 for a, b in zip(hf_shape, expected_shape) if a != b)
        if mismatches == 1 and hf_shape[0] != expected_shape[0]:
            ratio = abs(hf_shape[0] - expected_shape[0]) / expected_shape[0]
            if ratio <= tolerance:
                return True
    
    raise ValueError(
        f"{name} 形状不匹配: "
        f"期望 {expected_shape}, 实际 {hf_shape}"
    )


# =============================================================================
# 导出常用配置
# =============================================================================

__all__ = [
    "ModelConfig",
    "WeightMapping", 
    "DimensionCalculator",
    "get_weight_transform_func",
    "validate_shapes",
]
