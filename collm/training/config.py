from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RecEncoderConfig:
    encoder_type: str            # "mf" | "lightgcn" | "sasrec" | "din"
    checkpoint_path: str
    embedding_dim: int = 64
    use_collaborative_signal: bool = True
    # 執行時從資料集讀取
    user_num: int = 0
    item_num: int = 0
    # 序列模型共用
    max_seq_len: int = 50
    # SASRec 專用
    n_layers: int = 2
    n_heads: int = 2
    dropout: float = 0.2
    # LightGCN 專用
    n_gcn_layers: int = 3


@dataclass
class BackboneConfig:
    model_name_or_path: str
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    load_in_8bit: bool = False


@dataclass
class DataConfig:
    data_path: str
    dataset_type: str            # "movielens" | "amazon"
    max_history_len: int = 20
    max_seq_len: int = 512


@dataclass
class CoLLMConfig:
    rec: RecEncoderConfig
    backbone: BackboneConfig
    data: DataConfig
    # 協同信號注入層：0 = input embedding 層（原始 CoLLM 預設），
    # 正整數 N = 在第 N 個 transformer block 的輸出後注入（1-indexed）
    injection_layer: int = 0
    # CIE MLP 的 hidden dim 倍數（hidden = d_rec * proj_mid_times），對應原始論文設定
    proj_mid_times: int = 10
