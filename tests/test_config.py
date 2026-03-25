import yaml
import dacite
from collm.training.config import RecEncoderConfig, BackboneConfig, DataConfig, CoLLMConfig


def test_collm_config_from_dict():
    raw = {
        "rec": {
            "encoder_type": "sasrec",
            "checkpoint_path": "/tmp/rec.pth",
            "embedding_dim": 64,
            "use_collaborative_signal": True,
            "user_num": 1000,
            "item_num": 5000,
            "max_seq_len": 50,
            "n_layers": 2,
            "n_heads": 2,
            "dropout": 0.2,
            "n_gcn_layers": 3,
        },
        "backbone": {
            "model_name_or_path": "lmsys/vicuna-7b-v1.3",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_target_modules": ["q_proj", "v_proj"],
            "load_in_8bit": False,
        },
        "data": {
            "data_path": "/data/ml-1m",
            "dataset_type": "movielens",
            "max_history_len": 20,
            "max_seq_len": 512,
        },
    }
    cfg = dacite.from_dict(CoLLMConfig, raw, config=dacite.Config(cast=[list]))
    assert cfg.rec.encoder_type == "sasrec"
    assert cfg.backbone.lora_r == 8
    assert cfg.data.dataset_type == "movielens"
    assert cfg.rec.use_collaborative_signal is True
    # Verify lora_target_modules is properly cast to list (dacite cast=[list] guard)
    assert isinstance(cfg.backbone.lora_target_modules, list)
    assert cfg.backbone.lora_target_modules == ["q_proj", "v_proj"]


def test_config_from_yaml(tmp_path):
    yaml_content = """
rec:
  encoder_type: mf
  checkpoint_path: /tmp/mf.pth
  embedding_dim: 64
  use_collaborative_signal: false
  user_num: 100
  item_num: 200
  max_seq_len: 50
  n_layers: 2
  n_heads: 2
  dropout: 0.1
  n_gcn_layers: 3
backbone:
  model_name_or_path: lmsys/vicuna-7b-v1.3
  lora_r: 8
  lora_alpha: 16
  lora_target_modules:
    - q_proj
    - v_proj
  load_in_8bit: false
data:
  data_path: /data
  dataset_type: movielens
  max_history_len: 20
  max_seq_len: 512
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content)
    raw = yaml.safe_load(p.read_text())
    cfg = dacite.from_dict(CoLLMConfig, raw, config=dacite.Config(cast=[list]))
    assert cfg.rec.use_collaborative_signal is False
    assert cfg.rec.embedding_dim == 64
    assert isinstance(cfg.backbone.lora_target_modules, list)


def test_config_defaults():
    """Verify optional fields use correct defaults when omitted."""
    rec = RecEncoderConfig(encoder_type="mf", checkpoint_path="/tmp/mf.pth")
    assert rec.embedding_dim == 64
    assert rec.use_collaborative_signal is True
    assert rec.user_num == 0
    assert rec.item_num == 0
    assert rec.max_seq_len == 50
    assert rec.n_layers == 2
    assert rec.n_heads == 2
    assert rec.dropout == 0.2
    assert rec.n_gcn_layers == 3

    backbone = BackboneConfig(model_name_or_path="lmsys/vicuna-7b-v1.3")
    assert backbone.lora_r == 8
    assert backbone.lora_alpha == 16
    assert backbone.lora_target_modules == ["q_proj", "v_proj"]
    assert backbone.load_in_8bit is False

    data = DataConfig(data_path="/data", dataset_type="movielens")
    assert data.max_history_len == 20
    assert data.max_seq_len == 512
