# ══════════════════════════════════════════════════════════════════
#  tests/test_config.py
#  Tests for configuration loading and validation.
# ══════════════════════════════════════════════════════════════════
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import AgeGuardConfig, load_config

YAML_PATH = Path(__file__).parent.parent / "configs" / "base_config.yaml"


def test_config_loads():
    """Config loads from YAML without errors."""
    cfg = load_config(YAML_PATH)
    assert isinstance(cfg, AgeGuardConfig)


def test_config_model_params():
    """Model parameters are valid."""
    cfg = load_config(YAML_PATH)
    assert cfg.model.name == "efficientnetv2_s"
    assert cfg.model.img_size == 224
    assert 0 < cfg.model.dropout < 1


def test_config_training_params():
    """Training parameters are within valid ranges."""
    cfg = load_config(YAML_PATH)
    assert cfg.training.lr > 0
    assert cfg.training.batch_size > 0
    assert cfg.training.epochs > 0
    assert cfg.training.patience > 0


def test_config_business_params():
    """Business rules are correctly loaded."""
    cfg = load_config(YAML_PATH)
    assert cfg.business.legal_age == 21
    assert cfg.business.alert_threshold == 25
    assert cfg.business.mae_target == 5.0
    assert cfg.business.alert_threshold > cfg.business.legal_age


def test_config_paths_exist():
    """Critical directories exist."""
    cfg = load_config(YAML_PATH)
    assert cfg.base_dir.exists()
    assert cfg.images_dir.exists()
    assert cfg.processed_dir.exists()


def test_config_data_params():
    """Data split ratios are valid."""
    cfg = load_config(YAML_PATH)
    total = cfg.data.test_size + cfg.data.val_size
    assert total < 1.0  # Must leave room for train
    assert cfg.data.expected_samples == 7446
