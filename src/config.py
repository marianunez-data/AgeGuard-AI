# ══════════════════════════════════════════════════════════════════
#  src/config.py
#  Pydantic v2 Config: loads YAML, validates types at
#  runtime, exposes all derived paths as @property.
# ══════════════════════════════════════════════════════════════════
import json
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


# ── Sub-models (nested validation) ────────────────────────────────
class ModelCfg(BaseModel):
    name: str = "efficientnetv2_s"
    pretrained: bool = True
    img_size: int = Field(224, gt=0)
    dropout: float = Field(0.3, ge=0.0, le=1.0)


class TrainingCfg(BaseModel):
    batch_size: int = Field(32, gt=0)
    epochs: int = Field(20, gt=0)
    lr: float = Field(3e-4, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    num_workers: int = Field(4, ge=0)
    use_amp: bool = True
    patience: int = Field(5, gt=0)


class DataCfg(BaseModel):
    test_size: float = Field(0.15, gt=0, lt=1)
    val_size: float = Field(0.15, gt=0, lt=1)
    seed: int = 12345
    expected_samples: int = 7446


class BusinessCfg(BaseModel):
    legal_age: int = 21
    alert_threshold: int = 25
    mae_target: float = 5.0


# ── Root config ───────────────────────────────────────────────────
class AgeGuardConfig(BaseModel):
    base_dir: Path
    model: ModelCfg = ModelCfg()
    training: TrainingCfg = TrainingCfg()
    data: DataCfg = DataCfg()
    business: BusinessCfg = BusinessCfg()

    # ── Derived paths — use these everywhere, never raw strings ───
    @property
    def images_dir(self) -> Path:
        return self.base_dir / "data/final_files"

    @property
    def labels_path(self) -> Path:
        return self.base_dir / "data/labels.csv"

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models/checkpoints"

    @property
    def onnx_dir(self) -> Path:
        return self.base_dir / "models/onnx"

    @property
    def reports_eda(self) -> Path:
        return self.base_dir / "reports/eda"

    @property
    def reports_training(self) -> Path:
        return self.base_dir / "reports/training"

    @property
    def reports_eval(self) -> Path:
        return self.base_dir / "reports/evaluation"

    @property
    def reports_gradcam(self) -> Path:
        return self.base_dir / "reports/explainability"

    @property
    def artifacts_dir(self) -> Path:
        return self.base_dir / "artifacts"

    @property
    def configs_dir(self) -> Path:
        return self.base_dir / "configs"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def processed_dir(self) -> Path:
        return self.base_dir / "data" / "processed"

    def snapshot(self) -> Path:
        """Persist full config (fields + derived paths) to JSON — one per run."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.configs_dir / f"config_{run_id}.json"
        payload = self.model_dump()
        payload["base_dir"] = str(self.base_dir)
        payload["_derived_paths"] = {
            "images_dir": str(self.images_dir),
            "labels_path": str(self.labels_path),
            "models_dir": str(self.models_dir),
            "reports_eda": str(self.reports_eda),
            "reports_training": str(self.reports_training),
            "reports_eval": str(self.reports_eval),
        }
        out.write_text(json.dumps(payload, indent=2, default=str))
        return out

    def validate_paths(self) -> bool:
        checks = {
            "images_dir": self.images_dir.exists(),
            "labels_path": self.labels_path.exists(),
        }
        for name, ok in checks.items():
            status = "OK" if ok else "MISSING"
            print(f"    {status}  {name}")
        return all(checks.values())


# ── Loader — reads YAML, Pydantic validates on construction ───────
def load_config(yaml_path: Path) -> AgeGuardConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    return AgeGuardConfig(
        base_dir=raw["base_dir"],
        model=ModelCfg(**raw["model"]),
        training=TrainingCfg(**raw["training"]),
        data=DataCfg(**raw["data"]),
        business=BusinessCfg(**raw["business"]),
    )
