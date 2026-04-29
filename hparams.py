import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class FTHyperParams:
    model_name_or_path: str
    data_path: str
    save_model_dir: str
    
    layer: int
    rewrite_module: str  # 'model.layers.{}.mlp.down_proj.weight'
    
    batch_size: int = 8
    lr: float = 5e-5
    weight_decay: float = 0
    num_steps: int = 20
    
    device: int = 0
    
    # LoRA Parameters
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        valid_keys = cls.__annotations__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)
