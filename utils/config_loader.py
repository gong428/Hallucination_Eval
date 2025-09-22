import yaml
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# ===================================================================
# ⚙️ config.yaml의 구조에 맞는 데이터 클래스를 정의합니다.
# ===================================================================

@dataclass
class DatasetConfig:
    """datasets: 섹션의 각 데이터셋 설정을 담는 클래스"""
    type: str

@dataclass
class InferenceParams:
    """inference_params: 섹션의 설정을 담는 클래스"""
    system_prompt: str
    user_prompt_template: str
    scorers: List[str]
    max_new_tokens: int
    temperature: float
    top_p: float

@dataclass
class ModelConfig:
    """model: 섹션의 설정을 담는 클래스"""
    name: str
    path: str
    inference_method: str
    model_kwargs: Dict[str, Any]
    inference_params: InferenceParams

@dataclass
class PipelineConfig:
    """config.yaml 파일 전체의 구조를 담는 최상위 클래스"""
    model: ModelConfig
    datasets: Dict[str, DatasetConfig]  # ⬅️ 이 부분이 수정되었습니다.
    sample_num: Optional[int]

# ===================================================================
# 🚀 YAML 파일을 읽어 위 클래스 객체로 변환하는 함수
# ===================================================================

def load_config(path: str) -> PipelineConfig:
    """YAML 설정 파일을 로드하여 PipelineConfig 객체로 반환합니다."""
    try:
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"오류: 설정 파일을 찾을 수 없습니다. 경로를 확인하세요: {path}")
        raise

    # 딕셔너리를 데이터 클래스 객체로 변환
    return PipelineConfig(
        model=ModelConfig(
            name=config_dict['model']['name'],
            path=config_dict['model']['path'],
            inference_method=config_dict['model']['inference_method'],
            model_kwargs=config_dict['model']['model_kwargs'],
            inference_params=InferenceParams(**config_dict['model']['inference_params'])
        ),
        # ⬇️ datasets 딕셔너리를 올바르게 처리하도록 수정되었습니다.
        datasets={name: DatasetConfig(**cfg) for name, cfg in config_dict['datasets'].items()},
        sample_num=config_dict['sample_num']
    )

