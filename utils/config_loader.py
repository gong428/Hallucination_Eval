import yaml
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

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
    model_kwargs: Optional[Dict[str, Any]]
    inference_params: InferenceParams # 🆕 타입을 명확히 지정

@dataclass
class ResultsDirs:
    """results_dirs: 섹션의 설정을 담는 클래스"""
    model_outputs: str
    metrics: str

@dataclass
class PipelineConfig:
    """config.yaml 파일 전체의 구조를 담는 최상위 클래스"""
    model: ModelConfig
    datasets_to_evaluate: List[str]
    sample_num: Optional[int]
    results_dirs: ResultsDirs

def load_config(path: str) -> PipelineConfig:
    """YAML 설정 파일을 로드하여 PipelineConfig 객체로 반환합니다."""
    try:
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"오류: 설정 파일을 찾을 수 없습니다. 경로를 확인하세요: {path}")
        raise

    # 딕셔ner리를 데이터 클래스 객체로 변환
    return PipelineConfig(
        model=ModelConfig(
            name=config_dict['model']['name'],
            path=config_dict['model']['path'],
            inference_method=config_dict['model']['inference_method'],
            model_kwargs=config_dict['model'].get('model_kwargs'),
            # 🆕 명확한 타입의 InferenceParams 객체로 변환
            inference_params=InferenceParams(**config_dict['model']['inference_params'])
        ),
        datasets_to_evaluate=config_dict['datasets_to_evaluate'],
        sample_num=config_dict['sample_num'],
        results_dirs=ResultsDirs(**config_dict['results_dirs'])
    )

