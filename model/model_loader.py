import torch
from transformers import AutoTokenizer, pipeline
from typing import Optional, Dict, Any

def _get_torch_dtype(dtype_str: Optional[str] = None) -> Optional[torch.dtype]:
    """YAML에 명시된 문자열을 실제 torch.dtype 객체로 변환합니다."""
    if not dtype_str:
        return None
    
    # torch 객체에서 해당 데이터 타입 찾기
    return getattr(torch, dtype_str, None)

def load_model(model_name: str, model_path: str, model_kwargs: Optional[Dict[str, Any]] = None):
    """
    로컬 경로 또는 Hugging Face Hub에서 모델과 토크나이저를 로드합니다.
    config.yaml의 model_kwargs 설정을 동적으로 적용합니다.

    Args:
        model_name (str): 모델의 이름 (로그 출력용).
        model_path (str): 모델의 로컬 경로 또는 Hub ID.
        model_kwargs (dict, optional): 모델 로딩 시 사용할 추가 인자.

    Returns:
        tuple: 로드된 파이프라인 객체와 토크나이저.
    """
    print(f"--- Loading Model: {model_name} from '{model_path}' ---")

    # model_kwargs가 없으면 빈 딕셔너리로 초기화
    kwargs = model_kwargs if model_kwargs else {}

    # torch_dtype 문자열을 실제 torch.dtype 객체로 변환
    if 'torch_dtype' in kwargs and isinstance(kwargs['torch_dtype'], str):
        dtype = _get_torch_dtype(kwargs['torch_dtype'])
        if dtype:
            kwargs['torch_dtype'] = dtype
            print(f"Converted torch_dtype string to {dtype}")
        else:
            print(f"Warning: Could not find torch.dtype for '{kwargs['torch_dtype']}'. Ignoring.")
            del kwargs['torch_dtype']

    # bnb_4bit_compute_dtype 변환
    if 'bnb_4bit_compute_dtype' in kwargs and isinstance(kwargs['bnb_4bit_compute_dtype'], str):
        dtype = _get_torch_dtype(kwargs['bnb_4bit_compute_dtype'])
        if dtype:
            kwargs['bnb_4bit_compute_dtype'] = dtype
            print(f"Converted bnb_4bit_compute_dtype string to {dtype}")
        else:
            print(f"Warning: Could not find torch.dtype for '{kwargs['bnb_4bit_compute_dtype']}'. Ignoring.")
            del kwargs['bnb_4bit_compute_dtype']

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # text-generation 파이프라인 생성
        pipe = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=tokenizer,
            model_kwargs=kwargs,  # 🆕 동적으로 생성된 kwargs 전달
            device_map="auto"
        )
        print("✅ Model and Tokenizer loaded successfully with options:", kwargs)
        return pipe, tokenizer

    except Exception as e:
        print(f"❌ Error loading model '{model_path}': {e}")
        print("모델 경로가 올바른지, 필요한 라이브러리(예: bitsandbytes, accelerate)가 설치되었는지 확인해주세요.")
        raise

def load_model_path(model_name: str, model_path: str):
    """
    지정된 경로에서 transformers 파이프라인과 토크나이저를 로드합니다.
    4bit 양자화를 사용하여 메모리 사용량을 줄입니다.
    
    Returns:
        tuple: (pipe, tokenizer) 객체.
    """
    print(f"--- Loading model '{model_name}' from path: {model_path} ---")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 텍스트 생성 파이프라인 설정
    pipe = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "load_in_4bit": True, # 4bit 양자화 사용
        },
        device_map="auto" # 사용 가능한 GPU에 모델 자동 할당
    )
    
    print("Pipeline and tokenizer loaded successfully.")
    # ❗주의: 모델 자체 대신 파이프라인 객체를 반환합니다.
    return pipe, tokenizer