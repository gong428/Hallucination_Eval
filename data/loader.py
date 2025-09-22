# src/data_handler/loader.py

from datasets import DatasetDict
from uqlm.utils import load_example_dataset

def load_raw_dataset(dataset_name: str, sample_num: int | None) -> DatasetDict:
    """
    uqlm을 통해 지정된 이름의 데이터셋을 로드합니다.
    """
    print(f"Loading raw dataset: '{dataset_name}' with {sample_num or 'all'} samples...")
    
    # sample_num이 None이면 전체 데이터를 로드하도록 n=-1을 사용 (uqlm 라이브러리 사양에 따라 조정)
    # 만약 라이브러리가 n=None을 지원하면 그대로 사용
    num_to_load = -1 if sample_num is None else sample_num
    
    dataset = load_example_dataset(dataset_name, n=num_to_load)
    return dataset