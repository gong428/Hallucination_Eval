# src/data_handler/loader.py

from datasets import load_dataset, concatenate_datasets, Dataset
from uqlm.utils import load_example_dataset
from typing import Union
import pandas as pd

def _load_and_preprocess_halu_eval_qa(num : int = None) -> pd.DataFrame:
    """
    'pminervini/HaluEval'의 'qa' 데이터를 로드하고 전처리하여 pandas.DataFrame으로 반환합니다.
    - 모든 split(train, validation, test)을 하나로 합칩니다.
    - 'knowledge'와 'question'을 합쳐 새로운 'question'으로 만듭니다.
    - 'right_answer'를 'answer'로 이름을 변경합니다.
    """
    print("Loading and preprocessing 'pminervini/HaluEval' qa dataset...")
    
    try:
        # 1. Hugging Face Hub에서 DatasetDict 로드
        dataset_dict = load_dataset("pminervini/HaluEval", "qa")

        # 2. 모든 split을 단일 Dataset으로 통합
        dataset = concatenate_datasets([dataset_dict[s] for s in dataset_dict])

        # 3. 전처리 함수 정의
        def preprocess(example):
            example['question'] = f"{example['knowledge']}\n{example['question']}"
            example['answer'] = example['right_answer']
            return example

        # 4. 전처리 적용
        processed_dataset = dataset.map(
            preprocess, 
            remove_columns=['knowledge', 'right_answer', 'hallucinated_answer']
        )
        
        # 5. 최종 결과를 pandas.DataFrame으로 변환하여 반환
        df = processed_dataset.to_pandas()
        if isinstance(num, int):
            df = df.iloc[:num]
        print("✅ 'halu_eval_qa' dataset processed successfully.")
        return df

    except Exception as e:
        print(f"❌ Error loading or preprocessing 'halu_eval_qa': {e}")
        raise


def load_raw_dataset(dataset_name: str, sample_num: int | None) -> Union[Dataset, dict]:
    """
    uqlm을 통해 지정된 이름의 데이터셋을 로드합니다.
    """
    if dataset_name == 'halu_eval_qa':
        num_to_load = -1 if sample_num is None else sample_num
        dataset = _load_and_preprocess_halu_eval_qa(num=num_to_load)

        return dataset
    # --- ⬆️ ---
    else:
        # 기존 uqlm 데이터셋 로딩 로직
        print(f"Loading '{dataset_name}' using uqlm.utils.load_example_dataset with {sample_num or 'all'} samples...")
        num_to_load = -1 if sample_num is None else sample_num
        
        try:
            dataset = load_example_dataset(dataset_name, n=num_to_load)
            return dataset
        except Exception as e:
            print(f"❌ Error loading '{dataset_name}' via uqlm: {e}")
            raise