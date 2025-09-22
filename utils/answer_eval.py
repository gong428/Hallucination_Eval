import re
from typing import Dict, Callable,Union, List

# ===================================================================
# 유형별 후처리기 및 평가 함수
# ===================================================================

def _evaluate_math(prediction: str, answer: str) -> bool:
    """
    수학 문제 답변을 평가합니다.
    문자열에서 마지막 숫자를 추출하여 비교합니다.
    """
    def postprocess(text: str) -> str:
        text = str(text).strip().replace(",", "")
        # 음수, 소수점, 쉼표를 포함한 마지막 숫자를 찾습니다.
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return numbers[-1] if numbers else ""

    return postprocess(prediction) == postprocess(answer)

def _evaluate_open_domain(prediction: str, answer: Union[str, List[str]]) -> bool:
    """
    자유 형식 답변을 평가합니다.
    - answer가 리스트인 경우: prediction에 리스트 항목 중 하나라도 포함되면 True.
    - answer가 문자열인 경우: prediction에 해당 문자열이 포함되면 True.
    """
    prediction_lower = str(prediction).lower()
    
    if isinstance(answer, list):
        # 정답이 리스트인 경우, 하나라도 포함되면 정답으로 처리
        return any(str(ans).lower() in prediction_lower for ans in answer)
    else:
        # 정답이 단일 문자열인 경우, 해당 문자열 포함 여부 확인
        return str(answer).lower() in prediction_lower

def _evaluate_exact_match(prediction: str, answer: str) -> bool:
    """
    지문/객관식 답변을 평가합니다. (완전 일치 기준)
    정규화 후 문자열이 정확히 일치해야 정답으로 처리합니다.
    """
    def postprocess(text: str) -> str:
        text = str(text).strip().lower()
        # "A)", "(B", "C." 와 같은 형식을 "a", "b", "c"로 통일
        mcq_match = re.match(r'^\s*\(?([a-zA-Z])\)?', text)
        if mcq_match:
            return mcq_match.group(1).lower()
        return text

    return postprocess(prediction) == postprocess(answer)


# ===================================================================
# 메인 컨트롤러 함수
# ===================================================================

# 데이터셋 유형과 평가 함수를 매핑합니다.
EVALUATOR_MAP: Dict[str, Callable[[str, str], bool]] = {
    'math': _evaluate_math,
    'open_domain': _evaluate_open_domain,
    'exact_match': _evaluate_exact_match,
}

def is_correct(prediction: str, answer: str, dataset_type: str) -> bool:
    """
    데이터셋 유형에 맞는 평가 함수를 호출하여 정답 여부를 반환합니다.

    Args:
        prediction (str): 모델이 생성한 답변.
        answer (str): 실제 정답 (ground_truth).
        dataset_type (str): 'math', 'open_domain', 'exact_match' 중 하나.

    Returns:
        bool: 정답 여부 (True/False).
    """
    evaluator_func = EVALUATOR_MAP.get(dataset_type)
    
    if evaluator_func is None:
        raise ValueError(f"Unknown dataset_type: '{dataset_type}'. "
                         f"Available types are: {list(EVALUATOR_MAP.keys())}")
                         
    return evaluator_func(prediction, answer)
