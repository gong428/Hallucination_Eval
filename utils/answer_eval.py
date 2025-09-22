import re
from typing import Dict, Callable,Union, List
import json
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
    - answer가 리스트 또는 리스트 형태의 문자열인 경우: 
      prediction에 리스트 항목 중 하나라도 포함되면 True.
    - answer가 일반 문자열인 경우: 
      prediction에 해당 문자열이 포함되면 True.
    """
    prediction_lower = str(prediction).lower()
    
    # 1. answer가 이미 파이썬 리스트인 경우
    if isinstance(answer, list):
        return any(str(ans).lower() in prediction_lower for ans in answer)
    
    # 2. answer가 리스트 형태의 "문자열"인 경우 (예: "['a', 'b']")
    answer_str = str(answer).strip()
    if answer_str.startswith('[') and answer_str.endswith(']'):
        try:
            # 문자열을 실제 리스트로 파싱
            answer_list = json.loads(answer_str.replace("'", "\"")) # 작은따옴표도 처리
            if isinstance(answer_list, list):
                return any(str(ans).lower() in prediction_lower for ans in answer_list)
        except json.JSONDecodeError:
            # 파싱에 실패하면 일반 문자열로 취급하여 아래에서 처리
            pass

    # 3. answer가 일반 문자열인 경우 (기본 처리)
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
