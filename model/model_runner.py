import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from uqlm.scorers import WhiteBoxUQ
from typing import List, Dict

# 프롬프트 템플릿은 추론 방식과 무관하게 공통으로 사용될 수 있습니다.
FEW_SHOT_PROMPT_TEMPLATE = """When you solve this math problem only return the answer with no additional text.

Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
A: 72

Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
A: 10

Q: {question}
A:"""


def math_postprocessor(input_string: str) -> str:
    """
    Parameters
    ----------

    input_string: str
        The string from which the numerical answer will be extracted. Only the integer part is extracted.

    Returns
    -------
    str
        The postprocessed string containing the integer part of the answer.
    """
    result = ""
    for char in input_string:
        if char.isdigit():
            result += char
        elif char == ".":
            break
    return result

def _extract_logprobs(outputs, input_length: int, tokenizer) -> List[Dict[str, float]]:
    """
    Hugging Face 모델의 generate 출력에서 토큰별 logprob을 추출합니다.
    이 함수가 생성하는 [{'logprob': 값}] 형식은 WhiteBoxUQ 클래스의 
    get_logprobs, _get_probs 메소드가 기대하는 입력 형식과 일치합니다.
    """
    # 1. 생성된 시퀀스(답변) 부분만 분리합니다.
    generated_sequence = outputs.sequences[0][input_length:]
    
    # 2. 모델의 출력 점수(logits)를 로그 확률로 변환합니다.
    logprobs_list = [torch.nn.functional.log_softmax(score, dim=-1) for score in outputs.scores]
    
    # --- ⬇️ EOS 토큰 제외 로직 추가 ---
    sequence_to_process = generated_sequence
    # 마지막 토큰이 EOS 토큰이면, 처리할 시퀀스에서 제외합니다.
    if len(sequence_to_process) > 0 and sequence_to_process[-1] == tokenizer.eos_token_id:
        sequence_to_process = sequence_to_process[:-1]
    # --- ⬆️ EOS 토큰 제외 로직 끝 ---

    sequence_logprobs = []
    # 3. 생성된 각 토큰에 해당하는 로그 확률 값을 추출합니다.
    # 이제 EOS가 제외된 시퀀스를 순회합니다.
    for i, token_id in enumerate(sequence_to_process):
        token_logprob = logprobs_list[i][0, token_id].item()
        sequence_logprobs.append({'logprob': token_logprob})
        
    return sequence_logprobs

# ===================================================================
# 내부 함수 1: UQLM 추론 로직
# ===================================================================
def _run_uqlm_inference(pipe, tokenizer, data, params: dict):
    """uqlm.WhiteBoxUQ를 사용하여 신뢰도 점수를 계산합니다."""
    print("Executing inference using 'uqlm' method...")
    
    # --- 1. 설정 및 UQLM 객체 초기화 ---
    system_prompt = params.system_prompt
    user_prompt_template = params.user_prompt_template
    scorers = params.scorers
    wbuq = WhiteBoxUQ(llm=None, scorers=scorers)
    
    # --- 2. 결과를 담을 빈 리스트 초기화 ---
    prompts_for_scoring = []
    generated_texts = []
    all_logprobs_for_uqlm = []
    ground_truths_list = []

    # --- 3. 데이터셋을 한 줄씩 순회하며 추론 실행 ---
    for question, answer in tqdm(zip(data['question'], data['answer']), total=len(data['question']), desc="Generating responses (uqlm)"):
        
        user_prompt = user_prompt_template.format(question=question)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(pipe.device)
        
        with torch.no_grad():
            outputs = pipe.model.generate(
                inputs,
                max_new_tokens=params.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                temperature=params.temperature,
                top_p=params.top_p,
            )
        
        # --- 3-3. 생성된 텍스트와 로그 확률(logprobs) 추출 ---
        # ⬇️ 헬퍼 함수 호출 시 tokenizer를 전달합니다.
        sequence_logprobs = _extract_logprobs(outputs, inputs.shape[1], tokenizer)
        
        generated_sequence = outputs.sequences[0][inputs.shape[1]:]
        generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        
        # --- 3-4. 결과 리스트에 추가 ---
        prompts_for_scoring.append(user_prompt)
        generated_texts.append(generated_text)
        all_logprobs_for_uqlm.append(sequence_logprobs)
        ground_truths_list.append(answer)

    # --- 4. 모든 추론이 끝난 후, UQLM으로 일괄 점수 계산 ---
    results = wbuq.score(
        prompts=prompts_for_scoring, 
        responses=generated_texts, 
        logprobs_results=all_logprobs_for_uqlm
    )
    
    results_df = results.to_df()
    results_df.rename(columns={'prompts': 'question_prompt', 'response': 'prediction'}, inplace=True)
    results_df['ground_truth'] = ground_truths_list

    # --- 5. 후처리기를 적용하여 정답 여부 계산 ---
    is_correct = [math_postprocessor(p) == math_postprocessor(a) for p, a in zip(results_df["prediction"], results_df["ground_truth"])]
    results_df['response_correct'] = is_correct
    
    return results_df.to_dict(orient='records')
# ===================================================================
# 내부 함수 2: Ours 추론 로직 (Placeholder)
# ===================================================================
def _run_ours_inference(pipe, tokenizer, data, params: dict):
    """자체 개발한 아키텍처(ours)를 사용하여 신뢰도 점수를 계산합니다."""
    print("Executing inference using 'ours' method...")

    prompts = [FEW_SHOT_PROMPT_TEMPLATE.format(question=q) for q in data['question']]
    ground_truths = data['answer']
    
    # --- TODO: 이 부분에 자체 아키텍처 추론 및 점수 계산 로직을 구현하세요 ---
    
    # --- 아래는 최종 반환 형식에 맞춘 임시 Placeholder 코드입니다 ---
    num_rows = len(prompts)
    results_df = pd.DataFrame({
        'question': [p.split('\nQ: ')[-1].split('\nA:')[0] for p in prompts],
        'prediction': [f"OURS_DUMMY_ANSWER_{i}" for i in range(num_rows)],
        'ground_truth': ground_truths,
        'normalized_probability': np.random.rand(num_rows), 
        'min_probability': np.random.rand(num_rows),
    })
    
    return results_df.to_dict(orient='records')

# ===================================================================
# 메인 컨트롤러 함수
# ===================================================================
def model_inference(method: str, params: dict, pipe, tokenizer, data):
    """
    설정된 method에 따라 적절한 추론 함수를 호출하는 컨트롤 타워입니다.
    """
    if method == 'uqlm':
        return _run_uqlm_inference(pipe, tokenizer, data, params)
    
    elif method == 'ours':
        return _run_ours_inference(pipe, tokenizer, data, params)
        
    else:
        raise ValueError(f"Unsupported inference method: '{method}'. Choose from 'uqlm' or 'ours'.")
