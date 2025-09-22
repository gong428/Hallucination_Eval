
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from uqlm.scorers import WhiteBoxUQ
from typing import List, Dict


def _extract_logprobs_detailed(outputs, input_length: int, tokenizer) -> tuple[List[Dict[str, float]], Dict]:
    """
    generate()의 출력에서 '생성된 답변' 구간의 토큰별 로그확률을 추출해
    (1) WhiteBoxUQ에 맞는 최소 포맷, (2) 사람이 보기 좋은 상세 포맷을 함께 반환합니다.

    Returns
    -------
    tuple:
        - wbuq_logprobs: List[{'logprob': float}]
            WhiteBoxUQ.score(logprobs_results=...)에 바로 넣을 수 있는 최소 형식
        - detail_summary: Dict
            사람이 보기 좋은 상세 정보:
            {
              'tokens': [
                  {'idx': int, 'id': int, 'text': str, 'logprob': float, 'prob': float}
              ],
              'sum_logprob': float,
              'avg_logprob': float,
              'geom_mean_prob': float,
              'min_logprob': float,
              'min_token': {'idx':..., 'id':..., 'text':..., 'logprob':..., 'prob':...}
            }
    """
    # 1) 생성된 시퀀스만 분리
    generated_sequence = outputs.sequences[0][input_length:]

    # 2) 각 step의 로짓 -> 로그확률
    logprobs_list = [torch.nn.functional.log_softmax(score, dim=-1) for score in outputs.scores]

    # 3) EOS 토큰 제거
    seq_to_process = generated_sequence
    if len(seq_to_process) > 0 and tokenizer.eos_token_id is not None and seq_to_process[-1] == tokenizer.eos_token_id:
        seq_to_process = seq_to_process[:-1]

    tokens_detail = []
    wbuq_logprobs = []
    sum_lp = 0.0
    min_lp = float('inf')
    min_tok = None

    # 4) 토큰별 로그확률 수집
    for i, token_id in enumerate(seq_to_process):
        # i-th 생성 토큰은 i-th score와 매칭
        lp = logprobs_list[i][0, token_id].item()
        prob = float(np.exp(lp))
        text = tokenizer.decode([token_id], skip_special_tokens=True)

        entry = {'idx': i, 'id': int(token_id), 'text': text, 'logprob': lp, 'prob': prob}
        tokens_detail.append(entry)
        wbuq_logprobs.append({'logprob': lp})

        sum_lp += lp
        if lp < min_lp:
            min_lp = lp
            min_tok = entry

    # 5) 요약 통계
    n = max(len(tokens_detail), 1)
    avg_lp = sum_lp / n
    geom_mean_prob = float(np.exp(avg_lp))

    detail_summary = {
        'tokens': tokens_detail,
        'sum_logprob': sum_lp,
        'avg_logprob': avg_lp,
        'geom_mean_prob': geom_mean_prob,
        'min_logprob': min_lp if min_tok is not None else None,
        'min_token': min_tok,
    }
    return wbuq_logprobs, detail_summary

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