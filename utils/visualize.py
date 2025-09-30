import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List
import itertools
import argparse


def visualize_token_trajectory(file_path: str, sample_index: int):
    """
    특정 샘플의 토큰 생성 확률 궤적을 시각화합니다.

    x축: 생성된 토큰 시퀀스
    y축: 토큰 생성 확률
    - 회색 점: Top-10 후보 토큰들의 확률
    - 파란색 점/선: 모델이 실제로 선택한 토큰의 확률
    """
    # 1. 파일에서 특정 인덱스의 데이터 읽기
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line = next(itertools.islice(f, sample_index, sample_index + 1), None)
        if line is None:
            print(f"Error: Sample index {sample_index} is out of bounds for file {file_path}.")
            return
        sample_data = json.loads(line)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        print(f"Error reading or parsing sample {sample_index} from {file_path}: {e}")
        return

    # 2. 시각화할 데이터 추출
    token_stats = sample_data.get("token_stats", {})
    tokens = token_stats.get("tokens", [])
    prediction = sample_data.get("prediction", "N/A")
    question = sample_data.get("question", "N/A")
    ground_truth = sample_data.get("ground_truth", "N/A")
    

    if not tokens:
        print("No token data found in the selected sample.")
        return

    x_labels = []
    chosen_probs = []
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 9))

    # 3. 그래프 그리기
    for i, token_info in enumerate(tokens):
        x_labels.append(f"'{token_info['text']}'")
        chosen_probs.append(token_info['prob'])
        
        top_10 = token_info.get('top_10_tokens', [])
        if top_10:
            top_10_probs = [d['prob'] for d in top_10]
            # Top-10 후보들을 회색 점으로 표시
            ax.scatter([i] * len(top_10_probs), top_10_probs, c='gray', alpha=0.5, s=25, zorder=2)

    # 실제 선택된 토큰의 확률을 파란색 점과 선으로 강조
    ax.plot(range(len(x_labels)), chosen_probs, 'o-', color='blue', markersize=8, linestyle='--', label='Predicted Token Probability', zorder=3)

    # 4. 그래프 서식 설정
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=11)
    ax.set_xlabel("Predicted Token Sequence", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    print("\n" + "="*50)
    print(f"Sample #{sample_index} Details:")
    print(f"  Question: {question}")
    print(f"  Ground Truth Answer: {ground_truth}")
    print(f"  Prediction: {prediction}")
    print("="*50 + "\n")
    

    plt.show(fig)


def classify_indices_by_correctness(file_path: str):
    """
    주어진 JSON 결과 파일을 읽어 'response_correct' 값에 따라
    인덱스 번호를 'correct'와 'incorrect'로 분류하여 출력합니다.

    Args:
        file_path (str): 분석할 JSON 파일의 경로 (예: '..._output.json').
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다: {file_path}")
        return
    except json.JSONDecodeError:
        print(f"오류: JSON 파일을 파싱할 수 없습니다. 파일 형식을 확인해주세요: {file_path}")
        return
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    correct_indices = []
    incorrect_indices = []

    for i, item in enumerate(data):
        # 'response_correct' 키가 있는지 확인하고, 값을 가져옵니다.
        is_correct = item.get('response_correct')
        
        if is_correct is True:
            correct_indices.append(i)
        elif is_correct is False:
            incorrect_indices.append(i)
        # is_correct가 None이거나 다른 값이면 무시합니다.

    print("=" * 40)
    print(f"분석 파일: {Path(file_path).name}")
    print("=" * 40)
    
    print("\n✅ 정답 (Correct) 인덱스:")
    print(correct_indices)
    print(f"  (총 {len(correct_indices)}개)")

    print("\n❌ 오답 (Incorrect) 인덱스:")
    print(incorrect_indices)
    print(f"  (총 {len(incorrect_indices)}개)")
    print("-" * 40)
