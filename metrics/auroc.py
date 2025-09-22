import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from typing import List

def compute_auroc(scores: np.ndarray, correct_indicators: np.ndarray) -> float:
    """
    주어진 신뢰도 점수와 정답 여부를 바탕으로 AUROC를 계산합니다.
    (논문 기준에 따라 '정답'을 Positive Class로 고정)

    Args:
        scores (np.ndarray): 0~1 사이의 신뢰도 점수 (모델이 정답이라고 확신하는 확률).
        correct_indicators (np.ndarray): 실제 정답 여부 (1=정답, 0=오답).

    Returns:
        float: 계산된 AUROC 점수.
    """
    # y_true에 양성/음성 샘플이 모두 있어야 AUROC 계산 가능
    if len(np.unique(correct_indicators)) < 2:
        print("Warning: Cannot compute AUROC as only one class is present. Returning NaN.")
        return np.nan

    # 논문 기준: y_true = 정답(1/0), y_score = scores (그대로)
    return roc_auc_score(y_true=correct_indicators, y_score=scores)

def calculate_auroc_scores(results_df: pd.DataFrame, score_columns: list) -> dict:
    """
    결과 데이터프레임에서 여러 신뢰도 점수에 대한 AUROC를 일괄 계산합니다.

    Args:
        results_df (pd.DataFrame): 'response_correct'와 신뢰도 점수 컬럼들을 포함한 데이터프레임.
        score_columns (list): AUROC를 계산할 신뢰도 점수 컬럼 이름들의 리스트.

    Returns:
        dict: 각 신뢰도 점수 컬럼을 키로, 계산된 AUROC 점수를 값으로 갖는 딕셔너리.
              (예: {'min_probability': 0.67, 'normalized_probability': 0.62})
    """
    if "response_correct" not in results_df.columns:
        raise KeyError("DataFrame must contain a 'response_correct' column.")
        
    y_correct = results_df["response_correct"].astype(int).to_numpy()
    auroc_results = {}

    for col in score_columns:
        if col not in results_df.columns:
            print(f"Warning: Score column '{col}' not found in DataFrame. Skipping.")
            continue
        
        scores = results_df[col].to_numpy()
        
        # 'hallucination' 타겟을 제거하고 auroc_correct만 계산
        auroc_results[col] = compute_auroc(scores, y_correct)
    
    return auroc_results


def plot_roc_curve(results_df: pd.DataFrame, score_columns: List[str], save_path: str):
    """
    ROC 곡선을 그려 지정된 경로에 이미지 파일로 저장합니다.

    Args:
        results_df (pd.DataFrame): 'response_correct'와 신뢰도 점수 컬럼들을 포함한 데이터프레임.
        score_columns (list): ROC 곡선을 그릴 신뢰도 점수 컬럼 이름들의 리스트.
        save_path (str): 그래프를 저장할 파일 경로 (예: 'results/roc_curve.png').
    """
    if "response_correct" not in results_df.columns:
        raise KeyError("DataFrame must contain a 'response_correct' column.")

    plt.figure(figsize=(8, 6))
    
    y_true = results_df["response_correct"].astype(int)

    for col in score_columns:
        if col not in results_df.columns:
            continue
        
        y_score = results_df[col]
        
        # NaN 값 등 유효하지 않은 데이터 제외
        valid_indices = y_score.notna()
        y_true_valid = y_true[valid_indices]
        y_score_valid = y_score[valid_indices]

        if len(np.unique(y_true_valid)) < 2:
            print(f"Warning: Skipping ROC curve for '{col}' as only one class is present.")
            continue

        fpr, tpr, _ = roc_curve(y_true_valid, y_score_valid)
        auroc_value = roc_auc_score(y_true_valid, y_score_valid)
        plt.plot(fpr, tpr, label=f'{col} (AUROC = {auroc_value:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUROC = 0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    try:
        plt.savefig(save_path)
        print(f"ROC curve saved to: {save_path}")
    except Exception as e:
        print(f"Error saving ROC curve plot: {e}")
    plt.close()

