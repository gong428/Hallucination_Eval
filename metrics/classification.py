import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from uqlm.utils import Tuner
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path

def calculate_classification_metrics(results_df: pd.DataFrame, score_columns: list) -> dict:
    """
    최적 임계값을 찾아 Precision, Recall, F1-score를 계산합니다.

    Args:
        results_df (pd.DataFrame): 'response_correct'와 신뢰도 점수 컬럼들을 포함한 데이터프레임.
        score_columns (list): 평가할 신뢰도 점수 컬럼 이름들의 리스트.

    Returns:
        dict: 각 점수별로 계산된 메트릭과 최적 임계값.
    """
    t = Tuner()
    correct_indicators = results_df["response_correct"].astype(int)
    
    # 데이터를 튜닝셋과 평가셋으로 분리
    split_idx = len(results_df) // 2
    if split_idx == 0:
        raise ValueError("Cannot split data for tuning and evaluation. Need at least 2 data points.")

    metric_results = {}

    for scorer_name in score_columns:
        if scorer_name not in results_df.columns:
            print(f"Warning: Score column '{scorer_name}' not found in DataFrame. Skipping.")
            continue
            
        y_scores = results_df[scorer_name]
        
        # 튜닝셋으로 최적 임계값 찾기
        y_scores_tune = y_scores.iloc[0:split_idx]
        y_true_tune = correct_indicators.iloc[0:split_idx]
        best_threshold = t.tune_threshold(y_scores=y_scores_tune, correct_indicators=y_true_tune)

        # 전체 데이터에 임계값 적용하여 예측 생성
        y_pred = (y_scores > best_threshold).astype(int)

        # 평가셋으로 성능 측정
        y_true_eval = correct_indicators.iloc[split_idx:]
        y_pred_eval = y_pred.iloc[split_idx:]
        
        metric_results[scorer_name] = {
            "precision": precision_score(y_true=y_true_eval, y_pred=y_pred_eval, zero_division=0),
            "recall": recall_score(y_true=y_true_eval, y_pred=y_pred_eval, zero_division=0),
            "f1_score": f1_score(y_true=y_true_eval, y_pred=y_pred_eval, zero_division=0),
            "optimal_threshold": best_threshold
        }
        
    return metric_results

def plot_metrics_by_threshold(results_df: pd.DataFrame, score_columns: List[str], save_path_template: str):
    """
    임계값 변화에 따른 Precision, Recall, F1-score를 스코어러별로
    별도의 그래프로 생성하여 저장합니다.

    Args:
        results_df (pd.DataFrame): 'response_correct'와 신뢰도 점수 컬럼들을 포함한 데이터프레임.
        score_columns (list): 시각화할 신뢰도 점수 컬럼 이름들의 리스트.
        save_path_template (str): 그래프를 저장할 파일 경로의 템플릿.
    """
    if "response_correct" not in results_df.columns:
        raise KeyError("DataFrame must contain a 'response_correct' column.")

    thresholds = np.arange(0.0, 1.01, 0.05)
    y_true = results_df["response_correct"].astype(int)

    for col in score_columns:
        if col not in results_df.columns:
            continue

        y_score = results_df[col]
        
        precisions, recalls, f1s = [], [], []
        
        for thresh in thresholds:
            y_pred = (y_score > thresh).astype(int)
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1s.append(f1_score(y_true, y_pred, zero_division=0))
        
        # --- 각 스코어러별로 새로운 그래프 생성 ---
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        ax.plot(thresholds, precisions, marker='o', linestyle='-', label='Precision')
        ax.plot(thresholds, recalls, marker='x', linestyle='--', label='Recall')
        ax.plot(thresholds, f1s, marker='s', linestyle=':', label='F1-Score')
        
        ax.set_title(f'Metrics vs. Threshold for "{col}"')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend()
        
        # --- 파일 경로 수정 및 저장 ---
        p = Path(save_path_template)
        # 파일명에 스코어러 이름을 추가합니다 (예: ..._metrics_by_threshold_min_probability.png)
        new_save_path = p.parent / f"{p.stem}_{col}{p.suffix}"
        
        plt.tight_layout()
        try:
            plt.savefig(new_save_path)
            print(f"Metrics by threshold plot for '{col}' saved to: {new_save_path}")
        except Exception as e:
            print(f"Error saving metrics plot for '{col}': {e}")
        plt.close() # 다음 그래프를 위해 현재 그래프 창을 닫습니다.