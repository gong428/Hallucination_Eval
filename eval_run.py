import argparse
import json
from pathlib import Path
import pandas as pd

# 메트릭 모듈에서 필요한 함수들을 임포트합니다.
from metrics.auroc import calculate_auroc_scores, plot_roc_curve
from metrics.classification import calculate_classification_metrics, plot_metrics_by_threshold

def evaluate(results_dir: str):
    """
    지정된 결과 디렉토리의 모든 추론 결과를 평가하고,
    데이터셋별로 폴더를 생성하여 결과를 저장합니다.

    Args:
        results_dir (str): 평가할 실험 결과가 담긴 최상위 폴더
                           (예: 'results/codellama-7b-instruct_20250909_150000').
    """
    results_path = Path(results_dir)
    model_outputs_path = results_path / "model_outputs"
    metrics_path = results_path / "metrics"
    
    if not model_outputs_path.is_dir():
        print(f"Error: Directory not found: {model_outputs_path}")
        return

    # 최상위 메트릭 폴더 생성
    metrics_path.mkdir(parents=True, exist_ok=True)
    
    # 평가할 신뢰도 점수 컬럼 목록을 첫 번째 결과 파일로부터 동적으로 찾습니다.
    try:
        first_result_file = next(model_outputs_path.glob('*_output.json'))
        temp_df = pd.read_json(first_result_file)
        SCORE_COLUMNS = [col for col in temp_df.columns if 'probability' in col]
        print(f"Found score columns for evaluation: {SCORE_COLUMNS}")
    except (StopIteration, KeyError):
        print("No result files or score columns found. Exiting evaluation.")
        return
    
    final_summary = {}

    # 모든 데이터셋 결과 파일에 대해 반복
    for result_file in sorted(model_outputs_path.glob('*_output.json')):
        dataset_name = result_file.stem.replace('_output', '')
        print(f"\n--- Evaluating dataset: {dataset_name} ---")
        
        # --- ⬇️ 데이터셋별 결과 폴더 생성 ---
        dataset_metrics_path = metrics_path / dataset_name
        dataset_metrics_path.mkdir(exist_ok=True)
   
        
        results_df = pd.read_json(result_file)
        
        # 1. AUROC 점수 계산
        auroc_results = calculate_auroc_scores(results_df, SCORE_COLUMNS)
        print("AUROC Scores:", auroc_results)
        
        # 2. 분류 지표 계산
        class_results = calculate_classification_metrics(results_df, SCORE_COLUMNS)
        print("Classification Metrics:", class_results)

        dataset_summary = {
            "auroc_scores": auroc_results,
            "classification_metrics": class_results
        }
        final_summary[dataset_name] = dataset_summary

        # --- ⬇️ 데이터셋별 점수를 별도 파일로 저장 ---
        dataset_summary_path = dataset_metrics_path / f"{dataset_name}_scores.json"
        with open(dataset_summary_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_summary, f, indent=4)
        print(f"Dataset-specific scores saved to: {dataset_summary_path}")


        # 3. ROC 곡선 시각화 및 저장 (데이터셋 폴더에 저장)
        roc_save_path = dataset_metrics_path / "roc_curve.png"
        plot_roc_curve(results_df, SCORE_COLUMNS, str(roc_save_path))

        # 4. 임계값별 지표 시각화 및 저장 (데이터셋 폴더에 저장)
        thresh_save_path_template = dataset_metrics_path / "metrics_by_threshold.png"
        plot_metrics_by_threshold(results_df, SCORE_COLUMNS, str(thresh_save_path_template))

    # 최상위 폴더에 모든 데이터셋의 요약 결과 저장
    summary_path = metrics_path / "summary_eval.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=4)
    print(f"\n✅ Final summary of all datasets saved to: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model inference results and generate metrics and plots."
    )
    parser.add_argument(
        '--results_dir', 
        type=str, 
        required=True,
        help="Path to the experiment results directory (e.g., 'results/codellama_...')."
    )
    
    args = parser.parse_args()
    evaluate(results_dir=args.results_dir)

