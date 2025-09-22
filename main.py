import argparse
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

from utils.config_loader import load_config 
from data.loader import load_raw_dataset
from model.model_loader import load_model
from model.model_runner import model_inference # 🆕 추론 엔진 함수를 직접 임포트합니다.
from metrics.auroc import calculate_auroc_scores
from metrics.classification import calculate_classification_metrics



def main(config_path: str, save_dir_base: str | None):
    """
    주어진 설정 파일을 사용하여 전체 파이프라인을 실행합니다.
    """
    settings = load_config(config_path)
    print(f"===== Starting Pipeline with config: {config_path} =====")

    # 1. 저장 경로 설정 (옵션 우선)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{settings.model.name}_{timestamp}"
    experiment_dir = Path(save_dir_base) / experiment_name

    output_dir = experiment_dir / "model_outputs"
    metrics_dir = experiment_dir / "metrics"

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 2. 모델 로드 (최초 1회)
    pipe, tokenizer = load_model(settings.model.name, settings.model.path)

    # 3. 데이터셋별 추론 및 결과 저장
    for dataset_name, dataset_config in settings.datasets.items():
        try:
            print(f"\n--- Processing Dataset: {dataset_name} (type: {dataset_config.type}) ---")
            raw_data = load_raw_dataset(dataset_name, settings.sample_num)
            
            # 추론 실행
            results = model_inference(
                method=settings.model.inference_method,
                params=settings.model.inference_params,
                pipe=pipe,
                tokenizer=tokenizer,
                data=raw_data,
                dataset_type=dataset_config.type
            )
            #결과 저장
            output_path = output_dir / f"{dataset_name}_output.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            print(f"Results for '{dataset_name}' saved to: {output_path}")

        except Exception as e:
            print(f"!!! Error processing {dataset_name}: {e}")
            continue
    
    # 4. 모델 평가
    print("\n--- Evaluating Model Performance ---")
    final_summary = {}
    # 객체의 속성에 직접 접근
    SCORE_COLUMNS = settings.model.inference_params.scorers
    
    for dataset_name in settings.datasets_to_evaluate:
        results_path = output_dir / f"{dataset_name}_output.json"
        
        if not results_path.exists():
            print(f"Warning: Results file not found for '{dataset_name}'. Skipping evaluation.")
            continue
        
        results_df = pd.read_json(results_path)
        # 'prediction'과 'ground_truth'가 모두 문자열일 경우를 대비하여 타입 변환
        results_df['response_correct'] = results_df['prediction'].astype(str) == results_df['ground_truth'].astype(str)
        
        print(f"\n--- Metrics for {dataset_name} ---")
        auroc_results = calculate_auroc_scores(results_df, SCORE_COLUMNS)
        print("AUROC Scores:", auroc_results)

        class_results = calculate_classification_metrics(results_df, SCORE_COLUMNS)
        print("Classification Metrics:", class_results)


        final_summary[dataset_name] = {
            "auroc_scores": auroc_results, # '.to_dict(...)' 제거
            "classification_metrics": class_results
        }

    # 5. 최종 평가 결과 저장
    metrics_path = metrics_dir / "summary.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=4)
        
    print(f"\nFinal metrics summary saved to: {metrics_path}")
    print("\n===== Pipeline Complete =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Hallucination Detection Pipeline.")
    parser.add_argument('-c', '--config', type=str, default='configs/config.yaml', help='Path to the configuration YAML file.')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save all results, overriding config file settings.')
    args = parser.parse_args()
    main(config_path=args.config, save_dir_base=args.save_dir)

#python main.py -c configs/config.yaml --save_dir ./results