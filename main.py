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
    settings = load_config(config_path)
    print(f"===== Starting Pipeline with config: {config_path} =====")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{settings.model.name}_{timestamp}"
    experiment_dir = Path(save_dir_base) / experiment_name

    output_dir = experiment_dir / "model_outputs"
    metrics_dir = experiment_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 2. 모델 로드
    pipe, tokenizer = load_model(settings.model.name, settings.model.path)

    # 3. 데이터셋별 추론 및 결과 저장
    for dataset_name, dataset_config in settings.datasets.items():
        try:
            print(f"\n--- Processing Dataset: {dataset_name} (type: {dataset_config.type}) ---")
            raw_data = load_raw_dataset(dataset_name, settings.sample_num)

            # 추론 실행
            inf_result = model_inference(
                method=settings.model.inference_method,
                params=settings.model.inference_params,
                pipe=pipe,
                tokenizer=tokenizer,
                data=raw_data,
                dataset_type=dataset_config.type
            )

            # 하위호환: 예전 형식(List[Dict])도 허용
            if isinstance(inf_result, dict) and "records" in inf_result:
                records = inf_result["records"]
                token_details = inf_result.get("token_details", [])
            else:
                records = inf_result
                token_details = []

            # 결과 저장 (기존)
            output_path = output_dir / f"{dataset_name}_output.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=4, ensure_ascii=False)
            print(f"Results for '{dataset_name}' saved to: {output_path}")

            # === ★ 토큰 상세 저장: JSONL + FLAT CSV ===
            if token_details:
                # 1) JSONL: 샘플별 1라인
                token_jsonl_path = output_dir / f"{dataset_name}_token_details.jsonl"
                with open(token_jsonl_path, 'w', encoding='utf-8') as jf:
                    for item in token_details:
                        jf.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(f"Token details (JSONL) saved to: {token_jsonl_path}")

                # 2) FLAT CSV: (dataset, sample_idx, token_idx, token_id, token_text, logprob, prob)
                flat_rows = []
                for si, item in enumerate(token_details):
                    stats = item.get("token_stats", {}) or {}
                    tokens = stats.get("tokens", []) or []
                    for tok in tokens:
                        flat_rows.append({
                            "dataset": dataset_name,
                            "sample_idx": si,
                            "token_idx": tok.get("idx"),
                            "token_id": tok.get("id"),
                            "token_text": tok.get("text"),
                            "logprob": tok.get("logprob"),
                            "prob": tok.get("prob"),
                        })
                if flat_rows:
                    flat_df = pd.DataFrame(flat_rows)
                    flat_csv_path = output_dir / f"{dataset_name}_token_details_flat.csv"
                    flat_df.to_csv(flat_csv_path, index=False)
                    print(f"Token details (flat CSV) saved to: {flat_csv_path}")

        except Exception as e:
            print(f"!!! Error processing {dataset_name}: {e}")
            continue

    # 4. 모델 평가 (변경 없음)
    print("\n--- Evaluating Model Performance ---")
    final_summary = {}
    SCORE_COLUMNS = settings.model.inference_params.scorers

    for dataset_name, dataset_config in settings.datasets.items():
        results_path = output_dir / f"{dataset_name}_output.json"
        if not results_path.exists():
            print(f"Warning: Results file not found for '{dataset_name}'. Skipping evaluation.")
            continue

        results_df = pd.read_json(results_path)
        if 'response_correct' not in results_df.columns:
            print(f"Error: 'response_correct' column not found in results for {dataset_name}. Skipping evaluation.")
            continue

        print(f"\n--- Metrics for {dataset_name} ---")
        auroc_results = calculate_auroc_scores(results_df, SCORE_COLUMNS)
        print("AUROC Scores:", auroc_results)

        class_results = calculate_classification_metrics(results_df, SCORE_COLUMNS)
        print("Classification Metrics:", class_results)

        final_summary[dataset_name] = {
            "auroc_scores": auroc_results,
            "classification_metrics": class_results
        }

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