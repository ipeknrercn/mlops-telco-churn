"""
Telco Churn - Prefect Pipeline
===============================
Tek komutla train → evaluate → promote akışı

Kullanım:
    python pipelines/prefect_pipeline.py
"""

from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
import sys
from pathlib import Path
import subprocess
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

# Project root'u path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "training_config.yaml"


from scripts.train_improved import ChurnTrainer
from scripts.promote_model import ModelPromoter


@task(name=" Data Preparation", retries=2)
def prepare_data_task():
    """
    Veri hazırlama task'ı
    Şu an dummy data, ileride Kişi 1'den gerçek data gelecek
    """
    print(" Preparing data...")
    
    # TODO: Kişi 1'den gelecek
    # - Dataset yükleme
    # - High-cardinality kolonlar (customer_id, service_combo_id, geo_code)
    
    return {
        "status": "success",
        "message": "Dummy data ready (will be replaced with real Kaggle data)"
    }


@task(name=" Model Training", retries=1)
def train_model_task(data_info: dict, config_path: str = None):
    """
    Model eğitimi task'ı
    ChurnTrainer class'ını kullanır
    """
    print(" Training model...")
    
    try:
        # Config yoksa None geç (default config kullanılacak)
        if config_path:
            cfg_path = Path(config_path)
            if not cfg_path.is_absolute():
                cfg_path = (ROOT_DIR / cfg_path).resolve()
            if not cfg_path.exists():
                print(f" Warning: Config file not found: {cfg_path} (from '{config_path}'), using defaults")
                cfg_path = None
            else:
                cfg_path = DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None

            print(f" Using config: {cfg_path}")
            print(f" Working dir: {Path.cwd()}")

        trainer = ChurnTrainer(config_path=str(cfg_path) if cfg_path else None)
        
        # Düşük threshold ile override et (demo için)
        trainer.config["thresholds"] = {
            "production_min_roc_auc": 0.65,
            "production_min_pr_auc": 0.60,
            "production_min_recall": 0.45  # Düşük threshold - demo için
        }
        
        # Class weight balanced yap (recall için)
        trainer.config["model"]["class_weight"] = "balanced"
        
        run_id, metrics, production_check = trainer.run(register_model=True)
        
        return {
            "status": "success",
            "run_id": run_id,
            "metrics": metrics,
            "production_ready": production_check["ready_for_production"],
            "checks": production_check
        }
    except Exception as e:
        print(f" Warning: Training failed: {e}")
        import traceback
        traceback.print_exc()  # Detaylı hata göster
        return {
            "status": "failed",
            "error": str(e)
        }


@task(name=" Model Evaluation")
def evaluate_model_task(train_result: dict):
    """
    Model değerlendirme - metrics kontrolü
    """
    print(" Evaluating model...")
    
    if train_result["status"] != "success":
        return {
            "status": "skipped",
            "reason": "Training failed"
        }
    
    metrics = train_result["metrics"]
    
    # Kritik metrikleri kontrol et
    evaluation = {
        "status": "success",
        "run_id": train_result["run_id"],
        "test_roc_auc": metrics.get("test_roc_auc", 0),
        "test_pr_auc": metrics.get("test_pr_auc", 0),
        "test_recall": metrics.get("test_recall", 0),
        "production_ready": train_result["production_ready"],
        "recommendation": "PROMOTE" if train_result["production_ready"] else "DO_NOT_PROMOTE"
    }
    
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY:")
    print(f"  ROC-AUC: {evaluation['test_roc_auc']:.4f}")
    print(f"  PR-AUC:  {evaluation['test_pr_auc']:.4f}")
    print(f"  Recall:  {evaluation['test_recall']:.4f}")
    print(f"  Recommendation: {evaluation['recommendation']}")
    print(f"{'='*60}\n")
    
    return evaluation


@task(name=" Model Promotion")
def promote_model_task(evaluation: dict, auto_promote: bool = False):
    """
    Model promotion task'ı
    Staging → Production geçişi
    """
    print(" Checking promotion eligibility...")
    
    if evaluation["status"] != "success":
        return {
            "status": "skipped",
            "reason": "Evaluation failed"
        }
    
    if not evaluation["production_ready"] and not auto_promote:
        print(" Warning: Model is NOT ready for production")
        print("   Use --auto-promote to override")
        return {
            "status": "not_promoted",
            "reason": "Did not meet production thresholds"
        }
    
    # Promote
    try:
        promoter = ModelPromoter()
        staging_models = promoter.get_staging_models()
        
        if not staging_models:
            return {
                "status": "failed",
                "reason": "No models in staging"
            }
        
        latest_version = staging_models[0].version
        success = promoter.promote_to_production(latest_version)
        
        return {
            "status": "promoted" if success else "failed",
            "version": latest_version,
            "promoted_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f" Warning: Promotion failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }


@flow(
    name="Telco Churn Training Pipeline",
    description="End-to-end ML pipeline: Data → Train → Evaluate → Promote",
    task_runner=ConcurrentTaskRunner()
)
def churn_training_pipeline(
    config_path: str = None,
    auto_promote: bool = False
):
    """
    Ana pipeline flow
    
    Args:
        config_path: Training config YAML yolu
        auto_promote: Threshold'ları bypass et
    """
    
    print("\n" + "="*80)
    print(" STARTING TELCO CHURN TRAINING PIPELINE")
    print("="*80 + "\n")
    
    # 1. Data Preparation
    data_info = prepare_data_task()
    
    # 2. Model Training
    train_result = train_model_task(data_info, config_path)
    
    # 3. Model Evaluation
    evaluation = evaluate_model_task(train_result)
    
    # 4. Model Promotion (opsiyonel)
    if auto_promote or evaluation.get("production_ready"):
        promotion_result = promote_model_task(evaluation, auto_promote)
    else:
        print("\n Skipping promotion (model not ready)")
        promotion_result = {"status": "skipped"}
    
    # Final Summary
    print("\n" + "="*80)
    print(" PIPELINE COMPLETED")
    print("="*80)
    print(f"Training: {train_result.get('status', 'unknown')}")
    print(f"Evaluation: {evaluation.get('status', 'unknown')}")
    print(f"Promotion: {promotion_result.get('status', 'unknown')}")
    print("="*80 + "\n")
    
    return {
        "data": data_info,
        "training": train_result,
        "evaluation": evaluation,
        "promotion": promotion_result
    }


# Deployment version - scheduled runs için
@flow(name="Scheduled Churn Pipeline")
def scheduled_churn_pipeline():
    """
    Cron/schedule için otomatik çalışacak versiyon
    """
    return churn_training_pipeline(
        config_path="config/training_config.yaml",
        auto_promote=False  # Manuel approval gerekli
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Churn Training Pipeline")
    parser.add_argument("--config", type=str, default="config/training_config.yaml", help="Config file path")
    parser.add_argument("--auto-promote", action="store_true", help="Auto-promote to production")
    parser.add_argument("--schedule", action="store_true", help="Run as scheduled job")
    
    args = parser.parse_args()
    
    if args.schedule:
        # Prefect Cloud/Server'da schedule edilebilir
        scheduled_churn_pipeline.serve(
            name="churn-training-weekly",
            cron="0 2 * * 1"  # Her Pazartesi 02:00
        )
    else:
        # Tek seferlik çalıştır
        result = churn_training_pipeline(
            config_path=args.config,
            auto_promote=args.auto_promote
        )
        
        print("\n Final Pipeline Result:")
        print(result)