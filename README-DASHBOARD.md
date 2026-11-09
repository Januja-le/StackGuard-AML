# StackGuard AML dashboard

Dark analyst console for the Elliptic Bitcoin transaction experiment. This dashboard reads the project's saved **out-of-time test predictions** and evaluation artifacts; it does not fabricate live bank transactions or pretend the trained models are serialized for arbitrary online inference.

## Repository facts and artifact contract

The experiment runner writes these files into `results/`: `results.json`, `test_predictions.csv`, `table_main.csv`, and `confusion_matrix.csv`. Optional global explainability is served from `shap_xgboost_feature_importance.csv` (or `shap_feature_importance.csv`). The feature CSV is ignored by Git; place all three Elliptic CSVs under `data/` locally.

## Windows + VS Code

1. From the repository root, create/activate a virtual environment and install the existing ML requirements used by `run_experiments_cpu.py`, then install the API requirements:
   `py -m venv .venv`
   `.venv\\Scripts\\activate`
   `pip install -r backend/requirements.txt`
2. Ensure `data/elliptic_txs_features.csv`, `data/elliptic_txs_classes.csv`, and `data/elliptic_txs_edgelist.csv` exist.
3. Run the existing experiment if the saved artifacts are not present:
   `python run_experiments_cpu.py --data data --out results --skip-ablation --skip-shap`
   This is computationally expensive on CPU. Reuse existing results if you already have them.
4. Start API: `uvicorn backend.main:app --reload --port 8000`
5. In another terminal: `cd frontend`, `npm install`, `npm run dev`.
6. Open `http://localhost:5173`; API docs are at `http://localhost:8000/docs`.

## Pages

Overview, transaction monitoring, transaction investigation, model intelligence, global SHAP explainability, and reports/export.

## Important research/operational distinction

The displayed risk score is the saved ensemble score for the held-out test set (steps 35–49). The ground-truth label is included for offline research evaluation and must not be treated as an analyst-visible feature in a real deployment. The current experiment script does not persist fitted RF/XGBoost/LSTM/AE/GNN/meta-learner/scaler artifacts for serving arbitrary new transactions. A later model-packaging step is required before enabling true online inference. Global SHAP feature importance is not a per-transaction explanation.
