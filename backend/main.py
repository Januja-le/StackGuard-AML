from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io
import json

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.getenv("STACKGUARD_RESULTS_DIR", ROOT / "results"))
DATA = Path(os.getenv("STACKGUARD_DATA_DIR", ROOT / "data"))
app = FastAPI(title="StackGuard AML API", version="1.0.0", description="API for the StackGuard-AML Elliptic experiment artifacts.")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","), allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def load_json(name: str) -> dict[str, Any]:
    p = RESULTS / name
    if not p.exists():
        raise HTTPException(503, f"Missing {p.name}. Run the experiment first (see README).")
    return json.loads(p.read_text(encoding="utf-8"))

def predictions() -> pd.DataFrame:
    p = RESULTS / "test_predictions.csv"
    if not p.exists():
        raise HTTPException(503, "Missing test_predictions.csv. Run the experiment first.")
    return pd.read_csv(p)

def safe_num(v):
    try:
        n = float(v)
        return n if pd.notna(n) else None
    except (TypeError, ValueError):
        return None

@app.get("/api/health")
def health():
    required = ["results.json", "test_predictions.csv", "table_main.csv"]
    present = {f: (RESULTS / f).exists() for f in required}
    return {"status": "ok" if all(present.values()) else "setup_required", "artifacts": present, "results_dir": str(RESULTS)}

@app.get("/api/dashboard/summary")
def summary():
    res = load_json("results.json")
    df = predictions()
    threshold = float(res.get("threshold", 0.5))
    score_col = "Stacked_ensemble"
    if score_col not in df.columns:
        raise HTTPException(500, "Stacked_ensemble column is missing from test_predictions.csv")
    scores = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
    flagged = scores >= threshold
    model = res.get("models", {}).get("Stacked ensemble", {})
    return {
        "dataset": res.get("info", {}),
        "threshold": threshold,
        "total_test_transactions": int(len(df)),
        "flagged_transactions": int(flagged.sum()),
        "flag_rate": round(float(flagged.mean() * 100), 2) if len(df) else 0,
        "illicit_ground_truth": int((df["label"] == 1).sum()) if "label" in df else None,
        "model_metrics": model,
        "risk_distribution": [
            {"risk": "Critical", "count": int((scores >= max(threshold, 0.95)).sum())},
            {"risk": "High", "count": int(((scores >= threshold) & (scores < max(threshold, 0.95))).sum())},
            {"risk": "Low", "count": int((scores < threshold).sum())}
        ],
        "per_step": _per_step(df, threshold)
    }

def _per_step(df: pd.DataFrame, threshold: float):
    if "time_step" not in df or "Stacked_ensemble" not in df:
        return []
    out = []
    for step, group in df.groupby("time_step"):
        score = pd.to_numeric(group["Stacked_ensemble"], errors="coerce").fillna(0)
        out.append({"time_step": int(step), "transactions": int(len(group)), "flagged": int((score >= threshold).sum()), "illicit_actual": int((group["label"] == 1).sum())})
    return sorted(out, key=lambda x: x["time_step"])

@app.get("/api/models")
def models():
    res = load_json("results.json")
    return {"models": res.get("models", {}), "meta_coefficients": res.get("meta_coefficients", {}), "threshold": res.get("threshold"), "hardware": res.get("hardware", {})}

@app.get("/api/transactions")
def transactions(
    search: str = "",
    risk: str = "all",
    label: str = "all",
    step: int | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
):
    res = load_json("results.json")
    df = predictions()
    threshold = float(res.get("threshold", 0.5))
    if "Stacked_ensemble" not in df:
        raise HTTPException(500, "Stacked ensemble scores are unavailable.")
    df["risk_score"] = pd.to_numeric(df["Stacked_ensemble"], errors="coerce").fillna(0)
    df["risk_level"] = df["risk_score"].apply(lambda s: "high" if s >= threshold else "low")
    if search:
        df = df[df["txId"].astype(str).str.contains(search, case=False, na=False)]
    if step is not None:
        df = df[df["time_step"] == step]
    if label in ("licit", "illicit"):
        df = df[df["label"] == (1 if label == "illicit" else 0)]
    if risk in ("high", "low"):
        df = df[df["risk_level"] == risk]
    df = df.sort_values("risk_score", ascending=False)
    total = len(df)
    start = (page - 1) * page_size
    cols = [c for c in ["txId", "time_step", "label", "risk_score", "risk_level", "RF", "XGB", "LSTM", "AE", "GNN"] if c in df]
    rows = df.iloc[start:start + page_size][cols].replace({float("nan"): None}).to_dict(orient="records")
    for row in rows:
        row["txId"] = str(row["txId"])
        row["risk_score"] = safe_num(row.get("risk_score"))
        row["classification"] = "Illicit (ground truth)" if row.get("label") == 1 else "Licit (ground truth)" if row.get("label") == 0 else "Unknown"
    return {"items": rows, "total": int(total), "page": page, "page_size": page_size, "threshold": threshold}

@app.get("/api/transactions/{txid}")
def transaction_detail(txid: str):
    res = load_json("results.json")
    df = predictions()
    row = df[df["txId"].astype(str) == txid]
    if row.empty:
        raise HTTPException(404, "Transaction ID not found in the saved test predictions.")
    r = row.iloc[0].to_dict()
    threshold = float(res.get("threshold", 0.5))
    score = safe_num(r.get("Stacked_ensemble")) or 0
    base = {k: safe_num(r.get(k)) for k in ["RF", "XGB", "LSTM", "AE", "GNN"]}
    return {
        "txId": str(r["txId"]), "time_step": int(r["time_step"]),
        "ground_truth": int(r["label"]), "risk_score": score,
        "risk_level": "high" if score >= threshold else "low", "threshold": threshold,
        "model_scores": base,
        "model_metrics": res.get("models", {}).get("Stacked ensemble", {}),
        "note": "Risk score is the saved out-of-time test prediction. Ground truth is shown for research evaluation and would not be available to an operational analyst."
    }

@app.get("/api/explainability/global")
def explainability():
    p = RESULTS / "shap_xgboost_feature_importance.csv"
    if not p.exists():
        p = RESULTS / "shap_feature_importance.csv"
    if not p.exists():
        raise HTTPException(503, "SHAP feature-importance CSV not found. Run generate_shap.py and place its output in results/.")
    df = pd.read_csv(p).head(20)
    return {"scope": "global", "model": "XGBoost", "method": "mean absolute SHAP", "items": df.to_dict(orient="records"), "note": "Global feature importance, not transaction-specific SHAP explanations."}

@app.get("/api/transactions/{txid}/network")
def transaction_network(txid: str, depth: int = Query(1, ge=1, le=2), limit: int = Query(40, ge=5, le=100)):
    df = predictions()
    ids = set(df["txId"].astype(str))
    if txid not in ids:
        raise HTTPException(404, "Transaction ID not found in saved test predictions.")
    edge_file = DATA / "elliptic_txs_edgelist.csv"
    feature_file = DATA / "elliptic_txs_features.csv"
    if not edge_file.exists() or not feature_file.exists():
        raise HTTPException(503, "Graph data unavailable. Add the Elliptic edge-list and feature CSVs to data/.")
    features = pd.read_csv(feature_file, header=None, usecols=[0, 1], names=["txId", "time_step"])
    known = set(features["txId"].astype(str))
    edges = pd.read_csv(edge_file)
    edges["txId1"] = edges["txId1"].astype(str); edges["txId2"] = edges["txId2"].astype(str)
    frontier, seen = {txid}, {txid}
    links = []
    for _ in range(depth):
        subset = edges[edges["txId1"].isin(frontier) | edges["txId2"].isin(frontier)]
        subset = subset.head(limit * 4)
        next_frontier = set()
        for a, b in subset[["txId1", "txId2"]].itertuples(index=False, name=None):
            if a in seen and b in seen: continue
            links.append({"source": a, "target": b})
            if a not in seen: next_frontier.add(a)
            if b not in seen: next_frontier.add(b)
            seen.update([a, b])
            if len(seen) >= limit: break
        frontier = next_frontier
        if not frontier or len(seen) >= limit: break
    step_map = dict(zip(features["txId"].astype(str), features["time_step"].astype(int)))
    nodes = [{"id": x, "is_focus": x == txid, "time_step": step_map.get(x), "has_test_prediction": x in ids} for x in seen]
    return {"nodes": nodes, "links": links[:limit * 2], "focus": txid, "note": "Edges are Elliptic transaction-flow relationships. Neighbour risk scores are not inferred unless saved test predictions exist."}

@app.get("/api/reports/transactions.csv")
def export_transactions():
    df = predictions()
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=stackguard_test_predictions.csv"})
