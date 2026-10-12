"""
StackGuard-AML CPU experiment runner.

This version keeps the paper's core experimental protocol:
- Temporal split: labelled steps 1-34 train, 35-49 test
- StandardScaler fitted only on training transactions
- RF, XGBoost, flow-sequence LSTM, licit-only Autoencoder, GraphSAGE-style GNN
- 5-fold time-step-grouped OOF stacking
- Logistic-regression meta learner with OOF threshold selection
- Real test metrics and confusion matrix
- XGBoost TreeSHAP + meta LinearSHAP

CPU-oriented improvements:
- Limits BLAS/OpenMP oversubscription
- Saves OOF predictions after every completed fold
- Can resume completed OOF folds
- Saves final test base predictions before meta scoring
- Skips ablation and SHAP by default; run them later with flags
- Writes a paper_results.txt convenience file

IMPORTANT:
Do not report results from changed hyperparameters as the original paper's
results. The default model hyperparameters below match the supplied script.
"""

import argparse, json, os, platform, time
from pathlib import Path

# Avoid CPU thread oversubscription before importing numpy/sklearn/torch.
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "3")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import xgboost as xgb
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

SEED = 42
TRAIN_STEPS = (1, 34)
TEST_STEPS = (35, 49)
N_FOLDS = 5
BASE = ["RF", "XGB", "LSTM", "AE", "GNN"]

HP = dict(
    rf=dict(
        n_estimators=300,
        class_weight="balanced_subsample",
        min_samples_leaf=1,
        n_jobs=-1,
    ),
    xgb=dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=3,
    ),
    lstm=dict(
        hidden=64,
        layers=1,
        dropout=0.2,
        seq_len=5,
        epochs=20,
        lr=1e-3,
        batch=512,
    ),
    ae=dict(
        dims=[165, 64, 32, 64, 165],
        epochs=30,
        lr=1e-3,
        batch=512,
    ),
    gnn=dict(
        layers=2,
        hidden=64,
        dropout=0.3,
        epochs=200,
        lr=1e-2,
        weight_decay=5e-4,
    ),
    meta=dict(model="LogisticRegression", C=1.0, class_weight="balanced"),
)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def load_real(path):
    f = pd.read_csv(os.path.join(path, "elliptic_txs_features.csv"), header=None)
    c = pd.read_csv(os.path.join(path, "elliptic_txs_classes.csv"))
    e = pd.read_csv(os.path.join(path, "elliptic_txs_edgelist.csv"))

    ids = f[0].values
    step = f[1].values.astype(int)
    X = f.iloc[:, 2:].values.astype(np.float32)

    lab = (
        pd.Series(ids)
        .map(dict(zip(c["txId"], c["class"])))
        .fillna("unknown")
        .values
    )

    y = np.full(len(ids), -1)
    y[lab == "1"] = 1
    y[lab == "2"] = 0

    idx = {t: i for i, t in enumerate(ids)}
    src = e["txId1"].map(idx).dropna().astype(int).values
    dst = e["txId2"].map(idx).dropna().astype(int).values

    return X, y, step, src, dst


def feat_names():
    return [f"local_{i}" for i in range(1, 94)] + [
        f"agg_{i}" for i in range(1, 73)
    ]


def mean_adj(n, src, dst):
    A = sp.coo_matrix(
        (np.ones(len(src), np.float32), (dst, src)),
        shape=(n, n),
    ).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    return sp.diags(1.0 / deg) @ A


def flow_sequences(X, src, dst):
    Ain = mean_adj(X.shape[0], src, dst)
    Aout = mean_adj(X.shape[0], dst, src)

    u1 = Ain @ X
    u2 = Ain @ u1
    d1 = Aout @ X
    d2 = Aout @ d1

    return np.stack([u2, u1, X, d1, d2], axis=1).astype(np.float32)


def to_torch_sparse(M):
    M = M.tocoo()
    indices = np.vstack([M.row, M.col])
    return torch.sparse_coo_tensor(
        indices, M.data, M.shape
    ).coalesce().float().to(dev)


class LSTMNet(nn.Module):
    def __init__(self, d, h, layers, p):
        super().__init__()
        self.lstm = nn.LSTM(d, h, layers, batch_first=True)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(h, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.out(self.drop(o[:, -1])).squeeze(-1)


class AE(nn.Module):
    """165 -> 64 -> 32 -> 64 -> 165 autoencoder."""
    def __init__(self, dims):
        super().__init__()
        mid = len(dims) // 2

        enc_layers = []
        for a, b in zip(dims[:mid], dims[1:mid + 1]):
            enc_layers += [nn.Linear(a, b), nn.ReLU()]

        dec_layers = []
        for i in range(mid, len(dims) - 1):
            a, b = dims[i], dims[i + 1]
            dec_layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                dec_layers.append(nn.ReLU())

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.dec(self.enc(x))


class SAGE(nn.Module):
    def __init__(self, d, h, layers, p):
        super().__init__()
        dims = [d] + [h] * layers
        self.lin = nn.ModuleList(
            [nn.Linear(3 * a, b) for a, b in zip(dims[:-1], dims[1:])]
        )
        self.out = nn.Linear(h, 1)
        self.p = p

    def forward(self, x, Ain, Aout):
        for layer in self.lin:
            x = torch.relu(
                layer(
                    torch.cat(
                        [x, torch.sparse.mm(Ain, x), torch.sparse.mm(Aout, x)],
                        dim=1,
                    )
                )
            )
            x = nn.functional.dropout(x, self.p, self.training)
        return self.out(x).squeeze(-1)


def pos_weight(y):
    return torch.tensor(
        (y == 0).sum() / max((y == 1).sum(), 1),
        dtype=torch.float32,
        device=dev,
    )


def fit_lstm(Xs, y):
    h = HP["lstm"]
    m = LSTMNet(
        Xs.shape[2], h["hidden"], h["layers"], h["dropout"]
    ).to(dev)

    opt = torch.optim.Adam(m.parameters(), lr=h["lr"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight(y))

    Xt = torch.tensor(Xs, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)

    for _ in range(h["epochs"]):
        m.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), h["batch"]):
            b = perm[i:i + h["batch"]]
            opt.zero_grad()
            loss_fn(m(Xt[b]), yt[b]).backward()
            opt.step()

    return m


def pred_lstm(m, Xs):
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(
            m(torch.tensor(Xs, device=dev))
        ).cpu().numpy()


def fit_ae(Xl):
    h = HP["ae"]
    m = AE(h["dims"]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=h["lr"])
    Xt = torch.tensor(Xl, device=dev)

    for _ in range(h["epochs"]):
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), h["batch"]):
            b = Xt[perm[i:i + h["batch"]]]
            opt.zero_grad()
            nn.functional.mse_loss(m(b), b).backward()
            opt.step()

    return m


def pred_ae(m, X):
    m.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, device=dev)
        return ((m(Xt) - Xt) ** 2).mean(1).cpu().numpy()


def fit_gnn(Xall, Ain, Aout, tr_idx, y_tr):
    h = HP["gnn"]
    m = SAGE(
        Xall.shape[1],
        h["hidden"],
        h["layers"],
        h["dropout"],
    ).to(dev)

    opt = torch.optim.Adam(
        m.parameters(),
        lr=h["lr"],
        weight_decay=h["weight_decay"],
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight(y_tr))

    ti = torch.tensor(tr_idx, device=dev)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=dev)

    for epoch in range(h["epochs"]):
        m.train()
        opt.zero_grad()
        logits = m(Xall, Ain, Aout)
        loss = loss_fn(logits[ti], yt)
        loss.backward()
        opt.step()

        if (epoch + 1) % 25 == 0:
            print(f"      GNN epoch {epoch + 1}/{h['epochs']} loss={loss.item():.5f}")

    return m


def pred_gnn(m, Xall, Ain, Aout):
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(
            m(Xall, Ain, Aout)
        ).cpu().numpy()


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def best_threshold(y, s):
    ts = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197)))
    scores = [
        precision_recall_fscore_support(
            y, s >= t, average="binary", zero_division=0
        )[2]
        for t in ts
    ]
    return float(ts[int(np.argmax(scores))])


def illicit_metrics(y, s, thr):
    p, r, f, _ = precision_recall_fscore_support(
        y, s >= thr, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(
        y, s >= thr, labels=[0, 1]
    ).ravel()

    return dict(
        precision=float(p),
        recall=float(r),
        f1=float(f),
        pr_auc=float(average_precision_score(y, s)),
        roc_auc=float(roc_auc_score(y, s)),
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        accuracy=float((tp + tn) / (tp + tn + fp + fn)),
        threshold=float(thr),
    )


def save_predictions(path, ids, steps, y, scores):
    data = {
        "row_index": np.arange(len(ids)),
        "txId": ids,
        "time_step": steps,
        "label": y,
    }
    for name, value in scores.items():
        data[name] = value
    pd.DataFrame(data).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./elliptic_bitcoin_dataset")
    parser.add_argument("--out", default="./results")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    seed_all()

    print("=" * 72)
    print("StackGuard-AML CPU experiment")
    print(f"Device: {dev}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 72)

    X, y, step, src, dst = load_real(args.data)

    if X.shape[1] != 165:
        raise ValueError(
            f"Expected 165 transaction features, found {X.shape[1]}. "
            "Check that elliptic_txs_features.csv is the original no-header file."
        )

    ids = pd.read_csv(
        os.path.join(args.data, "elliptic_txs_features.csv"),
        header=None,
        usecols=[0],
    ).iloc[:, 0].values

    labelled = np.where(y >= 0)[0]
    tr = labelled[
        (step[labelled] >= TRAIN_STEPS[0])
        & (step[labelled] <= TRAIN_STEPS[1])
    ]
    te = labelled[
        (step[labelled] >= TEST_STEPS[0])
        & (step[labelled] <= TEST_STEPS[1])
    ]

    ytr, yte = y[tr], y[te]

    info = dict(
        n_nodes=int(len(y)),
        n_edges=int(len(src)),
        n_labelled=int(len(labelled)),
        n_illicit=int((y == 1).sum()),
        n_licit=int((y == 0).sum()),
        train_n=int(len(tr)),
        train_illicit=int(ytr.sum()),
        train_licit=int((ytr == 0).sum()),
        test_n=int(len(te)),
        test_illicit=int(yte.sum()),
        test_licit=int((yte == 0).sum()),
    )

    print("\nDATA")
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\nFitting StandardScaler on training transactions only...")
    scaler = StandardScaler().fit(X[tr])
    Xs = scaler.transform(X).astype(np.float32)

    print("Building flow sequences...")
    seq_path = out / "flow_sequences.npy"
    if seq_path.exists() and not args.fresh:
        seq = np.load(seq_path, mmap_mode="r")
        print("  Loaded cached flow sequences.")
    else:
        seq = flow_sequences(Xs, src, dst)
        np.save(seq_path, seq)
        print("  Saved flow sequences.")

    print("Building graph matrices...")
    Ain = to_torch_sparse(mean_adj(len(y), src, dst))
    Aout = to_torch_sparse(mean_adj(len(y), dst, src))
    Xall = torch.tensor(Xs, device=dev)

    def fit_predict(train_idx, ytrain, pred_idx):
        out_scores = {}
        models = {}
        times = {}

        t0 = time.time()
        rf = RandomForestClassifier(
            random_state=SEED, **HP["rf"]
        ).fit(Xs[train_idx], ytrain)
        train_t = time.time() - t0
        t1 = time.time()
        out_scores["RF"] = rf.predict_proba(Xs[pred_idx])[:, 1]
        pred_t = time.time() - t1
        models["RF"] = rf
        times["RF"] = [train_t, pred_t]
        print(f"    RF done: {train_t/60:.1f} min")

        spw = (ytrain == 0).sum() / max((ytrain == 1).sum(), 1)

        t0 = time.time()
        xg = xgb.XGBClassifier(
            random_state=SEED,
            scale_pos_weight=spw,
            **HP["xgb"],
        ).fit(Xs[train_idx], ytrain)
        train_t = time.time() - t0
        t1 = time.time()
        out_scores["XGB"] = xg.predict_proba(Xs[pred_idx])[:, 1]
        pred_t = time.time() - t1
        models["XGB"] = xg
        times["XGB"] = [train_t, pred_t]
        print(f"    XGB done: {train_t/60:.1f} min")

        t0 = time.time()
        ls = fit_lstm(seq[train_idx], ytrain)
        train_t = time.time() - t0
        t1 = time.time()
        out_scores["LSTM"] = pred_lstm(ls, seq[pred_idx])
        pred_t = time.time() - t1
        models["LSTM"] = ls
        times["LSTM"] = [train_t, pred_t]
        print(f"    LSTM done: {train_t/60:.1f} min")

        t0 = time.time()
        ae = fit_ae(Xs[train_idx][ytrain == 0])
        train_t = time.time() - t0
        t1 = time.time()
        out_scores["AE"] = pred_ae(ae, Xs[pred_idx])
        pred_t = time.time() - t1
        models["AE"] = ae
        times["AE"] = [train_t, pred_t]
        print(f"    AE done: {train_t/60:.1f} min")

        t0 = time.time()
        gn = fit_gnn(Xall, Ain, Aout, train_idx, ytrain)
        train_t = time.time() - t0
        t1 = time.time()
        all_gnn = pred_gnn(gn, Xall, Ain, Aout)
        out_scores["GNN"] = all_gnn[pred_idx]
        pred_t = time.time() - t1
        models["GNN"] = gn
        times["GNN"] = [train_t, pred_t]
        print(f"    GNN done: {train_t/60:.1f} min")

        return out_scores, models, times

    # -------------------------------------------------------------- OOF
    oof_file = out / "oof_predictions.npz"

    if oof_file.exists() and not args.fresh:
        z = np.load(oof_file)
        oof = {m: z[m] for m in BASE}
        print("\nLoaded completed OOF predictions from disk.")
    else:
        oof = {m: np.full(len(tr), np.nan, dtype=np.float32) for m in BASE}

        splitter = GroupKFold(N_FOLDS)
        folds = list(splitter.split(tr, ytr, groups=step[tr]))

        for k, (a_, b_) in enumerate(folds, start=1):
            fold_file = out / f"oof_fold_{k}.npz"

            if fold_file.exists() and not args.fresh:
                print(f"\nOOF fold {k}/{N_FOLDS}: loading cached fold")
                z = np.load(fold_file)
                for m in BASE:
                    oof[m][b_] = z[m]
                continue

            print(f"\n{'=' * 60}")
            print(f"OOF FOLD {k}/{N_FOLDS}")
            print(
                f"Train transactions: {len(a_)} | "
                f"validation transactions: {len(b_)}"
            )
            print(f"{'=' * 60}")

            seed_all()
            scores, _, _ = fit_predict(tr[a_], ytr[a_], tr[b_])

            np.savez(
                fold_file,
                **{m: scores[m].astype(np.float32) for m in BASE},
            )

            for m in BASE:
                oof[m][b_] = scores[m]

            np.savez(
                oof_file,
                **{m: oof[m] for m in BASE},
            )

            print(f"Saved OOF fold {k}.")

    if any(np.isnan(oof[m]).any() for m in BASE):
        raise RuntimeError("OOF predictions contain missing values.")

    print("\nAll OOF predictions complete.")

    # ---------------------------------------------------------- FINAL FIT
    test_file = out / "test_base_predictions.npz"
    final_models = None
    final_times = {}

    if test_file.exists() and not args.fresh:
        z = np.load(test_file)
        te_scores = {m: z[m] for m in BASE}
        print("Loaded cached final test predictions.")
    else:
        print("\n" + "=" * 72)
        print("FINAL FIT: all training steps 1-34 -> test steps 35-49")
        print("=" * 72)

        seed_all()
        te_scores, final_models, final_times = fit_predict(tr, ytr, te)

        np.savez(
            test_file,
            **{m: te_scores[m].astype(np.float32) for m in BASE},
        )
        print("Saved final test predictions.")

    # ------------------------------------------------------------- META
    def meta_feats(scores, cols):
        return np.column_stack(
            [
                logit(scores[m]) if m != "AE"
                else np.log1p(scores[m])
                for m in cols
            ]
        )

    def run_meta(cols):
        Ftr = meta_feats(oof, cols)
        meta_scaler = StandardScaler().fit(Ftr)

        lr = LogisticRegression(
            C=HP["meta"]["C"],
            class_weight="balanced",
            max_iter=1000,
        ).fit(
            meta_scaler.transform(Ftr),
            ytr,
        )

        s_oof = lr.predict_proba(
            meta_scaler.transform(Ftr)
        )[:, 1]

        threshold = best_threshold(ytr, s_oof)

        s_te = lr.predict_proba(
            meta_scaler.transform(meta_feats(te_scores, cols))
        )[:, 1]

        return lr, meta_scaler, threshold, s_oof, s_te

    lr, meta_scaler, threshold, s_oof, s_stack = run_meta(BASE)

    res = {
        "info": info,
        "hyperparameters": HP,
        "threshold": threshold,
        "models": {},
        "ablation": {},
        "timing_s": final_times,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "none"
            ),
        },
    }

    # Base-model thresholds are selected from OOF training scores only.
    for m in BASE:
        base_thr = best_threshold(ytr, oof[m])
        res["models"][m] = illicit_metrics(
            yte, te_scores[m], base_thr
        )

    res["models"]["Stacked ensemble"] = illicit_metrics(
        yte, s_stack, threshold
    )

    res["meta_coefficients"] = dict(
        zip(BASE, lr.coef_[0].tolist())
    )

    # ----------------------------------------------------------- outputs
    with open(out / "results.json", "w") as f:
        json.dump(res, f, indent=2)

    rows = []
    for model, values in res["models"].items():
        rows.append(
            {
                "model": model,
                "precision": round(values["precision"] * 100, 2),
                "recall": round(values["recall"] * 100, 2),
                "f1": round(values["f1"] * 100, 2),
                "pr_auc": round(values["pr_auc"] * 100, 2),
                "roc_auc": round(values["roc_auc"] * 100, 2),
                "tn": values["tn"],
                "fp": values["fp"],
                "fn": values["fn"],
                "tp": values["tp"],
            }
        )

    pd.DataFrame(rows).to_csv(
        out / "table_main.csv", index=False
    )

    save_predictions(
        out / "test_predictions.csv",
        ids[te],
        step[te],
        yte,
        {
            **te_scores,
            "Stacked_ensemble": s_stack,
        },
    )

    # ----------------------------------------------------- confusion matrix
    stack = res["models"]["Stacked ensemble"]
    cm = np.array(
        [[stack["tn"], stack["fp"]], [stack["fn"], stack["tp"]]]
    )

    pd.DataFrame(
        cm,
        index=["Actual_Licit", "Actual_Illicit"],
        columns=["Predicted_Licit", "Predicted_Illicit"],
    ).to_csv(out / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(3.0, 2.6), dpi=300)
    ax.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=9,
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Licit", "Illicit"], fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Licit", "Illicit"], fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_confusion_matrix.png")
    plt.close(fig)

    # ----------------------------------------------------- model comparison
    fig, ax = plt.subplots(figsize=(4.2, 2.8), dpi=300)
    model_names = list(res["models"].keys())
    width = 0.18

    for i, metric in enumerate(
        ["precision", "recall", "f1", "pr_auc"]
    ):
        ax.bar(
            np.arange(len(model_names))
            + (i - 1.5) * width,
            [res["models"][m][metric] for m in model_names],
            width,
            label="PR-AUC" if metric == "pr_auc"
            else metric.upper() if metric == "f1"
            else metric.capitalize(),
        )

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(
        [m.replace("Stacked ensemble", "Stack") for m in model_names],
        fontsize=6,
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Score", fontsize=7)
    ax.legend(fontsize=5, ncol=4)
    fig.tight_layout()
    fig.savefig(out / "fig2_model_comparison.png")
    plt.close(fig)

    # ------------------------------------------------------------ ablation
    if not args.skip_ablation:
        print("\nRunning ablation...")
        for drop in BASE:
            cols = [c for c in BASE if c != drop]
            _, _, thr2, _, s2 = run_meta(cols)
            res["ablation"][f"without_{drop}"] = illicit_metrics(
                yte, s2, thr2
            )

        pd.DataFrame(
            [
                {
                    "setting": k,
                    "precision": round(v["precision"] * 100, 2),
                    "recall": round(v["recall"] * 100, 2),
                    "f1": round(v["f1"] * 100, 2),
                    "pr_auc": round(v["pr_auc"] * 100, 2),
                }
                for k, v in res["ablation"].items()
            ]
        ).to_csv(out / "table_ablation.csv", index=False)

        with open(out / "results.json", "w") as f:
            json.dump(res, f, indent=2)

    # --------------------------------------------------------------- SHAP
    if not args.skip_shap:
        if final_models is None:
            print(
                "\nSHAP requested, but final model objects were not retained "
                "because test predictions were loaded from cache."
            )
            print("Run once with --fresh --skip-ablation to regenerate models.")
        else:
            print("\nGenerating SHAP plots...")
            rng = np.random.default_rng(SEED)
            sub = rng.choice(
                len(te),
                min(2000, len(te)),
                replace=False,
            )
            names = feat_names()

            try:
                explainer = shap.TreeExplainer(final_models["XGB"])
                sv = explainer.shap_values(Xs[te][sub])

                if isinstance(sv, list):
                    sv = sv[-1]

                plt.figure(figsize=(3.5, 3.2), dpi=300)
                shap.summary_plot(
                    sv,
                    Xs[te][sub],
                    feature_names=names,
                    max_display=12,
                    show=False,
                    plot_size=None,
                )
                plt.gcf().axes[0].tick_params(labelsize=6)
                plt.tight_layout()
                plt.savefig(
                    out / "fig4a_shap_xgboost.png",
                    bbox_inches="tight",
                )
                plt.close()
            except Exception as exc:
                print(f"XGBoost SHAP failed: {exc}")

            try:
                Ftr = meta_scaler.transform(
                    meta_feats(oof, BASE)
                )
                Fte = meta_scaler.transform(
                    meta_feats(te_scores, BASE)
                )

                linear_explainer = shap.LinearExplainer(
                    lr, Ftr
                )
                sv2 = linear_explainer.shap_values(Fte)

                plt.figure(figsize=(3.5, 2.0), dpi=300)
                shap.summary_plot(
                    sv2,
                    Fte,
                    feature_names=BASE,
                    show=False,
                    plot_size=None,
                )
                plt.gcf().axes[0].tick_params(labelsize=6)
                plt.tight_layout()
                plt.savefig(
                    out / "fig4b_shap_meta.png",
                    bbox_inches="tight",
                )
                plt.close()
            except Exception as exc:
                print(f"Meta SHAP failed: {exc}")

    # ------------------------------------------------------ paper summary
    s = res["models"]["Stacked ensemble"]

    summary = f"""StackGuard-AML experiment results

Temporal training steps: {TRAIN_STEPS[0]}-{TRAIN_STEPS[1]}
Temporal test steps: {TEST_STEPS[0]}-{TEST_STEPS[1]}

Test transactions: {info["test_n"]}
Test illicit: {info["test_illicit"]}
Test licit: {info["test_licit"]}

STACKED ENSEMBLE
Precision: {s["precision"] * 100:.2f}%
Recall:    {s["recall"] * 100:.2f}%
F1:        {s["f1"] * 100:.2f}%
PR-AUC:    {s["pr_auc"] * 100:.2f}%
ROC-AUC:   {s["roc_auc"] * 100:.2f}%

CONFUSION MATRIX
TN (licit correctly predicted): {s["tn"]}
FP (licit predicted illicit):    {s["fp"]}
FN (illicit predicted licit):    {s["fn"]}
TP (illicit correctly predicted):{s["tp"]}

OOF threshold: {threshold:.8f}

Hardware:
Python: {platform.python_version()}
PyTorch: {torch.__version__}
GPU: {"available" if torch.cuda.is_available() else "CPU only"}

IMPORTANT: These are actual experiment outputs. Do not replace them
with numbers from an earlier draft or with fabricated values.
"""

    (out / "paper_results.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("FINAL STACKED RESULTS")
    print("=" * 72)
    print(summary)
    print(f"All outputs saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
