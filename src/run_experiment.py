"""Full experiment: SMOTE-Tomek variants + Optuna + four tree ensembles on the CDC Diabetes Health Indicators data (BRFSS 2015).

Configurations: A baseline | B standard SMOTE-Tomek (ST) | C Optuna | D ST + Optuna | E Optuna + class weighting
                F discrete-aware SMOTE-Tomek (ST-D) + Optuna (proposed) | G ST-D with default hyperparameters (ablation)

Usage:  python src/run_experiment.py --data data/diabetes_binary_health_indicators_BRFSS2015.csv --out results [--quick]
Outputs (in --out): results.csv, best_params.json, bootstrap_ci.csv, importance.csv, info.json, probs.npz
"""
import json, time, warnings, sys, os, argparse
import numpy as np, pandas as pd, optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="results")
ap.add_argument("--quick", action="store_true", help="small subset and few trials for a smoke test")
ap.add_argument("--n_jobs", type=int, default=2); A_ = ap.parse_args()
os.makedirs(A_.out, exist_ok=True); OUT = lambda f: os.path.join(A_.out, f)
SEED, NJ = 42, A_.n_jobs
SMOKE = A_.quick
N_TUNE, N_FOLDS, N_TRIALS, TIMEOUT = (3000, 3, 2, 60) if SMOKE else (30000, 3, 25, 480)
log = lambda *a: (print(time.strftime("%H:%M:%S"), *a), sys.stdout.flush())

# ---------- 1. Data ----------
raw = pd.read_csv(A_.data)
info = {"n_raw": len(raw), "n_missing": int(raw.isna().sum().sum()), "n_dup": int(raw.duplicated().sum())}
df = raw.drop_duplicates().reset_index(drop=True)
if SMOKE: df = df.sample(20000, random_state=SEED).reset_index(drop=True)
X, y = df.drop(columns="Diabetes_binary"), df["Diabetes_binary"].astype(int)
info.update(n_clean=len(df), pos_clean=int(y.sum()), pos_rate=float(y.mean()))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
sc = StandardScaler().fit(Xtr)                       # fitted on training data only
Xtr_s = pd.DataFrame(sc.transform(Xtr), columns=X.columns); Xte_s = pd.DataFrame(sc.transform(Xte), columns=X.columns)
ytr, yte = ytr.reset_index(drop=True), yte.reset_index(drop=True)
info.update(n_train=len(Xtr), n_test=len(Xte), pos_train=int(ytr.sum()), pos_test=int(yte.sum()))
def smote_tomek_discrete(Xs_, y_):
    """SMOTE -> snap to valid discrete values on the original scale -> Tomek links (prevents fractional synthetic values)."""
    Xo, yo = SMOTE(random_state=SEED).fit_resample(Xs_, y_)
    raw_ = np.clip(np.round(sc.inverse_transform(Xo)), LO, HI)
    Xo = pd.DataFrame(sc.transform(raw_), columns=Xs_.columns)
    return TomekLinks(sampling_strategy="all", n_jobs=NJ).fit_resample(Xo, yo)   # same rule as standard SMOTETomek: remove both members of each link
LO, HI = Xtr.min().values, Xtr.max().values
spw = float((ytr == 0).sum() / (ytr == 1).sum())

t = time.time(); Xst, yst = SMOTETomek(random_state=SEED, n_jobs=NJ).fit_resample(Xtr_s, ytr)
info.update(n_train_st=len(Xst), st_counts=np.bincount(yst).tolist(), st_seconds=round(time.time() - t, 1))
info["frac_binary_rows_st"] = int((np.abs(sc.inverse_transform(Xst)[:, list(X.columns).index("HighBP")] % 1) > 1e-6).sum())
t = time.time(); Xsd, ysd = smote_tomek_discrete(Xtr_s, ytr)
info.update(n_train_std=len(Xsd), std_counts=np.bincount(ysd).tolist(), std_seconds=round(time.time() - t, 1))
log("data", info)

# tuning subset; folds are resampled once because resampling does not depend on hyperparameters
Xsub, _, ysub, _ = train_test_split(Xtr_s, ytr, train_size=N_TUNE, stratify=ytr, random_state=SEED)
Xsub, ysub = Xsub.reset_index(drop=True), ysub.reset_index(drop=True)
folds = []
for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(Xsub, ysub):
    Xr, yr = SMOTETomek(random_state=SEED, n_jobs=NJ).fit_resample(Xsub.iloc[tr], ysub.iloc[tr])
    Xd, yd = smote_tomek_discrete(Xsub.iloc[tr].reset_index(drop=True), ysub.iloc[tr].reset_index(drop=True))
    folds.append(dict(Xtr=Xsub.iloc[tr], ytr=ysub.iloc[tr], Xr=Xr, yr=yr, Xd=Xd, yd=yd, Xva=Xsub.iloc[va], yva=ysub.iloc[va]))

# ---------- 2. Models & search spaces ----------
def make(name, p=None, weighted=False):
    p = dict(p or {})
    if name == "XGBoost":
        return XGBClassifier(tree_method="hist", n_jobs=NJ, random_state=SEED, eval_metric="logloss",
                             scale_pos_weight=spw if weighted else 1.0, **p)
    if name == "LightGBM":
        return LGBMClassifier(n_jobs=NJ, random_state=SEED, verbose=-1,
                              class_weight="balanced" if weighted else None, **p)
    if name == "CatBoost":
        return CatBoostClassifier(thread_count=NJ, random_seed=SEED, verbose=0,
                                  auto_class_weights="Balanced" if weighted else None, **p)
    return RandomForestClassifier(n_jobs=NJ, random_state=SEED, class_weight="balanced" if weighted else None, **p)

def space(name, tr):
    if name == "XGBoost":
        return dict(n_estimators=tr.suggest_int("n_estimators", 100, 600, step=50),
                    max_depth=tr.suggest_int("max_depth", 3, 10),
                    learning_rate=tr.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    subsample=tr.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=tr.suggest_float("colsample_bytree", 0.6, 1.0),
                    min_child_weight=tr.suggest_int("min_child_weight", 1, 10),
                    gamma=tr.suggest_float("gamma", 0.0, 5.0),
                    reg_lambda=tr.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
    if name == "LightGBM":
        return dict(n_estimators=tr.suggest_int("n_estimators", 100, 600, step=50),
                    num_leaves=tr.suggest_int("num_leaves", 15, 255),
                    learning_rate=tr.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    min_child_samples=tr.suggest_int("min_child_samples", 5, 100),
                    subsample=tr.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
                    colsample_bytree=tr.suggest_float("colsample_bytree", 0.6, 1.0),
                    reg_lambda=tr.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
    if name == "CatBoost":
        return dict(iterations=tr.suggest_int("iterations", 200, 600, step=50),
                    depth=tr.suggest_int("depth", 4, 8),
                    learning_rate=tr.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    l2_leaf_reg=tr.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                    bagging_temperature=tr.suggest_float("bagging_temperature", 0.0, 1.0))
    return dict(n_estimators=tr.suggest_int("n_estimators", 100, 300, step=50),
                max_depth=tr.suggest_int("max_depth", 6, 30),
                min_samples_split=tr.suggest_int("min_samples_split", 2, 20),
                min_samples_leaf=tr.suggest_int("min_samples_leaf", 1, 10),
                max_features=tr.suggest_categorical("max_features", ["sqrt", "log2", 0.5]))

def tune(name, mode):   # mode: C original, D ST, E class weights, F ST-D
    def obj(tr):
        p = space(name, tr); s = []
        for f in folds:
            Xa, ya = {"D": (f["Xr"], f["yr"]), "F": (f["Xd"], f["yd"])}.get(mode, (f["Xtr"], f["ytr"]))
            m = make(name, p, weighted=(mode == "E")).fit(Xa, ya)
            s.append(f1_score(f["yva"], m.predict(f["Xva"])))
        return float(np.mean(s))
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=N_TRIALS, timeout=TIMEOUT)
    return st.best_params, st.best_value, len(st.trials)

def evaluate(m):
    pr = m.predict_proba(Xte_s)[:, 1]; yp = (pr >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, yp).ravel()
    return dict(accuracy=accuracy_score(yte, yp), precision=precision_score(yte, yp), recall=recall_score(yte, yp),
                specificity=tn / (tn + fp), f1=f1_score(yte, yp), roc_auc=roc_auc_score(yte, pr),
                pr_auc=average_precision_score(yte, pr), brier=brier_score_loss(yte, pr),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)), pr

# ---------- 3. Experiments ----------
rows, params, probs, models = [], {}, {}, {}
for name in ["XGBoost", "LightGBM", "CatBoost", "RandomForest"]:
    for cfg in ["A", "B", "C", "D", "E", "F", "G"]:
        t = time.time(); p, cvf, ntr = {}, None, 0
        key = f"{name}_{cfg}"
        if cfg in "CDEF":
            p, cvf, ntr = tune(name, cfg); tt = time.time() - t
        else: tt = 0.0
        Xa, ya = (Xst, yst) if cfg in "BD" else (Xsd, ysd) if cfg in "FG" else (Xtr_s, ytr)
        t2 = time.time(); m = make(name, p, weighted=(cfg == "E")).fit(Xa, ya); ft = time.time() - t2
        met, pr = evaluate(m)
        rows.append(dict(model=name, config=cfg, cv_f1=cvf, n_trials=ntr, tune_s=round(tt, 1), fit_s=round(ft, 1), **met))
        params[key] = p; probs[key] = pr; models[key] = m
        if cfg != "F": models.pop(key)   # keep only the proposed models in memory
        log(name, cfg, {k: round(v, 4) for k, v in met.items() if isinstance(v, float)}, f"tune={tt:.0f}s fit={ft:.0f}s trials={ntr}")
        pd.DataFrame(rows).to_csv(OUT("results.csv"), index=False)
        json.dump(params, open(OUT("best_params.json"), "w"), indent=1)
np.savez(OUT("probs.npz"), y=yte.values, **probs)

# ---------- 4. Bootstrap CI (1000x) ----------
rng = np.random.default_rng(SEED); B = 1000; yv = yte.values; ci = []
idx = [rng.integers(0, len(yv), len(yv)) for _ in range(B)]
for k, pr in probs.items():
    f1s, aucs = [], []
    for ii in idx:
        f1s.append(f1_score(yv[ii], pr[ii] >= 0.5)); aucs.append(roc_auc_score(yv[ii], pr[ii]))
    ci.append(dict(key=k, f1_lo=np.percentile(f1s, 2.5), f1_hi=np.percentile(f1s, 97.5),
                   auc_lo=np.percentile(aucs, 2.5), auc_hi=np.percentile(aucs, 97.5)))
pd.DataFrame(ci).to_csv(OUT("bootstrap_ci.csv"), index=False); log("bootstrap done")

# ---------- 5. Feature importance of the best proposed model (highest F1 in configuration F) ----------
res = pd.DataFrame(rows); best = res[res.config == "F"].sort_values("f1", ascending=False).iloc[0]
bk = f"{best.model}_F"; bm = models[bk]
if best.model == "CatBoost": imp = bm.get_feature_importance()
else: imp = bm.feature_importances_
sub = rng.choice(len(Xte_s), min(20000, len(Xte_s)), replace=False)
pi = permutation_importance(bm, Xte_s.iloc[sub], yte.iloc[sub], scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=1)
pd.DataFrame(dict(feature=X.columns, builtin=np.asarray(imp) / np.sum(imp), perm_mean=pi.importances_mean,
                  perm_std=pi.importances_std)).sort_values("perm_mean", ascending=False).to_csv(OUT("importance.csv"), index=False)
info["best_F"] = bk; json.dump(info, open(OUT("info.json"), "w"), indent=1); log("ALL DONE", bk)
