from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

EPS = 1e-10
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 0


# ---------------------------------------------------------------------------
# Point-estimate / surrogate construction
# ---------------------------------------------------------------------------

def _infer_demographic_cols(cells_df: pd.DataFrame) -> List[str]:
    return [c for c in cells_df.columns if c not in ("cell_id", "label")]


def posterior_mean_counts(reconstructed: Dict) -> np.ndarray:
    samples = np.asarray(reconstructed["atomic_posterior_samples"], dtype=float)
    return samples.mean(axis=0)


def atomic_counts_from_line_level(
    df: pd.DataFrame,
    cells_df: pd.DataFrame,
    demographic_cols: Optional[Sequence[str]] = None,
) -> np.ndarray:
    cols = list(demographic_cols) if demographic_cols is not None else _infer_demographic_cols(cells_df)
    grouped = df.groupby(cols).size()
    return np.array(
        [grouped.get(tuple(cells_df.iloc[i][cols]), 0) for i in range(len(cells_df))],
        dtype=float,
    )


def counts_to_line_level(
    counts: np.ndarray,
    cells_df: pd.DataFrame,
    demographic_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    cols = list(demographic_cols) if demographic_cols is not None else _infer_demographic_cols(cells_df)
    counts = np.clip(np.asarray(counts, dtype=float), 0.0, None)
    floor = np.floor(counts).astype(int)
    deficit = int(round(counts.sum())) - int(floor.sum())
    frac = counts - floor
    if deficit > 0:
        floor[np.argsort(-frac)[:deficit]] += 1
    elif deficit < 0:
        order = np.argsort(frac)
        k = 0
        while deficit < 0 and k < len(order):
            j = order[k]
            if floor[j] > 0:
                floor[j] -= 1
                deficit += 1
            k += 1

    cell_vals = cells_df[cols].to_numpy()
    rows: List[Dict[str, object]] = []
    for i, c in enumerate(floor):
        if c > 0:
            rows.extend([dict(zip(cols, cell_vals[i]))] * int(c))
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Pure metric primitives
# ---------------------------------------------------------------------------

def _safe_prob(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    return (counts + EPS) / (counts.sum() + EPS)


def jsd_from_counts(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    return float(jensenshannon(_safe_prob(p_counts), _safe_prob(q_counts)) ** 2)


def cramers_v(df: pd.DataFrame, col_a: str, col_b: str) -> float:
    table = pd.crosstab(df[col_a], df[col_b])
    if table.empty:
        return 0.0
    chi2 = chi2_contingency(table)[0]
    n = int(table.to_numpy().sum())
    k = min(table.shape) - 1
    if n == 0 or k <= 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * k)))


def _ci(point: float, boot: np.ndarray) -> Dict[str, float]:
    boot = np.asarray(boot, dtype=float)
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return {"point": float(point), "ci_lower": np.nan, "ci_upper": np.nan,
                "boot_mean": np.nan, "se": np.nan}
    return {
        "point": float(point),
        "ci_lower": float(np.percentile(boot, 2.5)),
        "ci_upper": float(np.percentile(boot, 97.5)),
        "boot_mean": float(boot.mean()),
        "se": float(boot.std(ddof=1)) if boot.size > 1 else np.nan,
    }


def fmt_ci(ci: Dict[str, float], prec: int = 4, point_kind: str = "plugin") -> str:
    point = ci["point"] if point_kind == "plugin" else ci.get("boot_mean", ci["point"])
    if not np.isfinite(ci.get("ci_lower", np.nan)):
        return f"{point:.{prec}f}"
    return f"{point:.{prec}f} ({ci['ci_lower']:.{prec}f}, {ci['ci_upper']:.{prec}f})"


# ---------------------------------------------------------------------------
# Analysis primitives (the statistics before any bootstrapping)
# ---------------------------------------------------------------------------

def marginal_probs(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].value_counts(normalize=True).sort_index()


def odds_ratios(df: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
    x_cols = [c for c in df.columns if c != outcome_col]
    rows = []
    for key, grp in df.groupby(x_cols):
        key = key if isinstance(key, tuple) else (key,)
        n_pos = int(grp[outcome_col].sum())
        n_neg = len(grp) - n_pos
        other = df[~((df[x_cols] == pd.Series(key, index=x_cols)).all(axis=1))]
        o_pos = int(other[outcome_col].sum())
        o_neg = len(other) - o_pos

        odds_g = n_pos / n_neg if n_neg else np.inf
        odds_o = o_pos / o_neg if o_neg else np.inf
        row = dict(zip(x_cols, key))
        row["odds_ratio"] = odds_g / odds_o if odds_o else np.inf
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("odds_ratio", ascending=False)
        .reset_index(drop=True)
    )


def safe_log_or(x, clip: float = 10.0) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(np.asarray(x, dtype=float))
    return np.clip(out, -clip, clip)


def or_flips(or_a, or_b) -> np.ndarray:
    a, b = np.asarray(or_a), np.asarray(or_b)
    return ((a > 1) & (b < 1)) | ((a < 1) & (b > 1))


def _encode(df, outcome_col, cat_cols, encoders=None):
    df = df.copy()
    fit = encoders is None
    if fit:
        encoders = {}
    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            unseen = set(df[col].unique()) - set(le.classes_)
            if unseen:
                le.classes_ = np.concatenate([le.classes_, np.array(sorted(unseen))])
            df[col + "_encoded"] = le.transform(df[col])
    X = df.drop([outcome_col] + list(cat_cols), axis=1).astype(int)
    y = df[outcome_col].astype(int)
    return X, y, encoders


def train_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    outcome_col: str,
    cat_cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    X_train, y_train, encs = _encode(train_df, outcome_col, cat_cols)
    X_test, y_test, _ = _encode(test_df, outcome_col, cat_cols, encs)

    lr = LogisticRegression(random_state=42, class_weight="balanced")
    lr.fit(X_train, y_train)
    return y_test.to_numpy(), lr.predict(X_test)


# ---------------------------------------------------------------------------
# Generic bootstrap drivers
# ---------------------------------------------------------------------------

def make_boot_indices(
    n: int, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _resolve_boot_idx(n, n_boot, seed, boot_idx) -> np.ndarray:
    if boot_idx is None:
        return make_boot_indices(n, n_boot, seed)
    boot_idx = np.asarray(boot_idx)
    if boot_idx.ndim != 2 or boot_idx.shape[1] != n:
        raise ValueError(
            f"boot_idx must have shape (n_boot, {n}) to match the data length; "
            f"got {boot_idx.shape}."
        )
    return boot_idx


def bootstrap_df(
    ref_df: pd.DataFrame,
    metric_fns: Dict[str, Callable[[pd.DataFrame], float]],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    boot_idx: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    boot_idx = _resolve_boot_idx(len(ref_df), n_boot, seed, boot_idx)
    point = {name: float(fn(ref_df)) for name, fn in metric_fns.items()}
    boot = {name: np.empty(len(boot_idx)) for name in metric_fns}
    for b, idx in enumerate(boot_idx):
        rdf = ref_df.iloc[idx]
        for name, fn in metric_fns.items():
            boot[name][b] = fn(rdf)
    return {name: _ci(point[name], boot[name]) for name in metric_fns}


def _patient_cell_map(
    df_original: pd.DataFrame,
    cells_df: pd.DataFrame,
    demographic_cols: Optional[Sequence[str]] = None,
) -> np.ndarray:
    cols = list(demographic_cols) if demographic_cols is not None else _infer_demographic_cols(cells_df)
    cell_vals = cells_df[cols].to_numpy()
    cell_pos = {tuple(cell_vals[i]): i for i in range(len(cell_vals))}
    return np.array(
        [cell_pos.get(tuple(r), -1) for r in df_original[cols].to_numpy()], dtype=int
    )


def cell_metric_patient_bootstrap(
    df_original: pd.DataFrame,
    cells_df: pd.DataFrame,
    surrogates: Dict[str, np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float],
    demographic_cols: Optional[Sequence[str]] = None,
    mask: Optional[np.ndarray] = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    boot_idx: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    cols = list(demographic_cols) if demographic_cols is not None else _infer_demographic_cols(cells_df)
    J = len(cells_df)
    pmap = _patient_cell_map(df_original, cells_df, cols)
    ref_full = np.bincount(pmap[pmap >= 0], minlength=J).astype(float)
    boot_idx = _resolve_boot_idx(len(df_original), n_boot, seed, boot_idx)

    m = slice(None) if mask is None else np.asarray(mask, dtype=bool)
    sur_m = {name: np.asarray(sur, dtype=float)[m] for name, sur in surrogates.items()}
    point = {name: metric(ref_full[m], sur_m[name]) for name in surrogates}
    boot = {name: np.empty(len(boot_idx)) for name in surrogates}
    for b, idx in enumerate(boot_idx):
        sel = pmap[idx]
        ref_b = np.bincount(sel[sel >= 0], minlength=J).astype(float)[m]
        for name in surrogates:
            boot[name][b] = metric(ref_b, sur_m[name])
    return {name: _ci(point[name], boot[name]) for name in surrogates}


def bootstrap_classification(
    y_true: np.ndarray,
    preds_by_method: Dict[str, np.ndarray],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    boot_idx: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Test-set bootstrap of classification metrics for fixed per-method models."""
    y_true = np.asarray(y_true)
    boot_idx = _resolve_boot_idx(len(y_true), n_boot, seed, boot_idx)

    funcs: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "Accuracy": accuracy_score,
        "Precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "Recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
        "F1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
    }

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, y_pred in preds_by_method.items():
        y_pred = np.asarray(y_pred)
        point = {m: f(y_true, y_pred) for m, f in funcs.items()}
        boot = {m: np.empty(len(boot_idx)) for m in funcs}
        for b, idx in enumerate(boot_idx):
            yt, yp = y_true[idx], y_pred[idx]
            for m, f in funcs.items():
                boot[m][b] = f(yt, yp)
        out[name] = {m: _ci(point[m], boot[m]) for m in funcs}
    return out


# ---------------------------------------------------------------------------
# Metric-set builders (each closure captures the fixed surrogate)
# ---------------------------------------------------------------------------

def _distribution_metric_fns(df_original, sur_df, demographic_cols):
    fns: Dict[str, Callable[[pd.DataFrame], float]] = {}
    for col in demographic_cols:
        cats = sorted(set(df_original[col].dropna().unique()) | set(sur_df[col].dropna().unique()))
        q = sur_df[col].value_counts().reindex(cats, fill_value=0).to_numpy(float)

        def f(ref_df, col=col, cats=cats, q=q):
            p = ref_df[col].value_counts().reindex(cats, fill_value=0).to_numpy(float)
            return jsd_from_counts(p, q)

        fns[col] = f

    cols = list(demographic_cols)
    sur_joint = sur_df.groupby(cols).size()

    def f_joint(ref_df, cols=cols, sur_joint=sur_joint):
        ref_joint = ref_df.groupby(cols).size()
        keys = sorted(set(ref_joint.index) | set(sur_joint.index))
        p = np.array([ref_joint.get(k, 0) for k in keys], dtype=float)
        q = np.array([sur_joint.get(k, 0) for k in keys], dtype=float)
        return jsd_from_counts(p, q)

    fns["Joint"] = f_joint
    return fns


def _marginal_prob_metric_fns(df_original, sur_df, demographic_cols):
    fns: Dict[str, Callable[[pd.DataFrame], float]] = {}
    for col in demographic_cols:
        cats = sorted(set(df_original[col].dropna().unique()) | set(sur_df[col].dropna().unique()))
        q = sur_df[col].value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy(float)

        def f(ref_df, col=col, cats=cats, q=q):
            p = ref_df[col].value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy(float)
            return float(np.mean(np.abs(p - q)))

        fns[col] = f
    return fns


# ---------------------------------------------------------------------------
# High-level comparators (one call per analysis, point + CI for every method)
# ---------------------------------------------------------------------------

def _surrogate_line_level(reconstructed, df_synthetic, demographic_cols):
    cells_df = reconstructed["cells_df"]
    cube_line = counts_to_line_level(
        posterior_mean_counts(reconstructed), cells_df, demographic_cols
    )
    return {"Cube": cube_line, "CTGAN": df_synthetic[list(demographic_cols)].copy()}


def distribution_fidelity_ci(
    df_original, reconstructed, df_synthetic, demographic_cols,
    n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED, boot_idx=None,
):
    surrogates = _surrogate_line_level(reconstructed, df_synthetic, demographic_cols)
    boot_idx = _resolve_boot_idx(len(df_original), n_boot, seed, boot_idx)
    rows_order = list(demographic_cols) + ["Joint"]
    out = {}
    for method, sur_df in surrogates.items():
        fns = _distribution_metric_fns(df_original, sur_df, demographic_cols)
        res = bootstrap_df(df_original, fns, boot_idx=boot_idx)
        out[method] = {k: res[k] for k in rows_order}
    return out


def compare_distribution_fidelity(
    df_original, reconstructed, df_synthetic, demographic_cols,
    n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED, point_kind="boot_mean", boot_idx=None,
):
    raw = distribution_fidelity_ci(
        df_original, reconstructed, df_synthetic, demographic_cols,
        n_boot=n_boot, seed=seed, boot_idx=boot_idx,
    )
    rows_order = list(demographic_cols) + ["Joint"]
    out = {method: {k: fmt_ci(raw[method][k], point_kind=point_kind) for k in rows_order}
           for method in raw}
    return pd.DataFrame(out).reindex(rows_order)


def plot_distribution_fidelity_compare(
    df_original, reconstructed, df_synthetic, demographic_cols,
    labels=("Cube", "CTGAN"), n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED,
    figsize=(9, 5), boot_idx=None, title="", point_kind="boot_mean",
    label_map=None,
):
    import matplotlib.pyplot as plt

    raw = distribution_fidelity_ci(
        df_original, reconstructed, df_synthetic, demographic_cols,
        n_boot=n_boot, seed=seed, boot_idx=boot_idx,
    )
    key = "boot_mean" if point_kind == "boot_mean" else "point"
    variables = list(demographic_cols) + ["Joint"]
    x = np.arange(len(variables))
    fig, ax = plt.subplots(figsize=figsize)
    for method, color, off in (("Cube", "steelblue", -0.18), ("CTGAN", "darkorange", 0.18)):
        means = np.array([raw[method][v][key] for v in variables])
        lo = np.array([raw[method][v]["ci_lower"] for v in variables])
        hi = np.array([raw[method][v]["ci_upper"] for v in variables])
        yerr = np.vstack([np.clip(means - lo, 0, None), np.clip(hi - means, 0, None)])
        ax.errorbar(x + off, means, yerr=yerr, fmt="o", capsize=4, color=color,
                    label=dict(zip(("Cube", "CTGAN"), labels))[method])
    ax.set_xticks(x)
    label_map = label_map or {}
    ax.set_xticklabels([label_map.get(v, v) for v in variables], rotation=30, ha="right")
    ax.set_ylabel("JS Divergence")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def _association_bootstrap(df_original, surrogates, pairs, boot_idx):
    v_orig_point = {p: cramers_v(df_original, *p) for p in pairs}
    v_sur = {m: {p: cramers_v(sdf, *p) for p in pairs} for m, sdf in surrogates.items()}
    boot = {p: np.empty(len(boot_idx)) for p in pairs}
    for b, idx in enumerate(boot_idx):
        rdf = df_original.iloc[idx]
        for p in pairs:
            boot[p][b] = cramers_v(rdf, *p)
    return v_orig_point, boot, v_sur


def compare_association_fidelity(
    df_original, reconstructed, df_synthetic, demographic_cols,
    n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED, boot_idx=None, prec=4,
):
    surrogates = _surrogate_line_level(reconstructed, df_synthetic, demographic_cols)
    boot_idx = _resolve_boot_idx(len(df_original), n_boot, seed, boot_idx)
    pairs = list(combinations(demographic_cols, 2))

    v_orig_point, boot, v_sur = _association_bootstrap(df_original, surrogates, pairs, boot_idx)

    col_order = ["Original (95% CI)"] + list(surrogates)
    rows = {}
    for p in pairs:
        row = {"Original (95% CI)": fmt_ci(_ci(v_orig_point[p], boot[p]), prec=prec, point_kind="boot_mean")}
        for m in surrogates:
            row[m] = f"{v_sur[m][p]:.{prec}f}"
        rows[f"{p[0]} x {p[1]}"] = row
    return pd.DataFrame(rows).T.reindex(index=[f"{p[0]} x {p[1]}" for p in pairs], columns=col_order)


def compare_association_mae(
    df_original, reconstructed, df_synthetic, demographic_cols,
    n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED, boot_idx=None, point_kind="boot_mean", prec=4,
):
    surrogates = _surrogate_line_level(reconstructed, df_synthetic, demographic_cols)
    boot_idx = _resolve_boot_idx(len(df_original), n_boot, seed, boot_idx)
    pairs = list(combinations(demographic_cols, 2))

    v_orig_point, boot, v_sur = _association_bootstrap(df_original, surrogates, pairs, boot_idx)

    out = {}
    for m in surrogates:
        point = float(np.mean([abs(v_orig_point[p] - v_sur[m][p]) for p in pairs]))
        boot_vals = np.mean([np.abs(boot[p] - v_sur[m][p]) for p in pairs], axis=0)
        out[m] = fmt_ci(_ci(point, boot_vals), prec=prec, point_kind=point_kind)
    return pd.DataFrame({"Mean |Δ| Cramér's V (95% CI)": out})


def compare_r2(
    df_original, reconstructed, df_synthetic, demographic_cols,
    n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED, boot_idx=None,
):
    cells_df = reconstructed["cells_df"]
    surrogate_counts = {
        "Cube": posterior_mean_counts(reconstructed),
        "CTGAN": atomic_counts_from_line_level(df_synthetic, cells_df, demographic_cols),
    }
    res = cell_metric_patient_bootstrap(
        df_original, cells_df, surrogate_counts, r2_score,
        demographic_cols=demographic_cols, n_boot=n_boot, seed=seed, boot_idx=boot_idx,
    )
    out = {method: fmt_ci(res[method], point_kind="boot_mean") for method in surrogate_counts}
    return pd.DataFrame({"R2 (95% CI)": out})


def compare_suppressed_mae(
    df_original, reconstructed, df_synthetic, demographic_cols,
    suppress_threshold=9, n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED,
    include_baselines=True, boot_idx=None,
):
    cells_df = reconstructed["cells_df"]
    true_counts = atomic_counts_from_line_level(df_original, cells_df, demographic_cols)
    mask = true_counts <= suppress_threshold

    estimates: Dict[str, np.ndarray] = {
        "Count Inference (Cube)": posterior_mean_counts(reconstructed),
    }
    if include_baselines:
        rng = np.random.default_rng(seed)
        estimates["Suppressed = 0"] = np.zeros_like(true_counts)
        estimates["Suppressed = 5"] = np.full_like(true_counts, 5.0)
        estimates["Suppressed = random (0-9)"] = rng.integers(
            0, suppress_threshold + 1, size=len(true_counts)
        ).astype(float)
    estimates["CTGAN synthetic data"] = atomic_counts_from_line_level(
        df_synthetic, cells_df, demographic_cols
    )

    res = cell_metric_patient_bootstrap(
        df_original, cells_df, estimates, mean_absolute_error,
        demographic_cols=demographic_cols, mask=mask,
        n_boot=n_boot, seed=seed, boot_idx=boot_idx,
    )
    out = {method: fmt_ci(res[method], point_kind="boot_mean") for method in estimates}
    return pd.DataFrame({"MAE (95% CI)": out})


def compare_marginal_prob_mae(
    df_original, reconstructed, df_synthetic, demographic_cols,
    n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED, point_kind="boot_mean", boot_idx=None,
):
    surrogates = _surrogate_line_level(reconstructed, df_synthetic, demographic_cols)
    boot_idx = _resolve_boot_idx(len(df_original), n_boot, seed, boot_idx)
    out = {}
    for method, sur_df in surrogates.items():
        fns = _marginal_prob_metric_fns(df_original, sur_df, demographic_cols)
        res = bootstrap_df(df_original, fns, boot_idx=boot_idx)
        out[method] = {col: fmt_ci(res[col], point_kind=point_kind) for col in demographic_cols}
    return pd.DataFrame(out).reindex(list(demographic_cols))


def cube_marginal_prob(reconstructed, col, demographic_cols=None, categories=None):
    cells_df = reconstructed["cells_df"]
    cols = list(demographic_cols) if demographic_cols is not None else _infer_demographic_cols(cells_df)
    cube = counts_to_line_level(posterior_mean_counts(reconstructed), cells_df, cols)
    s = cube[col].value_counts(normalize=True).sort_index()
    return s.reindex(categories, fill_value=0) if categories is not None else s


def compare_marginal_probs(
    df_original, reconstructed, df_synthetic, col,
    demographic_cols=None, n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED,
    categories=None, boot_idx=None, prec=4, point_kind="boot_mean",
):
    cols = list(demographic_cols) if demographic_cols is not None else _infer_demographic_cols(reconstructed["cells_df"])
    surrogates = _surrogate_line_level(reconstructed, df_synthetic, cols)
    if categories is None:
        cats = set(df_original[col].dropna().unique())
        for s in surrogates.values():
            cats |= set(s[col].dropna().unique())
        categories = sorted(cats)
    boot_idx = _resolve_boot_idx(len(df_original), n_boot, seed, boot_idx)

    p_orig = df_original[col].value_counts(normalize=True).reindex(categories, fill_value=0)
    p_sur = {m: s[col].value_counts(normalize=True).reindex(categories, fill_value=0)
             for m, s in surrogates.items()}

    fns: Dict[tuple, Callable[[pd.DataFrame], float]] = {}
    for m in surrogates:
        for cat in categories:
            qcat = float(p_sur[m][cat])
            fns[(m, cat)] = lambda ref_df, col=col, cat=cat, qcat=qcat: abs(
                float((ref_df[col] == cat).mean()) - qcat)
    res = bootstrap_df(df_original, fns, boot_idx=boot_idx)

    col_order = ["Original"]
    for m in surrogates:
        col_order += [m, f"{m} |Δ| (95% CI)"]

    rows = {}
    for cat in categories:
        row = {"Original": f"{float(p_orig[cat]):.{prec}f}"}
        for m in surrogates:
            row[m] = f"{float(p_sur[m][cat]):.{prec}f}"
            row[f"{m} |Δ| (95% CI)"] = fmt_ci(res[(m, cat)], prec=prec, point_kind=point_kind)
        rows[cat] = row
    return pd.DataFrame(rows).T.reindex(index=categories, columns=col_order)


def compare_odds_ratios(
    df_original, reconstructed, df_synthetic, demographic_cols, outcome_col,
    n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED, boot_idx=None, point_kind="boot_mean",
):
    surrogates = _surrogate_line_level(reconstructed, df_synthetic, demographic_cols)
    boot_idx = _resolve_boot_idx(len(df_original), n_boot, seed, boot_idx)
    out: Dict[str, Dict[str, str]] = {}

    for method, sur_df in surrogates.items():
        sur_or = odds_ratios(sur_df, outcome_col)
        key_cols = [c for c in sur_or.columns if c != "odds_ratio"]
        sur_map = sur_or.set_index(key_cols)["odds_ratio"]

        def both(ref_df):
            ref_or = odds_ratios(ref_df, outcome_col).set_index(key_cols)["odds_ratio"]
            common = ref_or.index.intersection(sur_map.index)
            a = safe_log_or(ref_or.loc[common].to_numpy())
            b = safe_log_or(sur_map.loc[common].to_numpy())
            flips = np.sum(or_flips(ref_or.loc[common].to_numpy(),
                                       sur_map.loc[common].to_numpy()))
            return float(np.mean(np.abs(a - b))), float(flips)

        p_mae, p_flips = both(df_original)
        b_mae, b_flips = np.empty(len(boot_idx)), np.empty(len(boot_idx))
        for b, idx in enumerate(boot_idx):
            b_mae[b], b_flips[b] = both(df_original.iloc[idx])
        out[method] = {
            "Log-OR MAE": fmt_ci(_ci(p_mae, b_mae), point_kind=point_kind),
            "OR sign flips": fmt_ci(_ci(p_flips, b_flips), prec=1, point_kind=point_kind),
        }
    return pd.DataFrame(out).reindex(["Log-OR MAE", "OR sign flips"])


def _subgroup_fisher_or_p(df, key_cols, key, outcome_col):
    target = pd.Series(key, index=key_cols)
    g = (df[key_cols] == target).all(axis=1).to_numpy()
    y = df[outcome_col].astype(int).to_numpy()
    n_pos = int(y[g].sum())
    n_neg = int(g.sum()) - n_pos
    o_pos = int(y[~g].sum())
    o_neg = int((~g).sum()) - o_pos
    or_val, p_val = fisher_exact([[n_pos, n_neg], [o_pos, o_neg]])
    return float(or_val), float(p_val)



def flipped_odds_ratio_significance(
    df_original, reconstructed, df_synthetic, demographic_cols, outcome_col,
    alpha=0.05, n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED,
    boot_idx=None, point_kind="boot_mean",
):
    surrogates = _surrogate_line_level(reconstructed, df_synthetic, demographic_cols)
    orig_or = odds_ratios(df_original, outcome_col)
    key_cols = [c for c in orig_or.columns if c != "odds_ratio"]
    orig_map = orig_or.set_index(key_cols)["odds_ratio"]
    boot_idx = _resolve_boot_idx(len(df_original), n_boot, seed, boot_idx)

    sur_maps = {m: odds_ratios(sdf, outcome_col).set_index(key_cols)["odds_ratio"]
                for m, sdf in surrogates.items()}

    def _family_p(ref_df, subgroups):
        raw = [
            _subgroup_fisher_or_p(ref_df, key_cols,
                                  k if isinstance(k, tuple) else (k,), outcome_col)[1]
            for k in subgroups
        ]
        return dict(zip(list(subgroups), raw))

    def _counts(ref_map, raw_p, sur_map):
        common = ref_map.index.intersection(sur_map.index)
        n_family_sig = sum(int(p < alpha) for p in raw_p.values())
        fl = or_flips(ref_map.loc[common].to_numpy(), sur_map.loc[common].to_numpy())
        flipped = [common[i] for i in np.where(fl)[0]]
        n_sig = sum(int(raw_p.get(k, 1.0) < alpha) for k in flipped)
        return float(len(common)), float(n_family_sig), float(len(flipped)), float(n_sig)

    # Point estimate on the full data (every subgroup in the original is tested).
    orig_raw_p = _family_p(df_original, list(orig_map.index))
    point = {m: _counts(orig_map, orig_raw_p, sur_maps[m]) for m in surrogates}

    # Bootstrap: the p-values depend only on the resampled reference, so compute
    # them once per resample and reuse across surrogates.
    count_keys = ("compared", "family_sig", "flips", "sig")
    boot = {m: {k: np.empty(len(boot_idx)) for k in count_keys} for m in surrogates}
    for b, idx in enumerate(boot_idx):
        ref_df = df_original.iloc[idx]
        ref_map = odds_ratios(ref_df, outcome_col).set_index(key_cols)["odds_ratio"]
        raw_p = _family_p(ref_df, list(ref_map.index))
        for m in surrogates:
            for k, v in zip(count_keys, _counts(ref_map, raw_p, sur_maps[m])):
                boot[m][k][b] = v

    summary: Dict[str, Dict[str, str]] = {}
    detail: Dict[str, pd.DataFrame] = {}
    for method in surrogates:
        p_compared, p_family_sig, p_flips, p_sig = point[method]
        summary[method] = {
            f"Significant in original, all subgroups, p<{alpha} (95% CI)":
                fmt_ci(_ci(p_family_sig, boot[method]["family_sig"]), prec=1, point_kind=point_kind),
            "OR sign flips (95% CI)":
                fmt_ci(_ci(p_flips, boot[method]["flips"]), prec=1, point_kind=point_kind),
            f"Significant flips, p<{alpha} (95% CI)":
                fmt_ci(_ci(p_sig, boot[method]["sig"]), prec=1, point_kind=point_kind),
        }

        # Point-estimate detail: the full-data flipped subgroups and their p-values.
        sur_map = sur_maps[method]
        common = orig_map.index.intersection(sur_map.index)
        flips = or_flips(orig_map.loc[common].to_numpy(), sur_map.loc[common].to_numpy())
        rows = []
        for i in np.where(flips)[0]:
            key = common[i]
            key_t = key if isinstance(key, tuple) else (key,)
            or_o, p_o = _subgroup_fisher_or_p(df_original, key_cols, key_t, outcome_col)
            or_s, p_s = _subgroup_fisher_or_p(surrogates[method], key_cols, key_t, outcome_col)
            row = dict(zip(key_cols, key_t))
            row.update({
                "Original OR": or_o,
                "Original p": p_o,
                f"{method} OR": or_s,
                f"{method} p": p_s,
                f"sig in original (p<{alpha})": p_o < alpha,
            })
            rows.append(row)
        det = pd.DataFrame(rows)
        if len(det):
            det = det.sort_values("Original p").reset_index(drop=True)
        detail[method] = det

    summary_df = pd.DataFrame(summary).T.reindex(list(surrogates))
    return summary_df, detail


def compare_classification(
    train_df, test_df, reconstructed_train, synthetic_train_df,
    outcome_col, cat_cols, demographic_cols=None,
    n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED, boot_idx=None,
):
    cells_df = reconstructed_train["cells_df"]
    cube_train = counts_to_line_level(
        posterior_mean_counts(reconstructed_train), cells_df, demographic_cols
    )
    train_sets = {
        "Original": train_df,
        "Cube": cube_train,
        "CTGAN": synthetic_train_df,
    }
    preds: Dict[str, np.ndarray] = {}
    y_true = None
    for name, tr in train_sets.items():
        y_true, preds[name] = train_and_predict(tr, test_df, outcome_col, cat_cols)

    res = bootstrap_classification(y_true, preds, n_boot=n_boot, seed=seed, boot_idx=boot_idx)
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    out = {name: {m: fmt_ci(res[name][m], point_kind="boot_mean") for m in metrics}
           for name in train_sets}
    return pd.DataFrame(out).reindex(metrics)
