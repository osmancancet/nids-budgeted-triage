"""Core utilities: models, calibration, conformal triage and metrics.

Every number reported in the manuscript is produced by run_experiments.py,
which imports this module.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb


# ----------------------------------------------------------------- features
class LogScaler:
    """Signed log1p followed by standardisation (fit on training data only)."""

    def fit(self, X):
        self.sc = StandardScaler().fit(self._log(X))
        return self

    @staticmethod
    def _log(X):
        X = np.asarray(X, dtype=np.float64)
        return np.sign(X) * np.log1p(np.abs(X))

    def transform(self, X):
        Z = self.sc.transform(self._log(X))
        return np.clip(Z, -50, 50)


# ------------------------------------------------------------------- models
MODEL_NAMES = ["LGBM", "RF", "MLP", "LR"]
NJOBS = int(__import__("os").environ.get("NJOBS", "4"))  # threads per model


def make_model(name, seed):
    if name == "LGBM":
        return lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  random_state=seed, n_jobs=NJOBS, verbose=-1)
    if name == "RF":
        return RandomForestClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=NJOBS,
                                      random_state=seed)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True,
                             max_iter=60, batch_size=512, random_state=seed)
    if name == "LR":
        return LogisticRegression(max_iter=2000, C=1.0)
    raise ValueError(name)


class Detector:
    """Binary detector: scaler + classifier returning P(attack)."""

    def __init__(self, name, seed):
        self.name, self.seed = name, seed

    def fit(self, X, y):
        self.scaler = LogScaler().fit(X)
        self.clf = make_model(self.name, self.seed).fit(self.scaler.transform(X), y)
        return self

    def score(self, X):
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]


# -------------------------------------------------------------- calibration
class Isotonic:
    def fit(self, s, y):
        self.ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s, y)
        return self

    def __call__(self, s):
        return self.ir.predict(s)


class Platt:
    def fit(self, s, y):
        z = _logit(s).reshape(-1, 1)
        self.lr = LogisticRegression(C=1e4, max_iter=1000).fit(z, y)
        return self

    def __call__(self, s):
        return self.lr.predict_proba(_logit(s).reshape(-1, 1))[:, 1]


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def ece(p, y, n_bins=15):
    """Expected calibration error of P(attack) with equal-width bins."""
    p, y = np.asarray(p), np.asarray(y)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return tot


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


# ------------------------------------------------------------------ novelty
class KNNNovelty:
    """Mean distance to the k nearest training flows (in LogScaler space)."""

    def __init__(self, k=5, n_ref=20000, seed=0):
        self.k, self.n_ref, self.seed = k, n_ref, seed

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(X), size=min(self.n_ref, len(X)), replace=False)
        self.scaler = LogScaler().fit(X)
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(self.scaler.transform(X[idx]))
        return self

    def score(self, X, batch=50000):
        Z = self.scaler.transform(X)
        out = np.empty(len(Z))
        for i in range(0, len(Z), batch):
            d, _ = self.nn.kneighbors(Z[i:i + batch])
            out[i:i + batch] = d.mean(axis=1)
        return out


# --------------------------------------------------------- conformal tools
def conformal_pvalues(cal_scores, test_scores):
    """Split-conformal p-values; larger nonconformity score = more unusual.

    p(x) = (1 + #{cal >= s(x)}) / (n + 1)
    """
    cal = np.sort(np.asarray(cal_scores))
    n = len(cal)
    ge = n - np.searchsorted(cal, test_scores, side="left")
    return (1.0 + ge) / (n + 1.0)


class ConformalTriage:
    """Budgeted triage of flows into an analyst review queue.

    Null hypothesis for a flow: "benign and typical of the training traffic".
    Two conformal p-values are computed against held-out *benign*
    calibration flows:
      p_det  -- from the detector score P(attack)            (known attacks)
      p_nov  -- from the kNN novelty score                    (unfamiliar traffic)
    and combined with a Bonferroni min: p = min(1, 2 * min(p_det, p_nov)).
    Flows with the smallest p are sent to the queue.  Under exchangeability of
    benign target and calibration flows the benign false-alarm rate of the rule
    p <= alpha is at most alpha.
    """

    def fit(self, det_cal_benign, nov_cal_benign):
        self.det_cal = np.asarray(det_cal_benign)
        self.nov_cal = np.asarray(nov_cal_benign)
        return self

    def pvalues(self, det, nov, mode="combined"):
        p_det = conformal_pvalues(self.det_cal, det)
        if mode == "detector":
            return p_det
        p_nov = conformal_pvalues(self.nov_cal, nov)
        if mode == "novelty":
            return p_nov
        if mode == "fisher":
            # Fisher's statistic -2 log(p_det p_nov) ~ chi2(4) under independence
            from scipy.stats import chi2
            return chi2.sf(-2.0 * (np.log(p_det) + np.log(p_nov)), df=4)
        return np.minimum(1.0, 2.0 * np.minimum(p_det, p_nov))


# --------------------------------------------------------------- metrics
def safe_auc(y, s):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, s)


def safe_ap(y, s):
    if len(np.unique(y)) < 2:
        return np.nan
    return average_precision_score(y, s)


def precision_at_prevalence(tpr, fpr, pi):
    """Precision a detector with (tpr, fpr) would reach at attack prevalence pi."""
    den = tpr * pi + fpr * (1 - pi)
    return np.nan if den == 0 else tpr * pi / den


def rates(flag, y):
    flag, y = np.asarray(flag, bool), np.asarray(y, int)
    tpr = flag[y == 1].mean() if (y == 1).any() else np.nan
    fpr = flag[y == 0].mean() if (y == 0).any() else np.nan
    return tpr, fpr


def queue_by_budget(priority, budget):
    """Boolean mask of the `budget` fraction of flows with highest priority."""
    k = int(np.floor(budget * len(priority)))
    mask = np.zeros(len(priority), bool)
    if k > 0:
        mask[np.argpartition(-priority, k - 1)[:k]] = True
    return mask
