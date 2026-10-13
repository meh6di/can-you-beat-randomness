"""Model zoo, fitted once per walk-forward fold.

Every model outputs an (n, 3) probability array in H, D, A order.

  lr       multinomial logistic regression on strength + market features + league
           dummies, standardised, with C tuned on the validation window
  blend    log-linear pooling  p ~ p_market^(1-w) * p_strength^w  where p_strength comes
           from a strength-only model (no market input). w in [0,1] is fitted on
           validation: w ~ 0 means the model adds nothing beyond the market.
  xgb      gradient-boosted trees (raw)
  xgb_cal  the same, recalibrated on the (later) validation window by a multinomial
           logistic regression on the log-probabilities (a full 3x3 map, L2 C=1; closer to
           matrix scaling than to vector scaling). Time-ordered, unlike 3-fold CV on the train set
  clim     class frequencies (reference only)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config as C
from common import OUTCOMES, OUTCOME_IDX

C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
W_GRID = np.linspace(0.0, 1.0, 21)
_XGB_LABEL = {"A": 0, "D": 1, "H": 2}


def log_loss(p, y_idx):
    return -np.mean(np.log(np.clip(p[np.arange(len(p)), y_idx], 1e-12, 1.0)))


def _dummies(df):
    return np.column_stack([(df["league"] == l).to_numpy(float) for l in C.LEAGUES])


class _LR:
    def __init__(self, cols, c):
        self.cols, self.c = cols, c

    def _X(self, df):
        return np.hstack([self.scaler.transform(df[self.cols].to_numpy(float)), _dummies(df)])

    def fit(self, df):
        self.scaler = StandardScaler().fit(df[self.cols].to_numpy(float))
        self.m = LogisticRegression(C=self.c, max_iter=3000).fit(self._X(df), df["FTR"])
        self.order = [list(self.m.classes_).index(o) for o in OUTCOMES]
        return self

    def predict(self, df):
        return self.m.predict_proba(self._X(df))[:, self.order]


def _tune_lr(train, val, cols):
    yv = val["FTR"].map(OUTCOME_IDX).to_numpy()
    best = None
    for c in C_GRID:
        m = _LR(cols, c).fit(train)
        ll = log_loss(m.predict(val), yv)
        if best is None or ll < best[0]:
            best = (ll, m, c)
    return best[1], best[2]


def blend(p_mkt, p_str, w):
    q = p_mkt ** (1 - w) * p_str ** w
    return q / q.sum(axis=1, keepdims=True)


def _xgb():
    return XGBClassifier(n_estimators=75, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, random_state=C.SEED, eval_metric="mlogloss")


def _xgb_X(df, cols):
    return np.hstack([df[cols].to_numpy(float), _dummies(df)])


def fit_predict_fold(train, val, test):
    """Return ({model: test probs}, meta). train < val < test in time."""
    cols_all = C.STRENGTH_FEATURES + C.MARKET_FEATURES
    y_val = val["FTR"].map(OUTCOME_IDX).to_numpy()
    out, meta = {}, {}

    lr, meta["lr_C"] = _tune_lr(train, val, cols_all)
    out["lr"] = lr.predict(test)

    strength, meta["strength_C"] = _tune_lr(train, val, C.STRENGTH_FEATURES)
    mk = ["market_prob_h", "market_prob_d", "market_prob_a"]
    pm_val, pm_test = val[mk].to_numpy(float), test[mk].to_numpy(float)
    ps_val, ps_test = strength.predict(val), strength.predict(test)
    lls = [log_loss(blend(pm_val, ps_val, w), y_val) for w in W_GRID]
    meta["blend_w"] = float(W_GRID[int(np.argmin(lls))])
    out["blend"] = blend(pm_test, ps_test, meta["blend_w"])

    y_tr = train["FTR"].map(_XGB_LABEL)
    xgb = _xgb().fit(_xgb_X(train, cols_all), y_tr)
    raw = lambda d: xgb.predict_proba(_xgb_X(d, cols_all))[:, [2, 1, 0]]   # A,D,H -> H,D,A
    p_val, out["xgb"] = raw(val), raw(test)
    z = lambda p: np.log(np.clip(p, 1e-6, 1.0))
    cal = LogisticRegression(C=1.0, max_iter=2000).fit(z(p_val), val["FTR"])
    order = [list(cal.classes_).index(o) for o in OUTCOMES]
    out["xgb_cal"] = cal.predict_proba(z(out["xgb"]))[:, order]

    freq = np.concatenate([train["FTR"].to_numpy(), val["FTR"].to_numpy()])
    out["clim"] = np.tile([(freq == o).mean() for o in OUTCOMES], (len(test), 1))
    return out, meta
