"""

  - GrubbsDetector      -> "Grubbs' test" slajd 
  - MahalanobisDetector -> "Multivariate data: Mahalanobis

Svaki detektor radi nad PRECISCENIM batch-om (numericki feature-i iz STAT_FEATURES).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


class GrubbsDetector:
    """Grubbs' test za detekciju outliera (predavanje 06).

    Iterativno uklanja najekstremniju vrednost dok G statistika
    prelazi kriticnu vrednost izvedenu iz t-raspodele.
    """

    def __init__(self, alpha=0.05, stat_features=None, log_features=None):
        self.alpha = alpha
        stat_features = stat_features or []
        log_features = log_features or []
        self._log_idx = [
            stat_features.index(f)
            for f in log_features
            if f in stat_features
        ]

    def _preprocess(self, batch):
        """log1p nad zakosenim kolonama; ostale ostaju netaknute."""
        X = np.asarray(batch, dtype=float).copy()
        if self._log_idx:
            cols = X[:, self._log_idx]
            X[:, self._log_idx] = np.log1p(np.clip(cols, 0, None))
        return X

    def _count_1d(self, x: np.ndarray, max_iter: int = 50) -> int:
        """Broj outliera u jednoj koloni, iterativnim Grubbs testom."""
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        removed = 0
        for _ in range(max_iter):
            n = x.size
            if n < 3:
                break
            s = x.std(ddof=1)
            if s == 0:
                break
            G = np.abs(x - x.mean()).max() / s
            # kriticna vrednost (dvostrani test)
            t_crit = stats.t.ppf(1 - self.alpha / (2 * n), n - 2)
            G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(
                t_crit**2 / (n - 2 + t_crit**2)
            )
            if G <= G_crit:
                break
            x = np.delete(x, np.abs(x - x.mean()).argmax())
            removed += 1
        return removed

    def score(self, batch: np.ndarray) -> dict:
        """batch: (n_uzoraka, n_feature). Ukupan broj outliera po feature-ima."""
        X = self._preprocess(batch)
        per_feature = [self._count_1d(X[:, j]) for j in range(X.shape[1])]
        return {"grubbs_outliers_total": int(sum(per_feature)),
                "grubbs_per_feature": per_feature}

class MahalanobisDetector:
    """Multivarijatni outlieri preko Mahalanobis distance.

    d^2 = (x - mu)^T * Sigma^-1 * (x - mu)
    Tacka je outlier ako d^2 > prag = hi-kvadrat kvantil sa p stepeni slobode.
    mu i Sigma se procenjuju nad CISTIM referentnim setom (ne nad batch-om).
    """

    def __init__(self, mean: np.ndarray, cov: np.ndarray, chi2_q: float = 0.975):
        self.mean = np.asarray(mean, dtype=float)
        cov = np.asarray(cov, dtype=float)
        # regularizacija za stabilan inverz
        cov = cov + np.eye(cov.shape[0]) * 1e-6
        self.inv_cov = np.linalg.pinv(cov)
        self.threshold = stats.chi2.ppf(chi2_q, df=self.mean.shape[0])

    def distances(self, batch: np.ndarray) -> np.ndarray:
        d = batch - self.mean
        return np.einsum("ij,jk,ik->i", d, self.inv_cov, d)

    def score(self, batch: np.ndarray) -> dict:
        d2 = self.distances(batch)
        outliers = int(np.sum(d2 > self.threshold))
        return {"mahalanobis_outlier_rate": float(outliers / max(len(batch), 1)),
                "mahalanobis_threshold": float(self.threshold),
                "mahalanobis_mean_d2": float(np.mean(d2))}
