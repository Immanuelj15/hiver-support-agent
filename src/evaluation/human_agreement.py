"""
src/evaluation/human_agreement.py

Computes agreement metrics between LLM Judge evaluations and human ground truth annotations.
Calculates Cohen's Quadratic Weighted Kappa (kappa_w), Spearman rank correlation,
and Pearson linear correlation.
"""

from typing import List, Dict, Any
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import cohen_kappa_score

def compute_judge_human_agreement(
    human_scores: List[float],
    judge_scores: List[float]
) -> Dict[str, Any]:
    """
    Computes statistical agreement metrics between human annotators and LLM judge:
    1. Quadratic Weighted Kappa (kappa_w): Standard for ordinal rating scales (1-5).
    2. Spearman Rank Correlation: Monotonic rank alignment.
    3. Pearson Correlation: Linear score alignment.
    4. Mean Absolute Error (MAE): Calibration gap.
    """
    assert len(human_scores) == len(judge_scores), "Scores list lengths must match."

    h = np.array(human_scores, dtype=np.float64)
    j = np.array(judge_scores, dtype=np.float64)

    # Discretize to nearest integer in [1, 5] for Cohen's Kappa
    h_int = np.clip(np.round(h), 1, 5).astype(int)
    j_int = np.clip(np.round(j), 1, 5).astype(int)

    kappa_quadratic = cohen_kappa_score(h_int, j_int, weights="quadratic")
    kappa_linear = cohen_kappa_score(h_int, j_int, weights="linear")

    spearman_corr, sp_val = spearmanr(h, j)
    pearson_corr, pe_val = pearsonr(h, j)
    mae = np.mean(np.abs(h - j))

    return {
        "sample_size": len(h),
        "cohen_kappa_quadratic": round(float(kappa_quadratic), 4),
        "cohen_kappa_linear": round(float(kappa_linear), 4),
        "spearman_rank_correlation": round(float(spearman_corr), 4),
        "spearman_p_value": float(sp_val),
        "pearson_correlation": round(float(pearson_corr), 4),
        "mean_absolute_error": round(float(mae), 4)
    }
