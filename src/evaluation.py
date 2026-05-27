"""
Statistik baholash va metrikalar moduli.

Quyidagi statistik testlar va metrikalarni o'z ichiga oladi:
    * Shapiro-Wilk     — normallik testi
    * t-test           — parametrik taqqoslash
    * Mann-Whitney U   — noparametrik taqqoslash
    * Cohen's d        — effect size
    * Pareto front + HV — multi-objective tahlil
    * Sensitivity      — alpha/beta sezgirligi
    * Business metrics — revenue, loyalty, churn, volatility
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None  # type: ignore


# ═══════════════════════════════════════════════════════════════
# Eksperiment ishga tushirish
# ═══════════════════════════════════════════════════════════════

def run_experiments(
    agents_dict: Dict[str, Any],
    env_factory: Callable[[], Any],
    n_runs: int = 30,
    progress_fn: Optional[Callable[[float, str], None]] = None,
) -> pd.DataFrame:
    """
    Barcha agentlar uchun n_runs ta mustaqil baholash o'tkazish.

    Parameters
    ----------
    agents_dict : dict
        {agent_name: BaseAgent obyekti}
    env_factory : callable
        Yangi muhit yaratuvchi funksiya (har baholash uchun)
    n_runs : int
        Har agent uchun mustaqil run lar soni (default: 30)
    progress_fn : callable, optional
        Streamlit progress bar uchun callback(fraction, message)

    Returns
    -------
    pandas.DataFrame
        Ustunlar: algorithm, run, revenue, loyalty, churn,
                  price_volatility, avg_price, total_reward
    """
    records: List[Dict[str, Any]] = []
    n_total = len(agents_dict) * n_runs
    counter = 0

    for algo_name, agent in agents_dict.items():
        for run_idx in range(n_runs):
            # Har baholash uchun yangi muhit
            agent.env = env_factory()
            # Bir epizodli baholash bu yerda
            res = agent.evaluate(n_episodes=1, deterministic=True)
            records.append({
                "algorithm": algo_name,
                "run": run_idx,
                "revenue": res["all_revenues"][0],
                "loyalty": res["all_loyalties"][0],
                "churn": res["all_churns"][0],
                "price_volatility": res["all_volatilities"][0],
                "avg_price": res["all_revenues"][0] / max(100, 1),  # taxminiy
                "total_reward": res["all_rewards"][0],
            })
            counter += 1
            if progress_fn:
                progress_fn(counter / n_total, f"{algo_name} run {run_idx + 1}/{n_runs}")

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════
# Normallik testi (Shapiro-Wilk)
# ═══════════════════════════════════════════════════════════════

def shapiro_wilk_test(
    data_dict: Dict[str, List[float]],
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """
    Har algoritm uchun Shapiro-Wilk normallik testi.

    H0: ma'lumotlar normal taqsimotga ega
    H0 rad qilinmasligi → t-test qo'llanishi mumkin

    Parameters
    ----------
    data_dict : dict
        {algorithm_name: [qiymatlar ro'yxati]}
    alpha : float
        Ahamiyatlilik darajasi

    Returns
    -------
    dict
        {algo: {statistic, p_value, is_normal}}
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy kerak: pip install scipy")

    results: Dict[str, Dict[str, Any]] = {}
    for algo, data in data_dict.items():
        arr = np.asarray(data, dtype=float)
        if len(arr) < 3:
            results[algo] = {"statistic": float("nan"), "p_value": float("nan"), "is_normal": False}
            continue
        stat, p_val = stats.shapiro(arr)
        results[algo] = {
            "statistic": float(stat),
            "p_value": float(p_val),
            "is_normal": bool(p_val > alpha),
            "n": int(len(arr)),
        }
    return results


# ═══════════════════════════════════════════════════════════════
# Cohen's d (effect size)
# ═══════════════════════════════════════════════════════════════

def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Cohen's d effect size hisoblash.

    d = (mean1 - mean2) / pooled_std

    Interpretatsiya:
        |d| < 0.2  — kichik effekt
        |d| < 0.5  — o'rtacha effekt
        |d| < 0.8  — katta effekt
        |d| >= 0.8 — juda katta effekt
    """
    a = np.asarray(group1, dtype=float)
    b = np.asarray(group2, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")

    s1_sq = float(np.var(a, ddof=1))
    s2_sq = float(np.var(b, ddof=1))
    pooled_sd = float(np.sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2)))
    if pooled_sd == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def interpret_effect_size(d: float) -> str:
    """Cohen's d qiymatini matn ko'rinishida tushuntirish."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "kichik"
    elif abs_d < 0.5:
        return "o'rtacha"
    elif abs_d < 0.8:
        return "katta"
    elif abs_d < 1.5:
        return "juda katta"
    else:
        return "ultra katta"


# ═══════════════════════════════════════════════════════════════
# Pairwise t-test
# ═══════════════════════════════════════════════════════════════

def pairwise_ttest(
    data_dict: Dict[str, List[float]],
    metric: str = "value",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Barcha algoritm juftliklari uchun independent t-test.

    Welch's t-test (variantlar teng emas deb taxmin qilinadi).

    Returns
    -------
    pandas.DataFrame
        Ustunlar: comparison, t_stat, p_value, cohens_d,
                  effect_size, significant, mean_diff
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy kerak")

    records: List[Dict[str, Any]] = []
    keys = list(data_dict.keys())

    for a, b in combinations(keys, 2):
        arr_a = np.asarray(data_dict[a], dtype=float)
        arr_b = np.asarray(data_dict[b], dtype=float)
        if len(arr_a) < 2 or len(arr_b) < 2:
            continue

        t_stat, p_val = stats.ttest_ind(arr_a, arr_b, equal_var=False)
        d = cohens_d(arr_a.tolist(), arr_b.tolist())
        records.append({
            "comparison": f"{a} vs {b}",
            "metric": metric,
            "mean_a": float(np.mean(arr_a)),
            "mean_b": float(np.mean(arr_b)),
            "mean_diff": float(np.mean(arr_a) - np.mean(arr_b)),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(d),
            "effect_size": interpret_effect_size(d),
            "significant": bool(p_val < alpha),
            "significance_label": _significance_stars(p_val),
        })

    return pd.DataFrame(records)


def _significance_stars(p_value: float) -> str:
    """p-value bo'yicha yulduzcha label qaytarish."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return "n.s."


# ═══════════════════════════════════════════════════════════════
# Mann-Whitney U (noparametrik)
# ═══════════════════════════════════════════════════════════════

def mann_whitney_test(
    data_dict: Dict[str, List[float]],
    metric: str = "value",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Noparametrik Mann-Whitney U testi.

    Distribution-free, normallikni talab qilmaydi.
    t-test natijalarini tasdiqlash uchun ishlatiladi.

    Returns
    -------
    pandas.DataFrame
        Ustunlar: comparison, u_stat, p_value, significant
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy kerak")

    records: List[Dict[str, Any]] = []
    keys = list(data_dict.keys())

    for a, b in combinations(keys, 2):
        arr_a = np.asarray(data_dict[a], dtype=float)
        arr_b = np.asarray(data_dict[b], dtype=float)
        if len(arr_a) < 2 or len(arr_b) < 2:
            continue

        u_stat, p_val = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
        records.append({
            "comparison": f"{a} vs {b}",
            "metric": metric,
            "median_a": float(np.median(arr_a)),
            "median_b": float(np.median(arr_b)),
            "u_stat": float(u_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < alpha),
            "significance_label": _significance_stars(p_val),
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════
# Pareto tahlili va Hyper-Volume
# ═══════════════════════════════════════════════════════════════

def pareto_analysis(
    revenue_data: Dict[str, List[float]],
    loyalty_data: Dict[str, List[float]],
) -> Dict[str, Any]:
    """
    Pareto frontasi va Hyper-Volume hisoblash.

    Pareto front — yaxshi yechimlar to'plami (boshqalar tomonidan ustun emas).
    Hyper-Volume — Pareto front sifati o'lchovi.

    Parameters
    ----------
    revenue_data : dict
        {algo: [revenue qiymatlari]}
    loyalty_data : dict
        {algo: [loyalty qiymatlari]}

    Returns
    -------
    dict
        {points: list, pareto_front: list, hypervolumes: dict}
    """
    points: List[Tuple[str, float, float]] = []
    for algo in revenue_data.keys():
        mean_rev = float(np.mean(revenue_data[algo]))
        mean_loy = float(np.mean(loyalty_data[algo]))
        points.append((algo, mean_rev, mean_loy))

    # Pareto frontasi (maksimallashtirish — har ikkala metrika)
    pareto_front: List[Tuple[str, float, float]] = []
    for i, (algo_i, rev_i, loy_i) in enumerate(points):
        is_dominated = False
        for j, (algo_j, rev_j, loy_j) in enumerate(points):
            if i == j:
                continue
            if rev_j >= rev_i and loy_j >= loy_i and (rev_j > rev_i or loy_j > loy_i):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append((algo_i, rev_i, loy_i))

    # Hyper-Volume hisoblash — normalize qilingan
    max_rev = max(p[1] for p in points) if points else 1.0
    max_loy = max(p[2] for p in points) if points else 1.0
    min_rev = min(p[1] for p in points) if points else 0.0
    min_loy = min(p[2] for p in points) if points else 0.0

    rev_range = max(max_rev - min_rev, 1e-9)
    loy_range = max(max_loy - min_loy, 1e-9)

    hypervolumes: Dict[str, float] = {}
    for algo, rev, loy in points:
        norm_rev = (rev - min_rev) / rev_range
        norm_loy = (loy - min_loy) / loy_range
        # Soddalashtirilgan HV: o'rtacha (referensiya = (0, 0))
        hv = (norm_rev + norm_loy) / 2.0
        hypervolumes[algo] = float(round(hv, 3))

    return {
        "points": [{"algorithm": a, "revenue": r, "loyalty": l} for a, r, l in points],
        "pareto_front": [{"algorithm": a, "revenue": r, "loyalty": l} for a, r, l in pareto_front],
        "hypervolumes": hypervolumes,
    }


# ═══════════════════════════════════════════════════════════════
# Sezgirlik tahlili (Sensitivity Analysis)
# ═══════════════════════════════════════════════════════════════

def sensitivity_analysis(
    env_class: type,
    agent_class: type,
    alpha_range: List[float],
    beta_range: List[float],
    n_episodes: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Alpha va beta parametrlari sezgirligini tahlil qilish.

    Grid search: har α/β juftligi uchun reward o'rtachasi.

    Parameters
    ----------
    env_class : type
        Muhit klassi (SingleAgentPricingEnv)
    agent_class : type
        Agent klassi (StaticAgent)
    alpha_range : list of float
        Alpha qiymatlari to'plami
    beta_range : list of float
        Beta qiymatlari to'plami
    n_episodes : int
        Har grid nuqtasi uchun epizodlar soni
    seed : int
        Reproducibility

    Returns
    -------
    dict
        {alpha_values, beta_values, reward_matrix (2D ndarray)}
    """
    reward_matrix = np.zeros((len(alpha_range), len(beta_range)))

    for i, alpha in enumerate(alpha_range):
        for j, beta in enumerate(beta_range):
            env = env_class(alpha=alpha, beta=beta, seed=seed)
            agent = agent_class(env)
            res = agent.evaluate(n_episodes=n_episodes, deterministic=True)
            reward_matrix[i, j] = res["mean_reward"]

    return {
        "alpha_values": alpha_range,
        "beta_values": beta_range,
        "reward_matrix": reward_matrix,
    }


# ═══════════════════════════════════════════════════════════════
# Biznes metrikalar
# ═══════════════════════════════════════════════════════════════

def compute_all_metrics(
    rewards: List[float],
    prices: List[float],
    loyalties: List[float],
) -> Dict[str, float]:
    """
    Bir epizod uchun barcha biznes metrikalarni hisoblash.

    Metrikalar:
        * total_revenue, mean_revenue, std_revenue
        * final_loyalty, mean_loyalty
        * churn_rate (loyalty < 0.1 bo'lgan qadamlar foizi)
        * price_volatility (std(price))
        * avg_price

    Parameters
    ----------
    rewards : list
        Epizod davomidagi reward lar
    prices : list
        Epizod davomidagi narxlar
    loyalties : list
        Epizod davomidagi sodiqlik qiymatlari

    Returns
    -------
    dict
        Barcha metrikalar
    """
    rewards_arr = np.asarray(rewards, dtype=float)
    prices_arr = np.asarray(prices, dtype=float)
    loyalties_arr = np.asarray(loyalties, dtype=float)

    return {
        "total_revenue": float(rewards_arr.sum()),
        "mean_revenue": float(rewards_arr.mean()) if len(rewards_arr) > 0 else 0.0,
        "std_revenue": float(rewards_arr.std(ddof=1)) if len(rewards_arr) > 1 else 0.0,
        "final_loyalty": float(loyalties_arr[-1]) if len(loyalties_arr) > 0 else 0.0,
        "mean_loyalty": float(loyalties_arr.mean()) if len(loyalties_arr) > 0 else 0.0,
        "churn_rate": float(np.mean(loyalties_arr < 0.1)) if len(loyalties_arr) > 0 else 0.0,
        "price_volatility": float(prices_arr.std(ddof=1)) if len(prices_arr) > 1 else 0.0,
        "avg_price": float(prices_arr.mean()) if len(prices_arr) > 0 else 0.0,
        "min_price": float(prices_arr.min()) if len(prices_arr) > 0 else 0.0,
        "max_price": float(prices_arr.max()) if len(prices_arr) > 0 else 0.0,
    }


def compute_improvement(
    treatment_value: float,
    baseline_value: float,
    higher_is_better: bool = True,
) -> Dict[str, float]:
    """
    Baseline ga nisbatan yaxshilanish foizini hisoblash.

    Returns
    -------
    dict
        {absolute_delta, relative_improvement_pct, is_improved}
    """
    abs_delta = treatment_value - baseline_value
    if abs(baseline_value) < 1e-9:
        rel_pct = 0.0
    else:
        rel_pct = (abs_delta / abs(baseline_value)) * 100.0

    if not higher_is_better:
        rel_pct = -rel_pct
        abs_delta = -abs_delta

    return {
        "absolute_delta": float(abs_delta),
        "relative_improvement_pct": float(rel_pct),
        "is_improved": bool(rel_pct > 0),
    }


# ═══════════════════════════════════════════════════════════════
# Konvergentsiya tahlili
# ═══════════════════════════════════════════════════════════════

def detect_convergence_episode(
    learning_curve: List[float],
    window: int = 20,
    threshold: float = 0.05,
) -> int:
    """
    Konvergentsiya nuqtasini aniqlash.

    Reward o'zgarish koeffitsienti (CV) belgilangan chegaradan past tushgan
    birinchi epizodni qaytaradi.

    Parameters
    ----------
    learning_curve : list
        Epizod reward lari
    window : int
        Slayd oyna kengligi
    threshold : float
        Maks. ruxsat etilgan CV

    Returns
    -------
    int
        Konvergentsiya epizodi (yoki -1 agar topilmasa)
    """
    arr = np.asarray(learning_curve, dtype=float)
    if len(arr) < window:
        return -1

    for i in range(window, len(arr)):
        sub = arr[i - window:i]
        if abs(np.mean(sub)) < 1e-9:
            continue
        cv = float(np.std(sub) / abs(np.mean(sub)))
        if cv < threshold:
            return i

    return -1
