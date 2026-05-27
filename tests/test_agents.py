"""
RL agentlari uchun unit testlar.

Ishga tushirish:
    $ pytest tests/test_agents.py -v
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from src.agents import (
    StaticAgent,
    AGENT_REGISTRY,
    create_agent,
)
from src.environment import SingleAgentPricingEnv


# ═══════════════════════════════════════════════════════════════
# StaticAgent testlari (SB3 dan mustaqil)
# ═══════════════════════════════════════════════════════════════

class TestStaticAgent:
    """StaticAgent uchun testlar — SB3 talab qilmaydi."""

    @pytest.fixture
    def env(self) -> SingleAgentPricingEnv:
        return SingleAgentPricingEnv(seed=42)

    @pytest.fixture
    def agent(self, env) -> StaticAgent:
        return StaticAgent(env)

    def test_static_agent_creation(self, agent):
        """StaticAgent yaratilsin."""
        assert agent.name == "Static"
        assert agent.is_trained is True
        assert agent.fixed_action == 3  # 0% o'zgarish

    def test_static_agent_always_same_action(self, agent):
        """Static agent doimo bir xil action qaytarishi kerak."""
        obs = np.array([0.5] * 5, dtype=np.float32)
        actions = [agent.get_action(obs) for _ in range(20)]
        assert all(a == agent.fixed_action for a in actions)

    def test_static_agent_evaluation(self, agent):
        """evaluate() to'g'ri formatda natija qaytaradi."""
        results = agent.evaluate(n_episodes=3)
        assert "mean_reward" in results
        assert "std_reward" in results
        assert "all_rewards" in results
        assert "mean_revenue" in results
        assert "mean_loyalty" in results
        assert "mean_churn" in results
        assert len(results["all_rewards"]) == 3

    def test_static_train_no_op(self, agent):
        """Static agent trening qilinmaydi (no-op)."""
        result = agent.train(timesteps=100)
        assert result["training_time"] == 0.0
        assert result["n_episodes"] == 0

    def test_custom_fixed_action(self, env):
        """Maxsus fixed_action ishlatilishi."""
        agent = StaticAgent(env, fixed_action=5)
        obs = np.zeros(5, dtype=np.float32)
        assert agent.get_action(obs) == 5


# ═══════════════════════════════════════════════════════════════
# Agent factory testlari
# ═══════════════════════════════════════════════════════════════

class TestAgentFactory:
    """Agent registry va factory testlari."""

    def test_registry_has_all_agents(self):
        """Barcha 4 ta agent registry da bo'lishi kerak."""
        assert "DQN" in AGENT_REGISTRY
        assert "PPO" in AGENT_REGISTRY
        assert "SAC" in AGENT_REGISTRY
        assert "Static" in AGENT_REGISTRY

    def test_create_static_agent(self):
        """Static agent yaratilsin."""
        env = SingleAgentPricingEnv(seed=42)
        agent = create_agent("Static", env)
        assert agent.name == "Static"

    def test_invalid_agent_name_raises(self):
        """Noto'g'ri nom bilan ValueError chiqarishi kerak."""
        env = SingleAgentPricingEnv(seed=42)
        with pytest.raises(ValueError):
            create_agent("UnknownAgent", env)


# ═══════════════════════════════════════════════════════════════
# Evaluation moduli testlari
# ═══════════════════════════════════════════════════════════════

class TestEvaluation:
    """src/evaluation.py funksiyalari uchun testlar."""

    def test_cohens_d_calculation(self):
        """Cohen's d to'g'ri hisoblanishi."""
        from src.evaluation import cohens_d
        # Aniq farqlangan ikki guruh — katta effekt size
        group1 = list(np.random.default_rng(42).normal(10, 1, 30))
        group2 = list(np.random.default_rng(43).normal(5, 1, 30))
        d = cohens_d(group1, group2)
        assert abs(d) > 2.0  # Juda katta effekt

    def test_effect_size_interpretation(self):
        """interpret_effect_size to'g'ri label qaytaradi."""
        from src.evaluation import interpret_effect_size
        assert interpret_effect_size(0.1) == "kichik"
        assert interpret_effect_size(0.3) == "o'rtacha"
        assert interpret_effect_size(0.6) == "katta"
        assert interpret_effect_size(1.0) == "juda katta"

    def test_shapiro_wilk_returns_proper_format(self):
        """Shapiro-Wilk to'g'ri formatda natija qaytaradi."""
        try:
            from src.evaluation import shapiro_wilk_test
        except ImportError:
            pytest.skip("scipy mavjud emas")

        data = {
            "A": list(np.random.default_rng(42).normal(0, 1, 30)),
            "B": list(np.random.default_rng(43).normal(0, 1, 30)),
        }
        result = shapiro_wilk_test(data)
        assert "A" in result
        assert "B" in result
        assert "statistic" in result["A"]
        assert "p_value" in result["A"]
        assert "is_normal" in result["A"]

    def test_pareto_analysis(self):
        """Pareto tahlili to'g'ri ishlashi."""
        from src.evaluation import pareto_analysis
        revenue = {"A": [10, 11, 12], "B": [5, 5, 5], "C": [8, 9, 10]}
        loyalty = {"A": [0.9, 0.85, 0.8], "B": [0.4, 0.4, 0.4], "C": [0.7, 0.7, 0.7]}
        result = pareto_analysis(revenue, loyalty)
        assert "points" in result
        assert "pareto_front" in result
        assert "hypervolumes" in result
        # A barcha o'lchovlar bo'yicha B dan ustun — B Pareto front da emas
        front_algos = [p["algorithm"] for p in result["pareto_front"]]
        assert "A" in front_algos
        assert "B" not in front_algos

    def test_compute_all_metrics(self):
        """compute_all_metrics barcha kalitlarni qaytaradi."""
        from src.evaluation import compute_all_metrics
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        prices = [10.0, 10.5, 11.0, 10.8, 11.2]
        loyalties = [0.5, 0.55, 0.6, 0.58, 0.62]
        result = compute_all_metrics(rewards, prices, loyalties)
        expected_keys = [
            "total_revenue", "mean_revenue", "std_revenue",
            "final_loyalty", "mean_loyalty", "churn_rate",
            "price_volatility", "avg_price", "min_price", "max_price",
        ]
        for key in expected_keys:
            assert key in result


# ═══════════════════════════════════════════════════════════════
# Utils moduli testlari
# ═══════════════════════════════════════════════════════════════

class TestUtils:
    """src/utils.py funksiyalari uchun testlar."""

    def test_load_precomputed_results(self):
        """Precomputed results yuklash."""
        from src.utils import load_precomputed_results
        results = load_precomputed_results()
        assert "experiments" in results
        assert "metadata" in results
        for algo in ["PPO", "DQN", "SAC", "Static"]:
            assert algo in results["experiments"]
            assert "revenues" in results["experiments"][algo]
            assert len(results["experiments"][algo]["revenues"]) == 30

    def test_color_palette_has_all_keys(self):
        """Rang palitra kerakli kalitlarga ega."""
        from src.utils import get_color_palette
        palette = get_color_palette()
        for key in ["PPO", "DQN", "SAC", "Static", "primary", "secondary", "accent"]:
            assert key in palette
            assert palette[key].startswith("#")

    def test_custom_css_returns_string(self):
        """create_custom_css str qaytaradi."""
        from src.utils import create_custom_css
        css = create_custom_css()
        assert isinstance(css, str)
        assert "<style>" in css
        assert "metric-card" in css

    def test_generate_sample_csv(self):
        """Sample CSV generatsiya qilinadi va to'g'ri ustunlarga ega."""
        from src.utils import generate_sample_csv
        df = generate_sample_csv(n_rows=50, seed=42)
        assert len(df) == 50
        required_cols = ["Invoice", "StockCode", "Description", "Quantity",
                         "InvoiceDate", "Price", "Customer ID", "Country"]
        for col in required_cols:
            assert col in df.columns
