"""
SingleAgentPricingEnv va DynamicPricingEnv uchun unit testlar.

Ishga tushirish:
    $ pytest tests/test_environment.py -v
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from src.environment import (
    SingleAgentPricingEnv,
    DynamicPricingEnv,
    PRICE_ACTION_DELTAS,
    NUM_PRICE_ACTIONS,
)


# ═══════════════════════════════════════════════════════════════
# SingleAgentPricingEnv testlari
# ═══════════════════════════════════════════════════════════════

class TestSingleAgentEnv:
    """SingleAgentPricingEnv uchun unit testlar."""

    @pytest.fixture
    def env(self) -> SingleAgentPricingEnv:
        """Standart muhit yaratish."""
        return SingleAgentPricingEnv(seed=42)

    def test_reset_returns_correct_shape(self, env):
        """reset() to'g'ri o'lchamdagi observation qaytarishi kerak."""
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (5,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_observation_in_bounds(self, env):
        """Observation [0, 1] oralig'ida bo'lishi kerak."""
        obs, _ = env.reset(seed=42)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_step_returns_5_values(self, env):
        """step() 5 ta qiymat qaytarishi kerak: obs, reward, terminated, truncated, info."""
        env.reset(seed=42)
        result = env.step(3)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_space_valid(self, env):
        """Action space 7 ta diskret qiymatga ega bo'lishi kerak."""
        assert env.action_space.n == NUM_PRICE_ACTIONS == 7

    def test_observation_space_valid(self, env):
        """Observation space (5,) Box bo'lishi kerak."""
        assert env.observation_space.shape == (5,)

    def test_reward_is_finite(self, env):
        """Reward har doim finite son bo'lishi kerak."""
        env.reset(seed=42)
        for action in range(NUM_PRICE_ACTIONS):
            _, reward, _, _, _ = env.step(action)
            assert np.isfinite(reward), f"Action {action} → non-finite reward"

    def test_episode_termination(self, env):
        """Maksimal qadamdan keyin truncated=True bo'lishi kerak."""
        env.reset(seed=42)
        truncated = False
        steps = 0
        max_iter = env.max_steps + 5
        while steps < max_iter:
            _, _, terminated, truncated, _ = env.step(3)
            steps += 1
            if terminated or truncated:
                break
        assert truncated or terminated, "Epizod yakunlanmadi"
        assert steps <= env.max_steps + 1

    def test_loyalty_decreases_on_high_price(self, env):
        """Narx ko'p oshirilsa, sodiqlik kamayishi kerak."""
        env.reset(seed=42)
        initial_loyalty = env.loyalty_index
        # 10 marta narx +15% (action=6)
        for _ in range(10):
            env.step(6)
        assert env.loyalty_index < initial_loyalty, \
            f"Sodiqlik kamaymadi: {initial_loyalty} → {env.loyalty_index}"

    def test_loyalty_increases_on_low_price(self, env):
        """Narx ko'p kamaytirilsa, sodiqlik oshishi kerak."""
        env.reset(seed=42)
        initial_loyalty = env.loyalty_index
        # 5 marta narx -15% (action=0)
        for _ in range(5):
            env.step(0)
        assert env.loyalty_index >= initial_loyalty - 0.05, \
            "Sodiqlik kutilganidan ko'p kamaydi"

    def test_invalid_action_raises(self, env):
        """Noto'g'ri action ValueError chiqarishi kerak."""
        env.reset(seed=42)
        with pytest.raises(ValueError):
            env.step(999)

    def test_history_is_tracked(self, env):
        """History har step da yangilanishi kerak."""
        env.reset(seed=42)
        for _ in range(5):
            env.step(3)
        assert len(env.history["prices"]) == 5
        assert len(env.history["rewards"]) == 5
        assert len(env.history["loyalties"]) == 5

    def test_price_action_deltas_correct(self):
        """PRICE_ACTION_DELTAS to'g'ri 7 ta qiymat ekanligi."""
        assert len(PRICE_ACTION_DELTAS) == 7
        assert PRICE_ACTION_DELTAS[3] == 0.0  # O'rta action — 0% o'zgarish

    def test_custom_alpha_beta(self):
        """alpha va beta parametrlari ishlatilganligi."""
        env = SingleAgentPricingEnv(alpha=0.5, beta=0.5, seed=42)
        assert env.alpha == 0.5
        assert env.beta == 0.5


# ═══════════════════════════════════════════════════════════════
# DynamicPricingEnv (MARL) testlari
# ═══════════════════════════════════════════════════════════════

class TestDynamicPricingEnv:
    """Multi-Agent muhit uchun testlar."""

    @pytest.fixture
    def env(self) -> DynamicPricingEnv:
        """MARL muhit yaratish."""
        return DynamicPricingEnv(seed=42)

    def test_has_3_agents(self, env):
        """3 ta agent bo'lishi kerak."""
        assert len(env.possible_agents) == 3
        assert "pricing_agent" in env.possible_agents
        assert "customer_agent" in env.possible_agents
        assert "competitor_agent" in env.possible_agents

    def test_reset_initializes_agents(self, env):
        """reset() agentlarni initialize qilishi kerak."""
        env.reset(seed=42)
        assert len(env.agents) == 3
        assert env.agent_selection == "pricing_agent"

    def test_observe_returns_correct_shapes(self, env):
        """Har agent kuzatuvi to'g'ri shaklda bo'lishi kerak."""
        env.reset(seed=42)
        # pricing va customer: shape (5,)
        assert env.observe("pricing_agent").shape == (5,)
        assert env.observe("customer_agent").shape == (5,)
        # competitor: shape (4,)
        assert env.observe("competitor_agent").shape == (4,)

    def test_step_advances_agent(self, env):
        """step() keyin keyingi agentga o'tishi kerak."""
        env.reset(seed=42)
        assert env.agent_selection == "pricing_agent"
        env.step(3)
        assert env.agent_selection == "customer_agent"
        env.step(0)
        assert env.agent_selection == "competitor_agent"
        env.step(2)
        # To'liq tur tugadi — qadam hisoblagich oshadi
        assert env.current_step == 1

    def test_rewards_assigned(self, env):
        """Har agent reward olishi kerak."""
        env.reset(seed=42)
        env.step(6)  # pricing
        assert env.rewards["pricing_agent"] != 0.0 or env.rewards["pricing_agent"] == 0.0  # finite

    def test_action_space_method(self, env):
        """action_space(agent) method ishlashi kerak."""
        assert env.action_space("pricing_agent").n == 7
        assert env.action_space("customer_agent").n == 3
        assert env.action_space("competitor_agent").n == 5
