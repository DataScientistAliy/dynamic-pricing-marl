"""
RL muhitlari moduli — Single-Agent va Multi-Agent.

Tarkib:
    * SingleAgentPricingEnv  — Gymnasium muhit (Stable-Baselines3 uchun)
    * DynamicPricingEnv       — PettingZoo AECEnv (3 ta agent)

Reward funksiyasi (har ikkala muhit uchun):
    R = α * revenue_norm + β * loyalty_delta * 100
    Default: α=0.7, β=0.3
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gym = None  # type: ignore
    spaces = None  # type: ignore


# ═══════════════════════════════════════════════════════════════
# Yordamchi: action diskret indeksini foiz o'zgartirishga aylantirish
# ═══════════════════════════════════════════════════════════════

# 7 ta discrete action: -15%, -10%, -5%, 0%, +5%, +10%, +15%
PRICE_ACTION_DELTAS: np.ndarray = np.array([-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15])
NUM_PRICE_ACTIONS: int = len(PRICE_ACTION_DELTAS)


def _gym_base():
    """Gymnasium mavjud bo'lsa Env, bo'lmasa object qaytaradi."""
    return gym.Env if GYMNASIUM_AVAILABLE else object


# ═══════════════════════════════════════════════════════════════
# Single-Agent muhit — Stable-Baselines3 uchun
# ═══════════════════════════════════════════════════════════════

class SingleAgentPricingEnv(_gym_base()):  # type: ignore[misc]
    """
    Dinamik narxlash uchun bir-agentli Gymnasium muhit.

    State (5 o'lchov, normallashtirilgan [0, 1]):
        [current_price, demand, competitor_price, loyalty_index, step_normalized]

    Action (7 ta Discrete):
        [-15%, -10%, -5%, 0%, +5%, +10%, +15%]

    Reward:
        R = α * revenue_norm + β * loyalty_delta * 100

    Parameters
    ----------
    initial_price : float
        Boshlang'ich narx (default: 10.0)
    base_demand : float
        Asosiy talab darajasi (default: 100.0)
    elasticity : float
        Narx elastikligi (default: -1.5)
    alpha : float
        Daromad reward og'irligi (default: 0.7)
    beta : float
        Sodiqliq reward og'irligi (default: 0.3)
    max_steps : int
        Epizod uzunligi (default: 100)
    competitor_volatility : float
        Raqib narxining tebranishi (default: 0.05)
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        initial_price: float = 10.0,
        base_demand: float = 100.0,
        elasticity: float = -1.5,
        alpha: float = 0.7,
        beta: float = 0.3,
        max_steps: int = 100,
        competitor_volatility: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        if not GYMNASIUM_AVAILABLE:
            raise ImportError(
                "Gymnasium kerak. O'rnatish: pip install gymnasium"
            )

        super().__init__()

        # Konfiguratsiya
        self.initial_price = float(initial_price)
        self.base_demand = float(base_demand)
        self.elasticity = float(elasticity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_steps = int(max_steps)
        self.competitor_volatility = float(competitor_volatility)

        # Action va observation spaces
        self.action_space = spaces.Discrete(NUM_PRICE_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # Random generator
        self._np_random: np.random.Generator = np.random.default_rng(seed)

        # State o'zgaruvchilari
        self.current_price: float = self.initial_price
        self.competitor_price: float = self.initial_price
        self.loyalty_index: float = 0.5
        self.current_demand: float = self.base_demand
        self.current_step: int = 0

        # Historiya (debugging va vizualizatsiya uchun)
        self.history: Dict[str, list] = {
            "prices": [], "demands": [], "rewards": [],
            "loyalties": [], "actions": [], "revenues": [],
        }

    # ───────────────────────────────────────────────────────────
    # Gymnasium API
    # ───────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Muhitni boshlang'ich holatga qaytarish.

        Returns
        -------
        observation : numpy.ndarray
            Boshlang'ich kuzatuv (shape=(5,))
        info : dict
            Qo'shimcha ma'lumot
        """
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self.current_price = self.initial_price
        self.competitor_price = self.initial_price * float(
            self._np_random.uniform(0.9, 1.1)
        )
        self.loyalty_index = 0.5
        self.current_demand = self._calculate_demand(self.current_price)
        self.current_step = 0

        # Tarixni tozalash
        self.history = {
            "prices": [], "demands": [], "rewards": [],
            "loyalties": [], "actions": [], "revenues": [],
        }

        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Bir qadam ijro etish.

        Parameters
        ----------
        action : int
            Discrete action (0-6)

        Returns
        -------
        observation : numpy.ndarray
        reward : float
        terminated : bool
            Epizod muvaffaqiyatli yakunlandi
        truncated : bool
            Vaqt tugadi
        info : dict
        """
        action = int(action)
        if not (0 <= action < NUM_PRICE_ACTIONS):
            raise ValueError(f"Yaroqsiz action: {action}")

        # Avvalgi holatni saqlash
        prev_loyalty = self.loyalty_index
        prev_price = self.current_price

        # 1. Narx o'zgarishi
        delta = PRICE_ACTION_DELTAS[action]
        self.current_price = float(np.clip(
            self.current_price * (1.0 + delta),
            self.initial_price * 0.3,    # Pastki chegara
            self.initial_price * 3.0,    # Tepa chegara
        ))

        # 2. Talab hisoblash (elastiklik formulasi)
        self.current_demand = self._calculate_demand(self.current_price)

        # 3. Sodiqlik o'zgarishi
        loyalty_change = self._calculate_loyalty_change(action, prev_price)
        self.loyalty_index = float(np.clip(
            self.loyalty_index + loyalty_change, 0.0, 1.0
        ))

        # 4. Raqib reaksiyasi (random walk)
        competitor_delta = float(self._np_random.normal(0, self.competitor_volatility))
        self.competitor_price = float(np.clip(
            self.competitor_price * (1.0 + competitor_delta),
            self.initial_price * 0.3,
            self.initial_price * 3.0,
        ))

        # 5. Reward hisoblash
        revenue = self.current_price * self.current_demand
        # Normallashtirish — maksimal mumkin bo'lgan daromadga nisbatan
        revenue_norm = revenue / (self.initial_price * self.base_demand * 1.5)
        revenue_norm = float(np.clip(revenue_norm, 0.0, 2.0))
        loyalty_delta = self.loyalty_index - prev_loyalty
        reward = float(self.alpha * revenue_norm + self.beta * loyalty_delta * 100.0)

        # 6. Qadam hisoblagich
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        terminated = self.loyalty_index <= 0.01  # Mijozlar yo'qotildi

        # 7. Tarix
        self.history["prices"].append(self.current_price)
        self.history["demands"].append(self.current_demand)
        self.history["rewards"].append(reward)
        self.history["loyalties"].append(self.loyalty_index)
        self.history["actions"].append(action)
        self.history["revenues"].append(revenue)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> str:
        """Joriy holatni matn ko'rinishida qaytarish."""
        return (
            f"Qadam {self.current_step:3d} | "
            f"Narx: {self.current_price:6.2f} | "
            f"Talab: {self.current_demand:6.1f} | "
            f"Sodiqlik: {self.loyalty_index:.3f} | "
            f"Raqib: {self.competitor_price:6.2f}"
        )

    def close(self) -> None:
        """Muhitni yopish (resource tozalash)."""
        pass

    # ───────────────────────────────────────────────────────────
    # Yordamchi metodlar
    # ───────────────────────────────────────────────────────────

    def _calculate_demand(self, price: float) -> float:
        """
        Narx elastikligi formulasi orqali talab hisoblash.

        Q = base_demand * (price / initial_price) ^ elasticity
        """
        ratio = price / self.initial_price
        demand = self.base_demand * (ratio ** self.elasticity)
        # Sodiqlik talabni kuchaytiradi
        demand *= 0.5 + self.loyalty_index
        return float(max(demand, 0.0))

    def _calculate_loyalty_change(self, action: int, prev_price: float) -> float:
        """
        Sodiqlik indeksining o'zgarishini hisoblash.

        Mantiq:
            * Narx pasaytirish → sodiqlik oshadi
            * Narx oshirish → sodiqlik kamayadi
            * Raqibga nisbatan narx muhim
        """
        delta = PRICE_ACTION_DELTAS[action]

        # Bazaviy effekt: narx pasayishi sodiqlikka ijobiy ta'sir
        base_effect = -delta * 0.5

        # Raqibga nisbatan narx
        price_ratio = self.current_price / max(self.competitor_price, 0.01)
        if price_ratio > 1.1:
            base_effect -= 0.02  # Raqibdan qimmat — sodiqlik kamayadi
        elif price_ratio < 0.95:
            base_effect += 0.015  # Raqibdan arzon — sodiqlik oshadi

        # Random shovqin (mijoz xulq-atvori noaniq)
        noise = float(self._np_random.normal(0, 0.01))
        return float(base_effect + noise)

    def _get_obs(self) -> np.ndarray:
        """Normallashtirilgan kuzatuv vektorini qaytarish."""
        price_norm = self.current_price / (self.initial_price * 3.0)
        demand_norm = self.current_demand / (self.base_demand * 2.0)
        comp_norm = self.competitor_price / (self.initial_price * 3.0)
        step_norm = self.current_step / max(self.max_steps, 1)

        obs = np.array([
            price_norm,
            demand_norm,
            comp_norm,
            self.loyalty_index,
            step_norm,
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    def _get_info(self) -> Dict[str, Any]:
        """Qo'shimcha info lug'ati."""
        return {
            "current_price": self.current_price,
            "demand": self.current_demand,
            "loyalty": self.loyalty_index,
            "competitor_price": self.competitor_price,
            "step": self.current_step,
        }


# ═══════════════════════════════════════════════════════════════
# Multi-Agent muhit — PettingZoo AECEnv
# ═══════════════════════════════════════════════════════════════

# 3 ta agent uchun maxsus action spacelar
CUSTOMER_ACTIONS = ["buy", "wait", "leave"]      # 3 ta
COMPETITOR_DELTAS = np.array([-0.10, -0.05, 0.0, 0.05, 0.10])  # 5 ta


class DynamicPricingEnv:
    """
    Multi-Agent dinamik narxlash muhiti (PettingZoo AECEnv standartiga muvofiq).

    3 ta agent:
        * pricing_agent    — narxni boshqaradi (action: 7 ta Discrete)
        * customer_agent   — xarid qarorlari (action: buy/wait/leave)
        * competitor_agent — raqobat strategiyasi (action: 5 ta narx delta)

    PettingZoo ixtiyoriy bog'liqlik — agar import qilinmasa,
    bu klass faqat asosiy logikani saqlaydi va testlarda ishlatiladi.
    """

    metadata = {
        "name": "dynamic_pricing_marl_v1",
        "render_modes": ["human"],
        "is_parallelizable": False,
    }

    def __init__(
        self,
        initial_price: float = 10.0,
        base_demand: float = 100.0,
        elasticity: float = -1.5,
        alpha: float = 0.7,
        beta: float = 0.3,
        max_steps: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        self.initial_price = float(initial_price)
        self.base_demand = float(base_demand)
        self.elasticity = float(elasticity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_steps = int(max_steps)
        self._np_random = np.random.default_rng(seed)

        # Agentlar
        self.possible_agents = ["pricing_agent", "customer_agent", "competitor_agent"]
        self.agents: list = []

        # Action spacelar (har agent uchun)
        self._action_spaces = self._build_action_spaces()
        self._observation_spaces = self._build_observation_spaces()

        # State o'zgaruvchilari
        self.current_price: float = self.initial_price
        self.competitor_price: float = self.initial_price
        self.loyalty_index: float = 0.5
        self.market_share: float = 0.5
        self.customer_budget: float = 50.0
        self.customer_need: float = 0.5
        self.current_step: int = 0
        self.agent_selection: str = "pricing_agent"

        # Reward va termination tracking
        self.rewards: Dict[str, float] = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards: Dict[str, float] = {agent: 0.0 for agent in self.possible_agents}
        self.terminations: Dict[str, bool] = {agent: False for agent in self.possible_agents}
        self.truncations: Dict[str, bool] = {agent: False for agent in self.possible_agents}
        self.infos: Dict[str, Dict] = {agent: {} for agent in self.possible_agents}

    # ───────────────────────────────────────────────────────────
    # Space yaratuvchilari
    # ───────────────────────────────────────────────────────────

    def _build_action_spaces(self) -> Dict[str, Any]:
        """Har agent uchun action space."""
        if not GYMNASIUM_AVAILABLE:
            return {}
        return {
            "pricing_agent": spaces.Discrete(NUM_PRICE_ACTIONS),       # 7 ta
            "customer_agent": spaces.Discrete(len(CUSTOMER_ACTIONS)),  # 3 ta
            "competitor_agent": spaces.Discrete(len(COMPETITOR_DELTAS)),  # 5 ta
        }

    def _build_observation_spaces(self) -> Dict[str, Any]:
        """Har agent uchun observation space."""
        if not GYMNASIUM_AVAILABLE:
            return {}
        return {
            # narx, talab, raqib, sodiqlik, vaqt
            "pricing_agent": spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32),
            # narx, tarix, sodiqlik, byudjet, ehtiyoj
            "customer_agent": spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32),
            # market_share, price_ratio, trend, time
            "competitor_agent": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
        }

    def action_space(self, agent: str):
        """Berilgan agent uchun action space qaytarish."""
        return self._action_spaces[agent]

    def observation_space(self, agent: str):
        """Berilgan agent uchun observation space qaytarish."""
        return self._observation_spaces[agent]

    # ───────────────────────────────────────────────────────────
    # AEC API
    # ───────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> None:
        """Muhitni boshlang'ich holatga qaytarish."""
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self.agents = list(self.possible_agents)
        self.current_price = self.initial_price
        self.competitor_price = self.initial_price * float(self._np_random.uniform(0.9, 1.1))
        self.loyalty_index = 0.5
        self.market_share = 0.5
        self.customer_budget = 50.0
        self.customer_need = 0.5
        self.current_step = 0
        self.agent_selection = "pricing_agent"

        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def observe(self, agent: str) -> np.ndarray:
        """
        Berilgan agent uchun kuzatuv qaytarish.

        Har agent o'z perspektivasidan dunyoni ko'radi.
        """
        if agent == "pricing_agent":
            return np.array([
                self.current_price / (self.initial_price * 3.0),
                self._calc_demand() / (self.base_demand * 2.0),
                self.competitor_price / (self.initial_price * 3.0),
                self.loyalty_index,
                self.current_step / max(self.max_steps, 1),
            ], dtype=np.float32).clip(0, 1)

        elif agent == "customer_agent":
            price_history_avg = self.current_price / (self.initial_price * 3.0)
            return np.array([
                self.current_price / (self.initial_price * 3.0),
                price_history_avg,
                self.loyalty_index,
                self.customer_budget / 100.0,
                self.customer_need,
            ], dtype=np.float32).clip(0, 1)

        elif agent == "competitor_agent":
            price_ratio = self.current_price / max(self.competitor_price, 0.01)
            trend = 0.5  # Soddalashtirilgan trend
            return np.array([
                self.market_share,
                min(price_ratio / 2.0, 1.0),
                trend,
                self.current_step / max(self.max_steps, 1),
            ], dtype=np.float32).clip(0, 1)

        raise ValueError(f"Noma'lum agent: {agent}")

    def step(self, action: int) -> None:
        """
        Joriy agent uchun action qabul qilish va navbatdagiga o'tish.

        AECEnv standartiga muvofiq, har bir step bitta agent uchun.
        """
        agent = self.agent_selection
        action = int(action)

        # Action ni qayta ishlash
        if agent == "pricing_agent":
            self._apply_pricing_action(action)
        elif agent == "customer_agent":
            self._apply_customer_action(action)
        elif agent == "competitor_agent":
            self._apply_competitor_action(action)

        # Navbatdagi agent
        self._advance_agent_selection()

        # Agar barcha agentlar bu turda harakat qilgan bo'lsa, qadam hisoblagich
        if agent == "competitor_agent":
            self.current_step += 1
            truncated = self.current_step >= self.max_steps
            for ag in self.agents:
                self.truncations[ag] = truncated

    def _apply_pricing_action(self, action: int) -> None:
        """Pricing agent harakatini qo'llash."""
        prev_loyalty = self.loyalty_index
        prev_price = self.current_price
        delta = PRICE_ACTION_DELTAS[action]
        self.current_price = float(np.clip(
            self.current_price * (1.0 + delta),
            self.initial_price * 0.3,
            self.initial_price * 3.0,
        ))

        # Sodiqlik effekti
        loyalty_change = -delta * 0.5
        if self.current_price > self.competitor_price * 1.1:
            loyalty_change -= 0.02
        self.loyalty_index = float(np.clip(
            self.loyalty_index + loyalty_change, 0.0, 1.0
        ))

        # Reward
        demand = self._calc_demand()
        revenue = self.current_price * demand
        revenue_norm = revenue / (self.initial_price * self.base_demand * 1.5)
        loyalty_delta = self.loyalty_index - prev_loyalty
        reward = self.alpha * revenue_norm + self.beta * loyalty_delta * 100.0
        self.rewards["pricing_agent"] = float(reward)
        self._cumulative_rewards["pricing_agent"] += float(reward)

    def _apply_customer_action(self, action: int) -> None:
        """Customer agent harakatini qo'llash."""
        action_name = CUSTOMER_ACTIONS[action]
        price_diff = self.current_price - self.initial_price

        if action_name == "buy":
            utility = (self.customer_need * 10.0) - price_diff
            self.customer_budget -= self.current_price
            reward = float(utility)
        elif action_name == "wait":
            reward = -0.1  # Kichik vaqt sarfi
        else:  # leave
            reward = -5.0  # Mijozni yo'qotish jarima

        self.rewards["customer_agent"] = reward
        self._cumulative_rewards["customer_agent"] += reward

        # Mijoz ehtiyoji va byudjeti yangilanadi
        self.customer_need = float(np.clip(
            self.customer_need + float(self._np_random.normal(0, 0.05)),
            0.0, 1.0,
        ))

    def _apply_competitor_action(self, action: int) -> None:
        """Competitor agent harakatini qo'llash."""
        prev_share = self.market_share
        delta = COMPETITOR_DELTAS[action]
        self.competitor_price = float(np.clip(
            self.competitor_price * (1.0 + delta),
            self.initial_price * 0.3,
            self.initial_price * 3.0,
        ))

        # Bozor ulushini yangilash
        if self.competitor_price < self.current_price:
            self.market_share = float(np.clip(self.market_share - 0.02, 0.0, 1.0))
        else:
            self.market_share = float(np.clip(self.market_share + 0.01, 0.0, 1.0))

        reward = (prev_share - self.market_share) * -10.0  # Ulush oshsa — yaxshi
        self.rewards["competitor_agent"] = float(reward)
        self._cumulative_rewards["competitor_agent"] += float(reward)

    def _advance_agent_selection(self) -> None:
        """Navbatdagi agentga o'tish."""
        order = ["pricing_agent", "customer_agent", "competitor_agent"]
        idx = order.index(self.agent_selection)
        self.agent_selection = order[(idx + 1) % len(order)]

    def _calc_demand(self) -> float:
        """Joriy narxda talab hisoblash."""
        ratio = self.current_price / self.initial_price
        demand = self.base_demand * (ratio ** self.elasticity)
        demand *= 0.5 + self.loyalty_index
        return float(max(demand, 0.0))

    def render(self) -> str:
        """Joriy holatni matn shaklida qaytarish."""
        return (
            f"[MARL] Qadam={self.current_step:3d} | "
            f"Narx={self.current_price:.2f} | "
            f"Raqib={self.competitor_price:.2f} | "
            f"Sodiqlik={self.loyalty_index:.3f} | "
            f"Bozor ulushi={self.market_share:.3f} | "
            f"Aktiv agent={self.agent_selection}"
        )

    def close(self) -> None:
        """Muhitni yopish."""
        self.agents = []
