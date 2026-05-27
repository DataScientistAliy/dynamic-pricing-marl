"""
RL agentlari moduli.

Tarkib:
    * BaseAgent   — umumiy interfeys
    * DQNAgent    — Deep Q-Network (off-policy, value-based)
    * PPOAgent    — Proximal Policy Optimization (on-policy)
    * SACAgent    — Soft Actor-Critic variant (A2C — discrete action uchun)
    * StaticAgent — Baseline (har doim narx o'zgartirmaydi)

BMI dan olingan hiperparametrlar bilan ishlaydi.
Stable-Baselines3 ixtiyoriy import — agar mavjud bo'lmasa, demo rejimi.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Stable-Baselines3 ixtiyoriy import
# ═══════════════════════════════════════════════════════════════

try:
    from stable_baselines3 import DQN, PPO, A2C  # SAC discrete uchun A2C ishlatamiz
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    DQN = PPO = A2C = None  # type: ignore
    BaseCallback = object  # type: ignore
    Monitor = None  # type: ignore


# ═══════════════════════════════════════════════════════════════
# BaseAgent — umumiy interfeys
# ═══════════════════════════════════════════════════════════════

class BaseAgent(ABC):
    """
    Barcha RL agentlar uchun asosiy interfeys.

    Sub-klasslar `train()`, `evaluate()`, `get_action()` ni amalga oshirishi kerak.

    Attributes
    ----------
    name : str
        Agent nomi (ko'rsatuv uchun)
    env : Any
        Bog'langan muhit
    model : Any
        Trained model (SB3 obyekti yoki maxsus)
    training_history : list
        Trening reward tarixi
    is_trained : bool
        Model trening qilinganmi
    """

    def __init__(self, env: Any, name: str = "BaseAgent") -> None:
        """
        Agent yaratish.

        Parameters
        ----------
        env : gymnasium.Env
            Bog'lanadigan muhit
        name : str
            Agent nomi
        """
        self.env = env
        self.name = name
        self.model: Any = None
        self.training_history: List[float] = []
        self.is_trained: bool = False
        self.training_time: float = 0.0

    @abstractmethod
    def train(
        self,
        timesteps: int = 30000,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Modelni trening qilish."""
        raise NotImplementedError

    @abstractmethod
    def get_action(self, observation: np.ndarray) -> int:
        """Joriy observation uchun action olish."""
        raise NotImplementedError

    def evaluate(self, n_episodes: int = 30, deterministic: bool = True) -> Dict[str, Any]:
        """
        Agentni baholash.

        Parameters
        ----------
        n_episodes : int
            Baholash epizodlari soni
        deterministic : bool
            Stochastic yoki deterministic policy

        Returns
        -------
        dict
            {mean_reward, std_reward, all_rewards, mean_revenue,
             mean_loyalty, mean_churn, all_episodes_data}
        """
        all_rewards: List[float] = []
        all_revenues: List[float] = []
        all_loyalties: List[float] = []
        all_churns: List[float] = []
        all_volatilities: List[float] = []
        episodes_data: List[Dict[str, Any]] = []

        for ep in range(n_episodes):
            obs, _ = self.env.reset(seed=ep)
            episode_reward = 0.0
            done = False
            truncated = False
            prices: List[float] = []
            loyalties: List[float] = []
            revenues: List[float] = []
            low_loyalty_steps = 0
            total_steps = 0

            while not (done or truncated):
                action = self.get_action(obs)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += float(reward)
                prices.append(float(info.get("current_price", 0.0)))
                loyalties.append(float(info.get("loyalty", 0.0)))
                rev = float(info.get("current_price", 0.0)) * float(info.get("demand", 0.0))
                revenues.append(rev)
                if float(info.get("loyalty", 1.0)) < 0.1:
                    low_loyalty_steps += 1
                total_steps += 1

            all_rewards.append(episode_reward)
            all_revenues.append(float(np.sum(revenues)))
            all_loyalties.append(float(np.mean(loyalties)) if loyalties else 0.0)
            all_churns.append(low_loyalty_steps / max(total_steps, 1))
            all_volatilities.append(float(np.std(prices)) if len(prices) > 1 else 0.0)
            episodes_data.append({
                "episode": ep,
                "reward": episode_reward,
                "revenue": float(np.sum(revenues)),
                "loyalty": float(np.mean(loyalties)) if loyalties else 0.0,
                "n_steps": total_steps,
            })

        return {
            "mean_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards)),
            "all_rewards": all_rewards,
            "mean_revenue": float(np.mean(all_revenues)),
            "mean_loyalty": float(np.mean(all_loyalties)),
            "mean_churn": float(np.mean(all_churns)),
            "mean_volatility": float(np.mean(all_volatilities)),
            "all_revenues": all_revenues,
            "all_loyalties": all_loyalties,
            "all_churns": all_churns,
            "all_volatilities": all_volatilities,
            "n_episodes": n_episodes,
            "episodes_data": episodes_data,
        }

    def save(self, path: str) -> None:
        """Modelni faylga saqlash."""
        if self.model is None:
            logger.warning("Saqlash uchun model yo'q (%s)", self.name)
            return
        if hasattr(self.model, "save"):
            self.model.save(path)
            logger.info("Model saqlandi: %s", path)

    def load(self, path: str) -> None:
        """Faylda modelni yuklash. Sub-klassda override qilinishi kerak."""
        raise NotImplementedError("Sub-klass o'z load() metodini amalga oshirishi kerak")


# ═══════════════════════════════════════════════════════════════
# Stable-Baselines3 dependent agentlar
# ═══════════════════════════════════════════════════════════════

class _SB3Callback(BaseCallback):  # type: ignore[misc]
    """
    Trening jarayonida reward tarixini saqlovchi yordamchi callback.

    Streamlit progress bar bilan integratsiya uchun ham ishlatiladi.
    """

    def __init__(
        self,
        history: List[float],
        progress_fn: Optional[Callable[[float], None]] = None,
        total_timesteps: int = 30000,
    ) -> None:
        super().__init__()
        self.history = history
        self.progress_fn = progress_fn
        self.total_timesteps = total_timesteps
        self.episode_rewards: List[float] = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        # Reward yig'ish
        if "rewards" in self.locals:
            self._current_reward += float(self.locals["rewards"][0])

        # Epizod tugasa
        if "dones" in self.locals and bool(self.locals["dones"][0]):
            self.history.append(self._current_reward)
            self._current_reward = 0.0

        # Progress callback
        if self.progress_fn and self.num_timesteps % 100 == 0:
            self.progress_fn(self.num_timesteps / max(self.total_timesteps, 1))

        return True


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent.

    BMI konfiguratsiyasi:
        learning_rate=1e-3, buffer_size=100000, batch_size=64,
        gamma=0.99, target_update_interval=1000,
        exploration_fraction=0.3, exploration_initial_eps=1.0,
        exploration_final_eps=0.05, net_arch=[128, 128]
    """

    def __init__(self, env: Any, **kwargs: Any) -> None:
        super().__init__(env, name="DQN")
        self.config = {
            "learning_rate": kwargs.get("learning_rate", 1e-3),
            "buffer_size": kwargs.get("buffer_size", 100_000),
            "batch_size": kwargs.get("batch_size", 64),
            "gamma": kwargs.get("gamma", 0.99),
            "target_update_interval": kwargs.get("target_update_interval", 1000),
            "exploration_fraction": kwargs.get("exploration_fraction", 0.3),
            "exploration_initial_eps": kwargs.get("exploration_initial_eps", 1.0),
            "exploration_final_eps": kwargs.get("exploration_final_eps", 0.05),
            "verbose": 0,
            "policy_kwargs": dict(net_arch=[128, 128]),
        }

    def train(
        self,
        timesteps: int = 30000,
        callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """DQN trening jarayoni."""
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 o'rnatilmagan")

        env = Monitor(self.env) if Monitor else self.env
        self.model = DQN("MlpPolicy", env, **self.config)
        cb = _SB3Callback(self.training_history, callback, timesteps)

        start = time.time()
        self.model.learn(total_timesteps=timesteps, callback=cb)
        self.training_time = time.time() - start
        self.is_trained = True

        logger.info("DQN trening yakunlandi: %.1f sek", self.training_time)
        return {
            "training_time": self.training_time,
            "history": self.training_history,
            "n_episodes": len(self.training_history),
        }

    def get_action(self, observation: np.ndarray) -> int:
        """Trained model dan action olish."""
        if not self.is_trained:
            return int(self.env.action_space.sample())
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)

    def load(self, path: str) -> None:
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 o'rnatilmagan")
        self.model = DQN.load(path)
        self.is_trained = True


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization agent.

    BMI konfiguratsiyasi:
        learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        net_arch=dict(pi=[128,128], vf=[128,128])
    """

    def __init__(self, env: Any, **kwargs: Any) -> None:
        super().__init__(env, name="PPO")
        self.config = {
            "learning_rate": kwargs.get("learning_rate", 3e-4),
            "n_steps": kwargs.get("n_steps", 2048),
            "batch_size": kwargs.get("batch_size", 64),
            "n_epochs": kwargs.get("n_epochs", 10),
            "gamma": kwargs.get("gamma", 0.99),
            "gae_lambda": kwargs.get("gae_lambda", 0.95),
            "clip_range": kwargs.get("clip_range", 0.2),
            "ent_coef": kwargs.get("ent_coef", 0.01),
            "verbose": 0,
            "policy_kwargs": dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        }

    def train(
        self,
        timesteps: int = 30000,
        callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """PPO trening jarayoni."""
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 o'rnatilmagan")

        env = Monitor(self.env) if Monitor else self.env
        self.model = PPO("MlpPolicy", env, **self.config)
        cb = _SB3Callback(self.training_history, callback, timesteps)

        start = time.time()
        self.model.learn(total_timesteps=timesteps, callback=cb)
        self.training_time = time.time() - start
        self.is_trained = True

        logger.info("PPO trening yakunlandi: %.1f sek", self.training_time)
        return {
            "training_time": self.training_time,
            "history": self.training_history,
            "n_episodes": len(self.training_history),
        }

    def get_action(self, observation: np.ndarray) -> int:
        if not self.is_trained:
            return int(self.env.action_space.sample())
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)

    def load(self, path: str) -> None:
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 o'rnatilmagan")
        self.model = PPO.load(path)
        self.is_trained = True


class SACAgent(BaseAgent):
    """
    SAC variant — discrete action space uchun A2C ishlatamiz.

    Sabab: SB3 SAC faqat Continuous action space ni qo'llab-quvvatlaydi.
    BMI da bu cheklov sabab A2C diskret variant ishlatilgan.

    Konfiguratsiya:
        learning_rate=7e-4, n_steps=5, gamma=0.99,
        net_arch=dict(pi=[128,128], vf=[128,128])
    """

    def __init__(self, env: Any, **kwargs: Any) -> None:
        super().__init__(env, name="SAC")
        self.config = {
            "learning_rate": kwargs.get("learning_rate", 7e-4),
            "n_steps": kwargs.get("n_steps", 5),
            "gamma": kwargs.get("gamma", 0.99),
            "ent_coef": kwargs.get("ent_coef", "auto"),
            "verbose": 0,
            "policy_kwargs": dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        }

    def train(
        self,
        timesteps: int = 30000,
        callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """SAC (A2C variant) trening jarayoni."""
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 o'rnatilmagan")

        env = Monitor(self.env) if Monitor else self.env
        self.model = A2C("MlpPolicy", env, **self.config)
        cb = _SB3Callback(self.training_history, callback, timesteps)

        start = time.time()
        self.model.learn(total_timesteps=timesteps, callback=cb)
        self.training_time = time.time() - start
        self.is_trained = True

        logger.info("SAC (A2C) trening yakunlandi: %.1f sek", self.training_time)
        return {
            "training_time": self.training_time,
            "history": self.training_history,
            "n_episodes": len(self.training_history),
        }

    def get_action(self, observation: np.ndarray) -> int:
        if not self.is_trained:
            return int(self.env.action_space.sample())
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)

    def load(self, path: str) -> None:
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 o'rnatilmagan")
        self.model = A2C.load(path)
        self.is_trained = True


# ═══════════════════════════════════════════════════════════════
# Static Baseline — har doim 0% o'zgarish
# ═══════════════════════════════════════════════════════════════

class StaticAgent(BaseAgent):
    """
    Statik baseline agent — har doim "0% o'zgarish" action ni tanlaydi.

    Bu baseline RL agentlarning samaradorligini taqqoslash uchun ishlatiladi.
    Trening kerak emas — darhol baholanishi mumkin.
    """

    def __init__(self, env: Any, fixed_action: int = 3, **kwargs: Any) -> None:
        super().__init__(env, name="Static")
        # 7 ta action ichida 3-indeks = 0% o'zgarish
        self.fixed_action = int(fixed_action)
        self.is_trained = True  # Trening shart emas

    def train(
        self,
        timesteps: int = 30000,
        callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """Trening yo'q — bu baseline."""
        self.is_trained = True
        self.training_time = 0.0
        return {"training_time": 0.0, "history": [], "n_episodes": 0}

    def get_action(self, observation: np.ndarray) -> int:
        """Doimo bir xil action qaytarish."""
        return self.fixed_action


# ═══════════════════════════════════════════════════════════════
# Agent factory
# ═══════════════════════════════════════════════════════════════

AGENT_REGISTRY: Dict[str, type] = {
    "DQN": DQNAgent,
    "PPO": PPOAgent,
    "SAC": SACAgent,
    "Static": StaticAgent,
}


def create_agent(name: str, env: Any, **kwargs: Any) -> BaseAgent:
    """
    Nomi bo'yicha agent yaratish (factory funksiya).

    Parameters
    ----------
    name : str
        "DQN", "PPO", "SAC" yoki "Static"
    env : Any
        Muhit
    **kwargs
        Agent uchun qo'shimcha parametrlar

    Returns
    -------
    BaseAgent
        Yaratilgan agent
    """
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Noma'lum agent: {name}. Mavjud: {list(AGENT_REGISTRY)}")
    return AGENT_REGISTRY[name](env, **kwargs)
