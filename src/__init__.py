"""
Multi-Agent Dynamic Pricing — asosiy kutubxona moduli.

Bu paket BMI loyihasi uchun barcha asosiy komponentlarni o'z ichiga oladi:
    * data_loader   — Online Retail II datasetini yuklash va RFM tahlili
    * environment   — Gym va PettingZoo muhitlari
    * agents        — DQN, PPO, SAC, Static agentlar
    * evaluation    — Statistik testlar va metrikalar
    * utils         — Yordamchi funksiyalar va precomputed natijalar
"""

__version__ = "1.0.0"
__author__ = "TDIU Data Science"
__all__ = ["data_loader", "environment", "agents", "evaluation", "utils"]
