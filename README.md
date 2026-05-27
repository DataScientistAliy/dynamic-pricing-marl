<div align="center">

# 🎯 Multi-Agent RL Dynamic Pricing System

### Ko'p-agentli kuchaytirilgan o'qitish yordamida e-commerce platformalari uchun intellektual dinamik narxlash tizimi

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.1+-FF6F00?style=for-the-badge)](https://stable-baselines3.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

[![Stars](https://img.shields.io/github/stars/DataScientistAliy/dynamic-pricing-marl?style=social)](https://github.com/DataScientistAliy/dynamic-pricing-marl)
[![Forks](https://img.shields.io/github/forks/DataScientistAliy/dynamic-pricing-marl?style=social)](https://github.com/DataScientistAliy/dynamic-pricing-marl)

<p align="center">
    <strong>PPO algoritmi static baseline ga nisbatan:</strong><br>
    💰 <strong>+37.5% daromad</strong> &nbsp;•&nbsp;
    ❤️ <strong>+42% sodiqlik</strong> &nbsp;•&nbsp;
    🚪 <strong>-54% churn</strong> &nbsp;•&nbsp;
    📊 <strong>p < 0.001</strong>
</p>

[🚀 Demo](https://share.streamlit.io) · [📖 Hujjatlar](#-loyiha-tuzilmasi) · [🐛 Bug](https://github.com/DataScientistAliy/dynamic-pricing-marl/issues) · [💡 Tavsiyalar](#-tavsiyalar)

</div>

---

## 📋 Mundarija

- [🎯 Loyiha haqida](#-loyiha-haqida)
- [✨ Asosiy xususiyatlar](#-asosiy-xususiyatlar)
- [📊 Natijalar](#-natijalar)
- [🏗️ Arxitektura](#️-arxitektura)
- [🚀 Boshlash](#-boshlash)
- [📁 Loyiha tuzilmasi](#-loyiha-tuzilmasi)
- [🧪 Testlar](#-testlar)
- [🛠️ Texnologiyalar](#️-texnologiyalar)
- [📚 Foydalanilgan adabiyotlar](#-foydalanilgan-adabiyotlar)
- [📄 Litsenziya](#-litsenziya)

---

## 🎯 Loyiha haqida

Bu loyiha **Toshkent Davlat Iqtisodiyot Universiteti** Data Science yo'nalishi bo'yicha **Bitiruv Malakaviy Ishi (BMI)** doirasida amalga oshirilgan. Asosiy maqsad — O'zbekiston e-commerce sektori uchun **Multi-Agent Reinforcement Learning (MARL)** yordamida intellektual dinamik narxlash tizimini ishlab chiqish.

### 🎓 BMI ma'lumotlari

| Parametr | Qiymat |
|---------|--------|
| **Dataset** | Online Retail II (UCI Machine Learning Repository) |
| **Tranzaksiyalar** | 1,067,371 |
| **Mijozlar** | 5,942 |
| **Algoritmlar** | DQN, PPO, SAC (A2C variant), Static baseline |
| **Statistik testlar** | Shapiro-Wilk, t-test, Mann-Whitney U, Cohen's d |
| **Experimentlar** | 30 ta mustaqil run, n=30 |
| **Reward funksiyasi** | R = 0.7·revenue_norm + 0.3·loyalty_delta·100 |

---

## ✨ Asosiy Xususiyatlar

### 🤖 3 ta Intellektual Agent

| Agent | Vazifasi | Action Space |
|-------|----------|--------------|
| 🎯 **Pricing Agent** | Narxni boshqaradi | 7 ta discrete (-15% → +15%) |
| 👤 **Customer Agent** | Mijoz xulqi simulyatsiya | 3 ta (buy/wait/leave) |
| 🏪 **Competitor Agent** | Raqobat strategiyasi | 5 ta (-10% → +10%) |

### 📊 Professional Dashboard

- ✅ **5 ta interaktiv sahifa** (EDA, Simulyatsiya, Natijalar, Statistika, Tavsiyalar)
- ✅ **Real-time grafiklar** (Plotly)
- ✅ **Dark theme** (glassmorphism, gradient)
- ✅ **Mobile responsive**
- ✅ **Streamlit Cloud uchun tayyor**

### 🔬 Chuqur Statistik Tahlil

- Shapiro-Wilk normallik testi
- Welch's t-test (parametrik)
- Mann-Whitney U testi (noparametrik)
- Cohen's d effect size
- Pareto frontasi va Hyper-Volume
- α/β sezgirlik tahlili (grid search)

---

## 📊 Natijalar

### 🏆 BMI 3.3-jadval: Algoritmlar Taqqoslash (n=30)

| Algoritm | 💰 Daromad (£) | ❤️ Sodiqlik | 🚪 Churn | 📈 Volatillik | 🎯 HV |
|----------|----------------|------------|----------|---------------|-------|
| 🥇 **PPO** | **38,672 ± 1,756** | **0.713 ± 0.048** | **0.083 ± 0.021** | 1.92 ± 0.31 | **0.687** |
| 🥈 SAC (A2C) | 36,141 ± 2,478 | 0.652 ± 0.069 | 0.103 ± 0.027 | 2.34 ± 0.39 | 0.589 |
| 🥉 DQN | 34,987 ± 2,184 | 0.621 ± 0.058 | 0.124 ± 0.029 | 2.78 ± 0.42 | 0.521 |
| Static | 28,134 ± 1,478 | 0.502 ± 0.041 | 0.182 ± 0.023 | **0.51 ± 0.09** | 0.214 |

### 📈 PPO vs Static (asosiy KPI)

```
💰 Daromad:    £28,134  →  £38,672    (+37.5%)
❤️ Sodiqlik:    0.502   →   0.713     (+42.0%)
🚪 Churn:       18.2%   →    8.3%     (-54.4%)
🎯 Hyper-Volume: 0.214  →   0.687     (3.2× yaxshilanish)
```

### 🔬 Statistik Ahamiyatlilik

```
PPO vs Static:  p < 0.001  ***  (Cohen's d = 6.42, ultra katta)
PPO vs SAC:     p < 0.001  ***  (Cohen's d = 1.18, juda katta)
PPO vs DQN:     p < 0.001  ***  (Cohen's d = 1.84, juda katta)
```

---

## 🏗️ Arxitektura

```
┌────────────────────────────────────────────────────────────┐
│           Online Retail II Dataset (1M+ tx)                │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│        Data Preprocessing (RFM + Elastiklik)               │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│                      MARL Muhiti                           │
│              (Gymnasium + PettingZoo AEC)                  │
├──────────────────┬──────────────────┬──────────────────────┤
│  🎯 Pricing      │   👤 Customer    │   🏪 Competitor      │
│  Agent           │   Agent          │   Agent              │
│  (PPO/DQN/SAC)   │   (buy/wait/     │   (5 ta strategiya)  │
│                  │    leave)        │                      │
└──────────────────┴──────────────────┴──────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│      Reward: R = 0.7·revenue_norm + 0.3·loyalty·100        │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│      Streamlit Dashboard (5 sahifa, real-time)             │
└────────────────────────────────────────────────────────────┘
```

---

## 🚀 Boshlash

### 1️⃣ Klonlash

```bash
git clone https://github.com/DataScientistAliy/dynamic-pricing-marl.git
cd dynamic-pricing-marl
```

### 2️⃣ O'rnatish

```bash
# Virtual environment (tavsiya)
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# Dependencylarni o'rnatish
pip install -r requirements.txt
```

### 3️⃣ Ishga tushirish

```bash
streamlit run app.py
```

Brauzerda **http://localhost:8501** ochiladi.

> 💡 **Tezkor Demo:** Hech qanday dataset yuklamasdan, ilova darhol ishlaydi —
> `data/sample_data.csv` (1000 qator) va `data/precomputed_results.json` avtomatik yaratiladi.

---

## 📁 Loyiha tuzilmasi

```
dynamic-pricing-marl/
│
├── 📄 README.md                         # Bu fayl
├── 📄 requirements.txt                  # Python kutubxonalar
├── 📄 .gitignore                        # Git ignore qoidalari
├── 📄 app.py                            # Streamlit asosiy sahifa
│
├── 📁 src/                              # Asosiy Python kutubxona
│   ├── __init__.py
│   ├── data_loader.py                   # Dataset + RFM
│   ├── environment.py                   # Gymnasium + PettingZoo
│   ├── agents.py                        # DQN, PPO, SAC, Static
│   ├── evaluation.py                    # Statistik testlar
│   └── utils.py                         # Yordamchi funksiyalar
│
├── 📁 pages/                            # Streamlit multipage
│   ├── 1_📊_EDA_va_Dataset.py
│   ├── 2_🤖_RL_Simulatsiya.py
│   ├── 3_📈_Natijalar.py
│   ├── 4_🔬_Statistik_Tahlil.py
│   └── 5_🗺️_Tavsiyalar.py
│
├── 📁 data/                             # Ma'lumotlar
│   ├── sample_data.csv                  # Namunaviy dataset (1000 qator)
│   └── precomputed_results.json         # Tayyor natijalar (demo uchun)
│
├── 📁 models/                           # Saqlangan modellar (.gitkeep)
│
├── 📁 notebooks/                        # Jupyter notebooklar
│   └── EDA_Analysis.ipynb               # EDA tahlili
│
└── 📁 tests/                            # Unit testlar
    ├── test_environment.py              # Muhit testlari
    └── test_agents.py                   # Agent testlari
```

---

## 🧪 Testlar

```bash
# Barcha testlar
pytest tests/ -v

# Faqat muhit testlari
pytest tests/test_environment.py -v

# Coverage hisoboti
pytest tests/ --cov=src --cov-report=html
```

**Test qamrovi:**
- ✅ `SingleAgentPricingEnv` — 12 ta test
- ✅ `DynamicPricingEnv` (MARL) — 6 ta test
- ✅ `StaticAgent` — 5 ta test
- ✅ Agent factory — 3 ta test
- ✅ Evaluation moduli — 5 ta test
- ✅ Utils moduli — 4 ta test

---

## 🛠️ Texnologiyalar

### Asosiy Stack

| Kategoriya | Texnologiya | Versiya |
|-----------|-------------|---------|
| 🐍 Til | Python | 3.10+ |
| 🧠 Deep Learning | PyTorch | 2.0+ |
| 🎯 RL Framework | Stable-Baselines3 | 2.1+ |
| 🏟️ RL Environments | Gymnasium, PettingZoo | 0.29+, 1.24+ |
| 📊 Dashboard | Streamlit | 1.32+ |
| 📈 Vizualizatsiya | Plotly, Matplotlib, Seaborn | latest |
| 🔬 Statistika | SciPy | 1.11+ |
| 📦 Data | Pandas, NumPy | latest |
| 🔧 Hyperparameter | Optuna | 3.4+ |

### Algoritmlar Konfiguratsiyasi

<details>
<summary><strong>PPO (Proximal Policy Optimization)</strong></summary>

```python
{
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "net_arch": dict(pi=[128, 128], vf=[128, 128])
}
```
</details>

<details>
<summary><strong>DQN (Deep Q-Network)</strong></summary>

```python
{
    "learning_rate": 1e-3,
    "buffer_size": 100_000,
    "batch_size": 64,
    "gamma": 0.99,
    "target_update_interval": 1000,
    "exploration_fraction": 0.3,
    "exploration_final_eps": 0.05,
    "net_arch": [128, 128]
}
```
</details>

<details>
<summary><strong>SAC (A2C variant)</strong></summary>

```python
# SB3 SAC faqat continuous action — A2C diskret variant uchun
{
    "learning_rate": 7e-4,
    "n_steps": 5,
    "gamma": 0.99,
    "net_arch": dict(pi=[128, 128], vf=[128, 128])
}
```
</details>

---

## 📚 Foydalanilgan Adabiyotlar

1. **Schulman, J., et al.** (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347
2. **Mnih, V., et al.** (2015). *Human-level control through deep reinforcement learning.* Nature.
3. **Haarnoja, T., et al.** (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL.* ICML.
4. **Terry, J., et al.** (2021). *PettingZoo: Gym for Multi-Agent Reinforcement Learning.* NeurIPS.
5. **Chen, D.** (2015). *Online Retail II Dataset.* UCI Machine Learning Repository.

### 📑 BMI Citation

```bibtex
@thesis{marl_pricing_2026,
    title  = {Multi-Agent Reinforcement Learning yordamida e-commerce platformalari
              uchun dinamik narxlash tizimi},
    author = {DataScientistAliy},
    school = {Toshkent Davlat Iqtisodiyot Universiteti},
    year   = {2026},
    type   = {Bitiruv Malakaviy Ishi},
    url    = {https://github.com/DataScientistAliy/dynamic-pricing-marl}
}
```

---

## 🗺️ Tavsiyalar

O'zbekiston e-commerce sektori uchun MARL ni amaliy joriy etish bo'yicha **4 bosqichli yo'l xaritasi**:

| Bosqich | Davomiyligi | Byudjet | Vazifalar |
|---------|------------|---------|-----------|
| 1️⃣ Ma'lumotlar va Infratuzilma | 1-2 oy | $20K | ETL, data warehouse |
| 2️⃣ MVP / Pilot Trening | 2-3 oy | $30K | PPO trening, env tuning |
| 3️⃣ A/B Test | 1-2 oy | $15K | 10% traffic, monitoring |
| 4️⃣ To'liq Joriy Etish | 3-4 oy | $25K | Production, auto-retrain |

**Jami investitsiya:** $80-120K  •  **Kutilgan ROI:** 350-420%  •  **Payback:** 6-8 oy

---

## 🤝 Hissa qo'shish

Hissa qo'shishni xohlaysizmi? Quyidagilarni amalga oshiring:

1. Repo ni fork qiling
2. Yangi branch yarating (`git checkout -b feature/AmazingFeature`)
3. O'zgartirishlarni commit qiling (`git commit -m 'Add AmazingFeature'`)
4. Branch ni push qiling (`git push origin feature/AmazingFeature`)
5. Pull Request oching

---

## 📄 Litsenziya

Bu loyiha **MIT Litsenziyasi** ostida tarqatiladi. Batafsil — [LICENSE](LICENSE) faylda.

---

<div align="center">

### 🎓 Toshkent Davlat Iqtisodiyot Universiteti
### Data Science · Bitiruv Malakaviy Ishi · 2026

<br>

**Made with ❤️ for BMI · Powered by Reinforcement Learning**

⭐ Bu loyiha sizga foydali bo'lsa, **star** qo'ying!

</div>
