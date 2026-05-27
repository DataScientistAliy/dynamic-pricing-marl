"""
Online Retail II datasetini yuklash va tahlil qilish moduli.

Asosiy funksiyalar:
    * Excel/CSV fayldan ma'lumot yuklash
    * Ma'lumotlarni tozalash (missing values, outliers, dublikatlar)
    * RFM segmentatsiya (Recency, Frequency, Monetary)
    * Narx elastikligini OLS regression yordamida hisoblash
    * Dashboard uchun xulosa statistikasi
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Modul logger sozlash
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class DataLoader:
    """
    Online Retail II datasetini yuklash, tozalash va tahlil qilish.

    Foydalanish:
        >>> loader = DataLoader()
        >>> df = loader.load_data("data/online_retail.xlsx")
        >>> df_clean = loader.clean_data(df)
        >>> rfm = loader.calculate_rfm(df_clean)

    Attributes
    ----------
    cleaning_stats : dict
        Oxirgi clean_data() chaqiruvi statistikasi
    """

    def __init__(self) -> None:
        """DataLoader instansiyasini yaratish."""
        self.cleaning_stats: Dict[str, int] = {}

    # ═══════════════════════════════════════════════════════════════
    # Ma'lumot yuklash
    # ═══════════════════════════════════════════════════════════════

    def load_data(self, path: str | Path) -> pd.DataFrame:
        """
        Excel yoki CSV fayldan ma'lumot yuklash.

        Avval fayl kengaytmasi bo'yicha format aniqlanadi.
        Agar yuklash muvaffaqiyatsiz bo'lsa, ikkala format ham sinab ko'riladi.

        Parameters
        ----------
        path : str or Path
            Fayl manzili

        Returns
        -------
        pandas.DataFrame
            Yuklangan ma'lumotlar

        Raises
        ------
        FileNotFoundError
            Fayl topilmasa
        ValueError
            Format aniqlanmasa
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Fayl topilmadi: {path}")

        suffix = path.suffix.lower()
        try:
            if suffix in {".xlsx", ".xls"}:
                df = pd.read_excel(path, engine="openpyxl")
            elif suffix == ".csv":
                df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
            else:
                # Fallback — ikkala formatni sinash
                try:
                    df = pd.read_csv(path)
                except Exception:
                    df = pd.read_excel(path)
        except Exception as exc:
            # Ikkilamchi fallback
            logger.warning("Birlamchi yuklash muvaffaqiyatsiz: %s. Boshqa formatni sinaymiz.", exc)
            try:
                df = pd.read_csv(path, encoding="latin-1")
            except Exception:
                df = pd.read_excel(path, engine="openpyxl")

        # Ustun nomlarini standartlashtirish
        df = self._standardize_columns(df)
        logger.info("Yuklandi: %d qator, %d ustun", len(df), len(df.columns))
        return df

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Turli versiyalardagi ustun nomlarini standartlashtirish.

        Online Retail va Online Retail II uchun har xil ustun nomlari mavjud.
        """
        column_mapping = {
            "InvoiceNo": "Invoice",
            "UnitPrice": "Price",
            "CustomerID": "Customer ID",
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        return df

    # ═══════════════════════════════════════════════════════════════
    # Ma'lumotlarni tozalash
    # ═══════════════════════════════════════════════════════════════

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Datasetni 5 bosqichda tozalash.

        Bosqichlar:
            1. Customer ID bo'sh qatorlarni o'chirish
            2. Manfiy miqdor (qaytarilgan mahsulotlar)
            3. Manfiy yoki nol narx
            4. Dublikat qatorlarni o'chirish
            5. Outlier narxlar (IQR metodi)

        Har bosqich statistikasi `self.cleaning_stats` ga yoziladi.

        Parameters
        ----------
        df : pandas.DataFrame
            Tozalanmagan dataset

        Returns
        -------
        pandas.DataFrame
            Tozalangan dataset
        """
        df = df.copy()
        initial_rows = len(df)
        stats: Dict[str, int] = {"initial": initial_rows}

        # Bosqich 1: Customer ID bo'sh qatorlar
        if "Customer ID" in df.columns:
            before = len(df)
            df = df.dropna(subset=["Customer ID"])
            stats["after_dropna_customer"] = len(df)
            stats["removed_no_customer"] = before - len(df)

        # Bosqich 2: Manfiy miqdor
        if "Quantity" in df.columns:
            before = len(df)
            df = df[df["Quantity"] > 0]
            stats["after_positive_quantity"] = len(df)
            stats["removed_negative_qty"] = before - len(df)

        # Bosqich 3: Manfiy yoki nol narx
        if "Price" in df.columns:
            before = len(df)
            df = df[df["Price"] > 0]
            stats["after_positive_price"] = len(df)
            stats["removed_non_positive_price"] = before - len(df)

        # Bosqich 4: Dublikatlar
        before = len(df)
        df = df.drop_duplicates()
        stats["after_dedup"] = len(df)
        stats["removed_duplicates"] = before - len(df)

        # Bosqich 5: Outlier narxlar (IQR)
        if "Price" in df.columns and len(df) > 0:
            q1 = df["Price"].quantile(0.25)
            q3 = df["Price"].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            before = len(df)
            df = df[(df["Price"] >= lower_bound) & (df["Price"] <= upper_bound)]
            stats["after_iqr_filter"] = len(df)
            stats["removed_outliers"] = before - len(df)

        # InvoiceDate ni datetime ga aylantirish
        if "InvoiceDate" in df.columns:
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
            df = df.dropna(subset=["InvoiceDate"])

        # TotalPrice ustuni
        if "Quantity" in df.columns and "Price" in df.columns:
            df["TotalPrice"] = df["Quantity"] * df["Price"]

        stats["final"] = len(df)
        stats["total_removed"] = initial_rows - len(df)
        stats["retention_pct"] = round(100.0 * len(df) / max(initial_rows, 1), 2)

        self.cleaning_stats = stats
        logger.info(
            "Tozalash yakunlandi: %d → %d qator (%.2f%% saqlandi)",
            initial_rows, len(df), stats["retention_pct"],
        )
        return df.reset_index(drop=True)

    # ═══════════════════════════════════════════════════════════════
    # RFM segmentatsiyasi
    # ═══════════════════════════════════════════════════════════════

    def calculate_rfm(
        self,
        df: pd.DataFrame,
        snapshot_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Har mijoz uchun RFM hisoblash va segmentatsiya.

        RFM o'lchovlari:
            * Recency  — oxirgi xariddan snapshot_date gacha kunlar soni
            * Frequency — unique invoice lar soni
            * Monetary — jami xarid summasi

        Har o'lchov quantile asosida 1-5 ball oladi.
        RFM_Score = R_Score + F_Score + M_Score (3-15)

        Segmentatsiya:
            * Champion  (12-15)
            * Loyal     (9-11)
            * Potential (6-8)
            * At_Risk   (4-5)
            * Lost      (3)

        Parameters
        ----------
        df : pandas.DataFrame
            Tozalangan dataset (TotalPrice ustuni bilan)
        snapshot_date : pandas.Timestamp, optional
            Hisoblash sanasi. None bo'lsa, dataset oxirgi sanasi + 1 kun.

        Returns
        -------
        pandas.DataFrame
            Customer ID indeksi bilan RFM jadval
        """
        required_cols = {"Customer ID", "Invoice", "InvoiceDate", "TotalPrice"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Ustunlar yetishmaydi: {missing}")

        if snapshot_date is None:
            snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

        # RFM hisoblash
        rfm = df.groupby("Customer ID").agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("TotalPrice", "sum"),
        ).reset_index()

        # 1-5 ball bo'yicha baholash (quantile-based)
        # Recency — kichik qiymat yaxshi (yaqinda xarid qildi)
        rfm["R_Score"] = self._safe_qcut(rfm["Recency"], labels=[5, 4, 3, 2, 1])
        # Frequency — katta qiymat yaxshi
        rfm["F_Score"] = self._safe_qcut(rfm["Frequency"], labels=[1, 2, 3, 4, 5])
        # Monetary — katta qiymat yaxshi
        rfm["M_Score"] = self._safe_qcut(rfm["Monetary"], labels=[1, 2, 3, 4, 5])

        rfm["RFM_Score"] = (
            rfm["R_Score"].astype(int) +
            rfm["F_Score"].astype(int) +
            rfm["M_Score"].astype(int)
        )

        # Segmentatsiya
        rfm["Segment"] = rfm["RFM_Score"].apply(self._classify_segment)

        return rfm

    @staticmethod
    def _safe_qcut(series: pd.Series, labels: List[int]) -> pd.Series:
        """
        Xavfsiz quantile cut — agar unique qiymatlar yetarli bo'lmasa.

        pd.qcut har doim ham 5 ta bin ga bo'la olmaydi (dublikat qiymatlar).
        Bu funksiya fallback bilan ishlaydi.
        """
        try:
            return pd.qcut(series, q=5, labels=labels, duplicates="drop").astype(int)
        except (ValueError, TypeError):
            # Ranking yondashuv — quantile o'rniga rank ishlatish
            ranks = series.rank(method="first", pct=True)
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
            return pd.cut(ranks, bins=bins, labels=labels, include_lowest=True).astype(int)

    @staticmethod
    def _classify_segment(score: int) -> str:
        """RFM_Score ga ko'ra segment nomi qaytarish."""
        if score >= 12:
            return "Champion"
        elif score >= 9:
            return "Loyal"
        elif score >= 6:
            return "Potential"
        elif score >= 4:
            return "At_Risk"
        else:
            return "Lost"

    # ═══════════════════════════════════════════════════════════════
    # Narx elastikligi
    # ═══════════════════════════════════════════════════════════════

    def calculate_elasticity(
        self,
        df: pd.DataFrame,
        group_by: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Narx elastikligini OLS regression yordamida hisoblash.

        Formula: ln(Quantity) = α + ε * ln(Price)
        ε — narx elastikligi koeffitsienti (odatda manfiy)

        Parameters
        ----------
        df : pandas.DataFrame
            Quantity va Price ustunlari bilan dataset
        group_by : str, optional
            Agar ko'rsatilsa, har guruh uchun alohida elastiklik hisoblanadi.

        Returns
        -------
        dict
            {"elasticity": float, "r_squared": float, "n_obs": int}
            Yoki group_by berilgan bo'lsa: {group_value: {"elasticity": ..., ...}, ...}
        """
        if "Quantity" not in df.columns or "Price" not in df.columns:
            raise ValueError("Quantity va Price ustunlari kerak")

        def _single(sub_df: pd.DataFrame) -> Dict[str, float]:
            """Bir guruh uchun elastiklik."""
            data = sub_df[(sub_df["Quantity"] > 0) & (sub_df["Price"] > 0)].copy()
            if len(data) < 10:
                return {"elasticity": float("nan"), "r_squared": 0.0, "n_obs": len(data)}

            log_qty = np.log(data["Quantity"].to_numpy())
            log_price = np.log(data["Price"].to_numpy())

            # OLS — eng kichik kvadratlar usuli
            x_mean = log_price.mean()
            y_mean = log_qty.mean()
            beta = (
                ((log_price - x_mean) * (log_qty - y_mean)).sum() /
                max(((log_price - x_mean) ** 2).sum(), 1e-12)
            )
            alpha = y_mean - beta * x_mean

            # R-squared
            y_pred = alpha + beta * log_price
            ss_res = ((log_qty - y_pred) ** 2).sum()
            ss_tot = ((log_qty - y_mean) ** 2).sum()
            r_sq = 1.0 - (ss_res / max(ss_tot, 1e-12))

            return {
                "elasticity": float(beta),
                "intercept": float(alpha),
                "r_squared": float(r_sq),
                "n_obs": int(len(data)),
            }

        if group_by is None:
            return _single(df)

        if group_by not in df.columns:
            raise ValueError(f"Ustun topilmadi: {group_by}")

        results: Dict[str, Any] = {}
        for value, sub in df.groupby(group_by):
            results[str(value)] = _single(sub)
        return results

    # ═══════════════════════════════════════════════════════════════
    # Namunaviy ma'lumotlar
    # ═══════════════════════════════════════════════════════════════

    def generate_sample_data(self, n: int = 1000, seed: int = 42) -> pd.DataFrame:
        """
        Online Retail II ga o'xshash sample dataset generatsiya qilish.

        Real dataset mavjud bo'lmasa ham loyiha ishlashi uchun.

        Parameters
        ----------
        n : int
            Qatorlar soni (default: 1000)
        seed : int
            Random seed

        Returns
        -------
        pandas.DataFrame
            Online Retail II formatidagi DataFrame
        """
        # utils modulida joylashgan generator dan foydalanamiz
        from src.utils import generate_sample_csv
        return generate_sample_csv(n_rows=n, seed=seed)

    # ═══════════════════════════════════════════════════════════════
    # Dashboard statistikasi
    # ═══════════════════════════════════════════════════════════════

    def get_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Dashboard uchun asosiy ko'rsatkichlar lug'ati.

        Parameters
        ----------
        df : pandas.DataFrame
            Tozalangan dataset

        Returns
        -------
        dict
            Barcha asosiy KPI larni o'z ichiga oluvchi lug'at
        """
        stats: Dict[str, Any] = {
            "total_transactions": int(len(df)),
            "unique_customers": int(df["Customer ID"].nunique()) if "Customer ID" in df.columns else 0,
            "unique_products": int(df["StockCode"].nunique()) if "StockCode" in df.columns else 0,
            "unique_countries": int(df["Country"].nunique()) if "Country" in df.columns else 0,
        }

        if "TotalPrice" in df.columns:
            stats["total_revenue"] = float(df["TotalPrice"].sum())
            stats["avg_order_value"] = float(df["TotalPrice"].mean())
        elif "Price" in df.columns and "Quantity" in df.columns:
            total = (df["Price"] * df["Quantity"]).sum()
            stats["total_revenue"] = float(total)
            stats["avg_order_value"] = float(total / max(len(df), 1))

        if "Price" in df.columns:
            stats["avg_price"] = float(df["Price"].mean())
            stats["median_price"] = float(df["Price"].median())
            stats["min_price"] = float(df["Price"].min())
            stats["max_price"] = float(df["Price"].max())

        if "Quantity" in df.columns:
            stats["total_quantity"] = int(df["Quantity"].sum())
            stats["avg_quantity"] = float(df["Quantity"].mean())

        if "InvoiceDate" in df.columns:
            stats["date_min"] = str(df["InvoiceDate"].min())
            stats["date_max"] = str(df["InvoiceDate"].max())

        if "Country" in df.columns:
            top_country = df["Country"].value_counts().head(1)
            if len(top_country) > 0:
                stats["top_country"] = str(top_country.index[0])
                stats["top_country_share"] = float(top_country.iloc[0] / len(df))

        return stats
