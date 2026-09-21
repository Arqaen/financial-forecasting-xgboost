"""Compara las estrategias de compare_strategies_simple con el capital igualado.

El backtest original deja que cada estrategia despliegue el capital que quiera
(RSI acaba invirtiendo ~$4k por ticker y MA MODE ~$52k), asi que su ROI sobre
capital invertido mide sobre todo la tasa de despliegue y no la calidad de la
senal. Aqui se reparte un presupuesto identico a partes iguales entre las
semanas en que cada senal dispara: todas invierten lo mismo y solo cambia el
*cuando*, de forma que el precio medio pagado aisla la calidad de la senal.

Con aportacion constante b en N semanas a precios p_i, el valor final es
P_end * b * sum(1/p_i), es decir proporcional al inverso de la media armonica de
los precios de compra. Esa media armonica es la metrica central del modulo.
"""

import datetime as dt
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_strategies_simple import (
    OUT_DIR,
    STRATEGY_ORDER,
    TICKERS,
    build_strategy_curves,
    calculate_rsi,
    fetch_weekly_close,
    simulate_signal_strategy,
)

# El original usa nz(rsi) < 30, y nz() convierte el NaN de warmup en 0, que
# cumple 0 < 30: la senal dispara en la primera semana de la serie por un fallo
# de arranque, no por sobreventa. Esta variante compara con la version correcta.
RSI_FIXED = "RSI MODE (warmup corregido)"

REPORT_ORDER = tuple(STRATEGY_ORDER) + (RSI_FIXED,)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
RED = "#e34948"


def harmonic_mean_price(prices: pd.Series) -> float:
    """Precio medio efectivo al repartir el mismo dinero en cada fecha dada."""
    values = prices.to_numpy(dtype=float)
    return float(len(values) / np.sum(1.0 / values))


def evaluate_ticker(ticker: str, prices: pd.Series, company_name: str) -> List[dict]:
    curves: Dict[str, pd.DataFrame] = build_strategy_curves(prices)

    # NaN < 30 es False, asi que esta senal no dispara durante el warmup.
    curves[RSI_FIXED] = simulate_signal_strategy(prices, calculate_rsi(prices) < 30.0)

    price_end = float(prices.iloc[-1])
    uniform_full = harmonic_mean_price(prices)
    rows: List[dict] = []

    for strategy in REPORT_ORDER:
        curve = curves.get(strategy)
        if curve is None:
            continue

        signal_dates = curve.index[curve["contribution"] > 0]
        if len(signal_dates) == 0:
            continue

        paid = harmonic_mean_price(prices.loc[signal_dates])
        # Compra uniforme en la misma ventana natural de la senal: neutraliza la
        # ventaja de limitarse a los primeros anios de una serie alcista.
        window = prices.loc[signal_dates[0] : signal_dates[-1]]
        uniform_window = harmonic_mean_price(window)

        rows.append(
            {
                "ticker": ticker,
                "company_name": company_name,
                "strategy": strategy,
                "n_signals": int(len(signal_dates)),
                "first_signal": signal_dates[0].date().isoformat(),
                "last_signal": signal_dates[-1].date().isoformat(),
                "avg_price_paid": paid,
                "uniform_window_price": uniform_window,
                "uniform_full_price": uniform_full,
                "price_end": price_end,
                # Mismo presupuesto para todas: quien acaba mas rico.
                "roi_normalized_pct": (price_end / paid - 1.0) * 100.0,
                # Calidad pura de la senal: comprar mas barato que repartir el
                # mismo dinero uniformemente en la misma ventana.
                "edge_vs_window_pct": (uniform_window / paid - 1.0) * 100.0,
                "edge_vs_full_pct": (uniform_full / paid - 1.0) * 100.0,
            }
        )

    return rows


def run() -> pd.DataFrame:
    rows: List[dict] = []

    for ticker in TICKERS:
        try:
            prices, company_name = fetch_weekly_close(ticker)
        except Exception as exc:
            print("SKIP {0}: {1}".format(ticker, exc))
            continue

        rows.extend(evaluate_ticker(ticker, prices, company_name))
        print("OK {0}: {1} semanas".format(ticker, len(prices)))

    return pd.DataFrame(rows)


def aggregate(detail: pd.DataFrame) -> pd.DataFrame:
    grouped = detail.groupby("strategy")
    summary = pd.DataFrame(
        {
            "n_signals_medio": grouped["n_signals"].mean(),
            "roi_norm_medio": grouped["roi_normalized_pct"].mean(),
            "roi_norm_mediana": grouped["roi_normalized_pct"].median(),
            "edge_ventana_medio": grouped["edge_vs_window_pct"].mean(),
            "edge_ventana_mediana": grouped["edge_vs_window_pct"].median(),
            "pct_tickers_con_edge": grouped["edge_vs_window_pct"].apply(
                lambda serie: float((serie > 0).mean() * 100.0)
            ),
        }
    )
    return summary.reindex(REPORT_ORDER).dropna(how="all")


def plot_summary(summary: pd.DataFrame, n_tickers: int, out_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=SURFACE)

    panels = (
        (
            axes[0],
            summary["roi_norm_mediana"].sort_values(),
            "Quien acaba mas rico con el mismo capital",
            "ROI mediano con presupuesto igualado (%)",
            False,
        ),
        (
            axes[1],
            summary["edge_ventana_mediana"].sort_values(),
            "Calidad de la senal (vs. comprar uniforme)",
            "Descuento mediano sobre compra uniforme (%)",
            True,
        ),
    )

    for ax, series, title, xlabel, diverging in panels:
        colors = [BLUE if value >= 0 else RED for value in series] if diverging else BLUE
        ax.barh(series.index, series.to_numpy(), color=colors, height=0.62)
        ax.set_facecolor(SURFACE)
        ax.set_title(title, color=INK, fontsize=12, pad=12)
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

        if diverging:
            ax.axvline(0, color=MUTED, linewidth=1.0)

        span = float(series.abs().max()) or 1.0
        for y, value in enumerate(series.to_numpy()):
            offset = span * 0.015
            ax.text(
                value + (offset if value >= 0 else -offset),
                y,
                "{0:+.1f}%".format(value) if diverging else "{0:.0f}%".format(value),
                va="center",
                ha="left" if value >= 0 else "right",
                color=INK,
                fontsize=8.5,
            )
        ax.margins(x=0.16)

    fig.suptitle(
        "Estrategias con capital desplegado igualado  ({0} tickers, semanal, 2000-{1})".format(
            n_tickers, dt.date.today().year
        ),
        color=INK,
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    detail = run()
    if detail.empty:
        raise RuntimeError("No se pudo calcular ningun ticker")

    summary = aggregate(detail)

    detail_path = OUT_DIR / "normalized_capital_detail.csv"
    summary_path = OUT_DIR / "normalized_capital_summary.csv"
    plot_path = OUT_DIR / "normalized_capital_by_strategy.png"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path)
    plot_summary(summary, detail["ticker"].nunique(), plot_path)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda value: "{0:,.1f}".format(value))
    print("\n=== Capital igualado: solo cambia el CUANDO ===")
    print(summary.to_string())
    print("\nDetalle: {0}".format(detail_path))
    print("Resumen: {0}".format(summary_path))
    print("Grafico: {0}".format(plot_path))


if __name__ == "__main__":
    main()
