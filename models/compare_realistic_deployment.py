"""Rentabilidad real: ahorras a ritmo fijo y decides cuando desplegar.

Los modulos anteriores igualaban el capital entre estrategias y solo cambiaban
el *cuando*, lo que aisla la calidad de la senal pero regala algo imposible:
saber de antemano cuantas veces va a disparar para repartir el presupuesto
entre esos disparos. Bajo ese supuesto las senales de profundidad baten al RSI
por 28 puntos.

Aqui se quita el supuesto. El dinero entra a razon fija, espera en efectivo y se
despliega entero cuando la senal dispara. Mismo total aportado en todas; lo
unico que cambia es cuanto tiempo estuvo el dinero invertido en vez de parado.

El resultado invierte el ranking: las senales que compran mas barato son las que
mas tiempo pasan esperando, y estar fuera del mercado cuesta mas que lo que se
ahorra en el precio de entrada. El optimo esta en ser POCO selectivo, dejando un
5-10% del capital en efectivo, no un 20-25%.

Dos advertencias sobre lo que el grafico no dice. La mediana del multiplo y el
porcentaje de series ganadas pueden discrepar, porque una estrategia puede ganar
a menudo y perder fuerte cuando pierde: por eso van los dos paneles. Y todo esto
mide 2000-2026, un periodo en el que todo acabo recuperandose, que es
exactamente el supuesto del que viven las senales de caida.
"""

import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from compare_entry_signals import (
    BLUE,
    GRID,
    INK,
    MARKET_OF,
    MUTED,
    OUT_DIR,
    RED,
    SURFACE,
    build_catalog,
    load_prices,
    sign_test,
)

WEEKLY = 100.0
DCA = "DCA (cada semana)"
# El efectivo a la espera renta algo; entre 2000 y 2026 el tipo corto ronda el
# 2%. Se calculan los dos extremos porque el ranking depende del supuesto.
CASH_RATES = (0.0, 0.02)
HEADLINE_RATE = 0.02

MIN_FIRES = 5  # una senal que dispara menos no es evaluable
MIN_SERIES = 40  # y debe serlo en casi todas las series
SHOWN = 18  # cuantas entran en el grafico

# Una estrategia que bate al DCA en poco mas de la mitad de las series no se
# distingue de una moneda al aire por muy alto que sea su ROI mediano, asi que
# se exige un margen claro antes de entrar en el grafico.
MIN_WIN_RATE = 60.0

# El DCA va siempre: es la vara de medir, no un competidor, y por definicion no
# puede superar su propio umbral de acierto.
ALWAYS_SHOWN = (DCA,)


def simulate(prices: pd.Series, mask: pd.Series, annual_cash_rate: float) -> dict:
    """Aporta WEEKLY cada semana y despliega todo el efectivo al disparar.

    Version de referencia, semana a semana. La usa el self-test para validar la
    vectorizada, que es la que barre el catalogo entero.
    """
    weekly_rate = (1.0 + annual_cash_rate) ** (1.0 / 52.0) - 1.0
    fires = mask.reindex(prices.index).fillna(False).to_numpy(dtype=bool)
    values = prices.to_numpy(dtype=float)

    cash = 0.0
    shares = 0.0
    weeks_invested = 0
    for position, price in enumerate(values):
        cash = cash * (1.0 + weekly_rate) + WEEKLY
        if fires[position]:
            shares += cash / price
            cash = 0.0
        if shares > 0.0:
            weeks_invested += 1

    contributed = WEEKLY * len(values)
    final = shares * values[-1] + cash
    return {
        "multiplo": final / contributed,
        "roi_pct": (final / contributed - 1.0) * 100.0,
        "pct_tiempo_invertido": weeks_invested / len(values) * 100.0,
        "efectivo_final_pct": cash / final * 100.0 if final else np.nan,
    }


def simulate_fast(prices: np.ndarray, fires: np.ndarray, annual_cash_rate: float) -> dict:
    """Lo mismo, vectorizado, para poder barrer el catalogo entero.

    El efectivo acumulado entre dos disparos es una suma geometrica: si se
    descuenta cada aporte al origen, lo acumulado en un tramo es una diferencia
    de sumas acumuladas y basta reinflarlo a la fecha del disparo.
    """
    n = len(prices)
    weekly_rate = (1.0 + annual_cash_rate) ** (1.0 / 52.0) - 1.0
    growth = (1.0 + weekly_rate) ** np.arange(n)
    accumulated = np.cumsum(WEEKLY / growth)
    contributed = WEEKLY * n

    idx = np.flatnonzero(fires)
    if len(idx) == 0:
        cash = accumulated[-1] * growth[-1]
        return {
            "multiplo": cash / contributed,
            "roi_pct": (cash / contributed - 1.0) * 100.0,
            "pct_tiempo_invertido": 0.0,
            "efectivo_final_pct": 100.0,
        }

    previous = np.concatenate(([0.0], accumulated[idx[:-1]]))
    cash_at_fire = growth[idx] * (accumulated[idx] - previous)
    shares = float(np.sum(cash_at_fire / prices[idx]))
    cash = growth[-1] * (accumulated[-1] - accumulated[idx[-1]])
    final = shares * prices[-1] + cash

    return {
        "multiplo": final / contributed,
        "roi_pct": (final / contributed - 1.0) * 100.0,
        "pct_tiempo_invertido": (n - int(idx[0])) / n * 100.0,
        "efectivo_final_pct": cash / final * 100.0 if final else np.nan,
    }


def self_test(prices: pd.Series, catalog: Dict) -> None:
    mask = catalog["RSI(14) < 30"][1](prices).reindex(prices.index).fillna(False)
    slow = simulate(prices, mask, HEADLINE_RATE)["multiplo"]
    fast = simulate_fast(prices.to_numpy(float), mask.to_numpy(bool), HEADLINE_RATE)["multiplo"]
    assert abs(fast - slow) / slow < 1e-9, (fast, slow)
    print("self-test OK: vectorizada == bucle semana a semana ({0:.4f}x)".format(fast))


def run(data: Dict[str, pd.Series], catalog: Dict, min_series: int = MIN_SERIES) -> pd.DataFrame:
    """Barre el catalogo ENTERO, no una seleccion a mano.

    Elegir a dedo que estrategias se comparan es la forma mas facil de
    esconder a las buenas, asi que la seleccion para el grafico se hace
    despues y por resultado.
    """
    arrays = {ticker: prices.to_numpy(float) for ticker, prices in data.items()}
    candidates = dict(catalog)
    candidates[DCA] = ("Pasiva", lambda prices: pd.Series(True, index=prices.index))

    rows: List[dict] = []
    for position, (name, (family, fn)) in enumerate(candidates.items(), start=1):
        measured: List[dict] = []
        for ticker, prices in data.items():
            fires = fn(prices).reindex(prices.index).fillna(False).to_numpy(bool)
            if fires.sum() < MIN_FIRES:
                continue
            for rate in CASH_RATES:
                measured.append(
                    {
                        "strategy": name,
                        "family": family,
                        "ticker": ticker,
                        # La serie puede no ser del universo fijo: el analisis de
                        # una sola bolsa por cohortes reutiliza este barrido y sus
                        # "tickers" son periodos, no simbolos.
                        "market": MARKET_OF.get(ticker, "Renta variable"),
                        "cash_rate": rate,
                        **simulate_fast(arrays[ticker], fires, rate),
                    }
                )
        if len(measured) >= min_series * len(CASH_RATES):
            rows.extend(measured)
        if position % 50 == 0:
            print("  ... {0}/{1} configuraciones".format(position, len(candidates)))

    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    equity = detail[detail["market"] != "Materias primas"]

    rows: List[dict] = []
    for rate, block in equity.groupby("cash_rate"):
        base = block[block["strategy"] == DCA].set_index("ticker")["multiplo"]
        for name, group in block.groupby("strategy"):
            multiples = group.set_index("ticker")["multiplo"]
            # Pareado contra el DCA dentro de cada serie, nunca entre medianas.
            diff = (multiples - base).dropna()
            rows.append(
                {
                    "strategy": name,
                    "cash_rate": rate,
                    "n_series": len(multiples),
                    "roi_mediano_pct": float(group["roi_pct"].median()),
                    "multiplo_mediano": float(multiples.median()),
                    "gana_a_dca_pct": float((diff > 0).mean() * 100.0),
                    "p_signo": sign_test(diff),
                    "peor_vs_dca_pct": float(((multiples / base - 1.0) * 100.0).min()),
                    "pct_tiempo_invertido": float(group["pct_tiempo_invertido"].median()),
                    "efectivo_final_pct": float(group["efectivo_final_pct"].median()),
                }
            )

    return pd.DataFrame(rows)


def select_for_plot(
    summary: pd.DataFrame,
    min_win_rate: float = MIN_WIN_RATE,
    tail: int = 0,
) -> List[str]:
    """Las mejores por resultado, mas las referencias. Nunca a dedo.

    El corte de acierto llega por parametro porque no siempre lo supera alguien:
    sobre una sola bolsa con un siglo de historia no lo pasa ninguna, y un
    grafico vacio dice menos que el ranking con el corte anotado aparte.

    `tail` anade las N peores por rentabilidad. Cuando ninguna supera el corte,
    el top por ROI se degenera en "las que disparan mas" y las 18 barras salen
    identicas; con la cola se ve el rango entero, que es donde esta el resultado.
    """
    headline = summary[summary["cash_rate"] == HEADLINE_RATE]
    reliable = headline[headline["gana_a_dca_pct"] >= min_win_rate]
    ordered = reliable.sort_values("roi_mediano_pct", ascending=False)
    picked = list(ordered.head(SHOWN - tail)["strategy"])
    if tail:
        picked += list(ordered.tail(tail)["strategy"])
    names = list(dict.fromkeys(picked + list(ALWAYS_SHOWN)))
    return [name for name in names if name in set(headline["strategy"])]


def plot(
    summary: pd.DataFrame,
    n_series: int,
    out_path: Path,
    sample: str = "series de renta variable",
    period: str = "2000-{0}".format(dt.date.today().year),
    unit: str = "series",
    min_win_rate: float = MIN_WIN_RATE,
    headline_note: Optional[str] = None,
    tail: int = 0,
) -> None:
    chosen = select_for_plot(summary, min_win_rate, tail)
    headline = summary[
        (summary["cash_rate"] == HEADLINE_RATE) & (summary["strategy"].isin(chosen))
    ].copy()
    headline = headline.sort_values("roi_mediano_pct")
    conservative = summary[summary["cash_rate"] == 0.0].set_index("strategy")

    dca_roi = float(headline[headline["strategy"] == DCA]["roi_mediano_pct"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(17, 8.5), facecolor=SURFACE, sharey=True)

    # --- Izquierda: la rentabilidad, que es lo que se pregunta --------------
    ax = axes[0]
    colors = [MUTED if name == DCA else BLUE for name in headline["strategy"]]
    values = headline["roi_mediano_pct"].to_numpy()
    ax.barh(headline["strategy"], values, color=colors, height=0.66)

    # El mismo calculo sin remunerar el efectivo: si el rombo se aleja de la
    # barra, la ventaja venia del interes y no de acertar la entrada.
    ax.scatter(
        [conservative.loc[name, "roi_mediano_pct"] for name in headline["strategy"]],
        range(len(headline)),
        marker="D",
        s=26,
        color=INK,
        zorder=4,
        label="Con efectivo al 0%",
    )
    ax.axvline(dca_roi, color=MUTED, linewidth=1.0, linestyle="--", zorder=1)

    for y, value in enumerate(values):
        ax.text(
            value + values.max() * 0.015,
            y,
            "{0:,.0f}%".format(value),
            va="center",
            ha="left",
            color=INK,
            fontsize=8.5,
        )
    ax.set_title("Rentabilidad sobre todo lo aportado", color=INK, fontsize=12, pad=10)
    ax.set_xlabel(
        "ROI mediano sobre el total aportado, efectivo al 2% (%)", color=MUTED, fontsize=9
    )
    legend = ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.02, 0.06))
    for text in legend.get_texts():
        text.set_color(INK)

    # --- Derecha: con que frecuencia gana, que es lo que decide -------------
    ax = axes[1]
    names = headline["strategy"].tolist()
    # El DCA es la referencia contra la que se mide, no un competidor: se deja
    # sin barra en vez de dibujarlo ganandose a si mismo el 0% de las veces.
    wins = np.array(
        [
            np.nan if name == DCA else win - 50.0
            for name, win in zip(names, headline["gana_a_dca_pct"])
        ]
    )
    ax.barh(
        names,
        np.nan_to_num(wins),
        color=[BLUE if win >= 0 else RED for win in np.nan_to_num(wins)],
        height=0.66,
    )
    ax.axvline(0, color=MUTED, linewidth=1.0)
    for y, (win, p_value, name) in enumerate(zip(wins, headline["p_signo"], names)):
        if name == DCA:
            ax.text(1.2, y, "referencia", va="center", ha="left", color=MUTED, fontsize=8.5)
            continue
        mark = "  p={0:.2f}".format(p_value) if p_value < 0.10 else ""
        ax.text(
            win + (1.2 if win >= 0 else -1.2),
            y,
            "{0:.0f}%{1}".format(win + 50.0, mark),
            va="center",
            ha="left" if win >= 0 else "right",
            color=INK,
            fontsize=8.5,
        )
    ax.set_title(
        "En cuantas {0} bate al DCA (50% = moneda al aire)".format(unit),
        color=INK,
        fontsize=12,
        pad=10,
    )
    ax.set_xlabel(
        "{0} ganadas frente al DCA, desviacion sobre el 50% (pp)".format(unit.capitalize()),
        color=MUTED,
        fontsize=9,
    )
    # Ticks derivados del rango real: con un filtro de acierto distinto, unos
    # valores fijos dejarian barras fuera del ultimo tick.
    finite = wins[np.isfinite(wins)]
    low = int(np.floor((finite.min() + 50.0) / 10.0) * 10)
    high = int(np.ceil((finite.max() + 50.0) / 10.0) * 10)
    ticks = list(range(min(low, 50), max(high, 50) + 1, 10))
    ax.set_xticks([tick - 50 for tick in ticks])
    ax.set_xticklabels(["{0}%".format(tick) for tick in ticks])

    for axis in axes:
        axis.set_facecolor(SURFACE)
        axis.grid(True, axis="x", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=MUTED, labelsize=9)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.spines["bottom"].set_color(GRID)
        axis.margins(x=0.16)

    for label in axes[0].get_yticklabels():
        if label.get_text() == DCA:
            label.set_color(INK)
            label.set_fontweight("bold")

    note = headline_note or "Solo las que baten al DCA en al menos el {0:.0f}% de las {1}".format(
        min_win_rate, unit
    )
    fig.suptitle(
        "Mejores estrategias por rentabilidad real, aportando 100 EUR/semana\n"
        "{0}   ({1} {2}, {3})".format(note, n_series, sample, period),
        color=INK,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_prices()
    catalog = build_catalog()
    self_test(next(iter(data.values())), catalog)

    detail = run(data, catalog)
    summary = summarize(detail)

    detail_path = OUT_DIR / "realistic_deployment_detail.csv"
    summary_path = OUT_DIR / "realistic_deployment_summary.csv"
    plot_path = OUT_DIR / "realistic_deployment_ranking.png"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    n_series = detail[detail["market"] != "Materias primas"]["ticker"].nunique()
    plot(summary, n_series, plot_path)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.float_format", lambda value: "{0:,.2f}".format(value))

    columns = [
        "strategy",
        "roi_mediano_pct",
        "gana_a_dca_pct",
        "p_signo",
        "peor_vs_dca_pct",
        "pct_tiempo_invertido",
        "efectivo_final_pct",
    ]
    for rate in CASH_RATES:
        block = summary[summary["cash_rate"] == rate].sort_values(
            "roi_mediano_pct", ascending=False
        )
        dca_roi = float(block[block["strategy"] == DCA]["roi_mediano_pct"].iloc[0])
        beats = block[block["roi_mediano_pct"] > dca_roi]
        often = block[block["gana_a_dca_pct"] > 50.0]
        solid = often[often["p_signo"] < 0.05]

        print("\n=== Efectivo a la espera al {0:.0%} · {1} series ===".format(rate, n_series))
        print("DCA de referencia: {0:.1f}% de ROI mediano".format(dca_roi))
        print("   con ROI por encima del DCA : {0:>3} de {1}".format(len(beats), len(block)))
        print("   ganan en >50% de las series: {0:>3} de {1}".format(len(often), len(block)))
        print("   ... y ademas con p < 0.05  : {0:>3} de {1}".format(len(solid), len(block)))
        reliable = block[block["gana_a_dca_pct"] >= MIN_WIN_RATE]
        print(
            "   ganan en >={0:.0f}% de las series: {1:>3} de {2}".format(
                MIN_WIN_RATE, len(reliable), len(block)
            )
        )
        print("\nLas que superan ese {0:.0f}%, por rentabilidad:".format(MIN_WIN_RATE))
        print(reliable[columns].to_string(index=False))

    print("\nDetalle: {0}".format(detail_path))
    print("Resumen: {0}".format(summary_path))
    print("Grafico: {0}".format(plot_path))


if __name__ == "__main__":
    main()
