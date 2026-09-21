"""ROI promedio por estrategia y comprobacion de que ningun mercado manda.

Dos preguntas que el barrido de compare_entry_signals no contestaba:

1. El grafico original de compare_strategies_simple ordenaba por ROI *medio*
   sobre capital invertido, y ese ROI media sobre todo cuanto capital desplegaba
   cada estrategia. Aqui se redibuja el mismo grafico con el capital ya igualado,
   asi que las barras solo reflejan el *cuando* se compra.

2. Con 27 tickers no se podia descartar que el resultado lo sostuviera un solo
   mercado, porque nunca se desglosaba. El universo sube a 57 series en seis
   grupos y se exige que la mejora frente al RSI se repita en cada grupo por
   separado; si dependiera de un mercado, aqui se veria.

Dos advertencias que conviene no perder de vista.

Los mercados de renta variable caen a la vez: 2008 y 2020 golpearon a todos los
grupos, asi que "gana en cinco mercados" no son cinco pruebas independientes,
es una prueba repetida en sitios correlacionados. Descarta el sesgo de un
mercado concreto, no la dependencia del regimen global.

Y la media aritmetica de ROI no sobrevive a este universo: ^MERV cotiza en pesos
argentinos y acumula un ROI de seis cifras, asi que por si solo mueve la media
de 301% a 11.437%. El grafico la dibuja igualmente, porque es la metrica del
original, pero al lado va la mediana para que se vea el tamaño del problema.

El resultado real del desglose es que las senales de profundidad ganan en los
cinco mercados de renta variable y PIERDEN en materias primas. Tiene sentido:
son ciclicas y sin tendencia secular, asi que una caida del 40% no implica que
vaya a recuperarse, que es justo el supuesto del que viven estas senales.
"""

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from compare_entry_signals import (
    AQUA,
    BASELINE,
    BLUE,
    GRID,
    INK,
    MUTED,
    ORANGE,
    OUT_DIR,
    RED,
    SURFACE,
    UNIVERSE,
    build_catalog,
    load_prices,
    run,
    summarize,
)

# Se dibujan la referencia, las mejores de cada familia y los dos baselines
# pasivos, para que el grafico sea legible en vez de un muro de 179 barras.
SHOWN = [
    "(Retorno 52s <= -30%) Y (Precio <= MA50 -20%)",
    "(Retorno 52s <= -30%) Y (Estocastico(52) <= 10)",
    "(Caida >=40% desde max 104s) Y (RSI(28) < 40)",
    "(Caida >=40% desde max 104s) Y (Precio <= MA50 -20%)",
    "Caida >=40% desde max 52s",
    "Caida >=40% desde max 104s",
    "Retorno 52s <= -30%",
    "Precio <= MA50 -20%",
    "Precio <= MA30 -20%",
    "Percentil <= 5% de 156s",
    "Z(104) <= -2.5",
    "Estocastico(52) <= 10",
    "RSI(28) < 40",
    BASELINE,
    "RSI(7) < 30",
]


def market_breakdown(detail: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
    """Mejora media frente al RSI dentro de cada grupo de mercado."""
    base = detail[detail["strategy"] == BASELINE].set_index("ticker")["edge_vs_dca_pct"]

    rows = []
    for name in strategies:
        if name == BASELINE:
            continue
        group = detail[detail["strategy"] == name].set_index("ticker")
        diff = (group["edge_vs_dca_pct"] - base).dropna()
        if diff.empty:
            continue
        entry: Dict[str, Any] = {"estrategia": name}
        markets = group.loc[diff.index, "market"]
        for market in UNIVERSE:
            values = diff[markets == market]
            entry[market] = float(values.mean()) if len(values) else np.nan
            entry["n_" + market] = len(values)
        entry["TODOS"] = float(diff.mean())
        entry["mercados_ganados"] = int(
            sum(1 for market in UNIVERSE if pd.notna(entry[market]) and entry[market] > 0)
        )
        rows.append(entry)

    return pd.DataFrame(rows)


def plot_average_roi(summary: pd.DataFrame, n_tickers: int, out_path: Path) -> None:
    """El grafico original, con el capital igualado y en sus dos versiones.

    Va en dos paneles a proposito. La media aritmetica de ROI es la metrica del
    grafico original, pero ^MERV cotiza en pesos argentinos y acumula un ROI de
    seis cifras, asi que arrastra la media de todas las estrategias por igual y
    el orden que produce no significa nada. Ambos paneles van ordenados por la
    media: que el panel de la mediana NO salga escalonado es justamente la
    prueba de que ese orden no se sostiene.
    """
    shown = summary[summary["strategy"].isin(SHOWN)].copy()
    shown = shown.sort_values("roi_medio_pct")

    palette = {"Profundidad": BLUE, "Oscilador": ORANGE, "Combinacion": AQUA, "Otras": MUTED}
    colors = [palette[group] for group in shown["grupo"]]

    fig, axes = plt.subplots(1, 2, figsize=(17, 8), facecolor=SURFACE, sharey=True)

    panels = (
        (
            axes[0],
            "roi_medio_pct",
            "ROI medio (lo que pediste)",
            "Media aritmetica del ROI por serie (%)",
        ),
        (
            axes[1],
            "roi_mediano_pct",
            "ROI mediano (resistente a outliers)",
            "Mediana del ROI por serie (%)",
        ),
    )

    for ax, column, title, xlabel in panels:
        values = shown[column].to_numpy()
        ax.barh(shown["strategy"], values, color=colors, height=0.66)
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
        ax.set_title(title, color=INK, fontsize=12, pad=10)
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.margins(x=0.20)

    for label in axes[0].get_yticklabels():
        if label.get_text() == BASELINE:
            label.set_color(INK)
            label.set_fontweight("bold")

    fig.suptitle(
        "ROI promedio por estrategia, con el capital desplegado igualado\n"
        "Azul: profundidad de la caida  ·  Naranja: osciladores  ·  Verde: combinaciones"
        "   ({0} series, 6 mercados, semanal, 2000-{1})".format(n_tickers, dt.date.today().year),
        color=INK,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def plot_by_market(breakdown: pd.DataFrame, out_path: Path) -> None:
    """Una barra por mercado y estrategia: el sesgo de mercado se veria aqui."""
    top = breakdown.sort_values("TODOS", ascending=False).head(8)
    markets = list(UNIVERSE)

    fig, axes = plt.subplots(1, len(markets), figsize=(19, 6.4), facecolor=SURFACE, sharey=True)

    order = top.sort_values("TODOS")["estrategia"].tolist()
    for ax, market in zip(axes, markets):
        values = [float(top[top["estrategia"] == name][market].iloc[0]) for name in order]
        # Cuantas series del mercado se llegan a medir: una senal selectiva no
        # alcanza MIN_SIGNALS en las series cortas, y varia por estrategia.
        measured = int(top["n_" + market].median())
        colors = [BLUE if value >= 0 else RED for value in values]
        ax.barh(order, values, color=colors, height=0.68)
        ax.axvline(0, color=MUTED, linewidth=1.0)
        ax.set_title(
            "{0}\n{1} de {2} series".format(market, measured, len(UNIVERSE[market])),
            color=INK,
            fontsize=10,
            pad=8,
        )
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    fig.suptitle(
        "La ventaja sobre RSI(14) < 30 se repite en cinco mercados y se invierte en materias primas"
        "   (mejora media del edge, en puntos porcentuales)",
        color=INK,
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_prices()
    detail = run(build_catalog(), data)
    summary = summarize(detail)
    breakdown = market_breakdown(detail, summary["strategy"].tolist())

    detail.to_csv(OUT_DIR / "entry_signals_detail.csv", index=False)
    summary.to_csv(OUT_DIR / "entry_signals_summary.csv", index=False)
    breakdown.to_csv(OUT_DIR / "entry_signals_by_market.csv", index=False)

    roi_path = OUT_DIR / "average_roi_equal_capital.png"
    market_path = OUT_DIR / "entry_signals_by_market.png"
    plot_average_roi(summary, detail["ticker"].nunique(), roi_path)
    plot_by_market(breakdown, market_path)

    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.float_format", lambda value: "{0:,.1f}".format(value))

    print(
        "\n=== ROI promedio con capital igualado ({0} series) ===".format(
            detail["ticker"].nunique()
        )
    )
    shown = summary[summary["strategy"].isin(SHOWN)].sort_values("roi_medio_pct", ascending=False)
    print(
        shown[
            [
                "strategy",
                "grupo",
                "disparos",
                "roi_medio_pct",
                "roi_mediano_pct",
                "edge_medio_pct",
                "mejora_media_pp",
                "gana_a_rsi_pct",
                "p_signo",
            ]
        ].to_string(index=False)
    )

    print("\n=== Mejora media frente a RSI(14)<30, por mercado (pp) ===")
    columns = ["estrategia"] + list(UNIVERSE) + ["TODOS", "mercados_ganados"]
    print(breakdown.sort_values("TODOS", ascending=False).head(15)[columns].to_string(index=False))

    print("\n=== Cuantas configuraciones ganan en los 6 mercados ===")
    counts = breakdown["mercados_ganados"].value_counts().sort_index(ascending=False)
    for won, how_many in counts.items():
        print("   {0} de 6 mercados: {1} configuraciones".format(won, how_many))

    print("\nROI:      {0}".format(roi_path))
    print("Mercados: {0}".format(market_path))


if __name__ == "__main__":
    main()
