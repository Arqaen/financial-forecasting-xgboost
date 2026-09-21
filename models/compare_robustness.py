"""Cuales sobreviven a un filtro anti-sobreajuste, y cuales solo lo parecian.

El ranking por rentabilidad de `compare_realistic_deployment` responde a "que
habria funcionado mejor entre 2000 y 2026". Esa pregunta se contesta sola: con
165 configuraciones probadas, la primera del ranking lo es en parte por merito y
en parte porque alguna tenia que salir primera. Aqui se le exige a cada una lo
que una casualidad no suele cumplir:

  1. Significancia   - el test de signos sobre las series da p < 0,05.
  2. Estabilidad     - gana al DCA en la primera mitad de la historia Y en la
                       segunda, evaluando cada mitad contra su propio DCA.
  3. Amplitud        - gana en al menos cuatro de los cinco mercados de renta
                       variable, no solo en el que mas pesa.
  4. Muestra         - dispara al menos 15 veces por serie; con 10 entradas en
                       26 anos el historial son diez anecdotas.
  5. Coherencia      - sus hermanas de familia con otro umbral tambien ganan.
                       Un pico aislado en la rejilla de parametros es ruido; una
                       meseta es estructura.

El resultado importa mas que la lista: las que pasan los cinco filtros son
justo las MENOS rentables del grafico anterior, porque son las que apenas se
separan del DCA. Las que prometian un 183% caen todas, y las de RSI caen por un
motivo concreto que conviene ver: dejan de dispararse despues de 2013.
"""

import datetime as dt
from pathlib import Path
from typing import Callable, Dict, List, Tuple

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
    SURFACE,
    build_catalog,
    load_prices,
    sign_test,
)
from compare_realistic_deployment import WEEKLY

CACHE = OUT_DIR / "prices_weekly.pkl"
CASH_RATE = 0.02  # mismo supuesto que el ranking de rentabilidad real
MIN_FIRES = 5  # por debajo de esto la serie no es evaluable
MIN_SERIES = 40  # y debe serlo en casi todas
MIN_FIRES_ERA = 3  # disparos minimos dentro de una mitad
MIN_SERIES_ERA = 30  # mitades evaluables para que la mitad cuente

# Los cinco filtros. Cada uno es un predicado sobre la fila de resultados, de
# modo que el grafico pueda dibujar la matriz de aprobados sin duplicar logica.
FILTERS: List[Tuple[str, str, Callable]] = [
    ("Significativa", "p < 0,05", lambda r: r["p"] < 0.05),
    ("1a mitad", "gana al DCA en 2000-2013", lambda r: r["era1"] > 50),
    ("2a mitad", "gana al DCA en 2013-2026", lambda r: r["era2"] > 50),
    ("Mercados", ">= 4 de 5 mercados", lambda r: r["mercados_ganados"] >= 4),
    ("Muestra", ">= 15 disparos", lambda r: r["fires"] >= 15),
    ("Familia", ">= 60% de sus hermanas gana", lambda r: r["familia_coherente_pct"] >= 60),
]

MIN_WIN_RATE = 60.0  # el corte del grafico anterior, para poder compararlos
SHOWN = 14  # candidatas por rentabilidad que entran en el grafico


def final_value(prices: np.ndarray, fires: np.ndarray) -> float:
    """Aporta WEEKLY cada semana y despliega todo el efectivo al disparar.

    El efectivo acumulado entre dos disparos es una suma geometrica: si se
    descuenta cada aporte al origen, lo acumulado es una diferencia de sumas
    acumuladas que basta reinflar a la fecha del disparo. Asi el barrido de 165
    configuraciones por 47 series sale sin bucle por semana.
    """
    n = len(prices)
    weekly = (1.0 + CASH_RATE) ** (1.0 / 52.0) - 1.0
    growth = (1.0 + weekly) ** np.arange(n)
    accumulated = np.cumsum(WEEKLY / growth)

    idx = np.flatnonzero(fires)
    if len(idx) == 0:
        return float(accumulated[-1] * growth[-1])

    previous = np.concatenate(([0.0], accumulated[idx[:-1]]))
    cash_at_fire = growth[idx] * (accumulated[idx] - previous)
    shares = float(np.sum(cash_at_fire / prices[idx]))
    leftover = growth[-1] * (accumulated[-1] - accumulated[idx[-1]])
    return float(shares * prices[-1] + leftover)


def win_rate(values: Dict[str, float], reference: Dict[str, float]) -> float:
    """Porcentaje de series en las que la estrategia acaba por encima del DCA."""
    diff = [values[t] - reference[t] for t in values if t in reference]
    return float(np.mean([d > 0 for d in diff]) * 100.0) if diff else np.nan


def evaluate_all(data: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, float]:
    """Mide las 165 configuraciones en la ventana completa y en cada mitad."""
    arrays = {t: p.to_numpy(float) for t, p in data.items()}
    contributed = {t: WEEKLY * len(p) for t, p in arrays.items()}
    halves = {t: len(p) // 2 for t, p in arrays.items()}

    # Cada mitad se compara contra un DCA corrido sobre esa misma mitad: de otro
    # modo se estaria comparando 13 anos de aportes contra 26.
    windows = {
        "full": {t: (p, 0) for t, p in arrays.items()},
        "era1": {t: (p[: halves[t]], 0) for t, p in arrays.items()},
        "era2": {t: (p[halves[t] :], halves[t]) for t, p in arrays.items()},
    }
    dca = {
        window: {t: final_value(p, np.ones(len(p), bool)) for t, (p, _) in series.items()}
        for window, series in windows.items()
    }

    rows: List[dict] = []
    for name, (family, fn) in build_catalog().items():
        fires = {t: fn(p).reindex(p.index).fillna(False).to_numpy(bool) for t, p in data.items()}
        usable = {t: m for t, m in fires.items() if m.sum() >= MIN_FIRES}
        if len(usable) < MIN_SERIES:
            continue

        full = {t: final_value(arrays[t], m) for t, m in usable.items()}
        diff = pd.Series({t: full[t] - dca["full"][t] for t in full})

        era: Dict[str, float] = {}
        for window in ("era1", "era2"):
            partial = {}
            for ticker, (prices, offset) in windows[window].items():
                mask = fires[ticker][offset : offset + len(prices)]
                if mask.sum() >= MIN_FIRES_ERA:
                    partial[ticker] = final_value(prices, mask)
            era[window] = (
                win_rate(partial, dca[window]) if len(partial) >= MIN_SERIES_ERA else np.nan
            )

        markets = {}
        for market in sorted({MARKET_OF[t] for t in full}):
            subset = {t: v for t, v in full.items() if MARKET_OF[t] == market}
            if len(subset) >= 4:
                markets[market] = win_rate(subset, dca["full"])

        rows.append(
            {
                "strategy": name,
                "family": family,
                "roi": float(np.median([full[t] / contributed[t] - 1.0 for t in full]) * 100.0),
                "wins": float((diff > 0).mean() * 100.0),
                "p": sign_test(diff),
                "fires": float(np.median([m.sum() for m in usable.values()])),
                "era1": era["era1"],
                "era2": era["era2"],
                "mercados_ganados": sum(v > 50 for v in markets.values()),
                "mercados": len(markets),
                "series": len(full),
            }
        )

    table = pd.DataFrame(rows)
    # La salud de una familia es que fraccion de sus umbrales bate al DCA. Se
    # calcula sobre la familia entera, no solo sobre las que llegan al grafico.
    health = table.groupby("family")["wins"].apply(lambda s: float((s > 50).mean() * 100.0))
    table["familia_coherente_pct"] = table["family"].map(health)
    table["familia_n"] = table["family"].map(table.groupby("family").size())

    for label, _, test in FILTERS:
        table[label] = table.apply(test, axis=1)
    table["filtros_superados"] = table[[label for label, _, _ in FILTERS]].sum(axis=1)

    dca_roi = float(np.median([dca["full"][t] / contributed[t] - 1.0 for t in arrays]) * 100.0)
    return table, dca_roi


def select_for_plot(
    table: pd.DataFrame,
    n_filters: int = len(FILTERS),
    min_win_rate: float = MIN_WIN_RATE,
    tail: int = 0,
) -> pd.DataFrame:
    """Las candidatas del ranking anterior, mas cualquiera que pase los cinco.

    La seleccion sale de los datos: las mejores por rentabilidad entre las que
    superan el corte de acierto, unidas a todas las supervivientes. Asi el
    grafico no puede omitir a una ganadora por como se escribio la lista.

    `tail` anade las N peores por rentabilidad, para cuando el top por si solo
    no distingue nada porque ninguna configuracion supera el corte.
    """
    candidates = table[table["wins"] >= min_win_rate].sort_values("roi", ascending=False)
    survivors = table[table["filtros_superados"] == n_filters]
    picked = [candidates.head(SHOWN - tail), survivors]
    if tail:
        picked.append(candidates.tail(tail))
    chosen = pd.concat(picked).drop_duplicates("strategy")
    return chosen.sort_values("roi", ascending=False)


def plot(
    chosen: pd.DataFrame,
    dca_roi: float,
    total: int,
    path: Path,
    filters: List[Tuple[str, str, Callable]] = FILTERS,
    sample: str = "47 series de renta variable, efectivo al 2%",
    period: str = "2000-2026",
    source: str = "models/compare_robustness.py",
) -> None:
    """Rentabilidad a la izquierda, matriz de aprobados a la derecha.

    Los filtros llegan por parametro porque no son los mismos cuando la muestra
    son 47 bolsas que cuando es una sola con un siglo de historia: alli se pide
    amplitud geografica, aqui amplitud temporal.
    """
    labels = list(chosen["strategy"])
    y = np.arange(len(labels))[::-1]
    passes_all = chosen["filtros_superados"] == len(filters)

    figure, (left, right) = plt.subplots(
        1,
        2,
        figsize=(15.5, 0.46 * len(labels) + 3.1),
        gridspec_kw={"width_ratios": [1.65, 1.0], "wspace": 0.06},
    )
    figure.patch.set_facecolor(SURFACE)
    # Los dos paneles comparten fila, asi que comparten limite vertical: si se
    # deja que cada uno lo calcule por su cuenta, las barras y los circulos se
    # desalinean y la matriz deja de leerse como continuacion de la barra.
    span = (y.min() - 0.7, y.max() + 0.7)

    # --- izquierda: rentabilidad, atenuada si la estrategia no sobrevive ----
    for axis in (left, right):
        axis.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(GRID)
        axis.tick_params(colors=MUTED, length=0)

    colors = [BLUE if ok else GRID for ok in passes_all]
    left.barh(y, chosen["roi"], height=0.62, color=colors, zorder=3)
    left.axvline(dca_roi, color=INK, linestyle="--", linewidth=1.0, zorder=4)
    left.text(
        dca_roi + 2.0,
        y.min() - 0.5,
        "DCA {0:.0f}%".format(dca_roi),
        color=INK,
        fontsize=9,
        ha="left",
        va="bottom",
    )
    for position, value in zip(y, chosen["roi"]):
        left.text(
            value + 2.5, position, "{0:.0f}%".format(value), va="center", fontsize=9, color=INK
        )

    left.set_yticks(y)
    left.set_yticklabels(labels, fontsize=9.5, color=INK)
    for tick, ok in zip(left.get_yticklabels(), passes_all):
        tick.set_color(INK if ok else MUTED)
        tick.set_fontweight("bold" if ok else "normal")
    left.set_xlim(0, float(chosen["roi"].max()) * 1.14)
    left.set_ylim(*span)
    left.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    left.set_axisbelow(True)
    left.set_xlabel("ROI mediano sobre el total aportado (%)", fontsize=9.5, color=MUTED)
    left.set_title("Rentabilidad {0}".format(period), fontsize=12.5, color=INK, pad=14)

    # --- derecha: matriz de aprobados; relleno/hueco, no solo color ---------
    right.set_xlim(-0.6, len(filters) - 0.4)
    right.set_ylim(*span)
    right.set_yticks([])
    right.set_xticks(range(len(filters)))
    # Las cabeceras de los filtros se tocan cuando son rangos de anos en vez de
    # una palabra, asi que el cuerpo se ajusta a la mas larga.
    headers = [label for label, _, _ in filters]
    right.set_xticklabels(
        headers,
        color=INK,
        fontsize=9.5 if max(len(header) for header in headers) <= 9 else 8.2,
    )
    right.tick_params(axis="x", pad=6)
    right.spines["bottom"].set_visible(False)

    for column, (label, _, _) in enumerate(filters):
        for position, ok in zip(y, chosen[label]):
            right.scatter(
                column,
                position,
                s=118,
                facecolor=BLUE if ok else SURFACE,
                edgecolor=BLUE if ok else GRID,
                linewidth=1.6,
                zorder=3,
            )
    for position in y:
        right.plot(
            [-0.45, len(filters) - 0.55], [position, position], color=GRID, linewidth=0.6, zorder=1
        )
    right.set_xlabel(
        "Circulo lleno = supera el filtro   ·   hueco = no",
        fontsize=9.5,
        color=MUTED,
    )
    right.set_title(
        "Los {0} filtros anti-sobreajuste".format(len(filters)),
        fontsize=12.5,
        color=INK,
        pad=14,
    )

    survivors = int(passes_all.sum())
    figure.suptitle(
        "Que queda cuando se le exige a la senal algo mas que haber funcionado\n"
        "Solo {0} de {1} configuraciones superan los {2} filtros   ({3})".format(
            survivors, total, len(filters), sample
        ),
        fontsize=13.5,
        color=INK,
        y=0.985,
    )
    figure.text(
        0.012,
        0.012,
        "Generado {0} · {1}".format(dt.date.today().isoformat(), source),
        fontsize=8,
        color=MUTED,
    )
    # tight_layout no sabe medir la matriz de circulos, asi que el margen
    # izquierdo se deduce de la etiqueta mas larga: los nombres de los combos
    # doblan en ancho a los de una senal simple y se salian del lienzo.
    widest = max(len(label) for label in labels)
    figure.subplots_adjust(
        left=min(0.33, 0.03 + 0.006 * widest), right=0.985, top=0.855, bottom=0.115
    )
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_prices(CACHE)
    equity = {t: p for t, p in data.items() if MARKET_OF[t] != "Materias primas"}

    table, dca_roi = evaluate_all(equity)
    chosen = select_for_plot(table)

    csv_path = OUT_DIR / "robustness_summary.csv"
    plot_path = OUT_DIR / "robustness_ranking.png"
    table.sort_values(["filtros_superados", "roi"], ascending=False).to_csv(csv_path, index=False)
    plot(chosen, dca_roi, len(table), plot_path)

    pd.set_option("display.width", 250)
    survivors = table[table["filtros_superados"] == len(FILTERS)]
    print("DCA de referencia: {0:.1f}% de ROI mediano".format(dca_roi))
    print("Configuraciones evaluables: {0}\n".format(len(table)))
    for label, description, _ in FILTERS:
        print("  {0:<15} {1:<32} pasan {2:>3}".format(label, description, int(table[label].sum())))
    print("\n  Superan los cinco a la vez: {0}\n".format(len(survivors)))
    columns = [
        "strategy",
        "family",
        "roi",
        "wins",
        "p",
        "fires",
        "era1",
        "era2",
        "mercados_ganados",
    ]
    print(survivors.sort_values("roi", ascending=False)[columns].round(1).to_string(index=False))
    print("\nEscrito {0}".format(csv_path))
    print("Escrito {0}".format(plot_path))


if __name__ == "__main__":
    main()
