"""El analisis completo sobre un solo activo con todo su historico disponible.

Los modulos multimercado arrancan en 2000 porque es donde empiezan a coexistir
los 57 indices del universo. Ese recorte deja fuera casi toda la historia y deja
dentro un periodo con dos crisis que se recuperaron rapido, que es exactamente el
supuesto del que viven las senales de caida. Con un solo activo se puede usar
todo lo que sirva la fuente.

El precio de usar una sola serie es que desaparece la muestra: no hay 47 bolsas
sobre las que contar victorias. Se sustituye por dos cosas:

  · COHORTES. Cada fecha de arranque define un ahorrador que aporta 100 EUR a la
    semana durante un horizonte fijo, comparado contra un DCA que corre en esa
    misma ventana. Se solapan, asi que describen bien pero su p-valor exagera.

  · PERMUTACION CIRCULAR. Para la significancia se rota el patron de disparos a
    lo largo de la serie entera. La rotacion conserva cuantas veces dispara la
    senal y como se agrupan los disparos, y solo cambia DONDE caen: es la
    hipotesis nula exacta de "el momento no aporta nada". El p-valor es la
    fraccion de rotaciones que iguala o supera al patron real.

El horizonte y el paso entre cohortes son por activo, porque no se puede pedir
cohortes de 20 anos a una serie de 12. Con BTC esa es justamente la limitacion
principal del resultado y conviene que este escrita en el propio catalogo.

Uso:  python3 compare_long_history.py [sp500|btc|all]
"""

import json
import pickle
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import compare_realistic_deployment as realistic
import compare_robustness as robustness
from compare_entry_signals import BLUE, GRID, INK, MUTED, ORANGE, OUT_DIR, SURFACE, build_catalog
from compare_realistic_deployment import WEEKLY


@dataclass(frozen=True)
class Asset:
    """Un activo y como se le puede preguntar por su historia.

    `horizon_years` es cuanto ahorra cada cohorte y `step_weeks` cada cuanto
    arranca una nueva. Con un siglo de datos se puede pedir un horizonte largo y
    un paso anual; con doce anos hay que acortar los dos, y el solapamiento
    resultante es mas severo.
    """

    key: str
    ticker: str
    label: str
    horizon_years: int
    step_weeks: int
    min_fires_cohort: int


ASSETS: Dict[str, Asset] = {
    "sp500": Asset("sp500", "^GSPC", "S&P 500", horizon_years=20, step_weeks=52, min_fires_cohort=8),
    "btc": Asset("btc", "BTC-USD", "Bitcoin", horizon_years=5, step_weeks=13, min_fires_cohort=5),
}

MIN_COHORT_SHARE = 0.5  # una senal debe ser evaluable en la mitad de las cohortes
CASH_RATE = 0.02
SHIFTS = 1000       # rotaciones del test de permutacion


def fetch_full_history(ticker: str) -> pd.Series:
    """Cierre semanal desde donde la fuente tenga datos."""
    params = urllib.parse.urlencode(
        {
            "period1": -2208988800,  # 1900-01-01; Yahoo recorta a lo que exista
            "period2": 2000000000,
            "interval": "1wk",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = "https://query1.finance.yahoo.com/v8/finance/chart/{0}?{1}".format(
        urllib.parse.quote(ticker, safe=""), params
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))

    result = payload["chart"]["result"][0]
    dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(None)
    prices = pd.Series(result["indicators"]["quote"][0]["close"], index=dates, name="Close")
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    return prices[prices > 0].sort_index()


def load_history(asset: Asset) -> pd.Series:
    cache = OUT_DIR / "{0}_weekly.pkl".format(asset.key)
    if cache.exists():
        with cache.open("rb") as handle:
            return pickle.load(handle)[asset.ticker]
    prices = fetch_full_history(asset.ticker)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as handle:
        pickle.dump({asset.ticker: prices}, handle)
    return prices


def build_cohorts(prices: pd.Series, asset: Asset) -> Dict[str, pd.Series]:
    """Una ventana por cada fecha de arranque, etiquetada por ano y trimestre.

    La etiqueta lleva el ano porque es lo que luego permite partir las cohortes
    en tercios: quien empieza en 1930 y quien empieza en 1995 no viven el mismo
    mercado ni de lejos.
    """
    horizon = asset.horizon_years * 52
    cohorts: Dict[str, pd.Series] = {}
    for start in range(0, len(prices) - horizon + 1, asset.step_weeks):
        window = prices.iloc[start : start + horizon]
        begins = window.index[0]
        label = str(begins.year) if asset.step_weeks >= 52 else "{0}T{1}".format(
            begins.year, (begins.month - 1) // 3 + 1
        )
        cohorts[label] = window
    return cohorts


def era_split(cohorts: Dict[str, pd.Series]) -> Tuple[Dict[str, str], Tuple[str, str, str]]:
    """Reparte las cohortes en tercios por ano de arranque.

    Los cortes salen de los cuantiles de los propios arranques, no de fechas
    fijas, para que el reparto siga siendo de un tercio cada uno sea cual sea el
    activo y su horizonte.
    """
    years = {name: window.index[0].year for name, window in cohorts.items()}
    low, high = np.quantile(list(years.values()), [1 / 3, 2 / 3])
    bounds = sorted(years.values())
    labels = (
        "{0}-{1:.0f}".format(bounds[0], low),
        "{0:.0f}-{1:.0f}".format(low, high),
        "{0:.0f}-{1}".format(high, bounds[-1]),
    )
    era_of = {
        name: labels[0] if year < low else (labels[1] if year < high else labels[2])
        for name, year in years.items()
    }
    return era_of, labels


def build_filters(labels: Tuple[str, str, str], n_cohorts: int) -> List[Tuple[str, str, callable]]:
    return [
        ("Significativa", "p < 0,05 por permutacion circular", lambda r: r["p_shift"] < 0.05),
        ("Cohortes", "gana en >= 60% de las {0} cohortes".format(n_cohorts),
         lambda r: r["wins"] >= 60.0),
        (labels[0], "gana en las cohortes mas antiguas", lambda r: r[labels[0]] > 50),
        (labels[1], "gana en las cohortes intermedias", lambda r: r[labels[1]] > 50),
        (labels[2], "gana en las cohortes recientes", lambda r: r[labels[2]] > 50),
        ("Familia", ">= 60% de sus hermanas gana", lambda r: r["familia_coherente_pct"] >= 60),
    ]


def shift_pvalue(prices: np.ndarray, fires: np.ndarray, rng: np.random.Generator) -> float:
    """Fraccion de rotaciones del patron que igualan o superan al patron real.

    Rotar circularmente conserva el numero de disparos y su agrupamiento, de modo
    que lo unico que se destruye es la coincidencia con los minimos reales. Si la
    senal solo sabe disparar mucho, el real quedara en medio del monton.
    """
    actual = robustness.final_value(prices, fires)
    lags = rng.choice(len(prices), size=min(SHIFTS, len(prices)), replace=False)
    null = np.array([robustness.final_value(prices, np.roll(fires, int(lag))) for lag in lags])
    return float((np.sum(null >= actual) + 1) / (len(null) + 1))


def evaluate(
    cohorts: Dict[str, pd.Series],
    prices: pd.Series,
    asset: Asset,
    filters: List[Tuple[str, str, callable]],
    era_of: Dict[str, str],
    labels: Tuple[str, str, str],
) -> Tuple[pd.DataFrame, float]:
    arrays = {name: window.to_numpy(float) for name, window in cohorts.items()}
    contributed = WEEKLY * asset.horizon_years * 52
    century = prices.to_numpy(float)
    rng = np.random.default_rng(20260921)

    dca = {name: robustness.final_value(values, np.ones(len(values), bool))
           for name, values in arrays.items()}
    # Exigir un numero fijo de cohortes no vale para dos activos con 80 y 29:
    # se pide la mitad, para que una senal no se juzgue en un submuestreo.
    min_cohorts = max(6, int(MIN_COHORT_SHARE * len(cohorts)))

    rows: List[dict] = []
    for position, (name, (family, fn)) in enumerate(build_catalog().items(), start=1):
        whole_fires = fn(prices).reindex(prices.index).fillna(False).to_numpy(bool)

        values, fire_counts = {}, []
        for cohort, window in cohorts.items():
            fires = fn(window).reindex(window.index).fillna(False).to_numpy(bool)
            if fires.sum() < asset.min_fires_cohort:
                continue
            values[cohort] = robustness.final_value(arrays[cohort], fires)
            fire_counts.append(int(fires.sum()))
        if len(values) < min_cohorts:
            continue

        diff = pd.Series({c: values[c] - dca[c] for c in values})
        era_wins = {}
        for era in labels:
            subset = [d for c, d in diff.items() if era_of[c] == era]
            era_wins[era] = float(np.mean([d > 0 for d in subset]) * 100.0) if subset else np.nan

        rows.append(
            {
                "strategy": name,
                "family": family,
                "roi": float(np.median([values[c] / contributed - 1.0 for c in values]) * 100.0),
                "wins": float((diff > 0).mean() * 100.0),
                "p_shift": shift_pvalue(century, whole_fires, rng),
                "fires": float(np.median(fire_counts)),
                "cohortes": len(values),
                **era_wins,
            }
        )
        if position % 60 == 0:
            print("  ... {0} configuraciones evaluadas".format(position))

    table = pd.DataFrame(rows)
    health = table.groupby("family")["wins"].apply(lambda s: float((s > 50).mean() * 100.0))
    table["familia_coherente_pct"] = table["family"].map(health)
    for label, _, test in filters:
        table[label] = table.apply(test, axis=1)
    table["filtros_superados"] = table[[label for label, _, _ in filters]].sum(axis=1)

    dca_roi = float(np.median([v / contributed - 1.0 for v in dca.values()]) * 100.0)
    return table, dca_roi


def deployment_chart(cohorts: Dict[str, pd.Series], asset: Asset, period: str) -> pd.DataFrame:
    """Reutiliza el barrido y el grafico de rentabilidad real con las cohortes.

    `compare_realistic_deployment` no sabe que sus "tickers" son ahora periodos;
    le basta con recibir un diccionario de series, que es lo que es una cohorte.
    """
    summary = realistic.summarize(
        realistic.run(cohorts, build_catalog(),
                      min_series=max(6, int(MIN_COHORT_SHARE * len(cohorts))))
    )

    headline = summary[summary["cash_rate"] == realistic.HEADLINE_RATE]
    strategies = headline[headline["strategy"] != realistic.DCA]
    reliable = int((strategies["gana_a_dca_pct"] >= realistic.MIN_WIN_RATE).sum())
    best = float(strategies["gana_a_dca_pct"].max())
    # Si el corte del 60% no lo pasa nadie, el grafico muestra el ranking entero
    # por rentabilidad y anota el corte en el titulo: una figura vacia esconderia
    # el resultado en vez de ensenarlo.
    note = (
        "Ninguna de las {0} bate al DCA en el 60% de las cohortes; la mejor llega al {1:.0f}%".format(
            len(strategies), best)
        if reliable == 0
        else "Solo las {0} que baten al DCA en al menos el 60% de las cohortes".format(reliable)
    )

    realistic.plot(
        summary,
        n_series=len(cohorts),
        out_path=OUT_DIR / "{0}_realistic_deployment_ranking.png".format(asset.key),
        sample="cohortes de {0} anos de {1}".format(asset.horizon_years, asset.label),
        period=period,
        unit="cohortes",
        min_win_rate=realistic.MIN_WIN_RATE if reliable else 0.0,
        headline_note=note,
        tail=0 if reliable else 8,
    )
    print("  Cohortes: {0} · con >=60% de acierto: {1} · mejor acierto: {2:.1f}%".format(
        len(cohorts), reliable, best))
    return summary


def analyse(asset: Asset) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    prices = load_history(asset)
    cohorts = build_cohorts(prices, asset)
    period = "{0}-{1}".format(prices.index[0].year, prices.index[-1].year)
    era_of, labels = era_split(cohorts)
    filters = build_filters(labels, len(cohorts))

    print("\n=== {0} ({1}) ===".format(asset.label, asset.ticker))
    print("{0} semanas, {1} · {2} cohortes de {3} anos cada {4} semanas".format(
        len(prices), period, len(cohorts), asset.horizon_years, asset.step_weeks))

    summary = deployment_chart(cohorts, asset, period)
    summary.to_csv(OUT_DIR / "{0}_realistic_deployment_summary.csv".format(asset.key), index=False)

    print("  Robustez: {0} rotaciones por configuracion...".format(SHIFTS))
    table, dca_roi = evaluate(cohorts, prices, asset, filters, era_of, labels)

    reachable = float(table["wins"].max()) >= robustness.MIN_WIN_RATE
    chosen = robustness.select_for_plot(
        table,
        n_filters=len(filters),
        min_win_rate=robustness.MIN_WIN_RATE if reachable else 0.0,
        tail=0 if reachable else 7,
    )
    robustness.plot(
        chosen, dca_roi, len(table),
        OUT_DIR / "{0}_robustness_ranking.png".format(asset.key),
        filters=filters,
        sample="{0}, {1} cohortes de {2} anos, efectivo al 2%".format(
            asset.label, len(cohorts), asset.horizon_years),
        period=period,
        source="models/compare_long_history.py",
    )
    table.sort_values(["filtros_superados", "roi"], ascending=False).to_csv(
        OUT_DIR / "{0}_robustness_summary.csv".format(asset.key), index=False)

    print("\n  DCA de referencia: {0:.1f}% de ROI mediano · {1} configuraciones".format(
        dca_roi, len(table)))
    for label, description, _ in filters:
        print("    {0:<14} {1:<42} pasan {2:>3}".format(label, description, int(table[label].sum())))
    print("    Superan los {0} a la vez: {1}".format(
        len(filters), int((table["filtros_superados"] == len(filters)).sum())))

    columns = ["strategy", "family", "roi", "wins", "p_shift", "fires", *labels, "filtros_superados"]
    pd.set_option("display.width", 250)
    print("\n  Mejores por rentabilidad:")
    print(table.sort_values("roi", ascending=False).head(10)[columns].round(2).to_string(index=False))
    print("\n  Mejores por tasa de acierto:")
    print(table.sort_values("wins", ascending=False).head(10)[columns].round(2).to_string(index=False))

    return table, summary, dca_roi


def compare(results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, float]]) -> None:
    """Un solo grafico con los dos activos: cuanto se gana y cuanto cuesta esperar.

    A la izquierda, donde cae cada configuracion en tasa de acierto. A la
    derecha, la relacion entre el efectivo que la senal deja parado y lo que
    pierde frente al DCA: es la misma pendiente en ambos activos, pero en BTC
    cada punto de efectivo parado cuesta mucho mas porque el subyacente sube
    mucho mas deprisa.
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=(15.5, 6.4), facecolor=SURFACE)
    colors = {"sp500": BLUE, "btc": ORANGE}
    markers = {"sp500": "o", "btc": "D"}

    for key, (_, summary, _) in results.items():
        asset = ASSETS[key]
        headline = summary[summary["cash_rate"] == realistic.HEADLINE_RATE]
        dca = headline[headline["strategy"] == realistic.DCA]
        rest = headline[headline["strategy"] != realistic.DCA]
        dca_roi = float(dca["roi_mediano_pct"].iloc[0])
        label = "{0} · {1} cohortes de {2} anos".format(
            asset.label, int(rest["n_series"].median()), asset.horizon_years)

        counts, edges = np.histogram(rest["gana_a_dca_pct"], bins=np.arange(0, 105, 5))
        left.step(edges[:-1] + 2.5, counts, where="mid", color=colors[key], linewidth=2.0, label=label)
        left.fill_between(edges[:-1] + 2.5, counts, step="mid", color=colors[key], alpha=0.13)

        right.scatter(
            rest["efectivo_final_pct"], rest["roi_mediano_pct"] / dca_roi * 100.0 - 100.0,
            s=34, marker=markers[key], facecolor=colors[key], edgecolor=SURFACE,
            linewidth=0.8, alpha=0.85, label=label, zorder=3,
        )

    left.axvline(50, color=INK, linestyle="--", linewidth=1.0, zorder=4)
    # El histograma llega hasta el 100% aunque nadie pase del 55%: sin recortar,
    # media figura queda en blanco y el detalle se comprime contra el eje.
    reach = max(
        float(block[block["cash_rate"] == realistic.HEADLINE_RATE]["gana_a_dca_pct"].max())
        for _, block, _ in results.values()
    )
    left.set_xlim(0, max(60.0, reach + 8.0))
    left.text(49.0, left.get_ylim()[1] * 0.5, "moneda al aire",
              color=INK, fontsize=9, ha="right", rotation=90, va="center")
    left.set_xlabel("Cohortes en las que la estrategia bate al DCA (%)", fontsize=9.5, color=MUTED)
    left.set_ylabel("Numero de configuraciones", fontsize=9.5, color=MUTED)
    left.set_title("Donde cae cada configuracion", fontsize=12.5, color=INK, pad=12)

    right.axhline(0, color=INK, linestyle="--", linewidth=1.0, zorder=4)
    right.text(right.get_xlim()[0], 1.5, "empate con el DCA", color=INK, fontsize=9, ha="left")
    right.set_xlabel("Efectivo que la senal deja parado al final (%)", fontsize=9.5, color=MUTED)
    right.set_ylabel("Riqueza final frente al DCA (%)", fontsize=9.5, color=MUTED)
    right.set_title("Lo que cuesta esperar", fontsize=12.5, color=INK, pad=12)

    for axis in (left, right):
        axis.set_facecolor(SURFACE)
        axis.grid(True, color=GRID, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.tick_params(colors=MUTED, labelsize=9)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        # La leyenda va al hueco de cada panel: arriba en el histograma, abajo
        # en la nube, donde la linea del empate ocupa el borde superior.
        legend = axis.legend(
            frameon=False, fontsize=9.5,
            loc="upper right" if axis is left else "lower left",
        )
        for text in legend.get_texts():
            text.set_color(INK)

    figure.suptitle(
        "El mismo barrido sobre dos activos con todo su historico\n"
        "Cuanto mas sube el subyacente, mas caro sale esperar a la caida",
        fontsize=13.5, color=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path = OUT_DIR / "long_history_comparison.png"
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    print("\nEscrito {0}".format(path))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    requested = sys.argv[1] if len(sys.argv) > 1 else "all"
    keys = list(ASSETS) if requested == "all" else [requested]

    results = {key: analyse(ASSETS[key]) for key in keys}
    if len(results) > 1:
        compare(results)


if __name__ == "__main__":
    main()
