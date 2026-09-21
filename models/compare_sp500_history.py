"""El mismo analisis sobre el S&P 500 con un siglo de historia, no con 26 anos.

Todo lo anterior arranca en 2000 por un motivo prosaico: es donde empiezan a
existir a la vez los 57 indices del universo. Ese recorte deja fuera casi toda
la historia interesante y, peor, deja dentro un periodo con dos crisis que se
recuperaron rapido, que es exactamente el supuesto del que viven las senales de
caida. Con una sola serie se puede ir hasta 1927: la Gran Depresion, la guerra,
la inflacion de los 70 y la decada perdida de 1966-1982 entran en la muestra.

El precio de usar una sola serie es que desaparece la muestra: no hay 47 bolsas
sobre las que contar victorias. Se sustituye por dos cosas:

  · COHORTES. Cada ano de arranque entre 1928 y 2006 define un ahorrador que
    aporta 100 EUR/semana durante 20 anos. Son 79 historias de inversion, cada
    una comparada contra un DCA que corre en esa misma ventana. Se solapan, asi
    que su tasa de acierto describe bien pero su p-valor exagera.

  · PERMUTACION CIRCULAR. Para la significancia se rota el patron de disparos a
    lo largo del siglo entero. La rotacion conserva cuantas veces dispara la
    senal y como se agrupan los disparos, y solo cambia DONDE caen: es la
    hipotesis nula exacta de "el momento no aporta nada". El p-valor es la
    fraccion de rotaciones que iguala o supera al patron real.

Se emiten los dos graficos del analisis multimercado con esta muestra:
`sp500_realistic_deployment_ranking.png` y `sp500_robustness_ranking.png`.
"""

import json
import pickle
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import compare_realistic_deployment as realistic
import compare_robustness as robustness
from compare_entry_signals import OUT_DIR, build_catalog, sign_test
from compare_realistic_deployment import WEEKLY

TICKER = "^GSPC"
CACHE = OUT_DIR / "sp500_weekly.pkl"

HORIZON_YEARS = 20          # cuanto ahorra cada cohorte
HORIZON = HORIZON_YEARS * 52
STEP = 52                   # una cohorte nueva por ano de arranque
MIN_FIRES_COHORT = 8        # disparos minimos para que la cohorte cuente
MIN_COHORTS = 40            # y cohortes minimas para que la config cuente

CASH_RATE = 0.02
SHIFTS = 1000               # rotaciones del test de permutacion
ERA_LABELS = ("1928-1954", "1954-1980", "1980-2006")

FILTERS: List[Tuple[str, str, callable]] = [
    ("Significativa", "p < 0,05 por permutacion circular", lambda r: r["p_shift"] < 0.05),
    ("Cohortes", "gana en >= 60% de las 79 cohortes", lambda r: r["wins"] >= 60.0),
    (ERA_LABELS[0], "gana en las cohortes de preguerra", lambda r: r[ERA_LABELS[0]] > 50),
    (ERA_LABELS[1], "gana en las cohortes de posguerra", lambda r: r[ERA_LABELS[1]] > 50),
    (ERA_LABELS[2], "gana en las cohortes modernas", lambda r: r[ERA_LABELS[2]] > 50),
    ("Familia", ">= 60% de sus hermanas gana", lambda r: r["familia_coherente_pct"] >= 60),
]


def fetch_full_history(ticker: str) -> pd.Series:
    """Cierre semanal desde donde Yahoo tenga datos; para ^GSPC, 1927."""
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


def load_history() -> pd.Series:
    if CACHE.exists():
        with CACHE.open("rb") as handle:
            return pickle.load(handle)[TICKER]
    prices = fetch_full_history(TICKER)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("wb") as handle:
        pickle.dump({TICKER: prices}, handle)
    return prices


def build_cohorts(prices: pd.Series) -> Dict[str, pd.Series]:
    """Una ventana de HORIZON semanas por cada ano de arranque.

    La clave es el ano de inicio, que es lo que luego permite partir las
    cohortes en tercios: un ahorrador que empieza en 1930 y otro que empieza en
    1995 no viven el mismo mercado ni de lejos.
    """
    cohorts: Dict[str, pd.Series] = {}
    for start in range(0, len(prices) - HORIZON + 1, STEP):
        window = prices.iloc[start : start + HORIZON]
        cohorts[str(window.index[0].year)] = window
    return cohorts


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


def evaluate(cohorts: Dict[str, pd.Series], prices: pd.Series) -> Tuple[pd.DataFrame, float]:
    arrays = {name: window.to_numpy(float) for name, window in cohorts.items()}
    contributed = WEEKLY * HORIZON
    century = prices.to_numpy(float)
    rng = np.random.default_rng(20260921)

    dca = {name: robustness.final_value(values, np.ones(len(values), bool))
           for name, values in arrays.items()}
    years = {name: int(name) for name in cohorts}
    edges = np.quantile(list(years.values()), [1 / 3, 2 / 3])
    era_of = {
        name: ERA_LABELS[0] if year < edges[0] else (ERA_LABELS[1] if year < edges[1] else ERA_LABELS[2])
        for name, year in years.items()
    }

    rows: List[dict] = []
    for position, (name, (family, fn)) in enumerate(build_catalog().items(), start=1):
        century_fires = fn(prices).reindex(prices.index).fillna(False).to_numpy(bool)

        values, fire_counts = {}, []
        for cohort, window in cohorts.items():
            fires = fn(window).reindex(window.index).fillna(False).to_numpy(bool)
            if fires.sum() < MIN_FIRES_COHORT:
                continue
            values[cohort] = robustness.final_value(arrays[cohort], fires)
            fire_counts.append(int(fires.sum()))
        if len(values) < MIN_COHORTS:
            continue

        diff = pd.Series({c: values[c] - dca[c] for c in values})
        era_wins = {}
        for era in ERA_LABELS:
            subset = [d for c, d in diff.items() if era_of[c] == era]
            era_wins[era] = float(np.mean([d > 0 for d in subset]) * 100.0) if subset else np.nan

        rows.append(
            {
                "strategy": name,
                "family": family,
                "roi": float(np.median([values[c] / contributed - 1.0 for c in values]) * 100.0),
                "wins": float((diff > 0).mean() * 100.0),
                "p_cohortes": sign_test(diff),
                "p_shift": shift_pvalue(century, century_fires, rng),
                "fires": float(np.median(fire_counts)),
                "cohortes": len(values),
                **era_wins,
            }
        )
        if position % 40 == 0:
            print("  ... {0} configuraciones evaluadas".format(position))

    table = pd.DataFrame(rows)
    health = table.groupby("family")["wins"].apply(lambda s: float((s > 50).mean() * 100.0))
    table["familia_coherente_pct"] = table["family"].map(health)
    for label, _, test in FILTERS:
        table[label] = table.apply(test, axis=1)
    table["filtros_superados"] = table[[label for label, _, _ in FILTERS]].sum(axis=1)

    dca_roi = float(np.median([v / contributed - 1.0 for v in dca.values()]) * 100.0)
    return table, dca_roi


def deployment_chart(cohorts: Dict[str, pd.Series], period: str) -> pd.DataFrame:
    """Reutiliza el barrido y el grafico de rentabilidad real con las cohortes.

    `compare_realistic_deployment` no sabe que sus "tickers" son ahora periodos;
    le basta con recibir un diccionario de series, que es justo lo que es una
    cohorte.
    """
    catalog = build_catalog()
    detail = realistic.run(cohorts, catalog)
    summary = realistic.summarize(detail)

    headline = summary[summary["cash_rate"] == realistic.HEADLINE_RATE]
    strategies = headline[headline["strategy"] != realistic.DCA]
    reliable = int((strategies["gana_a_dca_pct"] >= realistic.MIN_WIN_RATE).sum())
    best = float(strategies["gana_a_dca_pct"].max())
    # Sobre esta muestra el corte del 60% no lo pasa nadie, asi que el grafico
    # muestra el ranking completo por rentabilidad y anota el corte en el titulo:
    # una figura vacia esconderia el resultado en vez de ensenarlo.
    note = (
        "Ninguna de las {0} bate al DCA en el 60% de las cohortes; la mejor llega al {1:.0f}%".format(
            len(strategies), best
        )
        if reliable == 0
        else "Solo las {0} que baten al DCA en al menos el 60% de las cohortes".format(reliable)
    )

    realistic.plot(
        summary,
        n_series=len(cohorts),
        out_path=OUT_DIR / "sp500_realistic_deployment_ranking.png",
        sample="cohortes de {0} anos del S&P 500".format(HORIZON_YEARS),
        period=period,
        unit="cohortes",
        min_win_rate=realistic.MIN_WIN_RATE if reliable else 0.0,
        headline_note=note,
        tail=0 if reliable else 8,
    )
    print("\n  Cohortes: {0} · con >=60% de acierto: {1} · mejor acierto: {2:.1f}%".format(
        len(cohorts), reliable, best))
    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_history()
    cohorts = build_cohorts(prices)
    period = "{0}-{1}".format(prices.index[0].year, prices.index[-1].year)

    print("{0}: {1} semanas, {2}".format(TICKER, len(prices), period))
    print("{0} cohortes de {1} anos, arrancando de {2} a {3}\n".format(
        len(cohorts), HORIZON_YEARS, min(cohorts), max(cohorts)))

    summary = deployment_chart(cohorts, period)
    summary.to_csv(OUT_DIR / "sp500_realistic_deployment_summary.csv", index=False)

    print("\nRobustez: {0} rotaciones por configuracion...".format(SHIFTS))
    table, dca_roi = evaluate(cohorts, prices)
    # Mismo motivo que en el grafico de rentabilidad: si el corte de acierto no
    # lo pasa nadie, se muestra el ranking por rentabilidad con la matriz al lado.
    reachable = float(table["wins"].max()) >= robustness.MIN_WIN_RATE
    chosen = robustness.select_for_plot(
        table,
        n_filters=len(FILTERS),
        min_win_rate=robustness.MIN_WIN_RATE if reachable else 0.0,
        tail=0 if reachable else 7,
    )
    robustness.plot(
        chosen, dca_roi, len(table),
        OUT_DIR / "sp500_robustness_ranking.png",
        filters=FILTERS,
        sample="S&P 500, {0} cohortes de {1} anos, efectivo al 2%".format(
            len(cohorts), HORIZON_YEARS),
        period=period,
        source="models/compare_sp500_history.py",
    )
    table.sort_values(["filtros_superados", "roi"], ascending=False).to_csv(
        OUT_DIR / "sp500_robustness_summary.csv", index=False)

    pd.set_option("display.width", 250)
    print("\nDCA de referencia: {0:.1f}% de ROI mediano".format(dca_roi))
    print("Configuraciones evaluables: {0}\n".format(len(table)))
    for label, description, _ in FILTERS:
        print("  {0:<15} {1:<38} pasan {2:>3}".format(label, description, int(table[label].sum())))

    columns = ["strategy", "family", "roi", "wins", "p_shift", "fires",
               *ERA_LABELS, "filtros_superados"]
    survivors = table[table["filtros_superados"] == len(FILTERS)]
    print("\n  Superan los {0} a la vez: {1}".format(len(FILTERS), len(survivors)))

    print("\nMejores por rentabilidad:")
    print(table.sort_values("roi", ascending=False).head(12)[columns].round(2).to_string(index=False))

    print("\nMejores por tasa de acierto sobre las cohortes:")
    print(table.sort_values("wins", ascending=False).head(12)[columns].round(2).to_string(index=False))

    print("\nPeores por rentabilidad (las mas selectivas):")
    print(table.sort_values("roi").head(8)[columns].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
