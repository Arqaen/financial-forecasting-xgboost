"""Busca senales de entrada que batan al RSI, con el capital siempre igualado.

compare_strategies_normalized.py establecio que, repartiendo el mismo dinero
entre las semanas en que dispara una senal, RSI(14)<30 compra un 32% mas barato
que comprar uniformemente. Este modulo pregunta lo siguiente: existe algo mejor.

Barre 174 configuraciones de once familias de indicadores, mas sus combinaciones,
y las mide todas contra la misma referencia.

La metrica
----------
Con aportacion constante b en las N semanas de disparo a precios p_i, el valor
final es P_end * b * sum(1/p_i). Todo depende, por tanto, de la media de
x_t = 1/p_t sobre las fechas de compra, y el edge es exactamente una media
muestral normalizada:

    edge = media(x en las fechas de senal) / media(x en toda la serie) - 1

Esa cantidad es tambien, literalmente, el ratio de riqueza final frente a hacer
DCA con el mismo presupuesto, asi que un edge del 50% significa acabar 1,5 veces
mas rico que comprando cada semana.

Por que hace falta un test de significancia
-------------------------------------------
Cualquier senal que dispare en rachas durante un mercado alcista sale favorecida:
comprar agrupado cerca del principio ya bate a la media. Para separar la
habilidad del efecto de calendario se desplaza circularmente la mascara de la
senal por todos los desfases posibles. Eso conserva intactos el numero de
disparos Y su agrupamiento, y solo mueve *donde* cae el patron, que es justo la
hipotesis nula que interesa. Se calcula entera por FFT en una pasada.

Resultado
---------
El RSI pierde contra las senales de *profundidad* (caida desde maximo, retorno a
un ano, distancia a la media movil) a cualquier nivel de selectividad. El RSI
esta normalizado por la velocidad reciente, asi que dispara en cualquier regimen;
las de profundidad solo disparan cuando la caida es grande en terminos absolutos.
"""

import datetime as dt
import itertools
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_strategies_simple import OUT_DIR, calculate_rsi, fetch_weekly_close

# El universo original (27 tickers) ya tocaba doce paises, pero nunca se
# desglosaba, asi que no habia forma de saber si el resultado lo sostenia un
# solo mercado. Aqui se amplia a 57 series agrupadas, para poder exigir que el
# hallazgo se repita en cada grupo por separado.
UNIVERSE: Dict[str, List[str]] = {
    "EEUU": ["^GSPC", "^IXIC", "^DJI", "^RUT", "^NYA"],
    "Europa": [
        "^FTSE", "^GDAXI", "^FCHI", "^IBEX", "FTSEMIB.MI", "^AEX",
        "^SSMI", "^BFX", "^OMX", "^STOXX50E", "^ATX", "PSI20.LS",
    ],
    "Asia-Pacifico": [
        "^N225", "^HSI", "^BSESN", "^NSEI", "^KS11", "^TWII",
        "^STI", "^JKSE", "^AXJO", "^NZ50", "^KLSE", "000001.SS",
    ],
    "Otros mercados": ["^GSPTSE", "^BVSP", "^MXX", "^MERV", "^TA125.TA", "^JN0U.JO"],
    "Materias primas": ["CL=F", "GC=F", "SI=F", "HG=F", "NG=F", "ZC=F", "ZW=F", "ZS=F", "PL=F", "PA=F"],
    "Acciones": [
        "PFE", "IDR.MC", "ACX.MC", "REPYY", "SAN", "BKT.MC",
        "GEST.MC", "ITX.MC", "GE", "BB", "GM", "EGRNF",
    ],
}
MARKET_OF = {ticker: market for market, tickers in UNIVERSE.items() for ticker in tickers}
TICKERS = list(MARKET_OF)

BASELINE = "RSI(14) < 30"
MIN_SIGNALS = 12  # por debajo de esto el edge de un ticker es ruido
MIN_TICKERS = 44  # ~77% del universo: una config debe medirse en casi todo el
BUDGET = 10_000.0

# Familias cuya senal mide profundidad absoluta de la caida, frente a las que
# miden velocidad relativa reciente. Es el eje que decide el resultado.
DEPTH_FAMILIES = {"Caida desde maximo", "Momento", "Distancia a la media", "Percentil de precio"}
OSCILLATOR_FAMILIES = {"RSI", "Estocastico", "MACD", "Z-score"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"

SignalFn = Callable[[pd.Series], pd.Series]


# --------------------------------------------------------------------------
# Familias de senales. Todas causales: solo miran hasta la semana t inclusive.
# --------------------------------------------------------------------------


def sig_rsi(window: int, threshold: float) -> SignalFn:
    # NaN < umbral es False, asi que el warmup no dispara (el bug que tiene
    # compare_strategies_simple al pasar por nz() antes de comparar).
    return lambda prices: calculate_rsi(prices, window) < threshold


def sig_drawdown(lookback: Optional[int], pct: float) -> SignalFn:
    def fn(prices: pd.Series) -> pd.Series:
        peak = (
            prices.expanding().max()
            if lookback is None
            else prices.rolling(lookback, min_periods=lookback).max()
        )
        return (prices / peak - 1.0) <= -pct / 100.0

    return fn


def sig_below_ma(window: int, pct: float) -> SignalFn:
    def fn(prices: pd.Series) -> pd.Series:
        ma = prices.rolling(window, min_periods=window).mean()
        return (prices / ma - 1.0) <= -pct / 100.0

    return fn


def sig_zscore(window: int, z: float) -> SignalFn:
    def fn(prices: pd.Series) -> pd.Series:
        log_price = np.log(prices)
        mean = log_price.rolling(window, min_periods=window).mean()
        std = log_price.rolling(window, min_periods=window).std()
        return ((log_price - mean) / std) <= z

    return fn


def sig_pct_rank(window: int, pct: float) -> SignalFn:
    def fn(prices: pd.Series) -> pd.Series:
        return prices.rolling(window, min_periods=window).rank(pct=True) <= pct / 100.0

    return fn


def sig_consecutive_down(n: int) -> SignalFn:
    def fn(prices: pd.Series) -> pd.Series:
        return (prices.diff() < 0).astype(float).rolling(n, min_periods=n).sum() >= n

    return fn


def sig_roc(window: int, pct: float) -> SignalFn:
    return lambda prices: prices.pct_change(window) <= pct / 100.0


def sig_stochastic(window: int, threshold: float) -> SignalFn:
    def fn(prices: pd.Series) -> pd.Series:
        low = prices.rolling(window, min_periods=window).min()
        high = prices.rolling(window, min_periods=window).max()
        return (prices - low) / (high - low) * 100.0 <= threshold

    return fn


def sig_macd_hist(fast: int, slow: int, span: int, threshold: float) -> SignalFn:
    def fn(prices: pd.Series) -> pd.Series:
        macd = prices.ewm(span=fast, adjust=False).mean() - prices.ewm(span=slow, adjust=False).mean()
        hist = (macd - macd.ewm(span=span, adjust=False).mean()) / prices * 100.0
        hist.iloc[:slow] = np.nan  # ewm no produce NaN: hay que recortar el warmup
        return hist <= threshold

    return fn


def sig_vol_spike(window: int, quantile: float) -> SignalFn:
    def fn(prices: pd.Series) -> pd.Series:
        vol = np.log(prices).diff().rolling(window, min_periods=window).std()
        # Cuantil expansivo: cada semana solo se compara con su propio pasado.
        return vol >= vol.expanding(min_periods=104).quantile(quantile)

    return fn


def combine(first: SignalFn, second: SignalFn, how: str) -> SignalFn:
    def fn(prices: pd.Series) -> pd.Series:
        left, right = first(prices).fillna(False), second(prices).fillna(False)
        return (left & right) if how == "and" else (left | right)

    return fn


def build_catalog() -> Dict[str, Tuple[str, SignalFn]]:
    catalog: Dict[str, Tuple[str, SignalFn]] = {}

    def add(family: str, name: str, fn: SignalFn) -> None:
        catalog[name] = (family, fn)

    for window, threshold in itertools.product((2, 3, 4, 7, 14, 21, 28), (10, 15, 20, 25, 30, 35, 40)):
        add("RSI", "RSI({0}) < {1}".format(window, threshold), sig_rsi(window, threshold))

    for lookback, pct in itertools.product((52, 104, 156, 260, None), (10, 15, 20, 25, 30, 40, 50)):
        label = "max" if lookback is None else "max {0}s".format(lookback)
        add(
            "Caida desde maximo",
            "Caida >={0}% desde {1}".format(pct, label),
            sig_drawdown(lookback, pct),
        )

    for window, pct in itertools.product((20, 30, 50, 100, 200), (0, 2, 5, 10, 15, 20)):
        add(
            "Distancia a la media",
            "Precio <= MA{0} -{1}%".format(window, pct),
            sig_below_ma(window, pct),
        )

    for window, z in itertools.product((26, 52, 104), (-1.0, -1.5, -2.0, -2.5)):
        add("Z-score", "Z({0}) <= {1}".format(window, z), sig_zscore(window, z))

    for window, pct in itertools.product((52, 104, 156, 260), (5, 10, 20, 30)):
        add(
            "Percentil de precio",
            "Percentil <= {0}% de {1}s".format(pct, window),
            sig_pct_rank(window, pct),
        )

    for n in (2, 3, 4, 5):
        add("Rachas", "{0} semanas seguidas bajando".format(n), sig_consecutive_down(n))

    for window, pct in itertools.product((4, 13, 26, 52), (-10, -20, -30)):
        add("Momento", "Retorno {0}s <= {1}%".format(window, pct), sig_roc(window, pct))

    for window, threshold in itertools.product((14, 26, 52), (10, 20)):
        add(
            "Estocastico",
            "Estocastico({0}) <= {1}".format(window, threshold),
            sig_stochastic(window, threshold),
        )

    for threshold in (-0.5, -1.0, -2.0, -3.0):
        add("MACD", "MACD hist <= {0}%".format(threshold), sig_macd_hist(12, 26, 9, threshold))

    for window, quantile in itertools.product((13, 26), (0.80, 0.90, 0.95)):
        add(
            "Volatilidad",
            "Volatilidad({0}) >= p{1:.0f}".format(window, quantile * 100),
            sig_vol_spike(window, quantile),
        )

    # Intersecciones entre las mejores de familias distintas: una confirma a la
    # otra y el disparo exige profundidad y sobreventa a la vez.
    pairs = (
        ("Caida >=40% desde max 104s", "RSI(28) < 40"),
        ("Caida >=40% desde max 104s", "Precio <= MA50 -20%"),
        ("Retorno 52s <= -30%", "Precio <= MA50 -20%"),
        ("Retorno 52s <= -30%", "RSI(28) < 40"),
        ("Retorno 52s <= -30%", "Estocastico(52) <= 10"),
    )
    for left, right in pairs:
        add("Combo", "({0}) Y ({1})".format(left, right), combine(catalog[left][1], catalog[right][1], "and"))

    return catalog


# --------------------------------------------------------------------------
# Medicion
# --------------------------------------------------------------------------


def circular_null(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Suma de x bajo la mascara para los N desfases circulares, via FFT.

    Conserva cuantas veces dispara la senal y como se agrupa; solo cambia donde
    cae el patron sobre la serie.
    """
    spectrum = np.fft.rfft(x) * np.conj(np.fft.rfft(mask.astype(float)))
    return np.fft.irfft(spectrum, n=len(x))


def evaluate(prices: pd.Series, signal: pd.Series) -> dict:
    mask = signal.reindex(prices.index).fillna(False).to_numpy(dtype=bool)
    k = int(mask.sum())
    if k < MIN_SIGNALS:
        return {}

    x = 1.0 / prices.to_numpy(dtype=float)
    mean_all = float(x.mean())
    mean_signal = float(x[mask].mean())
    edge = mean_signal / mean_all - 1.0

    null_edges = circular_null(x, mask) / k / mean_all - 1.0
    spread = float(null_edges.std())

    # Techo teorico: comprar las k semanas mas baratas de toda la serie.
    oracle = float(np.sort(x)[-k:].mean()) / mean_all - 1.0

    price_end = float(prices.iloc[-1])
    return {
        "n_signals": k,
        "edge_vs_dca_pct": edge * 100.0,
        "z_score": (edge - float(null_edges.mean())) / spread if spread else np.nan,
        "p_value": float((null_edges >= edge - 1e-12).mean()),
        "captura_oracle_pct": edge / oracle * 100.0 if oracle > 0 else np.nan,
        # ROI sobre el capital invertido, que es la metrica del grafico original
        # de compare_strategies_simple, pero ya con el capital igualado.
        "roi_pct": (price_end * mean_signal - 1.0) * 100.0,
        "roi_dca_pct": (price_end * mean_all - 1.0) * 100.0,
        "valor_final": BUDGET * mean_signal * price_end,
        "valor_final_dca": BUDGET * mean_all * price_end,
    }


def sign_test(differences: pd.Series) -> float:
    """p bilateral de que las mejoras ticker a ticker sean moneda al aire."""
    from math import comb

    values = differences.dropna()
    n = int((values != 0).sum())
    if n == 0:
        return np.nan
    wins = int((values > 0).sum())
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(wins, n + 1)) / 2**n))


def load_prices(cache: Optional[Path] = None) -> Dict[str, pd.Series]:
    if cache and cache.exists():
        with cache.open("rb") as handle:
            return pickle.load(handle)

    data: Dict[str, pd.Series] = {}
    for ticker in TICKERS:
        try:
            prices, _ = fetch_weekly_close(ticker)
        except Exception as exc:
            print("SKIP {0}: {1}".format(ticker, exc))
            continue
        data[ticker] = prices
        print("OK {0}: {1} semanas".format(ticker, len(prices)))

    if cache:
        with cache.open("wb") as handle:
            pickle.dump(data, handle)
    return data


def run(catalog: Dict[str, Tuple[str, SignalFn]], data: Dict[str, pd.Series]) -> pd.DataFrame:
    rows: List[dict] = []
    for position, (name, (family, fn)) in enumerate(catalog.items(), start=1):
        for ticker, prices in data.items():
            try:
                metrics = evaluate(prices, fn(prices))
            except Exception as exc:
                print("  ERR {0} / {1}: {2}".format(name, ticker, exc))
                continue
            if metrics:
                rows.append(
                    {
                        "strategy": name,
                        "family": family,
                        "ticker": ticker,
                        "market": MARKET_OF[ticker],
                        **metrics,
                    }
                )
        if position % 50 == 0:
            print("  ... {0}/{1} configuraciones".format(position, len(catalog)))
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    """Agrega por configuracion y la compara, ticker a ticker, con la referencia."""
    base = detail[detail["strategy"] == BASELINE].set_index("ticker")["edge_vs_dca_pct"]

    rows: List[dict] = []
    for (name, family), group in detail.groupby(["strategy", "family"]):
        edges = group.set_index("ticker")["edge_vs_dca_pct"]
        if len(edges) < MIN_TICKERS:
            continue
        # Pareado: la diferencia se mide dentro de cada ticker, nunca entre
        # medianas de universos distintos.
        diff = (edges - base).dropna()
        rows.append(
            {
                "strategy": name,
                "family": family,
                "grupo": classify(family),
                "n_tickers": len(edges),
                "disparos": float(group["n_signals"].median()),
                # Media y mediana juntas: si divergen mucho, un puñado de
                # tickers esta mandando y hay que mirar la dispersion.
                "edge_medio_pct": float(edges.mean()),
                "edge_vs_dca_pct": float(edges.median()),
                "roi_medio_pct": float(group["roi_pct"].mean()),
                "roi_mediano_pct": float(group["roi_pct"].median()),
                "z_mediano": float(group["z_score"].median()),
                "pct_tickers_p05": float((group["p_value"] < 0.05).mean() * 100.0),
                "captura_oracle_pct": float(group["captura_oracle_pct"].median()),
                "mejora_media_pp": float(diff.mean()) if len(diff) else np.nan,
                "mejora_vs_rsi_pp": float(diff.median()) if len(diff) else np.nan,
                "gana_a_rsi_pct": float((diff > 0).mean() * 100.0) if len(diff) else np.nan,
                "p_signo": sign_test(diff),
            }
        )

    return pd.DataFrame(rows).sort_values("mejora_media_pp", ascending=False)


def classify(family: str) -> str:
    if family in DEPTH_FAMILIES:
        return "Profundidad"
    if family in OSCILLATOR_FAMILIES:
        return "Oscilador"
    if family == "Combo":
        return "Combinacion"
    return "Otras"


def wealth_on_common_universe(detail: pd.DataFrame, finalists: List[str]) -> pd.DataFrame:
    """Euros de verdad, sobre los tickers donde TODAS las finalistas son medibles."""
    subset = detail[detail["strategy"].isin(finalists)]
    counts = subset.groupby("ticker")["strategy"].nunique()
    universe = counts[counts == len(finalists)].index
    common = subset[subset["ticker"].isin(universe)]

    base = common[common["strategy"] == BASELINE].set_index("ticker")["valor_final"]
    dca = common.groupby("ticker")["valor_final_dca"].first()

    rows = []
    for name, group in common.groupby("strategy"):
        values = group.set_index("ticker")["valor_final"]
        diff = (values - base).dropna()
        rows.append(
            {
                "estrategia": name,
                "disparos": int(group["n_signals"].median()),
                "cartera_total": float(values.sum()),
                "x_veces_dca": float((values / dca).median()),
                "gana_a_rsi_pct": float((diff > 0).mean() * 100.0),
                "p_signo": sign_test(diff),
            }
        )

    table = pd.DataFrame(rows).sort_values("cartera_total", ascending=False)
    table.attrs["n_tickers"] = len(universe)
    table.attrs["dca_total"] = float(dca.sum())
    return table


# --------------------------------------------------------------------------
# Grafico
# --------------------------------------------------------------------------


def plot(summary: pd.DataFrame, n_tickers: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=SURFACE)

    # --- Izquierda: todo el barrido, selectividad frente a edge --------------
    ax = axes[0]
    # La forma del marcador duplica la identidad del grupo: el aqua queda por
    # debajo de 3:1 sobre esta superficie y el color no puede ir solo.
    styles = {
        "Profundidad": (BLUE, "o", 46, 1.0),
        "Oscilador": (ORANGE, "^", 46, 1.0),
        "Combinacion": (AQUA, "s", 52, 1.0),
        "Otras": (MUTED, "x", 26, 0.5),
    }
    for group in ("Otras", "Oscilador", "Profundidad", "Combinacion"):
        points = summary[summary["grupo"] == group]
        color, marker, size, alpha = styles[group]
        ax.scatter(
            points["disparos"],
            points["edge_medio_pct"],
            s=size,
            c=color,
            marker=marker,
            alpha=alpha,
            edgecolors=SURFACE,
            linewidths=0.8,
            label="{0} ({1})".format(group, len(points)),
            zorder=3 if group != "Otras" else 2,
        )

    reference = summary[summary["strategy"] == BASELINE].iloc[0]
    ax.scatter(
        reference["disparos"],
        reference["edge_medio_pct"],
        s=200,
        facecolors="none",
        edgecolors=INK,
        linewidths=1.8,
        zorder=4,
    )
    ax.annotate(
        "RSI(14) < 30\n{0:.0f} disparos, {1:.0f}%".format(
            reference["disparos"], reference["edge_medio_pct"]
        ),
        xy=(reference["disparos"], reference["edge_medio_pct"]),
        xytext=(14, -34),
        textcoords="offset points",
        color=INK,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.9},
    )
    ax.axhline(reference["edge_medio_pct"], color=MUTED, linewidth=0.9, linestyle="--", zorder=1)
    ax.set_xscale("log")
    ax.set_title(
        "Las senales de profundidad estan por encima a cualquier selectividad",
        color=INK,
        fontsize=12,
        pad=12,
    )
    ax.set_xlabel("Numero de disparos por ticker (escala log)", color=MUTED, fontsize=9)
    ax.set_ylabel("Edge medio sobre DCA con el mismo capital (%)", color=MUTED, fontsize=9)
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK)

    # --- Derecha: finalistas, mejora pareada contra el RSI -------------------
    ax = axes[1]
    finalists = summary.head(12).sort_values("mejora_media_pp")
    values = finalists["mejora_media_pp"].to_numpy()
    colors = [BLUE if value >= 0 else RED for value in values]
    ax.barh(finalists["strategy"], values, color=colors, height=0.62)
    ax.axvline(0, color=MUTED, linewidth=1.0)
    ax.set_title(
        "Cuanto baten al RSI, midiendo ticker a ticker", color=INK, fontsize=12, pad=12
    )
    ax.set_xlabel(
        "Mejora media del edge frente a RSI(14) < 30 (puntos porcentuales)",
        color=MUTED,
        fontsize=9,
    )

    span = float(np.abs(values).max()) or 1.0
    for y, (value, wins) in enumerate(zip(values, finalists["gana_a_rsi_pct"])):
        ax.text(
            value + span * 0.02,
            y,
            "{0:+.1f} pp   gana en {1:.0f}% de tickers".format(value, wins),
            va="center",
            ha="left",
            color=INK,
            fontsize=8.5,
        )
    ax.margins(x=0.34)

    for axis in axes:
        axis.set_facecolor(SURFACE)
        axis.grid(True, axis="x" if axis is axes[1] else "both", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=MUTED, labelsize=9)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.spines["bottom"].set_color(GRID)

    fig.suptitle(
        "Que bate al RSI cuando el capital desplegado es el mismo"
        "   ({0} tickers, semanal, 2000-{1})".format(n_tickers, dt.date.today().year),
        color=INK,
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_prices()
    catalog = build_catalog()
    print("\n{0} configuraciones x {1} tickers".format(len(catalog), len(data)))

    detail = run(catalog, data)
    summary = summarize(detail)

    finalists = [BASELINE] + [
        name for name in summary.head(8)["strategy"].tolist() if name != BASELINE
    ]
    wealth = wealth_on_common_universe(detail, finalists)

    detail_path = OUT_DIR / "entry_signals_detail.csv"
    summary_path = OUT_DIR / "entry_signals_summary.csv"
    wealth_path = OUT_DIR / "entry_signals_wealth.csv"
    plot_path = OUT_DIR / "entry_signals_vs_rsi.png"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    wealth.to_csv(wealth_path, index=False)
    plot(summary, detail["ticker"].nunique(), plot_path)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.float_format", lambda value: "{0:,.1f}".format(value))

    print("\n=== Las 15 que mas baten a RSI(14)<30 (ordenadas por mejora MEDIA) ===")
    columns = [
        "strategy", "grupo", "disparos", "edge_medio_pct", "edge_vs_dca_pct",
        "roi_medio_pct", "z_mediano", "mejora_media_pp", "mejora_vs_rsi_pp",
        "gana_a_rsi_pct", "p_signo",
    ]
    print(summary.head(15)[columns].to_string(index=False))

    print("\n=== Referencia ===")
    print(summary[summary["strategy"] == BASELINE][columns].to_string(index=False))

    print(
        "\n=== En euros: {0:,.0f} EUR por ticker, {1} tickers comunes ===".format(
            BUDGET, wealth.attrs["n_tickers"]
        )
    )
    print(wealth.to_string(index=False))
    print("DCA con el mismo presupuesto: {0:,.0f} EUR".format(wealth.attrs["dca_total"]))

    print("\nDetalle: {0}".format(detail_path))
    print("Resumen: {0}".format(summary_path))
    print("Euros:   {0}".format(wealth_path))
    print("Grafico: {0}".format(plot_path))


if __name__ == "__main__":
    main()
