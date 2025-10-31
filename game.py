# file: game_gauss_report.py
from __future__ import annotations

import argparse
import math
import random
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Backend headless se solicitado
def _base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--no-show", action="store_true")
    return p

_pre_args, _ = _base_parser().parse_known_args()
if _pre_args.no_show:
    import matplotlib
    matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from scipy import stats as spstats  # opcional para QQ-plot
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ===================== Simulação =====================
def simulate_game(v: float, start: float, rng: random.Random) -> float:
    pot = start
    while rng.random() < 0.5:
        pot *= 2.0
    return pot - v


def run_sim(n: int, v: float, start: float, seed: Optional[int]) -> List[float]:
    rng = random.Random(seed)
    return [simulate_game(v, start, rng) for _ in range(n)]


# ===================== Estatísticas =====================
def _quantile(xs_sorted: Sequence[float], p: float) -> float:
    n = len(xs_sorted)
    i = p * (n - 1)
    lo, hi = int(i), math.ceil(i)
    if lo == hi:
        return xs_sorted[lo]
    w = i - lo
    return xs_sorted[lo] * (1 - w) + xs_sorted[hi] * w


def summarize(data: Sequence[float]) -> Dict[str, float]:
    xs = sorted(map(float, data))
    n = len(xs)
    mean = float(sum(xs) / n)
    stdev = float(statistics.pstdev(xs))
    # Momentos p/ skew/kurtosis (excesso)
    m2 = sum((x - mean) ** 2 for x in xs) / n
    m3 = sum((x - mean) ** 3 for x in xs) / n
    m4 = sum((x - mean) ** 4 for x in xs) / n
    if m2 <= 0:
        skew = 0.0
        ex_kurt = 0.0
    else:
        skew = m3 / (m2 ** 1.5)
        ex_kurt = m4 / (m2 ** 2) - 3.0
    # Jarque–Bera ~ χ²(2)
    JB = n / 6.0 * (skew ** 2 + (ex_kurt ** 2) / 4.0)
    # p-valor sem dependência externa (aprox. cauda de χ²2): p = exp(-JB/2)
    # (Exato seria 1 - CDF_chi2(JB, df=2) = exp(-JB/2))
    pval_jb = math.exp(-JB / 2.0)

    return {
        "n": float(n),
        "mean": mean,
        "stdev": stdev,
        "median": float(statistics.median(xs)),
        "min": float(xs[0]),
        "max": float(xs[-1]),
        "p_win": float(sum(1 for x in xs if x > 0) / n),
        "p50": _quantile(xs, 0.50),
        "p75": _quantile(xs, 0.75),
        "p90": _quantile(xs, 0.90),
        "p95": _quantile(xs, 0.95),
        "p99": _quantile(xs, 0.99),
        "skew": float(skew),
        "ex_kurt": float(ex_kurt),
        "jb": float(JB),
        "jb_pvalue": float(pval_jb),
    }


# ===================== Gráficos (1 figura por gráfico) =====================
def _maybe_save(fig: plt.Figure, outdir: Optional[Path], name: str) -> None:
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / f"{name}.png", dpi=150, bbox_inches="tight")


def plot_hist_with_gauss(saldos: Sequence[float], stats: Dict[str, float], outdir: Optional[Path]) -> Tuple[plt.Figure, str]:
    xs = np.asarray(saldos, dtype=float)
    mu, sigma = stats["mean"], max(stats["stdev"], 1e-12)
    # histograma em densidade
    fig = plt.figure()
    counts, bin_edges, _ = plt.hist(xs, bins=200, density=True, alpha=0.6)
    # curva normal ajustada (mesma escala de densidade)
    grid = np.linspace(float(bin_edges[0]), float(bin_edges[-1]), 1000)
    norm_pdf = 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((grid - mu) / sigma) ** 2)
    plt.plot(grid, norm_pdf)
    plt.yscale("log")  # log para mostrar cauda vs Gauss
    plt.title("Histograma (densidade) + Curva de Gauss ajustada (escala log)")
    plt.xlabel("Saldo por jogo (R$)")
    plt.ylabel("Densidade (log)")
    plt.tight_layout()
    _maybe_save(fig, outdir, "01_hist_gauss_log")
    explain = (
        "Como ler: barras = densidade empírica; linha = normal ajustada pela média/σ dos dados. "
        "Importância: contrasta o corpo da amostra com a Gauss; em escala log, a cauda empírica "
        "decai muito mais lentamente que a normal. "
        "Significado: a suposição normal é inadequada — o jogo tem cauda direita pesada."
    )
    return fig, explain


def plot_hist_body_p99(saldos: Sequence[float], outdir: Optional[Path]) -> Tuple[plt.Figure, str]:
    xs = np.asarray(saldos, dtype=float)
    p99 = float(np.percentile(xs, 99))
    fig = plt.figure()
    plt.hist(xs[xs <= p99], bins=100)
    plt.title("Distribuição dos saldos (até p99)")
    plt.xlabel("Saldo por jogo (R$)")
    plt.ylabel("Frequência")
    plt.tight_layout()
    _maybe_save(fig, outdir, "02_hist_body_p99")
    explain = (
        "Como ler: histograma truncado no percentil 99 foca no típico. "
        "Importância: destaca que quase sempre os resultados ficam perto de perdas (~−v). "
        "Significado: 99% dos jogos não acessam a cauda extrema."
    )
    return fig, explain


def plot_ecdf(saldos: Sequence[float], outdir: Optional[Path]) -> Tuple[plt.Figure, str]:
    xs = np.sort(np.asarray(saldos, dtype=float))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    fig = plt.figure()
    plt.plot(xs, ys, drawstyle="steps-post")
    plt.title("ECDF dos saldos")
    plt.xlabel("Saldo por jogo (R$)")
    plt.ylabel("Probabilidade acumulada")
    plt.tight_layout()
    _maybe_save(fig, outdir, "03_ecdf")
    explain = (
        "Como ler: para X no eixo, a curva mostra fração de jogos ≤ X. "
        "Importância: lê diretamente percentis (P50, P90, P99). "
        "Significado: subida rápida nas perdas, subida lenta na cauda de ganhos raros."
    )
    return fig, explain


def plot_cumulative(saldos: Sequence[float], outdir: Optional[Path], k: int) -> Tuple[plt.Figure, str]:
    k = max(1, min(k, len(saldos)))
    cum = []
    s = 0.0
    for x in saldos[:k]:
        s += x
        cum.append(s)
    fig = plt.figure()
    plt.plot(cum)
    plt.title(f"Saldo cumulativo ({k:,} jogos)")
    plt.xlabel("Jogo")
    plt.ylabel("Saldo acumulado (R$)")
    plt.tight_layout()
    _maybe_save(fig, outdir, "04_cumulative")
    explain = (
        f"Como ler: trajetória do saldo nos primeiros {k} jogos. "
        "Importância: mostra risco temporal (longas sequências negativas). "
        "Significado: muitos degraus para baixo e raros saltos positivos grandes."
    )
    return fig, explain


def plot_qq_normal(saldos: Sequence[float], outdir: Optional[Path]) -> Tuple[Optional[plt.Figure], str]:
    if not _HAVE_SCIPY:
        return None, (
            "QQ-plot normal requer SciPy; não encontrado. Instale scipy para gerar esta verificação visual "
            "de normalidade (alinhamento com a linha de 45° indicaria normalidade)."
        )
    xs = np.asarray(saldos, dtype=float)
    fig = plt.figure()
    spstats.probplot(xs, dist="norm", plot=plt)  # usa média/σ empíricos
    plt.title("QQ-plot Normal")
    plt.tight_layout()
    _maybe_save(fig, outdir, "05_qq_normal")
    explain = (
        "Como ler: se os pontos seguem a linha, os dados se parecem com normal; desvios nas caudas indicam não-normalidade. "
        "Importância: evidencia cauda direita pesada (pontos no topo acima da linha). "
        "Significado: reforça que a normal subestima probabilidades de ganhos extremos."
    )
    return fig, explain


# ===================== PDF & Report =====================
def _text_page(title: str, body: str) -> plt.Figure:
    fig = plt.figure()
    plt.axis("off")
    plt.text(0.02, 0.95, title, fontsize=14, fontweight="bold", va="top", ha="left")
    plt.text(0.02, 0.90, body, fontsize=11, va="top", ha="left", wrap=True)
    return fig


def save_pdf(
    pdf_path: Path,
    params: Dict[str, float | int],
    stats: Dict[str, float],
    sections: List[Tuple[str, str]],
    figs_and_expl: List[Tuple[plt.Figure, str]],
) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        # Capa
        cover = _text_page(
            "Relatório – Histograma, Curva de Gauss e Análises",
            f"Parâmetros: n={params['n']:,}, v=R${params['v']:.2f}, start=R${params['start']:.2f}, seed={params['seed']}",
        )
        pdf.savefig(cover); plt.close(cover)
        # Estatísticas
        stats_body = (
            f"Média: R$ {stats['mean']:.2f} | Mediana (P50): R$ {stats['median']:.2f} | σ: {stats['stdev']:.2f}\n"
            f"Min/Máx: R$ {stats['min']:.2f} / R$ {stats['max']:.2f} | P(win): {stats['p_win']:.2%}\n"
            f"P75: {stats['p75']:.2f} | P90: {stats['p90']:.2f} | P95: {stats['p95']:.2f} | P99: {stats['p99']:.2f}\n"
            f"Skewness: {stats['skew']:.3f} | Excesso de curtose: {stats['ex_kurt']:.3f}\n"
            f"Jarque–Bera: {stats['jb']:.2f} (p≈{stats['jb_pvalue']:.2e})"
        )
        stats_page = _text_page("Estatísticas principais", stats_body)
        pdf.savefig(stats_page); plt.close(stats_page)
        # Seções textuais
        for title, body in sections:
            pg = _text_page(title, body); pdf.savefig(pg); plt.close(pg)
        # Gráficos + explicações
        for i, (fig, expl) in enumerate(figs_and_expl, 1):
            pdf.savefig(fig); plt.close(fig)
            pg = _text_page(f"Explicação do Gráfico {i}", expl); pdf.savefig(pg); plt.close(pg)


def print_stdout_summary(params: Dict[str, float | int], stats: Dict[str, float], sections: List[Tuple[str, str]], graph_texts: List[str]) -> None:
    print("-" * 78)
    print(f"Parâmetros: n={params['n']:,} | v=R${params['v']:.2f} | start=R${params['start']:.2f} | seed={params['seed']}")
    print(
        f"Mean={stats['mean']:.2f} | Median(P50)={stats['median']:.2f} | σ={stats['stdev']:.2f} | "
        f"Min/Max={stats['min']:.2f}/{stats['max']:.2f} | P(win)={stats['p_win']:.2%}"
    )
    print(
        f"P75={stats['p75']:.2f} | P90={stats['p90']:.2f} | P95={stats['p95']:.2f} | P99={stats['p99']:.2f} | "
        f"Skew={stats['skew']:.3f} | ExKurt={stats['ex_kurt']:.3f} | JB={stats['jb']:.2f}, p≈{stats['jb_pvalue']:.2e}"
    )
    print("-" * 78)
    for title, body in sections:
        print(f"[{title}] {body}")
        print("-" * 78)
    for i, t in enumerate(graph_texts, 1):
        print(f"[Gráfico {i}] {t}")
    print("-" * 78)


# ===================== CLI =====================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        parents=[_base_parser()],
        description="Histograma com curva de Gauss e análises pertinentes (ECDF, QQ-plot, cumulativo, JB).",
    )
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--v", type=float, default=50.0)
    p.add_argument("--start", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--savefig", type=Path, default=None)
    p.add_argument("--pdf", type=Path, default=None)
    p.add_argument("--cum-n", type=int, default=1000)
    return p.parse_args()


# ===================== Main =====================
def main() -> None:
    args = parse_args()
    saldos = run_sim(args.n, args.v, args.start, args.seed)
    st = summarize(saldos)

    # Seções textuais (explicações)
    percentis_text = (
        "Percentil Pxx: valor abaixo do qual estão xx% dos saldos. P50=mediana; P99 deixa 1% acima. "
        "Útil para descrever o 'típico' ignorando caudas raras."
    )
    gauss_vs_real = (
        "Curva de Gauss é simétrica e de cauda fina; este jogo é assimétrico com cauda direita pesada. "
        "A média/σ existem na amostra, mas a normal subestima fortemente a probabilidade de ganhos extremos."
    )
    long_run = (
        "No longo prazo, a experiência é dominada por muitas perdas pequenas e raros ganhos enormes. "
        "Mesmo com média amostral menos negativa (ou positiva em algumas execuções), a mediana e P90/P95 "
        "ficam negativas: o jogador tipicamente perde por muito tempo e depende de eventos raros para recuperar."
    )

    sections = [
        ("Percentis (P50, P90, P99…)", percentis_text),
        ("Gauss vs realidade", gauss_vs_real),
        ("Longo prazo: o que esperar", long_run),
    ]

    figs_texts: List[Tuple[plt.Figure, str]] = []
    f1, t1 = plot_hist_with_gauss(saldos, st, args.savefig);      figs_texts.append((f1, t1))
    f2, t2 = plot_hist_body_p99(saldos, args.savefig);             figs_texts.append((f2, t2))
    f3, t3 = plot_ecdf(saldos, args.savefig);                      figs_texts.append((f3, t3))
    f4, t4 = plot_cumulative(saldos, args.savefig, args.cum_n);    figs_texts.append((f4, t4))
    f5, t5 = plot_qq_normal(saldos, args.savefig);                 # pode ser None

    # stdout
    graph_texts = [t1, t2, t3, t4]
    if f5 is None:
        graph_texts.append(t5)  # só o texto explicativo
    else:
        figs_texts.append((f5, t5)); graph_texts.append(t5)

    print_stdout_summary(
        params={"n": args.n, "v": args.v, "start": args.start, "seed": args.seed},
        stats=st,
        sections=sections,
        graph_texts=graph_texts,
    )

    # PDF
    if args.pdf:
        save_pdf(
            pdf_path=Path(args.pdf),
            params={"n": args.n, "v": args.v, "start": args.start, "seed": args.seed},
            stats=st,
            sections=sections,
            figs_and_expl=figs_texts,
        )
        print(f"PDF salvo em: {args.pdf}")

    # Mostrar janelas caso permitido
    if not _pre_args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
