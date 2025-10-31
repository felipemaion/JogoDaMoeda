# file: bank_bust_report_parallel.py
from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# --- Pre-parse para backend headless quando --no-show for usado ---
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
from concurrent.futures import ProcessPoolExecutor, as_completed


# =========================== Modelo ===========================
@dataclass(frozen=True)
class SessionResult:
    winner: str           # "player" ou "bank"
    rounds: int
    final_player: float
    final_bank: float


def simulate_round_fast(start: float, bank_roll: float, rng: random.Random) -> Tuple[float, float, bool]:
    """Rápido: usa getrandbits(1). Encerra ao primeiro 0 (coroa)."""
    pot = start
    while rng.getrandbits(1) == 1:
        pot *= 2.0
        if pot >= bank_roll:  # banca não cobre
            return bank_roll, 0.0, True
    payout = pot
    if payout >= bank_roll:
        return bank_roll, 0.0, True
    return payout, bank_roll - payout, False


def simulate_session_fast(
    player0: float,
    bank0: float,
    v: float,
    start: float,
    rng: random.Random,
) -> SessionResult:
    player = float(player0)
    bank = float(bank0)
    rounds = 0

    while player >= v and bank > 0.0:
        player -= v
        bank += v
        paid, bank_after, broke_bank = simulate_round_fast(start=start, bank_roll=bank, rng=rng)
        player += paid
        bank = bank_after
        rounds += 1
        if broke_bank:
            return SessionResult("player", rounds, player, bank)

    if bank <= 0.0:
        return SessionResult("player", rounds, player, bank)
    return SessionResult("bank", rounds, player, bank)


# =========================== Estatística ===========================
def _quantiles(xs: Sequence[float], probs: Sequence[float]) -> List[float]:
    if not xs:
        return [float("nan")] * len(probs)
    s = sorted(xs)
    n = len(s)
    out: List[float] = []
    for p in probs:
        p = min(max(p, 0.0), 1.0)
        i = p * (n - 1)
        lo, hi = int(i), math.ceil(i)
        if lo == hi:
            out.append(float(s[lo]))
        else:
            w = i - lo
            out.append(float(s[lo] * (1 - w) + s[hi] * w))
    return out


@dataclass(frozen=True)
class Aggregate:
    n: int
    p_player_win: float
    p_bank_win: float
    mean_rounds: float
    median_rounds: float
    p95_rounds: float
    p99_rounds: float
    mean_final_player: float
    median_final_player: float
    p95_final_player: float
    p99_final_player: float
    mean_final_bank: float
    median_final_bank: float


def summarize_results(results: Sequence[SessionResult]) -> Aggregate:
    n = len(results)
    wins = sum(1 for r in results if r.winner == "player")
    rounds_list = [r.rounds for r in results]
    final_p_list = [r.final_player for r in results]
    final_b_list = [r.final_bank for r in results]

    med_r, p95_r, p99_r = _quantiles(rounds_list, [0.5, 0.95, 0.99])
    med_fp, p95_fp, p99_fp = _quantiles(final_p_list, [0.5, 0.95, 0.99])
    med_fb = _quantiles(final_b_list, [0.5])[0]

    return Aggregate(
        n=n,
        p_player_win=wins / n,
        p_bank_win=1.0 - (wins / n),
        mean_rounds=float(statistics.mean(rounds_list)),
        median_rounds=float(med_r),
        p95_rounds=float(p95_r),
        p99_rounds=float(p99_r),
        mean_final_player=float(statistics.mean(final_p_list)),
        median_final_player=float(med_fp),
        p95_final_player=float(p95_fp),
        p99_final_player=float(p99_fp),
        mean_final_bank=float(statistics.mean(final_b_list)),
        median_final_bank=float(med_fb),
    )


# =========================== Paralelismo ===========================
def _run_chunk(
    m: int,
    seed: int,
    player0: float,
    bank0: float,
    v: float,
    start: float,
) -> List[SessionResult]:
    rng = random.Random(seed)
    out: List[SessionResult] = []
    append = out.append
    for _ in range(m):
        res = simulate_session_fast(player0, bank0, v, start, rng)
        append(res)
    return out


def _print_progress(done: int, total: int, start_time: float) -> None:
    # por que: feedback de execução em ambientes longos
    frac = done / total if total else 1.0
    pct = int(frac * 100)
    bar_len = 30
    filled = int(bar_len * frac)
    bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
    elapsed = time.time() - start_time
    eta = (elapsed / frac - elapsed) if frac > 0 else 0.0
    msg = f"\r{bar} {pct:3d}%  chunks {done}/{total}  elapsed {elapsed:6.1f}s  ETA {eta:6.1f}s"
    sys.stderr.write(msg)
    sys.stderr.flush()
    if done == total:
        sys.stderr.write("\n")
        sys.stderr.flush()


def run_parallel(
    n: int,
    player0: float,
    bank0: float,
    v: float,
    start: float,
    seed: int,
    workers: int,
    chunk: int,
    show_progress: bool,
) -> List[SessionResult]:
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 1))
    if chunk <= 0:
        chunk = 1000

    # Quebra n em lotes
    chunks: List[int] = []
    remaining = n
    while remaining > 0:
        c = min(chunk, remaining)
        chunks.append(c)
        remaining -= c

    results_all: List[SessionResult] = []
    start_time = time.time()
    done_chunks = 0
    total_chunks = len(chunks)
    if show_progress:
        _print_progress(done_chunks, total_chunks, start_time)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = []
        for i, m in enumerate(chunks):
            futs.append(
                ex.submit(
                    _run_chunk,
                    m,
                    seed + i,     # seed diferente por chunk
                    player0,
                    bank0,
                    v,
                    start,
                )
            )
        for f in as_completed(futs):
            results_all.extend(f.result())
            done_chunks += 1
            if show_progress:
                _print_progress(done_chunks, total_chunks, start_time)
    return results_all


# =========================== Plots (1 figura por gráfico) ===========================
def _maybe_save(fig: plt.Figure, outdir: Optional[Path], name: str) -> None:
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / f"{name}.png", dpi=150, bbox_inches="tight")


def plot_outcome_bars(results: Sequence[SessionResult], outdir: Optional[Path]) -> Tuple[plt.Figure, str]:
    wins_player = sum(1 for r in results if r.winner == "player")
    wins_bank = len(results) - wins_player
    labels = ["Jogador", "Banca"]
    vals = [wins_player, wins_bank]
    fig = plt.figure()
    plt.bar(labels, vals)
    plt.title("Resultado das sessões (contagem)")
    plt.ylabel("Sessões")
    plt.tight_layout()
    _maybe_save(fig, outdir, "01_outcomes")
    expl = (
        "Como ler: barras mostram quantas sessões terminaram com vitória do jogador ou da banca. "
        "Importância: fornece a probabilidade empírica de cada lado vencer. "
        "Significado: com banca muito grande e stake fixo, quase todas as sessões terminam com o jogador quebrando."
    )
    return fig, expl


def plot_hist_rounds(results: Sequence[SessionResult], outdir: Optional[Path]) -> Tuple[plt.Figure, str]:
    rounds = [r.rounds for r in results]
    fig = plt.figure()
    plt.hist(rounds, bins=50)
    plt.title("Partidas por sessão (histograma)")
    plt.xlabel("Partidas até quebrar")
    plt.ylabel("Frequência")
    plt.tight_layout()
    _maybe_save(fig, outdir, "02_rounds_hist")
    expl = (
        "Como ler: distribuição do número de partidas até o término. "
        "Importância: mede duração típica das sessões e risco de sequência de perdas. "
        "Significado: o pico próximo a saldo_inicial/stake indica que o jogador geralmente quebra sem grandes prêmios."
    )
    return fig, expl


def plot_ecdf_rounds(results: Sequence[SessionResult], outdir: Optional[Path]) -> Tuple[plt.Figure, str]:
    rounds = np.sort(np.array([r.rounds for r in results], dtype=float))
    ys = np.arange(1, len(rounds) + 1) / len(rounds)
    fig = plt.figure()
    plt.plot(rounds, ys, drawstyle="steps-post")
    plt.title("ECDF – partidas por sessão")
    plt.xlabel("Partidas até quebrar")
    plt.ylabel("Probabilidade acumulada")
    plt.tight_layout()
    _maybe_save(fig, outdir, "03_rounds_ecdf")
    expl = (
        "Como ler: para um X, a curva dá a fração de sessões com duração ≤ X. "
        "Importância: lê percentis de duração (P50, P90, P95) diretamente. "
        "Significado: maior parte das sessões acaba rapidamente; cauda longa aparece quando ocorre ganho raro."
    )
    return fig, expl


def plot_hist_final_player_p99(results: Sequence[SessionResult], outdir: Optional[Path]) -> Tuple[plt.Figure, str]:
    vals = np.array([r.final_player for r in results], dtype=float)
    p99 = float(np.percentile(vals, 99))
    fig = plt.figure()
    plt.hist(vals[vals <= p99], bins=80)
    plt.title("Saldo final do jogador (até p99)")
    plt.xlabel("R$ saldo ao final da sessão")
    plt.ylabel("Frequência")
    plt.tight_layout()
    _maybe_save(fig, outdir, "04_final_player_p99")
    expl = (
        "Como ler: distribuição do saldo final do jogador truncada em p99 para focar no típico. "
        "Importância: mostra onde o jogador normalmente termina. "
        "Significado: a massa perto de valores baixos indica que, sem um grande prêmio, o jogador tende a quebrar."
    )
    return fig, expl


def plot_hist_final_bank_p99(results: Sequence[SessionResult], outdir: Optional[Path]) -> Tuple[plt.Figure, str]:
    vals = np.array([r.final_bank for r in results], dtype=float)
    p99 = float(np.percentile(vals, 99))
    fig = plt.figure()
    plt.hist(vals[vals <= p99], bins=80)
    plt.title("Saldo final da banca (até p99)")
    plt.xlabel("R$ saldo ao final da sessão")
    plt.ylabel("Frequência")
    plt.tight_layout()
    _maybe_save(fig, outdir, "05_final_bank_p99")
    expl = (
        "Como ler: distribuição do saldo final da banca truncada em p99. "
        "Importância: evidencia variação típica do caixa da banca por sessão. "
        "Significado: com banca enorme, a variação percentual é mínima; raramente há perda grande."
    )
    return fig, expl


# =========================== PDF + Report ===========================
def _text_page(title: str, body: str) -> plt.Figure:
    fig = plt.figure()
    plt.axis("off")
    plt.text(0.02, 0.95, title, fontsize=14, fontweight="bold", va="top", ha="left")
    plt.text(0.02, 0.90, body, fontsize=11, va="top", ha="left", wrap=True)
    return fig


def save_pdf(
    pdf_path: Path,
    params: Dict[str, float | int],
    agg: Aggregate,
    figures: List[Tuple[plt.Figure, str]],
) -> None:
    percentis_text = (
        "Percentil Pxx é o valor abaixo do qual estão xx% das observações. "
        "P50 (mediana) separa ao meio; P90 deixa 10% acima; P99 deixa 1% acima. "
        "Em distribuições com cauda pesada, percentis descrevem o 'típico' melhor que a média."
    )
    long_run_text = (
        "Longo prazo: com banca muito maior que o jogador e stake fixo, a probabilidade de o jogador quebrar "
        "é muito alta. Ocasionalmente, ocorre um prêmio raro e enorme que prolonga a sessão ou até quebra a banca, "
        "mas com banca de R$1e9 isso é praticamente impossível em horizontes modestos."
    )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        cover = _text_page(
            "Relatório – Sessões até quebrar (paralelo)",
            f"Parâmetros: n={params['n']:,}, jogador=R${params['player']:.2f}, banca=R${params['bank']:.2f}, "
            f"stake=R${params['v']:.2f}, start=R${params['start']:.2f}, seed={params['seed']}, "
            f"workers={params['workers']}, chunk={params['chunk']}",
        )
        pdf.savefig(cover); plt.close(cover)

        stats_body = (
            f"Probabilidades:\n"
            f"- P(jogador vence / banca quebra): {agg.p_player_win:.6%}\n"
            f"- P(banca vence / jogador quebra): {agg.p_bank_win:.6%}\n\n"
            f"Duração da sessão (partidas): mean={agg.mean_rounds:.2f} | P50={agg.median_rounds:.0f} | "
            f"P95={agg.p95_rounds:.0f} | P99={agg.p99_rounds:.0f}\n"
            f"Saldo final do jogador: mean=R$ {agg.mean_final_player:,.2f} | "
            f"P50=R$ {agg.median_final_player:,.2f} | P95=R$ {agg.p95_final_player:,.2f} | "
            f"P99=R$ {agg.p99_final_player:,.2f}\n"
            f"Saldo final da banca: mean=R$ {agg.mean_final_bank:,.2f} | "
            f"P50=R$ {agg.median_final_bank:,.2f}\n"
        )
        sp = _text_page("Estatísticas principais", stats_body)
        pdf.savefig(sp); plt.close(sp)

        perc = _text_page("Percentis (P50, P90, P99…)", percentis_text)
        pdf.savefig(perc); plt.close(perc)

        for i, (fig, expl) in enumerate(figures, 1):
            pdf.savefig(fig); plt.close(fig)
            ep = _text_page(f"Explicação do Gráfico {i}", expl)
            pdf.savefig(ep); plt.close(ep)

        longp = _text_page("Longo prazo – interpretação", long_run_text)
        pdf.savefig(longp); plt.close(longp)


def print_stdout(params: Dict[str, float | int], agg: Aggregate, texts: List[str]) -> None:
    print("-" * 80)
    print(f"Parâmetros: n={params['n']:,} | jogador=R${params['player']:.2f} | banca=R${params['bank']:.2f} | "
          f"stake=R${params['v']:.2f} | start=R${params['start']:.2f} | seed={params['seed']} | "
          f"workers={params['workers']} | chunk={params['chunk']}")
    print(f"P(jogador vence): {agg.p_player_win:.6%} | P(banca vence): {agg.p_bank_win:.6%}")
    print(f"Partidas/sessão: mean={agg.mean_rounds:.2f} | P50={agg.median_rounds:.0f} | "
          f"P95={agg.p95_rounds:.0f} | P99={agg.p99_rounds:.0f}")
    print(f"Saldo final jogador: mean=R$ {agg.mean_final_player:,.2f} | "
          f"P50=R$ {agg.median_final_player:,.2f} | P95=R$ {agg.p95_final_player:,.2f} | "
          f"P99=R$ {agg.p99_final_player:,.2f}")
    print(f"Saldo final banca:   mean=R$ {agg.mean_final_bank:,.2f} | "
          f"P50=R$ {agg.median_final_bank:,.2f}")
    print("-" * 80)
    for i, t in enumerate(texts, 1):
        print(f"[Gráfico {i}] {t}")
    print("-" * 80)


# =========================== CLI ===========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        parents=[_base_parser()],
        description="Simula sessões até quebrar em paralelo e gera PDF/plots.",
    )
    p.add_argument("--n", type=int, default=100_000, help="Número de sessões")
    p.add_argument("--player", type=float, default=1_000.0, help="Caixa inicial do jogador (R$)")
    p.add_argument("--bank", type=float, default=1_000_000_000.0, help="Caixa inicial da banca (R$)")
    p.add_argument("--v", type=float, default=50.0, help="Stake por partida (R$)")
    p.add_argument("--start", type=float, default=1.0, help="Pote inicial (R$)")
    p.add_argument("--seed", type=int, default=42, help="Seed RNG base")
    p.add_argument("--workers", type=int, default=0, help="Número de processos (0=auto)")
    p.add_argument("--chunk", type=int, default=2000, help="Tamanho do lote por tarefa")
    p.add_argument("--progress", action="store_true", help="Mostrar barra de progresso e ETA")
    p.add_argument("--savefig", type=Path, default=None, help="Diretório para salvar PNGs")
    p.add_argument("--pdf", type=Path, default=None, help="Salvar relatório em PDF")
    return p.parse_args()


# =========================== Main ===========================
def main() -> None:
    args = parse_args()

    results = run_parallel(
        n=args.n,
        player0=args.player,
        bank0=args.bank,
        v=args.v,
        start=args.start,
        seed=args.seed,
        workers=args.workers,
        chunk=args.chunk,
        show_progress=args.progress,
    )
    agg = summarize_results(results)

    figs: List[Tuple[plt.Figure, str]] = []
    f1, t1 = plot_outcome_bars(results, args.savefig);            figs.append((f1, t1))
    f2, t2 = plot_hist_rounds(results, args.savefig);             figs.append((f2, t2))
    f3, t3 = plot_ecdf_rounds(results, args.savefig);             figs.append((f3, t3))
    f4, t4 = plot_hist_final_player_p99(results, args.savefig);   figs.append((f4, t4))
    f5, t5 = plot_hist_final_bank_p99(results, args.savefig);     figs.append((f5, t5))

    print_stdout(
        params={"n": args.n, "player": args.player, "bank": args.bank, "v": args.v, "start": args.start,
                "seed": args.seed, "workers": args.workers or (os.cpu_count() or 1), "chunk": args.chunk},
        agg=agg,
        texts=[t1, t2, t3, t4, t5],
    )

    if args.pdf:
        save_pdf(
            pdf_path=args.pdf,
            params={"n": args.n, "player": args.player, "bank": args.bank, "v": args.v, "start": args.start,
                    "seed": args.seed, "workers": args.workers or (os.cpu_count() or 1), "chunk": args.chunk},
            agg=agg,
            figures=figs,
        )
        print(f"PDF salvo em: {args.pdf}")

    if not _pre_args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
