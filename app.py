# file: app.py
# Executar: streamlit run app.py
from __future__ import annotations

import math
import os
import random
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------- Modelo ----------------
@dataclass(frozen=True)
class SessionResult:
    winner: str
    rounds: int
    final_player: float
    final_bank: float

def simulate_round_fast(start: float, bank_roll: float, rng: random.Random) -> Tuple[float, float, bool]:
    pot = start
    while rng.getrandbits(1) == 1:
        pot *= 2.0
        if pot >= bank_roll:
            return bank_roll, 0.0, True
    payout = pot
    if payout >= bank_roll:
        return bank_roll, 0.0, True
    return payout, bank_roll - payout, False

def simulate_session_fast(player0: float, bank0: float, v: float, start: float, rng: random.Random) -> SessionResult:
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

# ---------------- Estatística ----------------
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
    rounds = [r.rounds for r in results]
    pfinal = [r.final_player for r in results]
    bfinal = [r.final_bank for r in results]
    med_r, p95_r, p99_r = _quantiles(rounds, [0.5, 0.95, 0.99])
    med_fp, p95_fp, p99_fp = _quantiles(pfinal, [0.5, 0.95, 0.99])
    med_fb = _quantiles(bfinal, [0.5])[0]
    return Aggregate(
        n=n,
        p_player_win=wins / n,
        p_bank_win=1.0 - (wins / n),
        mean_rounds=float(statistics.mean(rounds)),
        median_rounds=float(med_r),
        p95_rounds=float(p95_r),
        p99_rounds=float(p99_r),
        mean_final_player=float(statistics.mean(pfinal)),
        median_final_player=float(med_fp),
        p95_final_player=float(p95_fp),
        p99_final_player=float(p99_fp),
        mean_final_bank=float(statistics.mean(bfinal)),
        median_final_bank=float(med_fb),
    )

# ---------------- Paralelismo ----------------
def _run_chunk(m: int, seed: int, player0: float, bank0: float, v: float, start: float) -> List[SessionResult]:
    rng = random.Random(seed)
    out: List[SessionResult] = []
    append = out.append
    for _ in range(m):
        append(simulate_session_fast(player0, bank0, v, start, rng))
    return out

def run_parallel(n: int, player0: float, bank0: float, v: float, start: float, seed: int, workers: int, chunk: int, progress_cb) -> List[SessionResult]:
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 1))
    if chunk <= 0:
        chunk = 1000
    chunks: List[int] = []
    remaining = n
    while remaining > 0:
        c = min(chunk, remaining)
        chunks.append(c)
        remaining -= c
    results_all: List[SessionResult] = []
    done, total = 0, len(chunks)
    progress_cb(done, total)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run_chunk, m, seed + i, player0, bank0, v, start) for i, m in enumerate(chunks)]
        for f in as_completed(futs):
            results_all.extend(f.result())
            done += 1
            progress_cb(done, total)
    return results_all

# ---------------- Visualizações (Plotly) ----------------
def fig_outcome_bars(results: Sequence[SessionResult]) -> go.Figure:
    wins_player = sum(1 for r in results if r.winner == "player")
    wins_bank = len(results) - wins_player
    df = pd.DataFrame({"lado": ["Jogador", "Banca"], "sessões": [wins_player, wins_bank]})
    return px.bar(df, x="lado", y="sessões", title="Resultado das sessões")

def fig_hist_rounds(results: Sequence[SessionResult]) -> go.Figure:
    rounds = [r.rounds for r in results]
    return px.histogram(pd.DataFrame({"partidas": rounds}), x="partidas", nbins=50, title="Partidas por sessão (histograma)")

def fig_ecdf_rounds(results: Sequence[SessionResult]) -> go.Figure:
    xs = np.sort(np.array([r.rounds for r in results], dtype=float))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line_shape="hv", name="ECDF"))
    fig.update_layout(title="ECDF – partidas por sessão", xaxis_title="Partidas até quebrar", yaxis_title="Probabilidade acumulada")
    return fig

def _hist_p99(values: Sequence[float], title: str, bins: int = 80) -> go.Figure:
    arr = np.array(values, dtype=float)
    p99 = float(np.percentile(arr, 99)) if len(arr) > 0 else 0.0
    df = pd.DataFrame({"valor": arr[arr <= p99]})
    return px.histogram(df, x="valor", nbins=bins, title=title)

def fig_hist_final_player_p99(results: Sequence[SessionResult]) -> go.Figure:
    vals = [r.final_player for r in results]
    return _hist_p99(vals, "Saldo final do jogador (até p99)")

def fig_hist_final_bank_p99(results: Sequence[SessionResult]) -> go.Figure:
    vals = [r.final_bank for r in results]
    return _hist_p99(vals, "Saldo final da banca (até p99)")

def _norm_ppf_acklam(p: np.ndarray) -> np.ndarray:
    a = np.array([-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00])
    b = np.array([-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01])
    c = np.array([-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00])
    d = np.array([7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00])
    plow = 0.02425; phigh = 1 - plow
    x = np.empty_like(p, dtype=float)
    mask_low = p < plow
    if np.any(mask_low):
        q = np.sqrt(-2 * np.log(p[mask_low]))
        x[mask_low] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    mask_high = p > phigh
    if np.any(mask_high):
        q = np.sqrt(-2 * np.log(1 - p[mask_high]))
        x[mask_high] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    mask_mid = (~mask_low) & (~mask_high)
    if np.any(mask_mid):
        q = p[mask_mid] - 0.5; r = q*q
        x[mask_mid] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    return x

def fig_scatter_rounds_vs_player(results: Sequence[SessionResult]) -> go.Figure:
    df = pd.DataFrame({"partidas": [r.rounds for r in results], "saldo_player": [r.final_player for r in results], "vencedor": [r.winner for r in results]})
    fig = px.scatter(df, x="partidas", y="saldo_player", color="vencedor", title="Dispersão: partidas vs. saldo final do jogador", labels={"partidas": "Partidas até quebrar", "saldo_player": "Saldo final do jogador (R$)"}, opacity=0.7)
    fig.update_traces(marker=dict(size=6), selector=dict(mode="markers"))
    return fig

def fig_qq_rounds(results: Sequence[SessionResult]) -> go.Figure:
    xs = np.sort(np.array([r.rounds for r in results], dtype=float))
    n = len(xs)
    if n == 0:
        return go.Figure()
    mean = float(xs.mean())
    std = float(xs.std(ddof=0)) or 1.0
    ps = (np.arange(1, n + 1) - 0.5) / n
    z = _norm_ppf_acklam(ps)
    theo = mean + std * z
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theo, y=xs, mode="markers", name="Dados"))
    lo = float(min(theo.min(), xs.min())); hi = float(max(theo.max(), xs.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="45°"))
    fig.update_layout(title="QQ-plot Normal – partidas por sessão", xaxis_title="Quantis teóricos (Normal)", yaxis_title="Quantis observados")
    return fig

# ---------------- UI ----------------
st.set_page_config(page_title="Jogo da Moeda – Dashboard", layout="wide")
st.title("🎲 Jogo da Moeda – Sessões até quebrar (Dashboard)")

# Estado: nunca roda ao abrir
if "run" not in st.session_state:
    st.session_state.run = False
if "params" not in st.session_state:
    st.session_state.params = None
if "results" not in st.session_state:
    st.session_state.results = None
if "agg" not in st.session_state:
    st.session_state.agg = None

with st.sidebar:
    st.header("Parâmetros obrigatórios")
    player0 = st.number_input("Quantia do Jogador (R$)", min_value=1.0, value=1_000.0, step=100.0, format="%.2f")
    bank0 = st.number_input("Quantia da Banca (R$)", min_value=1.0, value=1_000_000_000.0, step=1_000_000.0, format="%.2f")
    v = st.number_input("Valor da partida (R$)", min_value=0.01, value=50.0, step=1.0, format="%.2f")
    start = st.number_input("Quantia inicial da mesa (R$)", min_value=0.01, value=1.0, step=0.5, format="%.2f")
    st.divider()
    st.subheader("Execução")
    n = st.number_input("Sessões (n)", min_value=1, max_value=5_000_000, value=100_000, step=10_000)
    seed = st.number_input("Seed (inteiro)", min_value=0, value=42, step=1)
    workers = st.number_input("Processos (0=auto)", min_value=0, value=0, step=1)
    chunk = st.number_input("Tamanho do lote (chunk)", min_value=100, value=5000, step=100)
    st.caption("Dica: para valores grandes de jogador/banca, aumente o chunk.")
    st.divider()
    cA, cB = st.columns(2)
    with cA:
        play = st.button("▶️ Play", type="primary", use_container_width=True)
    with cB:
        reset = st.button("⏹ Reset", use_container_width=True)

if reset:
    st.session_state.run = False
    st.session_state.params = None
    st.session_state.results = None
    st.session_state.agg = None
    st.success("Estado resetado. Ajuste os parâmetros e clique Play.")

if play:
    st.session_state.run = True
    st.session_state.params = dict(n=int(n), player=float(player0), bank=float(bank0), v=float(v), start=float(start), seed=int(seed), workers=int(workers), chunk=int(chunk))
    st.session_state.results = None
    st.session_state.agg = None

# Só roda após Play
if st.session_state.run and st.session_state.params is not None:
    p = st.session_state.params
    st.info(f"Rodando: n={p['n']:,}, Jogador=R$ {p['player']:,.2f}, Banca=R$ {p['bank']:,.2f}, v=R$ {p['v']:,.2f}, start=R$ {p['start']:,.2f}, seed={p['seed']}, workers={p['workers']}, chunk={p['chunk']}")

    progress = st.progress(0.0, text="Preparando...")
    def _cb(done: int, total: int):
        frac = 0.0 if total == 0 else done / total
        progress.progress(frac, text=f"Simulando... {done}/{total} lotes concluídos")

    results = run_parallel(p["n"], p["player"], p["bank"], p["v"], p["start"], p["seed"], p["workers"], p["chunk"], _cb)
    agg = summarize_results(results)
    progress.progress(1.0, text="Concluído ✅")

    st.session_state.results = results
    st.session_state.agg = agg

# Renderiza somente se já executou
if st.session_state.results is None or st.session_state.agg is None:
    st.warning("Defina os parâmetros na **barra lateral** e clique **▶️ Play**. Nada é executado automaticamente.")
else:
    results = st.session_state.results
    agg = st.session_state.agg

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("P(jogador vence)", f"{agg.p_player_win:.6%}")
    k2.metric("P(banca vence)", f"{agg.p_bank_win:.6%}")
    k3.metric("Partidas – P50 / P95", f"{agg.median_rounds:.0f} / {agg.p95_rounds:.0f}")
    k4.metric("Saldo jogador – P50 / P99", f"R$ {agg.median_final_player:,.2f} / R$ {agg.p99_final_player:,.2f}")

    # Gráficos principais
    st.plotly_chart(fig_outcome_bars(results), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_hist_rounds(results), use_container_width=True)
    with c2:
        st.plotly_chart(fig_ecdf_rounds(results), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(fig_hist_final_player_p99(results), use_container_width=True)
    with c4:
        st.plotly_chart(fig_hist_final_bank_p99(results), use_container_width=True)

    # Extras
    st.subheader("Análises adicionais")
    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(fig_scatter_rounds_vs_player(results), use_container_width=True)
    with c6:
        st.plotly_chart(fig_qq_rounds(results), use_container_width=True)

    # Resumo + downloads
    st.subheader("Resumo estatístico")
    df_summary = pd.DataFrame(
        {
            "métrica": [
                "Sessões","P(jogador vence)","P(banca vence)","Partidas (média)","Partidas (P50)",
                "Partidas (P95)","Partidas (P99)","Saldo jogador (média)","Saldo jogador (P50)",
                "Saldo jogador (P95)","Saldo jogador (P99)","Saldo banca (média)","Saldo banca (P50)",
            ],
            "valor": [
                f"{agg.n:,}", f"{agg.p_player_win:.6%}", f"{agg.p_bank_win:.6%}",
                f"{agg.mean_rounds:.2f}", f"{agg.median_rounds:.0f}", f"{agg.p95_rounds:.0f}", f"{agg.p99_rounds:.0f}",
                f"R$ {agg.mean_final_player:,.2f}", f"R$ {agg.median_final_player:,.2f}",
                f"R$ {agg.p95_final_player:,.2f}", f"R$ {agg.p99_final_player:,.2f}",
                f"R$ {agg.mean_final_bank:,.2f}", f"R$ {agg.median_final_bank:,.2f}",
            ],
        }
    )
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    st.subheader("Exportar dados")
    df_sessions = pd.DataFrame(
        {"vencedor": [r.winner for r in results], "partidas": [r.rounds for r in results], "final_player": [r.final_player for r in results], "final_bank": [r.final_bank for r in results]}
    )
    st.download_button("⬇️ Baixar CSV das sessões", df_sessions.to_csv(index=False).encode("utf-8"), file_name="sessoes.csv", mime="text/csv", use_container_width=True)
    st.download_button("⬇️ Baixar resumo (JSON)", pd.Series(asdict(agg)).to_json().encode("utf-8"), file_name="resumo.json", mime="application/json", use_container_width=True)

    with st.expander("O que cada gráfico mostra (explicação)"):
        st.markdown(
            """
- **Resultado das sessões**: contagem de vitórias do Jogador vs Banca → probabilidades empíricas.
- **Partidas por sessão (histograma)**: duração típica; picos perto de `jogador / v`.
- **ECDF**: percentis de duração (P50, P95, P99) sem supor distribuição.
- **Saldo final do jogador/banca (p99)**: onde tipicamente terminam; truncagem evita cauda rara.
- **Dispersão (partidas vs saldo)**: sessões longas aparecem junto de ganhos grandes.
- **QQ-plot Normal**: mostra não-normalidade nas caudas (linha 45° seria normal).
"""
        )
