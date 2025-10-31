# Jogo da Moeda – Dashboard, Simuladores e Relatórios

**Comece pelo `app.py` (dashboard web Streamlit).** Depois, se quiser rodar por linha de comando, use `game.py` e `game_until_win_brake.py`.

---

## 🚀 Como instalar (usando `requirements.txt`)

```bash
# (Opcional) ambiente virtual
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# dependências
pip install -r requirements.txt
```

> O arquivo `requirements.txt` contém: `streamlit`, `plotly`, `numpy`, `pandas`, `matplotlib`.  
> (Opcional) Para QQ-plot baseado em SciPy em alguns scripts, instale também: `pip install scipy`.

---

## 🌐 `app.py` — Dashboard web interativo (Streamlit)

**O que faz**
- Executa **sessões “até quebrar”** (jogador vs banca) **somente após você clicar ▶️ Play**.
- Paraleliza a simulação (usa múltiplos processos) e exibe **barra de progresso**.
- Mostra KPIs & gráficos: **barras (vencedor)**, **histograma de partidas**, **ECDF**, **saldo final p/ jogador/banca (p99)**, **dispersão (partidas × saldo)**, **QQ-plot**.
- Permite **download** de **CSV** (sessões) e **JSON** (resumo).

**Como executar**
```bash
streamlit run app.py
# Abra o link exibido (ex.: http://localhost:8501)
```

**Uso**
1. Na **barra lateral**, preencha **QUATRO parâmetros obrigatórios**:  
   - **Quantia do Jogador (R$)**  
   - **Quantia da Banca (R$)**  
   - **Valor da partida (R$ 50)**  
   - **Quantia inicial da mesa (R$ 1)**  
   (Ajuste também `Sessões (n)`, `Seed`, `Processos (0=auto)` e `Chunk`.)
2. Clique **▶️ Play**. Nada roda automaticamente ao abrir.
3. Acompanhe o **progresso** e veja os **KPIs + gráficos**.
4. Baixe os dados: **CSV** das sessões e **JSON** do resumo.

**Dicas de performance**
- Para **n** grande ou **Quantia do Jogador** alta, use **Processos = 0 (auto)** e aumente **Chunk** (ex.: 5.000–20.000).
- Teste primeiro com `n` menor; aumente depois.

---

## 🧪 `game.py` — Jogos independentes + Gauss/ECDF/Cumulativo/QQ

**O que faz**
- Simula **n jogos independentes** (sem caixa do jogador), saldo por jogo = **pote − v**.
- Gera gráficos: **Histograma + Curva de Gauss** (escala log), **histograma até p99**, **ECDF**, **saldo cumulativo** e, se disponível SciPy, **QQ-plot**.
- Pode exportar **PDF** com capa, estatísticas, seções explicativas e todos os gráficos; também salva **PNGs**.

**Exemplos**
```bash
# Relatório completo, headless
python game.py --n 100000 --v 50 --start 1 --seed 42   --pdf out/gauss_report.pdf --savefig out/ --no-show

# Visualização rápida com janelas
python game.py --n 2000 --v 50 --start 1 --seed 42
```

**Parâmetros principais**
`--n`, `--v`, `--start`, `--seed`, `--cum-n`, `--savefig DIR`, `--pdf ARQ.pdf`, `--no-show`

**Saídas**
- **STDOUT**: média, mediana, Pxx, skew/curtose, Jarque–Bera + explicações.
- **PNG/PDF**: gráficos com “Como ler / Por que importa / Significado”.

---

## 🏦 `game_until_win_brake.py` — Sessões **até quebrar** (paralelo, PDF/plots)

**O que faz**
- Simula **n sessões** com **caixa do Jogador** e **caixa da Banca**. Cada partida: paga-se **v**; o pote dobra em “cara” e paga-se em “coroa”; banca quebra se não cobrir o prêmio.
- Termina quando **jogador** não consegue pagar **v** ou quando a **banca** não consegue pagar o pote.
- Suporta **ProcessPoolExecutor** (paralelo) e **barra de progresso** no terminal.
- Gera gráficos (resultado, partidas, ECDF, saldos finais p99) e **PDF** com explicações.

**Exemplos**
```bash
# 100k sessões, paralelo auto, progresso, sem abrir janelas
python game_until_win_brake.py --n 100000 --player 1000 --bank 1000000000 --v 50 --start 1   --workers 0 --chunk 5000 --progress --pdf out/relatorio_parallel.pdf --savefig out/ --no-show

# Teste menor com jogador alto
python game_until_win_brake.py --n 20000 --player 1000000 --bank 1000000000 --v 50 --start 1   --workers 0 --progress
```

**Parâmetros principais**
`--n`, `--player`, `--bank`, `--v`, `--start`, `--seed`, `--workers`, `--chunk`, `--progress`, `--savefig`, `--pdf`, `--no-show`

**Saídas**
- **STDOUT**: P(jogador vence) vs P(banca vence), percentis/médias de partidas e saldos.
- **PNG/PDF**: figuras e relatório com seções explicativas (percentis, longo prazo).

---

## 📂 Estrutura do repositório

```
.
├── app.py                      # Dashboard Streamlit
├── game.py                     # Jogos independentes + Gauss/ECDF/QQ/cumulativo
├── game_until_win_brake.py     # Sessões até quebrar (paralelo, progresso, PDF/plots)
├── requirements.txt            # Dependências (pip install -r requirements.txt)
└── out/                        # (opcional) PDFs/PNGs gerados
```

---

## ❓ FAQ / Solução de problemas

- **“O dashboard abre mas não roda”** → comportamento esperado: defina os parâmetros e clique **▶️ Play**.
- **Execução lenta** → use **workers=0** (auto) e aumente **chunk**; reduza `n` para testes; `--no-show` para headless.
- **PDF/PNGs não aparecem** → verifique diretórios de `--pdf`/`--savefig` (crie `out/` se necessário).
- **Windows + processos** → execute o script via `python arquivo.py` (não REPL).

---

## 📜 Licença
Projeto educacional. Adapte e evolua conforme sua necessidade.
