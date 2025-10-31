# Jogo da Moeda – Simulações, Relatórios e Dashboard

Este repositório contém três componentes para estudar o jogo da moeda com **pote que dobra a cada “cara”** e termina em **“coroa”**:

- **`game.py`** — Simulador de jogos independentes com relatório estatístico (histograma, curva de Gauss, ECDF, cumulativo, QQ-plot opcional) e exportação em PDF/PNGs.
- **`game_until_win_brake.py`** — Simulador de **sessões até quebrar** (jogador vs banca) com **paralelismo** e barra de progresso opcional, gerando gráficos e PDF.
- **`app.py`** — Dashboard **Streamlit** interativo para rodar as sessões até quebrar em paralelo, visualizar KPIs, histogramas, ECDF, dispersão e QQ-plot, e baixar CSV/JSON.

> Regra básica do jogo: para jogar paga-se **v** (ex.: R$ 50). O pote começa em **start** (ex.: R$ 1) e **dobra** a cada “cara” até sair “coroa”; ao sair “coroa” o jogador recebe o pote. Nos scripts de **sessão**, o jogador tem um caixa e joga até **quebrar** ou **quebrar a banca**.

---

## 1) `game.py` — jogos independentes + Gauss, ECDF, cumulativo, QQ

**O que faz**  
- Simula **n** jogos independentes (sem caixa do jogador), computa o **saldo por jogo = pote − v**.
- Gera gráficos: **Histograma + Curva de Gauss ajustada (escala log)**, **histograma até p99**, **ECDF**, **saldo cumulativo (primeiros k)** e **QQ-plot normal** (quando SciPy estiver instalado).
- Pode gerar **PDF** com capa, estatísticas, seções explicativas e todos os gráficos, além de salvar PNGs.

**Execução rápida**

```bash
# Ambiente headless (não abre janelas), salvando PDF e PNGs
python game.py --n 100000 --v 50 --start 1 --seed 42 --pdf out/gauss_report.pdf --savefig out/ --no-show

# Execução com janelas de gráficos
python game.py --n 1000 --v 50 --start 1 --seed 42
```

**Principais parâmetros**
- `--n`: quantidade de jogos a simular.
- `--v`: valor pago por jogo (stake).
- `--start`: pote inicial.
- `--seed`: semente do RNG para reprodutibilidade.
- `--cum-n`: tamanho da janela do gráfico cumulativo.
- `--savefig DIR`: salva PNGs no diretório informado.
- `--pdf ARQ.pdf`: exporta o relatório em PDF.
- `--no-show`: evita abrir janelas (útil em servidor).

**Saídas**  
- **STDOUT**: resumo estatístico (média, mediana, Pxx, skew, curtose, Jarque–Bera) + explicações.
- **PNG/PDF**: gráficos com explicação de leitura e importância.

---

## 2) `game_until_win_brake.py` — sessões até quebrar (paralelo, gráficos, PDF)

**O que faz**  
- Simula **n sessões** onde o **jogador** começa com um caixa (ex.: R$ 1.000) e a **banca** com outro (ex.: R$ 1.000.000.000).  
- Em cada partida: o jogador paga **v** à banca; resolve-se o jogo do pote; a banca paga o pote (se conseguir).  
- A sessão termina quando **o jogador não consegue pagar v** (jogador quebra) **ou** quando **a banca não consegue pagar o pote** (banca quebra).  
- Suporta **paralelismo** via `ProcessPoolExecutor` e **barra de progresso** no terminal.  
- Gera gráficos (barras de resultado, histogramas, ECDF, saldos finais p99) e **PDF** com explicações.

**Execução rápida**

```bash
# 100k sessões, auto-CPUs, com progresso, sem abrir janelas
python game_until_win_brake.py --n 100000 --player 1000 --bank 1000000000 --v 50 --start 1   --workers 0 --chunk 5000 --progress --pdf out/relatorio_parallel.pdf --savefig out/ --no-show

# Execução menor para testes locais
python game_until_win_brake.py --n 20000 --player 1000000 --bank 1000000000 --v 50 --start 1 --workers 0 --progress
```

**Principais parâmetros**
- `--n`: número de **sessões** (independentes).
- `--player`: caixa inicial do jogador (R$).
- `--bank`: caixa inicial da banca (R$).
- `--v`: valor por partida (stake).
- `--start`: pote inicial na mesa (R$).
- `--seed`: semente RNG base (cada **chunk** recebe `seed+i`).
- `--workers`: nº de processos (0 = auto).
- `--chunk`: tamanho do lote por tarefa (ajuste para throughput).
- `--progress`: habilita barra de progresso + ETA no terminal.
- `--savefig DIR` / `--pdf ARQ.pdf` / `--no-show` conforme acima.

**Saídas**  
- **STDOUT**: probabilidades empíricas de vitória (jogador vs banca), percentis e médias de **partidas** e **saldos**.
- **PNG/PDF**: figuras e relatório com explicações (percentis, longo prazo, etc.).

---

## 3) `app.py` — Dashboard web (Streamlit)

**O que faz**  
- Interface web para **rodar as sessões só depois de clicar “▶️ Play”**.
- Ajuste na barra lateral: **Quantia do Jogador**, **Quantia da Banca**, **Valor da partida (v)**, **Quantia inicial da mesa (start)**, `n`, `seed`, `workers`, `chunk`.
- Execução **em paralelo** com barra de progresso.
- Mostra **KPIs** (probabilidades, percentis), gráficos (**barras, hist, ECDF, dispersão, QQ-plot**) e permite **baixar CSV/JSON**.

**Instalação**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install streamlit plotly numpy pandas
```

**Como rodar**

```bash
streamlit run app.py
# Abra o link (ex.: http://localhost:8501)
```

**Uso**  
1. Defina na barra lateral: **Quantia do Jogador**, **Quantia da Banca**, **Valor da partida (R$ 50)**, **Quantia inicial da mesa (R$ 1)**, além de `n`, `seed`, `workers`, `chunk`.
2. Clique **▶️ Play**. Nada roda automaticamente ao abrir.
3. Acompanhe o **progresso** e veja KPIs + gráficos.
4. Baixe os dados em **CSV** e o **resumo JSON**.

**Dicas de performance**  
- Use **Processos = 0 (auto)** e aumente **chunk** (ex.: 5.000–20.000) para `n` grandes e **caixa do jogador** alto.
- Evite `n` gigantes com **workers=1**.

---

## Conceitos e leituras dos gráficos

- **Barras: Resultado das sessões** — aproxima a probabilidade do **jogador quebrar a banca** vs **banca quebrar o jogador**.
- **Histograma de partidas / ECDF** — mostram a **duração típica** de uma sessão e permitem ler percentis (**P50**, **P95**, **P99**).
- **Histograma dos saldos (p99)** — foca no **comportamento típico**, truncando caudas raras para não distorcer a leitura.
- **Dispersão (partidas vs saldo final do jogador)** — sessões longas costumam associar-se a **ganhos raros**.
- **QQ-plot Normal** — evidencia **não-normalidade** (caudas pesadas); a linha de 45° seria a normal teórica.

**Percentis**  
- **P50 (mediana)**: metade das observações **≤** esse valor.  
- **P90/P95/P99**: 90/95/99% **≤** esse valor; úteis para riscos e caudas pesadas.

---

## Exemplos completos

### A) Relatório Gauss + ECDF (1e5 jogos), sem janelas
```bash
python game.py --n 100000 --v 50 --start 1 --seed 123   --pdf out/gauss_report.pdf --savefig out/ --no-show
```

### B) 100k sessões até quebrar (player=R$1.000.000), paralelo, com progresso
```bash
python game_until_win_brake.py --n 100000 --player 1000000 --bank 1000000000 --v 50 --start 1   --workers 0 --chunk 10000 --progress --no-show --pdf out/relatorio_parallel.pdf --savefig out/
```

### C) Dashboard web
```bash
pip install streamlit plotly numpy pandas
streamlit run app.py
```

---

## Estrutura do repositório

```
.
├── app.py                      # Dashboard Streamlit
├── game.py                     # Jogos independentes + Gauss/ECDF/QQ/cumulativo
├── game_until_win_brake.py     # Sessões até quebrar (paralelo, progresso, PDF/plots)
└── out/                        # (opcional) PDFs/PNGs gerados
```

---

## Solução de problemas (FAQ)

- **“Nada acontece quando abro o dashboard”**: é intencional — defina os parâmetros na barra lateral e clique **▶️ Play**.
- **Execução muito lenta**: aumente `--workers` (ou `0` para auto) e **`--chunk`**; reduza `n` para testes; rode sem janelas (**`--no-show`**).
- **Memória/tempo em `player` muito alto**: prefira `chunk` maiores (ex.: 10k–20k) e mantenha `workers=auto`.
- **PDF/PNG não aparecem**: verifique o caminho passado em `--pdf`/`--savefig` e se o diretório existe. Use caminhos relativos como `out/`.
- **Windows + paralelismo**: execute os scripts via `python ...` (não interativo REPL) para que o `ProcessPoolExecutor` funcione corretamente.

---

## Licença
Este projeto é disponibilizado para fins educacionais/aprendizado. Ajuste conforme sua necessidade.
