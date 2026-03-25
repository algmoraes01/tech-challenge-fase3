# Tech Challenge — Fase 3 · Machine Learning Engineering

**André Luiz Gomes de Moraes**  
Pós-graduação FIAP — Machine Learning Engineering  

---

## Apresentação em vídeo

O vídeo do Tech Challenge com a explicação do trabalho, os resultados e conclusões pode ser acessado no seguinte link:

**Link do vídeo:**

`[Assistir à apresentação](https://youtu.be/owZyia1Ef3g)`

---

## Sobre este repositório

Trabalho da **Fase 3** sobre **atrasos em voos comerciais nos EUA**. O fluxo começa na análise exploratória, passa por modelos supervisionados (classificação e regressão), técnicas não supervisionadas (PCA e agrupamento de aeroportos), semi-supervisionado, detecção de anomalias e visualização geográfica, e termina com artefatos prontos para inspeção ou para o painel Streamlit.

A ideia central da modelagem supervisionada é usar só o que estaria disponível **antes** da partida programada (calendário, horário de saída previsto, companhia, rota, distância, estado dos aeroportos quando o cadastro existe, indicadores de feriado nos EUA e engenharia de features derivada disso). Campos observados só depois da operação do voo não entram como preditores, para evitar vazamento conceitual num cenário de uso real.

---

## O que o enunciado pedia e como foi atendido

**Exploração (EDA):** estatísticas descritivas e taxa de missing exportadas; gráficos de distribuição de atraso, taxa de atraso > 15 min por dia da semana, hora programada, companhia, aeroporto de origem, mês e **estado de origem** (quando `airports.csv` está presente). Ausentes documentados; imputação mediana onde o pipeline usa `SCHEDULED_TIME` e `DISTANCE`; voos cancelados ou desviados ficam fora do conjunto principal de modelagem de atraso na partida.

**Supervisionado:** classificação binária (atraso na partida **> 15 min**) com **quatro** algoritmos (regressão logística, floresta aleatória, gradient boosting, XGBoost), validação cruzada estratificada na fase de treino para ROC-AUC, e métricas no holdout (acurácia, precisão, revocação, F1, ROC-AUC, PR-AUC) mais matrizes de confusão. **Regressão** sobre minutos de atraso com recorte numérico para suavizar cauda pesada, com três modelos (RF, GB, XGB) e MAE, RMSE e R².

**Não supervisionado:** PCA em duas componentes sobre variáveis numéricas do voo; **K-Means** sobre agregados por aeroporto de origem (volume, atraso médio, mediana, proporção > 15 min), com gráficos e CSVs.

**Além do obrigatório:** variáveis derivadas (hora, fim de semana, picos manhã/noite, estação, seno/cosseno temporal), **feriados federais americanos** e **véspera de feriado** (biblioteca `holidays`), **estado de origem e destino** via merge com `airports.csv`, **mapa Folium** por aeroporto de origem, **segundo mapa** com **linhas** entre os principais pares origem–destino, **Isolation Forest** em agregados por aeroporto, **SelfTrainingClassifier** com base logística (semi-supervisionado) e **Streamlit** consolidando métricas, figuras e mapas.

---

## Dados necessários

| Arquivo | Função |
|---------|--------|
| `data/flights.csv` | Tabela principal de voos (material FIAP / MLET). |
| `data/airports.csv` | Cadastro com `IATA_CODE`, `STATE`, `LATITUDE`, `LONGITUDE`. Usado para estado nos modelos, EDA por estado e mapa de rotas. Se ausente, estados viram `UNK` e o mapa de rotas não é gerado. |
| `data/airports_geo.csv` | Já incluído no projeto: IATA com lat/lon para o mapa de **bolhas** por aeroporto de origem. |
| `data/airlines.csv` | Opcional; o pipeline atual não depende dele. |

Quem clonar o repositório precisa baixar **`flights.csv`** (e recomenda-se **`airports.csv`**) no ambiente da disciplina e colocar em `data/`.

Variáveis de ambiente úteis no PowerShell:

```powershell
$env:FLIGHTS_CSV = "D:\dados\flights.csv"
$env:FLIGHTS_NROWS = "500000"
python run_all.py
```

`FLIGHTS_NROWS` limita a leitura às primeiras N linhas do CSV (testes mais rápidos). Para resultado final e vídeo, rode sem esse limite se a máquina aguentar.

---

## Ambiente e reprodução

Python **3.10** ou **3.11**. Dependências em `requirements.txt` (pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, folium, streamlit, **holidays**, joblib, scipy, entre outras).

```powershell
cd <pasta-do-clone>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_all.py
```

Gera ou sobrescreve **`outputs/`** com todas as tabelas, figuras, JSON, HTML dos mapas e modelos `.joblib`. Em seguida, para o painel:

```powershell
streamlit run streamlit_app.py
```

Na **primeira execução** do Streamlit no seu usuário, pode aparecer um convite de e-mail: deixe em branco e pressione **Enter** para continuar. Para reduzir prompts em automação ou gravação, use antes: `$env:CI = "true"` (opcional).

O `run_all.py` fixa o backend **Agg** do Matplotlib para evitar conflito de thread com interface gráfica no Windows ao salvar PNGs. O `run_all.py` não imprime progresso no terminal; para acompanhar a execução, observe arquivos aparecendo em **`outputs/`** ou use `FLIGHTS_NROWS` menor em testes.

---

## Organização do código

- **`run_all.py`** — Orquestra merge de aeroportos, feriados, EDA, PCA, clusters, anomalias, classificação, regressão, semi-supervisionado e os dois mapas; grava `run_metadata.json`.
- **`src/features.py`** — Leitura do `flights`, filtro de voos utilizáveis, merge com `airports.csv`, feriados US, features derivadas, agregações por aeroporto, coordenadas para mapas.
- **`src/eda.py`** — Describe, missing e figuras (incluindo taxa de atraso por estado de origem).
- **`src/supervised.py`** — Pré-processamento (numéricas escaladas, categóricas com one-hot), treino e avaliação supervisionados.
- **`src/semisupervised.py`** — SelfTraining sobre parte dos rótulos do treino tratada como não rotulada (`-1`).
- **`src/unsupervised.py`** — PCA em voos e K-Means em perfis de aeroporto.
- **`src/anomalies.py`** — Isolation Forest nos agregados.
- **`src/maps_folium.py`** — Mapa por aeroporto (`airports_geo.csv`) e mapa de polilinhas O–D (`airports.csv`).
- **`streamlit_app.py`** — Lê `outputs/` e exibe métricas, imagens e HTML dos mapas.

---

## Saídas principais (`outputs/`)

| Pasta / arquivo | Conteúdo |
|-----------------|----------|
| `eda_tables/` | `describe_all.csv`, `missing_rate.csv`, resumo JSON de missing |
| `eda_figures/` | Histogramas, séries por dia/hora/mês, companhia, aeroporto, **estado de origem** |
| `supervised_classification/` | `classification_metrics.json`, matrizes de confusão, relatórios texto, pipelines `.joblib` |
| `supervised_regression/` | `regression_metrics.json`, dispersão observado × predito |
| `semi_supervised/` | Métricas, relatório e matriz de confusão do experimento semi-supervisionado |
| `unsupervised/` | PCA, clusters de aeroportos, CSVs e JSON de resumo |
| `anomalies/` | Figura e tabela de anomalias por aeroporto |
| `maps/delay_mean_by_airport.html` | Mapa interativo por origem |
| `maps/routes_top_od.html` | Principais rotas como linhas entre aeroportos |
| `run_metadata.json` | Metadados da última execução (caminho dos dados, contagens, resumos) |

---

## Entrega acadêmica

Conforme enunciado do Tech Challenge: **repositório** com código reproduzível e **vídeo** dentro do prazo e duração pedidos.

---

## Uso dos dados FIAP

O `flights.csv` e demais bases fornecidas pela instituição devem ser usados **apenas** no contexto acadêmico da disciplina, respeitando as normas da FIAP.
