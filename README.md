# Hybrid Online Zero-Day Detection Pipeline

Pipeline híbrido de Machine Learning online para detecção de intrusões e reconhecimento de ataques *Zero-Day* em fluxos de dados contínuos (Data Streams). O sistema combina detecção de anomalias não-supervisionada, classificação multi-classe incremental e *Open Set Recognition* (OSR) dual — tudo treinado e avaliado de forma **online/incremental**, sem retreinamento em lote.

> **Contexto de aplicação:** ambientes IoT/Edge com tráfego de rede rotulado (ML-EdgeIIoT, ERENO-2.0).

---

## Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura do Pipeline](#arquitetura-do-pipeline)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Módulos Principais](#módulos-principais)
- [Configuração](#configuração)
- [Datasets](#datasets)
- [Como Executar](#como-executar)
- [Saída e Métricas](#saída-e-métricas)
- [Dependências](#dependências)

---

## Visão Geral

A proposta central do projeto é detectar **ataques inéditos (Zero-Day)** — classes de ataque nunca vistas durante o treinamento — usando exclusivamente modelos treinados online de forma incremental.

O pipeline trata cada classe de ataque como potencial Zero-Day em experimentos independentes e paralelos: para cada rodada, uma classe é omitida do treino e injetada durante a avaliação. O desempenho é medido pelo **F1-Score da classe Zero-Day** e comparado contra um baseline AdaBoost com limiar de confiança fixo.

---

## Arquitetura do Pipeline

O fluxo de decisão por amostra é:

```
Amostra de rede
       │
       ▼
┌──────────────────┐
│  HalfSpaceTrees  │  ← Detector de anomalia online (não-supervisionado)
│  (HST)           │
└────────┬─────────┘
         │ score > threshold?
    NÃO  │            SIM
 (Normal)│             │
         │             ▼
         │   ┌──────────────────┐
         │   │ AdaBoost Online  │  ← Classificador multi-classe incremental
         │   │ (HoeffdingTrees) │     (River)
         │   └────────┬─────────┘
         │            │ predicted_cls
         │            ▼
         │   ┌──────────────────────────────┐
         │   │    PrequentialSelector       │  ← Seleciona melhor OSR por classe
         │   │  (OSR-A vs OSR-B via F1)     │     usando janela deslizante
         │   └──────┬───────────────────────┘
         │          │
         │    ┌─────┴──────┐
         │    ▼            ▼
         │  OSR-A        OSR-B
         │ Centroid      Entropy
         │   OSR           OSR
         │    └─────┬──────┘
         │          │ Zero-Day ou Classe Conhecida?
         │          ▼
         │   ┌──────────────────┐
         │   │ ConservativeAuto │  ← Pseudo-labeling com alta confiança
         │   │    Labeler       │     retroalimenta clf + OSR + HST
         │   └──────────────────┘
         │
         ▼
     Predição Final
```

### Etapas detalhadas

| # | Etapa | Módulo |
|---|-------|--------|
| 1 | **Split** — separa treino (90%), teste e amostras Zero-Day | `experiment.py` |
| 2 | **Encoding** — OrdinalEncoder em colunas categóricas | `features.py` |
| 3 | **Warm-up HST** — treina HST apenas com amostras normais | `anomaly_detector.py` |
| 4 | **Calibração de limiar HST** — Índice de Youden sobre candidatos | `anomaly_detector.py` |
| 5 | **Treino online clf + OSR** — AdaBoost e centroides aprendem incrementalmente | `experiment.py` |
| 6 | **Calibração OSR-A/B** — Youden (Centroid) e percentil 95 (Entropy) | `centroid.py`, `entropy.py` |
| 7 | **Avaliação híbrida** — inferência amostra a amostra com auto-labeling | `experiment.py` |
| 8 | **Baseline** — AdaBoost com limiar de confiança 0.70 | `experiment.py` |
| 9 | **Métricas** — F1 Zero-Day, Delta F1, latência média, taxa de auto-label | `experiment.py` |

---

## Estrutura do Projeto

```
.
├── main.py                    # Entry point (argparse + ProcessPoolExecutor)
├── src/
│   ├── config.py              # ExperimentConfig — todos os hiperparâmetros
│   ├── models/
│   │   ├── anomaly_detector.py    # HalfSpaceTrees: build + calibração de limiar
│   │   └── classifier.py          # AdaBoost com HoeffdingTreeClassifier
│   ├── osr/
│   │   ├── centroid.py            # CentroidOSR — distância normalizada ao centroide
│   │   └── entropy.py             # EntropyOSR — entropia das probabilidades
│   ├── pipeline/
│   │   ├── experiment.py          # run_experiment() — loop principal por classe ZD
│   │   ├── selector.py            # PrequentialSelector — escolha OSR por F1 em janela
│   │   └── auto_labeler.py        # ConservativeAutoLabeler — pseudo-labeling
│   ├── preprocessing/
│   │   ├── features.py            # load_dataset, preprocess_features, encode_labels…
│   │   └── scalers.py             # Factories get_sklearn_scaler / get_river_scaler
│   └── utils/
│       └── logger.py              # log() com prefixo de classe
├── ablation/                  # Experimentos de ablação por componente
│   ├── noautolaber/
│   ├── nocentroid/
│   ├── noentropy/
│   ├── nohst/
│   ├── resultado-ereno.txt
│   └── resultado-ML.txt
├── db/                        # Datasets CSV
│   ├── ERENO-2.0-100K.csv
│   ├── ML_EdgeIIoT_BALANCEADO_FINAL.csv
│   └── ML_EdgeIIoT_SMOTE.csv
└── scripts/                   # Utilitários de pré-processamento
    ├── balance_db.py
    └── dataset.py
```

---

## Módulos Principais

### `HalfSpaceTrees` — Detector de Anomalia

Modelo não-supervisionado online da biblioteca **River**. Treinado exclusivamente com amostras normais (warm-up). O limiar de decisão é calibrado pelo **Índice de Youden** sobre `hst_threshold_candidates` pontos candidatos no conjunto de treino, maximizando `TPR + TNR - 1`.

### `CentroidOSR` — OSR por Distância ao Centroide

Mantém média e variância online (algoritmo de Welford) por classe. A distância de uma amostra ao centroide é normalizada pelo desvio padrão de cada feature (distância MAD-like). O limiar por classe é calibrado pelo Índice de Youden sobre distâncias **intra-classe** vs **inter-classe** usando `n_candidates` candidatos.

```
is_zero_day → True se a distância da amostra a TODOS os centroides supera seus limiares
```

### `EntropyOSR` — OSR por Entropia das Probabilidades

Calcula a entropia de Shannon das probabilidades de saída do classificador. Mantém uma janela deslizante de entropias e usa o **percentil 95** como limiar adaptativo. Alta entropia = incerteza elevada = possível Zero-Day.

```python
H(p) = -Σ p_i · log2(p_i)
is_zero_day → True se H(probas) > threshold
```

### `PrequentialSelector` — Seleção Dinâmica de OSR

Para cada classe conhecida, mantém duas janelas deslizantes de predições (OSR-A e OSR-B). A cada seleção, calcula o F1-Score da classe Zero-Day em cada janela e escolhe o detector com melhor desempenho recente. Requer `min_samples` exemplos antes de comparar (fallback para OSR-B).

### `ConservativeAutoLabeler` — Auto-rotulação Conservadora

Aceita uma amostra como pseudo-rótulo somente se **todas** as condições são satisfeitas:
1. OSR-A e OSR-B concordam na predição (classe ≠ Zero-Day)
2. Confiança máxima do classificador ≥ `autolabel_confidence` (default 0.90)
3. A classe acordada aparece ≥ `min_history_count` vezes nas últimas `history_window` amostras aceitas

Amostras aceitas retroalimentam `clf`, `osr_a`, `osr_b` e `hst`.

---

## Configuração

Todos os hiperparâmetros são centralizados em `src/config.py` via `ExperimentConfig`:

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `scaler` | `"standard"` | Normalizador: `standard`, `minmax`, `robust` |
| `train_split` | `0.90` | Proporção de dados conhecidos usada no treino |
| `n_workers` | `-1` | Workers paralelos (-1 = `cpu_count - 2`) |
| `hst_n_trees` | `10` | Número de árvores no HalfSpaceTrees |
| `hst_height` | `11` | Profundidade das árvores HST |
| `hst_window_size` | `100` | Tamanho da janela de aprendizado do HST |
| `hst_threshold_candidates` | `300` | Candidatos para calibração do limiar HST |
| `adaboost_n_models` | `15` | Estimadores no AdaBoostClassifier |
| `hoeffding_max_depth` | `15` | Profundidade máxima das HoeffdingTrees |
| `centroid_osr_window` | `2000` | Janela de distâncias para calibração CentroidOSR |
| `centroid_osr_candidates` | `400` | Candidatos para limiar Youden no CentroidOSR |
| `entropy_osr_window` | `2000` | Janela de entropias para calibração EntropyOSR |
| `selector_window_size` | `200` | Janela prequential de cada OSR no Selector |
| `selector_min_samples` | `30` | Mínimo de amostras antes de comparar OSRs |
| `autolabel_confidence` | `0.90` | Limiar mínimo de confiança para auto-label |
| `autolabel_history_window` | `50` | Janela histórica de aceites do AutoLabeler |
| `autolabel_min_history_count` | `3` | Mínimo de ocorrências para aceitar auto-label |
| `expert_queue_size` | `100` | Tamanho da fila de atualização do Selector |

> **Nota sobre `scaler=robust`:** O RobustScaler é usado no pipeline sklearn. No River (HST), não há RobustScaler nativo — o sistema faz fallback para StandardScaler nesses casos.

---

## Datasets

Os datasets ficam em `db/` e são detectados automaticamente pelo `main.py`:

| Arquivo | Descrição |
|---------|-----------|
| `ML_EdgeIIoT_SMOTE.csv` | ML-EdgeIIoT com balanceamento SMOTE (prioridade de auto-detecção) |
| `ML_EdgeIIoT_BALANCEADO_FINAL.csv` | ML-EdgeIIoT balanceado manualmente |
| `ERENO-2.0-100K.csv` | Dataset ERENO 2.0 com 100K amostras |

A coluna-alvo é detectada automaticamente por heurísticas de nome (`label`, `attack`, `class`, etc.).

---

## Como Executar

### Instalação

```bash
pip install numpy pandas scikit-learn river
```

### Execução básica

```bash
# Scaler padrão (StandardScaler), dataset auto-detectado
python main.py

# Com MinMaxScaler
python main.py --scaler minmax

# Com RobustScaler e dataset específico
python main.py --scaler robust --dataset db/ERENO-2.0-100K.csv

# Controlar paralelismo (ex: 4 workers)
python main.py --workers 4
```

### Argumentos

| Argumento | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `--scaler` | `standard` \| `minmax` \| `robust` | `standard` | Normalizador |
| `--dataset` | `str` | auto | Caminho para o CSV |
| `--workers` | `int` | `-1` | Workers paralelos |

---

## Saída e Métricas

Ao final da execução, o pipeline exibe uma tabela com uma linha por classe Zero-Day testada:

| Coluna | Descrição |
|--------|-----------|
| `Zero_Day_Class` | Nome da classe usada como Zero-Day |
| `Hybrid_F1` | F1-Score do pipeline híbrido para a classe Zero-Day |
| `Baseline_F1` | F1-Score do baseline (AdaBoost + limiar 0.70) |
| `Delta_F1` | `Hybrid_F1 - Baseline_F1` (ganho/perda do híbrido) |
| `Hybrid_Latency_us` | Latência média por amostra no pipeline híbrido (µs) |
| `Baseline_Latency_us` | Latência média por amostra no baseline (µs) |
| `AutoLabel_AcceptRate` | Taxa de amostras aceitas pelo AutoLabeler [0, 1] |

Exemplo de saída:

```
============================================================================
RESUMO FINAL
============================================================================
 Zero_Day_Class  Hybrid_F1  Baseline_F1  Delta_F1  Hybrid_Latency_us  ...
        DDoS         0.8341       0.6120    +0.2221              12.4
        MITM         0.7892       0.5988    +0.1904              13.1
       Ransomware    0.7105       0.6301    +0.0804              11.8
============================================================================
Média F1 Híbrido: 0.7779 | Delta: +0.1643
============================================================================
```

---

## Dependências

| Biblioteca | Uso |
|------------|-----|
| `river` | HalfSpaceTrees, AdaBoostClassifier, HoeffdingTreeClassifier, StandardScaler online |
| `scikit-learn` | OrdinalEncoder, StandardScaler/MinMaxScaler/RobustScaler, F1-Score, classification_report |
| `numpy` | Operações vetoriais, calibração de limiares |
| `pandas` | Carregamento e manipulação de datasets |

```bash
pip install numpy pandas scikit-learn river
```

> Testado com Python 3.10+.
