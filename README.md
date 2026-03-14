# 🛡️ Hybrid Zero-Day Anomaly Detection Pipeline

Este repositório contém um pipeline de Machine Learning híbrido projetado para detecção de intrusões e reconhecimento de ataques de dia zero (*Zero-Day*) em fluxos de dados contínuos (Data Streams). O foco principal do projeto é atuar em ambientes IoT/Edge (utilizando o ML-EdgeIIoT dataset) combinando detecção de anomalias com algoritmos de *Open Set Recognition* (OSR).

O projeto utiliza a biblioteca **River** para aprendizado online (incremental) e **Scikit-Learn** para avaliação e métricas.

## ✨ Principais Funcionalidades

* **Online Machine Learning**: Modelos treinados de forma incremental, adaptando-se a novos dados sem necessidade de retreinar todo o conjunto.
* **Dual Open Set Recognition (OSR)**: 
    * *CentroidOSR*: Avalia amostras baseadas na distância aos centroides das classes conhecidas.
    * *EntropyOSR*: Detecta ataques *Zero-Day* analisando a incerteza (entropia) das probabilidades de predição do classificador.
* **Prequential Selector**: Um mecanismo dinâmico que avalia continuamente e escolhe o melhor detector OSR (Centroide ou Entropia) por classe, otimizando o F1-Score em tempo real.
* **Conservative Auto-Labeler**: Um sistema de auto-rotulação (pseudo-labeling) que adiciona novas amostras de ataques desconhecidos ao treinamento quando há alta confiança e consistência temporal.
* **Processamento Paralelo**: Avaliação concorrente de diferentes classes como *Zero-Day* usando `concurrent.futures`.

## 🧠 Arquitetura do Pipeline

1.  **Detecção de Anomalias (Filtro Inicial)**: Um modelo `HalfSpaceTrees` identifica se o tráfego é normal ou anômalo.
2.  **Classificação Multi-classe**: Se uma anomalia é detectada, um `AdaBoostClassifier` (com árvores de Hoeffding) tenta classificar o tipo de ataque.
3.  **Avaliação OSR**: Os detectores *CentroidOSR* e *EntropyOSR* avaliam se a amostra pertence a uma classe conhecida ou se é um ataque inédito (*Zero-Day*).
4.  **Auto-rotulação**: Amostras classificadas como *Zero-Day* com altíssima confiança retroalimentam os modelos para adaptação contínua.

## 📊 Dataset

O script está configurado para utilizar o **ML-EdgeIIoT-dataset** . 
O caminho padrão no código aponta para: `../../ML-EdgeIIoT-dataset-CLEANED.csv`.

> **Nota:** Certifique-se de ajustar o caminho do arquivo `csv` na função principal (`if __name__ == "__main__":`) de acordo com a estrutura de diretórios do seu ambiente local.

## 📦 Requisitos e Dependências

Certifique-se de ter o Python 3.8+ instalado. As principais bibliotecas utilizadas são:

* `numpy`
* `pandas`
* `scikit-learn`
* `river`

Você pode instalá-las rodando:
```bash
pip install numpy pandas scikit-learn river
