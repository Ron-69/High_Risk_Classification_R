# High_Risk_Classification_R
Este projeto demonstra o uso de técnicas Classificação Binária (Regressão Logística/SVM) , com o objetivo de Classificar um cliente como sendo de "Alto Custo" (Top 25% dos valores de charges) ou "Custo Normal" (Bottom 75%).

# 🚨 02_High_Risk_Classification: Identificação de Clientes de Alto Risco (Regressão Logística)

## 🎯 Objetivo do Projeto

Este projeto implementa uma solução de **Classificação Binária** usando **Regressão Logística** em R para identificar clientes de seguro saúde que representam **Alto Risco de Custo**.

O objetivo é prever se o custo anual de um cliente estará no **Top 25%** do dataset. A avaliação do modelo é focada em **Recall** e **AUC**, métricas essenciais para lidar com o desbalanceamento de classes e garantir a captura máxima de clientes de alto risco real.

## 🔑 Metodologia e Decisões Estratégicas

| Aspecto | Detalhe da Implementação |
| :--- | :--- |
| **Variável Alvo ($Y$)** | `high_cost`: Criada usando o 75º Percentil (Q3: R$ 16.639,91) do custo (`charges`). |
| **Desbalanceamento** | Dataset desbalanceado: **1003 Normal** vs. **335 Alto Risco** (25%). |
| **Particionamento** | **Estratificado** (`caret::createDataPartition`) para manter a proporção da classe `Alto_Risco` nos conjuntos de Treino e Teste. |
| **Técnica** | **Regressão Logística (GLM, family=binomial)**: Utilizada para modelar a probabilidade do evento de Alto Risco. |
| **Pré-processamento** | Variáveis numéricas padronizadas (`center` e `scale`) para otimizar o treinamento do modelo. |

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** R
* **Pacotes Principais:** `caret`, `dplyr`, `pROC`, `readr`

---

## 📊 Fluxo de Trabalho e Resultados Consolidados

### Célula 1 & 2: Setup e Engenharia da Variável Alvo

A variável alvo `high_cost` foi criada e confirmou o desbalanceamento.

#### Código R (Essencial)

```R
# --- CÉLULA 1 & 2: SETUP E FEATURE ENGINEERING ---
# Carrega dados, calcula o 75º percentil (Q3) e cria a variável high_cost
q3_charges <- quantile(insurance_data$charges, 0.75)
insurance_data <- insurance_data %>%
  mutate(high_cost = ifelse(charges > q3_charges, "Alto_Risco", "Normal")) 
# ... código continua ...
```

## Saída Obtida

O Limiar de Alto Custo (Q3) é: 16639.91 
Balanceamento da Classe 'high_cost':
    Normal Alto_Risco 
      1003        335

### Célula 3: Pré-processamentoDivisão 70/30 (estratificada) e aplicação da padronização (center e scale).

Código R (Essencial)

R# --- CÉLULA 3: PRÉ-PROCESSAMENTO E PARTICIONAMENTO ---

# Código para createDataPartition e preProcess

## Saída Obtida

Dimensões do Set de Treino Processado: 938 7
Célula 4: Treinamento da Regressão Logística
O modelo foi treinado no conjunto processado.

Código R (Essencial)

R# --- CÉLULA 4: TREINAMENTO DA REGRESSÃO LOGÍSTICA ---

# logreg_model <- glm(high_cost ~ ..., family = "binomial", ...)

## Saída e Análise de Coeficientes

O summary demonstra que o Hábito de Fumar é o preditor mais forte.Sumário do Modelo de Regressão Logística:

Coefficients:
                 Estimate Std. Error z value Pr(>|z|)    
(Intercept)      -2.56700   0.30082  -8.533 < 2e-16 ***
age               0.47801   0.14208   3.364 0.000767 ***
sexmale          -0.11175   0.26304  -0.425 0.670941    
bmi               0.31932   0.13674   2.335 0.019531 * children          0.13752   0.12432   1.106 0.268658    
smokeryes         5.78865   0.39074  14.815 < 2e-16 ***
regionnorthwest  -0.01247   0.37653  -0.033 0.973588    
regionsoutheast   0.10613   0.36625   0.290 0.771996    
regionsouthwest  -0.45854   0.38851  -1.180 0.237903    
---

## Conclusão da Significância

As variáveis **`smokeryes`**, **`age`** e **`bmi`** são estatisticamente significativas ($P$-Value baixo) na previsão do Alto Risco. O coeficiente de $\mathbf{5.79}$ para `smokeryes` indica um aumento exponencial na probabilidade de ser classificado como Alto Risco.

---

### Célula 5: Avaliação Focada em Recall e AUC

A avaliação no Set de Teste (30%) confirma a robustez do modelo para identificar a classe positiva (`Alto_Risco`).

#### Matriz de Confusão e Métricas ($Threshold \ P=0.5$)

O resultado da Matriz de Confusão no Set de Teste é o seguinte:
## Confusion Matrix and Statistics

             Reference
Prediction    Normal Alto_Risco
  Normal         291         25
  Alto_Risco       9         75

Recall (Sensibilidade): 0.75 
Área Sob a Curva ROC (AUC): 0.8669 
#### Análise das Métricas Chave

| Métrica | Valor | Objetivo Estratégico |
| :--- | :--- | :--- |
| **Recall (Sensibilidade)** | $\mathbf{0.7500}$ | **Sucesso:** O modelo capturou 75% dos clientes que realmente são de alto risco (Minimização de Falsos Negativos). |
| **AUC** | $\mathbf{0.8669}$ | **Forte Desempenho:** Indica que o modelo tem um excelente poder de discriminação entre as classes. |
| **Especificidade** | $0.9700$ | $97\%$ dos clientes normais foram corretamente identificados. |

---

### Curva ROC e AUC 📈

A Área Sob a Curva (AUC) é de **0.8669**, um valor forte que confirma o excelente poder discriminatório do modelo, independentemente do *threshold* de classificação escolhido.

O gráfico a seguir ilustra o *trade-off* entre a Taxa de Verdadeiros Positivos (Recall) e a Taxa de Falsos Positivos em todos os *thresholds* possíveis:

![Curva ROC para Identificação de Alto Risco](./assets/Curve_ROC_High_Risk_Identifier.png)
**Nota sobre a Curva ROC:** A **Curva ROC** valida o forte desempenho do modelo, pois a área sob a curva (AUC) está bem próxima de 1.0, confirmando a robustez da classificação.

## 📈 Conclusão e Próximos Passos

O projeto alcançou o objetivo de construir um modelo de classificação de Alto Risco com alta performance:
O modelo identificou $\mathbf{75\%}$ dos clientes de Alto Risco (Recall), minimizando o risco financeiro de Falsos Negativos.
O AUC de $\mathbf{0.8669}$ confirma que o modelo é altamente eficaz e generalizável.
A Análise de Coeficientes identificou Fumar como o fator de risco mais crítico.
Para otimizar o modelo em um ambiente de produção (minimizar os 25 Falsos Negativos restantes), os próximos passos envolveriam: ajustar o threshold de classificação para um valor menor que $0.5$ (priorizando o Recall sobre a Especificidade) ou aplicar técnicas avançadas de Balanceamento de Classes (como SMOTE) no conjunto de treino.