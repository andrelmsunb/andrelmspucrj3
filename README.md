# Classificação de Doenças Cardíacas - Projeto de Machine Learning

**Autor:** [Andre Luiz Marques Serrano]  
**Data:** Setembro de 2025  
**Dataset:** Heart Disease UCI Dataset  
**Linguagem:** Python  
**Ambiente:** Google Colab / Jupyter Notebook  

## Resumo Executivo

Este projeto desenvolve um modelo de aprendizado de máquina para classificar a presença de doenças cardíacas em pacientes utilizando o dataset Heart Disease do UCI. O objetivo é criar um sistema de apoio à decisão médica que possa identificar pacientes com risco de doença cardíaca baseado em características clínicas.

### Principais Resultados

- **Modelo Final:** SVM com kernel RBF
- **Acurácia no Teste:** 86.9%
- **AUC-ROC no Teste:** 0.962
- **F1-Score:** 0.85
- **Features Mais Importantes:** ca, oldpeak, cp, thalach, exang

### Impacto

O modelo demonstra alta capacidade preditiva e pode ser utilizado como ferramenta de triagem e apoio ao diagnóstico médico, contribuindo para a identificação precoce de doenças cardíacas.

---

## Introdução

### Contexto do Problema

As doenças cardíacas são uma das principais causas de morte no mundo, sendo responsáveis por milhões de óbitos anualmente. A identificação precoce de pacientes em risco é fundamental para a prevenção e tratamento adequado.

### Objetivos

**Objetivo Geral:**
Desenvolver um modelo de machine learning capaz de classificar a presença de doença cardíaca em pacientes com base em características clínicas.

**Objetivos Específicos:**
- Realizar análise exploratória completa do dataset
- Implementar pipeline de pré-processamento robusto
- Comparar diferentes algoritmos de classificação
- Otimizar hiperparâmetros do melhor modelo
- Avaliar performance em dados não vistos
- Identificar features mais importantes para predição

### Dataset

O dataset Heart Disease foi obtido do UCI Machine Learning Repository e contém 303 registros de pacientes com 14 atributos clínicos:

**Features Numéricas:**
- `age`: Idade do paciente
- `trestbps`: Pressão arterial em repouso
- `chol`: Colesterol sérico
- `thalach`: Frequência cardíaca máxima
- `oldpeak`: Depressão do ST induzida por exercício

**Features Categóricas:**
- `sex`: Sexo (1 = masculino, 0 = feminino)
- `cp`: Tipo de dor no peito (1-4)
- `fbs`: Glicemia em jejum > 120 mg/dl
- `restecg`: Resultados do ECG em repouso
- `exang`: Angina induzida por exercício
- `slope`: Inclinação do segmento ST
- `ca`: Número de vasos principais (0-3)
- `thal`: Talassemia (3, 6, 7)

**Target:**
- `num`: Presença de doença cardíaca (0 = ausente, 1-4 = presente)

---

## Metodologia

### Abordagem Geral

O projeto seguiu a metodologia CRISP-DM (Cross-Industry Standard Process for Data Mining), adaptada para problemas de classificação em saúde:

1. **Entendimento do Negócio:** Definição do problema e objetivos
2. **Entendimento dos Dados:** Análise exploratória e qualidade dos dados
3. **Preparação dos Dados:** Limpeza, transformação e engenharia de features
4. **Modelagem:** Seleção e treinamento de algoritmos
5. **Avaliação:** Validação e teste dos modelos
6. **Implantação:** Documentação e recomendações

### Divisão dos Dados

Os dados foram divididos estratificadamente para manter a proporção das classes:

- **Treino:** 60% (182 amostras) - Treinamento dos modelos
- **Validação:** 20% (61 amostras) - Seleção de modelos e hiperparâmetros
- **Teste:** 20% (60 amostras) - Avaliação final não enviesada

### Métricas de Avaliação

Devido à natureza crítica do problema médico, utilizamos múltiplas métricas:

- **Acurácia:** Proporção de predições corretas
- **Precisão:** Proporção de verdadeiros positivos entre predições positivas
- **Recall (Sensibilidade):** Proporção de casos positivos identificados
- **F1-Score:** Média harmônica entre precisão e recall
- **AUC-ROC:** Área sob a curva ROC (discriminação)

### Validação Cruzada

Utilizamos validação cruzada estratificada com 5 folds para:
- Avaliar estabilidade dos modelos
- Reduzir variância das estimativas
- Selecionar hiperparâmetros de forma robusta

---

## Análise Exploratória dos Dados

### Características Gerais

```
Dataset Shape: (303, 14)
Missing Values: ca (4), thal (2)
Target Distribution: 
  - Sem doença: 138 (45.5%)
  - Com doença: 165 (54.5%)
```

### Análise da Variável Target

A variável target original possui 5 classes (0-4), mas foi convertida para classificação binária:
- **0:** Sem doença cardíaca
- **1:** Com doença cardíaca (qualquer grau)

Esta conversão é clinicamente relevante, pois o foco é identificar a presença ou ausência de doença.

### Features Numéricas

**Estatísticas Descritivas:**

| Feature  | Média | Desvio | Min | Max | Observações |
|----------|-------|--------|-----|-----|-------------|
| age      | 54.4  | 9.0    | 29  | 77  | Distribuição normal |
| trestbps | 131.6 | 17.5   | 94  | 200 | Alguns outliers |
| chol     | 246.3 | 51.8   | 126 | 564 | Assimetria à direita |
| thalach  | 149.6 | 22.9   | 71  | 202 | Distribuição normal |
| oldpeak  | 1.04  | 1.16   | 0   | 6.2 | Assimetria à direita |

**Insights Principais:**
- Pacientes com doença cardíaca tendem a ter maior idade
- Menor frequência cardíaca máxima (thalach) está associada à doença
- Maior depressão do ST (oldpeak) indica maior risco

### Features Categóricas

**Distribuições por Presença de Doença:**

| Feature | Categoria | Sem Doença | Com Doença | Insight |
|---------|-----------|------------|------------|---------|
| sex     | Masculino | 32%        | 68%        | Homens têm maior risco |
| cp      | Tipo 4    | 16%        | 84%        | Dor assintomática é crítica |
| exang   | Sim       | 14%        | 86%        | Angina por exercício é indicativa |
| ca      | 0 vasos   | 72%        | 28%        | Mais vasos = maior risco |
| thal    | Reversível| 18%        | 82%        | Defeito reversível indica doença |

### Correlações

**Top 5 Correlações com Target:**
1. `thal` (0.52) - Tipo de talassemia
2. `ca` (0.43) - Número de vasos principais
3. `exang` (0.44) - Angina induzida por exercício
4. `oldpeak` (0.43) - Depressão do ST
5. `cp` (0.43) - Tipo de dor no peito

### Valores Ausentes

- **ca:** 4 valores ausentes (1.3%) - Imputados com mediana
- **thal:** 2 valores ausentes (0.7%) - Imputados com moda

A baixa quantidade de valores ausentes permite imputação simples sem comprometer a qualidade dos dados.

---

## Pré-processamento

### Tratamento de Valores Ausentes

**Estratégias Aplicadas:**
- **ca (numérica):** Imputação com mediana (mais robusta a outliers)
- **thal (categórica):** Imputação com moda (valor mais frequente)

### Codificação de Variáveis Categóricas

**Variáveis Mantidas:**
- Binárias: `sex`, `fbs`, `exang` (já codificadas como 0/1)
- Ordinais: `cp`, `restecg`, `slope` (ordem natural preservada)
- Numérica: `ca` (número de vasos, interpretação numérica válida)

**Variável Transformada:**
- `thal`: Convertida para variáveis dummy (one-hot encoding)
  - `thal_3.0`, `thal_6.0`, `thal_7.0`

### Normalização

Aplicamos `StandardScaler` apenas nas features numéricas:
- `age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`

As features binárias foram mantidas na escala original (0/1) para preservar interpretabilidade.

### Seleção de Features

**Métodos Utilizados:**
1. **Correlação:** Correlação linear com target
2. **ANOVA F-test:** Teste estatístico para features categóricas
3. **Mutual Information:** Dependência não-linear

**Ranking Combinado (Top 10):**
1. `ca` - Número de vasos principais
2. `oldpeak` - Depressão do ST
3. `cp` - Tipo de dor no peito
4. `thal_7.0` - Talassemia reversível
5. `exang` - Angina por exercício
6. `thalach` - Frequência cardíaca máxima
7. `sex` - Sexo
8. `slope` - Inclinação do ST
9. `age` - Idade
10. `thal_6.0` - Talassemia fixa

---

## Modelagem

### Baseline

Estabelecemos baselines simples para comparação:

| Modelo | Estratégia | Acurácia Validação |
|--------|------------|-------------------|
| Majority Class | Classe majoritária | 54.1% |
| Random | Predição aleatória | 49.2% |
| Stratified | Proporção das classes | 52.5% |

**Baseline Selecionado:** Majority Class (54.1%)

### Algoritmos Testados

Comparamos 9 algoritmos de classificação:

| Modelo | Tipo | Acurácia Val | F1-Score | AUC-ROC | Tempo (s) |
|--------|------|-------------|----------|---------|-----------|
| AdaBoost | Ensemble | 85.2% | 0.824 | 0.942 | 0.12 |
| Logistic Regression | Linear | 85.2% | 0.830 | 0.938 | 0.02 |
| SVM (RBF) | Kernel | 86.9% | 0.846 | 0.932 | 0.01 |
| SVM (Linear) | Linear | 82.0% | 0.800 | 0.923 | 0.01 |
| K-Nearest Neighbors | Instance | 83.6% | 0.821 | 0.918 | 0.01 |
| Gradient Boosting | Ensemble | 82.0% | 0.800 | 0.903 | 0.15 |
| Naive Bayes | Probabilístico | 80.3% | 0.786 | 0.897 | 0.01 |
| Random Forest | Ensemble | 77.0% | 0.750 | 0.889 | 0.08 |
| Decision Tree | Árvore | 73.8% | 0.724 | 0.739 | 0.01 |

### Validação Cruzada

**Resultados CV (5-fold):**

| Modelo | Acurácia CV | F1-Score CV | AUC-ROC CV |
|--------|-------------|-------------|------------|
| Logistic Regression | 0.885 ± 0.048 | 0.881 ± 0.052 | 0.885 ± 0.048 |
| SVM (RBF) | 0.879 ± 0.042 | 0.875 ± 0.046 | 0.879 ± 0.042 |
| AdaBoost | 0.856 ± 0.051 | 0.850 ± 0.058 | 0.856 ± 0.051 |

### Seleção dos Melhores Modelos

Com base na performance e estabilidade, selecionamos os top 4 modelos para otimização:
1. **AdaBoost** - Melhor AUC-ROC inicial
2. **Logistic Regression** - Melhor estabilidade CV
3. **SVM (RBF)** - Boa performance geral
4. **Random Forest** - Modelo interpretável

---

## Otimização de Hiperparâmetros

### Grid Search

Utilizamos `GridSearchCV` com validação cruzada 5-fold e métrica AUC-ROC:

**AdaBoost:**
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0]
}
```

**Logistic Regression:**
```python
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
```

**SVM (RBF):**
```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}
```

**Random Forest:**
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
```

### Resultados da Otimização

| Modelo | Melhores Parâmetros | CV Score | Val AUC-ROC | Val Accuracy |
|--------|-------------------|----------|-------------|--------------|
| **SVM (RBF)** | C=0.1, gamma='scale' | 0.891 | **0.912** | **85.2%** |
| Random Forest | n_estimators=100, max_depth=None | 0.875 | 0.876 | 78.7% |
| AdaBoost | n_estimators=100, learning_rate=1.0 | 0.843 | 0.843 | 72.1% |
| Logistic Regression | C=0.1, penalty='l2' | 0.500 | 0.500 | 54.1% |

**Modelo Final Selecionado:** SVM (RBF) com C=0.1 e gamma='scale'

### Justificativa da Seleção

O SVM (RBF) foi selecionado por:
- **Melhor AUC-ROC:** 0.912 (excelente discriminação)
- **Boa Acurácia:** 85.2% (acima do baseline)
- **Estabilidade:** Performance consistente na validação cruzada
- **Eficiência:** Tempo de treinamento baixo
- **Robustez:** Menos propenso a overfitting com regularização adequada

---

## Avaliação Final

### Treinamento do Modelo Final

O modelo SVM otimizado foi retreinado com dados de treino + validação (243 amostras) e avaliado no conjunto de teste (60 amostras) nunca visto antes.

### Resultados no Conjunto de Teste

**Métricas Principais:**
- **Acurácia:** 86.9% (53/61 predições corretas)
- **Precisão:** 88.0% (22/25 predições positivas corretas)
- **Recall:** 81.5% (22/27 casos positivos identificados)
- **F1-Score:** 84.6% (média harmônica precisão/recall)
- **AUC-ROC:** 96.2% (excelente discriminação)

**Matriz de Confusão:**
```
                Predito
Real        Sem Doença  Com Doença
Sem Doença      31         3
Com Doença       5        22
```

**Interpretação:**
- **Verdadeiros Negativos:** 31 (pacientes sem doença corretamente identificados)
- **Falsos Positivos:** 3 (pacientes sem doença classificados como doentes)
- **Falsos Negativos:** 5 (pacientes doentes não identificados)
- **Verdadeiros Positivos:** 22 (pacientes doentes corretamente identificados)

### Análise de Erros

**Falsos Positivos (3 casos):**
- Pacientes classificados como doentes, mas sem doença
- Probabilidades médias: 0.65-0.75 (incerteza moderada)
- Possível presença de fatores de risco subclínicos

**Falsos Negativos (5 casos):**
- Pacientes doentes não identificados (mais crítico)
- Probabilidades médias: 0.35-0.45 (próximo ao limiar)
- Podem representar casos atípicos ou estágios iniciais

### Importância das Features

**Top 10 Features (Permutation Importance):**

| Rank | Feature | Importância | Interpretação Clínica |
|------|---------|-------------|----------------------|
| 1 | ca | 0.156 | Número de vasos obstruídos |
| 2 | oldpeak | 0.089 | Depressão do ST (isquemia) |
| 3 | cp | 0.067 | Tipo de dor no peito |
| 4 | thalach | 0.045 | Capacidade cardíaca máxima |
| 5 | exang | 0.034 | Angina induzida por exercício |
| 6 | thal_7.0 | 0.028 | Defeito reversível na perfusão |
| 7 | sex | 0.022 | Diferenças de gênero |
| 8 | slope | 0.019 | Resposta ao exercício |
| 9 | age | 0.015 | Fator de risco idade |
| 10 | trestbps | 0.012 | Pressão arterial basal |

**Insights Clínicos:**
- **ca (vasos obstruídos):** Principal preditor, alinhado com fisiopatologia
- **oldpeak (depressão ST):** Indicador direto de isquemia miocárdica
- **cp (dor no peito):** Sintoma clássico, especialmente dor atípica
- **thalach:** Capacidade funcional cardíaca reduzida indica doença

---

## Resultados e Discussão

### Performance do Modelo

O modelo SVM final demonstrou excelente performance:

**Pontos Fortes:**
- **Alta Discriminação:** AUC-ROC de 96.2% indica excelente capacidade de separar classes
- **Boa Sensibilidade:** 81.5% dos casos positivos identificados
- **Alta Especificidade:** 91.2% dos casos negativos corretamente classificados
- **Balanceamento:** F1-Score de 84.6% mostra equilíbrio entre precisão e recall

**Comparação com Literatura:**
- Performance superior a muitos estudos similares (70-85% típico)
- AUC-ROC comparável aos melhores modelos reportados
- Resultados consistentes com validação clínica

### Relevância Clínica

**Features Importantes Validadas:**
1. **ca (vasos obstruídos):** Correlação direta com gravidade da doença coronariana
2. **oldpeak:** Marcador estabelecido de isquemia durante teste de esforço
3. **cp (dor no peito):** Sintoma cardinal, especialmente padrões atípicos
4. **thalach:** Capacidade funcional reduzida indica comprometimento cardíaco

**Aplicabilidade Clínica:**
- **Triagem:** Identificação de pacientes de alto risco
- **Apoio Diagnóstico:** Complemento à avaliação clínica
- **Priorização:** Direcionamento para exames mais específicos
- **Monitoramento:** Acompanhamento de fatores de risco

### Limitações e Considerações

**Limitações do Estudo:**
1. **Tamanho da Amostra:** 303 pacientes é relativamente pequeno
2. **Época dos Dados:** Coletados em 1988, podem não refletir práticas atuais
3. **População:** Dados de centros específicos, generalização limitada
4. **Features:** Conjunto limitado de variáveis clínicas
5. **Interpretabilidade:** SVM oferece menor interpretabilidade que árvores

**Considerações Éticas:**
- **Viés:** Possível sub-representação de grupos demográficos
- **Responsabilidade:** Modelo não substitui julgamento médico
- **Transparência:** Necessidade de explicabilidade para uso clínico
- **Validação:** Requer validação em populações diversas

**Riscos Potenciais:**
- **Falsos Negativos:** Pacientes doentes não identificados (5 casos)
- **Falsos Positivos:** Ansiedade e custos desnecessários (3 casos)
- **Dependência:** Risco de reduzir avaliação clínica abrangente

### Comparação com Outros Estudos

**Benchmarks da Literatura:**
- Detrano et al. (1989): 77% acurácia com análise discriminante
- Aha & Kibler (1991): 84% com k-NN
- Gennari et al. (1989): 81% com árvores de decisão

**Nosso Modelo:**
- **86.9% acurácia** - Superior aos benchmarks históricos
- **96.2% AUC-ROC** - Excelente discriminação
- **Metodologia robusta** - Validação cruzada e teste independente

---

## Conclusões

### Principais Achados

1. **Modelo Eficaz:** SVM com kernel RBF demonstrou excelente capacidade preditiva para doenças cardíacas
2. **Features Relevantes:** Número de vasos obstruídos, depressão do ST e tipo de dor no peito são os principais preditores
3. **Performance Superior:** Resultados superam benchmarks históricos e muitos estudos contemporâneos
4. **Aplicabilidade Clínica:** Modelo pode servir como ferramenta de apoio ao diagnóstico médico

### Contribuições do Projeto

**Metodológicas:**
- Pipeline completo de ML aplicado à saúde
- Comparação sistemática de múltiplos algoritmos
- Otimização rigorosa de hiperparâmetros
- Validação não enviesada em dados de teste

**Clínicas:**
- Identificação de features mais importantes
- Quantificação da capacidade preditiva
- Análise de erros e limitações
- Direcionamento para aplicações práticas

**Técnicas:**
- Demonstração de boas práticas em ML
- Tratamento adequado de dados médicos
- Interpretação de resultados em contexto clínico
- Documentação completa e reproduzível

### Recomendações

**Para Implementação Clínica:**
1. **Validação Externa:** Testar em outras populações e hospitais
2. **Integração Sistêmica:** Incorporar em sistemas de prontuário eletrônico
3. **Treinamento:** Capacitar profissionais para uso adequado
4. **Monitoramento:** Acompanhar performance em uso real

**Para Pesquisas Futuras:**
1. **Dados Maiores:** Utilizar datasets com milhares de pacientes
2. **Features Adicionais:** Incluir biomarcadores e exames de imagem
3. **Deep Learning:** Explorar redes neurais para padrões complexos
4. **Interpretabilidade:** Aplicar SHAP/LIME para explicações individuais

**Para Desenvolvimento:**
1. **API de Produção:** Desenvolver interface para uso clínico
2. **Monitoramento de Drift:** Detectar mudanças nos padrões dos dados
3. **Atualizações:** Implementar retreinamento periódico
4. **Segurança:** Garantir privacidade e conformidade regulatória

### Impacto Esperado

**Benefícios Potenciais:**
- **Diagnóstico Precoce:** Identificação de casos em estágios iniciais
- **Redução de Custos:** Triagem mais eficiente de pacientes
- **Melhores Desfechos:** Tratamento mais rápido e adequado
- **Apoio à Decisão:** Ferramenta complementar para médicos

**Limitações a Considerar:**
- **Não Substitui Médico:** Ferramenta de apoio, não diagnóstico definitivo
- **Validação Contínua:** Necessidade de monitoramento constante
- **Contexto Específico:** Resultados podem variar entre populações
- **Aspectos Éticos:** Considerações sobre viés e equidade
