# AGENTS.md

## Contexto do Projeto

Este repositório contém o desenvolvimento do projeto de TCC I/TCC II intitulado:

"Sistema Inteligente de Alerta Climático Baseado em Dados Meteorológicos e Aprendizado de Máquina".

O objetivo é desenvolver um sistema inteligente capaz de coletar dados meteorológicos de fontes públicas, tratar essas informações, aplicar modelos supervisionados de aprendizado de máquina e gerar alertas preventivos relacionados a eventos climáticos extremos.

## Escopo do Sistema

O sistema deve utilizar dados meteorológicos provenientes das seguintes fontes:

* INMET;
* OpenWeather.

A granularidade inicial do sistema será por cidade.

As variáveis meteorológicas previstas incluem:

* temperatura;
* sensação térmica;
* umidade relativa do ar;
* precipitação;
* velocidade dos ventos;
* pressão atmosférica.

## Eventos Climáticos Considerados

O sistema deverá classificar riscos associados a eventos como:

* ondas de calor;
* ondas de frio;
* geadas;
* chuvas intensas;
* enchentes;
* granizo;
* ventos extremos;
* condições favoráveis à formação de tornados;
* risco de incêndios florestais;
* umidade crítica.

## Tecnologias Definidas

O projeto deverá utilizar:

* Python 3.12;
* Pandas;
* NumPy;
* Scikit-Learn;
* Streamlit;
* APIs do INMET e OpenWeather.

## Modelos de Aprendizado de Máquina

Devem ser considerados inicialmente os seguintes modelos supervisionados:

* Regressão Logística;
* Random Forest;
* XGBoost.

A Regressão Logística será usada como baseline interpretável. O Random Forest será utilizado por sua robustez com relações não lineares. O XGBoost será avaliado por seu desempenho preditivo e capacidade de correção sequencial de erros.

## Avaliação dos Modelos

As métricas utilizadas serão:

* acurácia;
* precisão;
* recall;
* F1-Score;
* matriz de confusão.

O principal critério de comparação será o F1-Score, pois equilibra precisão e recall e é mais adequado para cenários com classes possivelmente desbalanceadas.

## Convenções de Código

Use as seguintes regras:

* Código em Python seguindo PEP 8.
* Nomes de arquivos em snake_case.
* Nomes de funções em snake_case.
* Nomes de classes em PascalCase.
* Variáveis em snake_case.
* Constantes em UPPER_CASE.
* Separar responsabilidades por módulo.
* Evitar arquivos grandes demais.
* Preferir funções pequenas e testáveis.
* Usar type hints sempre que possível.
* Documentar funções principais com docstrings curtas.

## Arquitetura Esperada

O fluxo geral do sistema deve seguir esta ordem:

1. Coleta de dados meteorológicos.
2. Tratamento e preparação dos dados.
3. Treinamento dos modelos supervisionados.
4. Classificação do risco climático.
5. Geração de alertas.
6. Visualização em dashboard Streamlit.

## Regras para o Codex

Antes de implementar, sempre explique brevemente o plano.

Não criar funcionalidades fora do escopo do TCC.

Não utilizar serviços pagos sem necessidade.

Não expor chaves de API no código.

Usar arquivo .env para variáveis sensíveis.

Criar código simples, acadêmico e fácil de explicar para banca.

Priorizar clareza, organização e rastreabilidade em vez de complexidade excessiva.

Sempre que criar uma funcionalidade, atualizar o README.md se necessário.

Sempre que alterar estrutura de pastas, explicar o motivo.
