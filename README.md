# Sistema Inteligente de Alerta Climatico

Projeto academico de TCC para um sistema de alerta climatico baseado em dados
meteorologicos e aprendizado de maquina.

O objetivo e coletar dados de fontes publicas, preparar variaveis
meteorologicas, treinar modelos supervisionados e apresentar alertas
preventivos em um dashboard Streamlit. Nesta etapa, o projeto possui uma base
simples e executavel, sem funcionalidades complexas de coleta ou treinamento em
producao.

## Arquitetura

```text
app/                 Dashboard Streamlit
src/alerts/          Regras e classificacao de risco climatico
src/config/          Leitura de variaveis de ambiente
src/data/            Clientes para INMET e OpenWeather
src/processing/      Limpeza e preparacao de dados
src/ml/              Criacao, treinamento e avaliacao de modelos
src/utils/           Utilitarios compartilhados
data/raw/            Dados brutos locais
data/processed/      Dados tratados locais
docs/                Documentacao do projeto
notebooks/           Experimentos academicos
tests/               Testes automatizados basicos
```

Fluxo esperado:

1. Coleta de dados meteorologicos.
2. Tratamento e preparacao dos dados.
3. Treinamento de modelos supervisionados.
4. Classificacao do risco climatico.
5. Geracao de alertas.
6. Visualizacao no Streamlit.

## Tecnologias

- Python 3.12
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Streamlit
- Requests
- Python Dotenv
- Pytest

## Configuracao

Copie `.env.example` para `.env` e preencha a chave da OpenWeather quando for
usar a API externa.

```powershell
Copy-Item .env.example .env
```

Para executar somente o prototipo local com entrada manual de dados, a chave de
API nao e obrigatoria.

## Instalacao

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Execucao

```powershell
streamlit run app\streamlit_app.py
```

## Testes

```powershell
python -m pytest
```
