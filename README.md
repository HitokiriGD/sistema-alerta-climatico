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

No arquivo `.env`, configure:

```env
OPENWEATHER_API_KEY=sua_chave_da_openweather
DEFAULT_CITY=Brasilia
DEFAULT_COUNTRY=BR
OPENWEATHER_BASE_URL=https://api.openweathermap.org/data/2.5
INMET_HISTORICAL_ZIP_DIR=data/raw/inmet/zips
INMET_HISTORICAL_START_YEAR=2020
INMET_HISTORICAL_END_YEAR=2026
```

Para executar somente o prototipo local com entrada manual de dados, a chave de
API nao e obrigatoria. Se `OPENWEATHER_API_KEY` nao estiver configurada, o
dashboard exibira um aviso amigavel ao selecionar OpenWeather e o modo manual
continuara disponivel.

## Fontes de Dados

O projeto separa as fontes por finalidade:

- `OpenWeather`: usada para dados meteorologicos atuais por cidade.
- `INMET Historico`: usado como base historica oficial para consulta,
  comparacao e futuro treinamento dos modelos.

O INMET nao e usado como fonte de tempo real neste projeto, pois o portal
publico depende de validacoes como `seed`, `gcap` e reCAPTCHA. Em vez disso,
use os ZIPs anuais da base historica oficial do INMET.

Baixe os ZIPs anuais no portal de dados historicos do INMET e salve os arquivos
em:

```text
data/raw/inmet/zips/
```

O dashboard usa por padrao o intervalo de 2020 a 2026 para evitar lentidao ao
carregar muitos anos de dados. Esse intervalo pode ser alterado no `.env` ou
nos campos da secao historica do dashboard.

Os dados brutos nao sao versionados no GitHub. A pasta `data/raw/`, arquivos
`*.zip` e arquivos `*.har` ficam ignorados pelo Git.

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

No dashboard, escolha a fonte dos dados na barra lateral:

- `Entrada manual`: usa os valores preenchidos diretamente na tela.
- `OpenWeather`: busca dados meteorologicos atuais pela cidade informada usando
  a chave configurada no `.env`.

A secao `Base historica INMET` fica separada dos dados atuais. Informe o codigo
da estacao, como `A001`, e o intervalo de anos para carregar os ZIPs locais,
visualizar o periodo disponivel, a quantidade de registros e medias historicas.
Esses dados historicos nao geram alerta climatico diretamente nesta etapa.

## Testes

```powershell
python -m pytest
```
