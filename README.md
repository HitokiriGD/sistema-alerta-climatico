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
scripts/             Scripts operacionais locais
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
INMET_PROCESSED_DATA_PATH=data/processed/inmet_hourly.parquet
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
`*.zip` e arquivos `*.har` ficam ignorados pelo Git. Ela deve conter os ZIPs
baixados localmente e outros insumos originais que podem ser recriados.

A pasta `data/processed/` armazena datasets tratados gerados localmente pelo
pipeline de pre-tratamento. Arquivos grandes como `*.parquet` e `*.csv` nessa
pasta tambem ficam ignorados pelo Git; apenas a estrutura da pasta e mantida no
repositorio.

## Pre-tratamento dos Dados

A etapa de pre-tratamento centraliza a preparacao dos dados meteorologicos em
`src/processing/preprocessing.py`. O pipeline:

- padroniza nomes de colunas meteorologicas;
- converte numeros com virgula decimal para `float`;
- valida colunas obrigatorias;
- aplica limites plausiveis para temperatura, sensacao termica, umidade,
  precipitacao, vento e pressao;
- remove registros sem `datetime` ou sem `temperature`;
- preserva registros parcialmente uteis com a coluna `quality_flag`;
- ordena os dados por `datetime`;
- remove duplicidades por `station_code` e `datetime`, quando houver estacao.

Para gerar a base historica tratada do INMET a partir dos ZIPs locais, execute:

```powershell
python scripts\build_inmet_dataset.py --station A001 --start-year 2020 --end-year 2026
```

Por padrao, o arquivo tratado e salvo em:

```text
data/processed/inmet_hourly.parquet
```

Se o ambiente nao tiver uma dependencia de parquet instalada, o script salva
automaticamente em:

```text
data/processed/inmet_hourly.csv
```

Esse dataset tratado sera usado nas proximas etapas para comparacao historica e
futura modelagem com aprendizado de maquina.

## Classificador por Regras

A etapa atual implementa um classificador inicial por regras em
`src/alerts/risk_classifier.py`. Ele usa os dados meteorologicos atuais
padronizados, vindos da `Entrada manual` ou da `OpenWeather`, para gerar uma
saida explicavel antes da etapa de aprendizado de maquina.

A resposta do classificador contem:

- `risk_level`: `baixo`, `moderado`, `alto` ou `critico`;
- `event_type`: principal tipo de evento climatico identificado;
- `reason`: justificativa textual da classificacao;
- `triggered_rules`: regras acionadas;
- `variables`: variaveis meteorologicas consideradas;
- `recommendations`: orientacoes gerais curtas.

Os tipos iniciais de evento sao `sem_risco_relevante`, `baixa_umidade`,
`calor_extremo`, `frio_intenso`, `chuva_intensa`, `vento_forte` e
`risco_incendio`. As regras avaliam limites simples de umidade, temperatura,
sensacao termica, precipitacao e vento. Quando mais de uma regra e acionada, o
evento principal e definido pela maior severidade, mantendo todas as regras na
lista de evidencias.

Essas regras sao heuristicas iniciais do prototipo academico. Elas nao
substituem alertas oficiais de defesa civil ou de orgaos meteorologicos. O
INMET historico continua separado como base de consulta, comparacao e futura
modelagem; ele nao gera alerta atual sozinho nesta etapa.

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
