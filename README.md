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
- DuckDB
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
INMET_STATION_CATALOG_PATH=data/processed/inmet_station_catalog.csv
INMET_DATABASE_PATH=data/processed/inmet_historical.duckdb
INMET_DATABASE_URL=
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

O fluxo historico pode usar duas formas de dados:

- ZIPs brutos do INMET em `data/raw/inmet/zips/`: usados para gerar bases
  locais e como fallback quando nao existe banco processado.
- Banco processado DuckDB em `data/processed/inmet_historical.duckdb`: usado
  preferencialmente pelo dashboard por permitir consulta filtrada por estacao e
  intervalo de anos, sem carregar todo o historico nacional na memoria.

O arquivo DuckDB tambem nao e versionado no Git. Ele pode ser gerado localmente
ou distribuido por fora do repositorio, por exemplo em GitHub Releases.

O INMET historico pode ser consultado por um seletor pesquisavel de estacao no
dashboard. O sistema resolve internamente o codigo da estacao usando um
catalogo local construido a partir dos nomes dos CSVs presentes nos ZIPs anuais
brutos do INMET, complementando latitude, longitude e altitude com metadados
quando disponiveis. Por exemplo, as opcoes aparecem como `MANAUS - AM | A101`
e `BRASILIA - DF | A001`, quando essas estacoes existirem nos ZIPs locais.

### Pressao atmosferica na comparacao historica

A pressao da OpenWeather em `main.pressure` representa a pressao ao nivel do
mar. Esse valor e preservado no sistema como `pressure_sea_level_hpa` e tambem
permanece no campo legado `pressure` para compatibilidade com partes antigas
do prototipo.

O historico do INMET usa pressao atmosferica ao nivel da estacao. Por isso, a
comparacao historica nao compara diretamente `main.pressure` da OpenWeather
com a coluna `pressure` do INMET. Quando a OpenWeather retorna `main.grnd_level`,
o sistema usa esse valor diretamente como `pressure_station_hpa`.

Quando `main.grnd_level` nao esta disponivel, o sistema estima
`pressure_station_hpa` a partir de `pressure_sea_level_hpa` e da altitude da
estacao INMET em metros, vinda do catalogo local de estacoes:

```text
pressure_station_hpa = pressure_sea_level_hpa * (1 - 0.0065 * altitude_m / 288.15) ** 5.255
```

Se nao houver `grnd_level` nem altitude da estacao, a pressao fica marcada como
nao avaliada na analise historica. Isso evita falso positivo por comparar
pressao ao nivel do mar com pressao ao nivel da estacao.

O dataset tratado em `data/processed/inmet_hourly.parquet` nao e usado para
descobrir estacoes, pois ele pode ter sido gerado apenas para uma unica
estacao. A descoberta do catalogo sempre parte dos ZIPs brutos em
`data/raw/inmet/zips/`.

Para gerar um CSV local com o catalogo de estacoes, execute:

```powershell
python scripts\build_inmet_station_catalog.py
```

O arquivo sera salvo por padrao em:

```text
data/processed/inmet_station_catalog.csv
```

Esse catalogo e derivado dos nomes/metadados dos CSVs nos ZIPs e nao e
versionado no Git.

### Base historica processada em DuckDB

Para gerar a base historica processada a partir dos ZIPs locais, execute:

```powershell
python scripts\build_inmet_duckdb.py --start-year 2000 --end-year 2026
```

Para gerar apenas algumas estacoes, informe os codigos separados por virgula:

```powershell
python scripts\build_inmet_duckdb.py --start-year 2000 --end-year 2026 --stations A001,A101,A312
```

Por padrao, a base e salva em:

```text
data/processed/inmet_historical.duckdb
```

O dashboard verifica `INMET_DATABASE_PATH`. Se o DuckDB existir, ele usa a base
processada como fonte principal e consulta apenas `station_code` e intervalo de
anos selecionados. Se o DuckDB nao existir, o dashboard continua usando os
ZIPs locais como fallback.

Para baixar uma base pronta, configure `INMET_DATABASE_URL` no `.env` e execute:

```powershell
python scripts\download_inmet_database.py
```

A URL real nao e obrigatoria no prototipo. A estrutura existe para permitir
distribuir o arquivo processado por GitHub Releases ou outro armazenamento
publico sem commitar o DuckDB no repositorio.

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

## Comparacao Historica no Dashboard

O dashboard tambem exibe a secao `Comparacao com historico INMET`. Ela usa os
dados atuais carregados por `Entrada manual` ou `OpenWeather` e compara esses
valores com a serie historica de uma estacao INMET.

No modo `OpenWeather`, o fluxo principal e automatico: informe a cidade uma
vez, clique em `Buscar dados atuais` e o sistema tenta associar a estacao INMET
mais adequada. Primeiro ele usa latitude e longitude retornadas pela
OpenWeather para escolher a estacao mais proxima no catalogo local do INMET. Se
as coordenadas nao estiverem disponiveis, tenta encontrar uma estacao pelo nome
da cidade. Em seguida carrega o historico da estacao e executa o
`HistoricalAnalyzer`.

Na tela, o sistema mostra o fluxo executado:

- dados atuais obtidos via OpenWeather;
- estacao INMET associada automaticamente;
- historico INMET carregado para comparacao;
- analise historica executada.

A estacao manual continua disponivel em `Opcao avancada: alterar estacao INMET
manualmente`. Ela serve para corrigir a associacao automatica ou para escolher
outra estacao de referencia.

No modo `Entrada manual`, nao ha coordenadas automaticas da cidade. Por isso, o
dashboard solicita a estacao INMET de referencia para comparacao historica. O
historico e carregado automaticamente apos a selecao.

O app mostra nome da estacao, UF, codigo, altitude quando disponivel, distancia
aproximada ate a cidade atual quando a associacao usa coordenadas, periodo
carregado e quantidade de registros. Quando ha dados atuais e historico, o
`HistoricalAnalyzer` calcula anomalias estatisticas e exibe:

- anomalias historicas identificadas, com tipo, severidade, valor atual,
  referencia historica e justificativa;
- resumo por variavel com valor atual, media historica, percentil usado e
  interpretacao;
- detalhes tecnicos da pressao atmosferica em um expander.

Essa comparacao historica complementa o alerta principal por regras. Ela ajuda
a explicar se o dado atual esta fora do padrao observado para a estacao, mas
ainda nao e Machine Learning e nao substitui o classificador por regras nem
alertas oficiais.

A pressao segue o mesmo cuidado de referencial descrito acima: o historico do
INMET usa pressao ao nivel da estacao. Quando a OpenWeather fornece
`grnd_level`, o sistema usa esse valor como `pressure_station_hpa`; quando nao
fornece, o sistema estima a pressao ao nivel da estacao com a altitude da
estacao INMET. A tela de detalhes mostra a pressao ao nivel do mar, quando
existir, e a pressao de estacao usada na comparacao. Assim, a comparacao nao
usa diretamente a pressao ao nivel do mar contra a pressao historica da
estacao.

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

A secao `Comparacao com historico INMET` fica abaixo dos dados atuais e do
alerta por regras. No modo `OpenWeather`, digite uma cidade, como `Manaus`, e
clique em `Buscar dados atuais`; o sistema tentara associar automaticamente
uma estacao como `MANAUS - AM | A101` ou a mais proxima pelas coordenadas. No
modo `Entrada manual`, selecione a estacao de referencia na secao historica.
Esses dados historicos nao geram alerta climatico diretamente nesta etapa.

## Testes

```powershell
python -m pytest
```
