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
INMET_DATABASE_RELEASE_REPO=HitokiriGD/sistema-alerta-climatico
INMET_DATABASE_RELEASE_TAG=inmet-db-v1
INMET_DATABASE_ASSET_NAME=inmet_historical.duckdb
INMET_DATABASE_SHA256=2621a5ada2f5b1d2f598690a3868639a406c4efe13fd36dd076f0a08eaa6edbe
ML_ARTIFACTS_RELEASE_REPO=HitokiriGD/sistema-alerta-climatico
ML_ARTIFACTS_RELEASE_TAG=ml-artifacts-v1
ML_MODEL_ASSET_NAME=risk_level_model.joblib
ML_MODEL_METADATA_ASSET_NAME=risk_level_model_metadata.json
ML_EVALUATION_REPORT_ASSET_NAME=risk_level_robust_evaluation_temporal_report.json
ML_MODEL_SHA256=COLE_AQUI_O_SHA256_DO_JOBLIB
ML_MODEL_METADATA_SHA256=COLE_AQUI_O_SHA256_DO_METADATA_JSON
ML_EVALUATION_REPORT_SHA256=COLE_AQUI_O_SHA256_DO_REPORT_JSON
GITHUB_TOKEN=
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

## Base Propria OpenWeather em Postgres

A partir desta etapa, o projeto tambem possui uma base operacional propria em
Postgres para armazenar snapshots coletados da OpenWeather ao longo do tempo.
Essa base fica separada do INMET historico:

- `INMET DuckDB`: base historica oficial processada, usada para comparacao
  historica e futuro treinamento inicial.
- `OpenWeather Postgres`: base propria operacional do sistema, preenchida por
  coletas periodicas realizadas pelo projeto.

A coleta padrao usa as 27 capitais brasileiras. Essa escolha mantem o escopo
controlado, demonstravel no TCC e compativel com o limite do Supabase Free. A
lista pode ser expandida futuramente por meio de `OPENWEATHER_COLLECTION_CITIES`.

O Supabase Free possui limite aproximado de 500 MB por projeto. Como o Supabase
nao apaga registros antigos automaticamente, o sistema executa retencao apos as
coletas:

- remove observacoes com `collected_at` mais antigo que
  `OPENWEATHER_RETENTION_DAYS`;
- se ainda houver mais que `OPENWEATHER_MAX_ROWS`, remove os registros mais
  antigos ate ficar dentro do limite;
- `OPENWEATHER_STORE_RAW_PAYLOAD=false` por padrao para economizar espaco.

Configure no `.env` local:

```env
DATABASE_URL=postgresql://usuario:senha@host:porta/database
OPENWEATHER_COLLECTION_CITIES=Rio Branco:BR,Maceio:BR,Macapa:BR,Manaus:BR,Salvador:BR,Fortaleza:BR,Brasilia:BR,Vitoria:BR,Goiania:BR,Sao Luis:BR,Cuiaba:BR,Campo Grande:BR,Belo Horizonte:BR,Belem:BR,Joao Pessoa:BR,Curitiba:BR,Recife:BR,Teresina:BR,Rio de Janeiro:BR,Natal:BR,Porto Alegre:BR,Porto Velho:BR,Boa Vista:BR,Florianopolis:BR,Sao Paulo:BR,Aracaju:BR,Palmas:BR
OPENWEATHER_RETENTION_DAYS=180
OPENWEATHER_MAX_ROWS=100000
OPENWEATHER_STORE_RAW_PAYLOAD=false
```

Nunca versionar `.env`, banco real ou dados coletados. `DATABASE_URL`,
`OPENWEATHER_API_KEY` e tokens devem ficar apenas no ambiente local ou em
secrets.

Inicialize o banco Postgres:

```powershell
python scripts\init_weather_database.py
```

Coleta manual de uma cidade:

```powershell
python scripts\collect_openweather_snapshot.py --city Brasilia --country BR
```

Coleta de todas as capitais configuradas:

```powershell
python scripts\collect_openweather_snapshot.py
```

O resumo no terminal mostra apenas informacoes seguras: cidades processadas,
registros inseridos, falhas, limpeza de retencao e registros restantes. A URL
do banco e a chave da API nao sao impressas.

Para automatizar no GitHub Actions, configure os secrets do repositorio:

- `OPENWEATHER_API_KEY`
- `DATABASE_URL`

O workflow `.github/workflows/collect-openweather.yml` executa a coleta manual
por `workflow_dispatch` e tambem 4 vezes ao dia pelo cron:

```text
0 0,6,12,18 * * *
```

O cron do GitHub Actions usa UTC. Ajuste a interpretacao dos horarios de acordo
com o fuso desejado para a apresentacao.

Essa base propria ainda nao e usada como fonte principal de treino de Machine
Learning, pois comecara pequena. Ela prepara o sistema para coleta continua,
armazenamento externo e validacao futura.

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

Para baixar a base pronta publicada em GitHub Release, execute:

```powershell
python scripts\download_inmet_database.py
```

Por padrao, o script usa a release:

```text
HitokiriGD/sistema-alerta-climatico | tag inmet-db-v1 | inmet_historical.duckdb
```

Se `INMET_DATABASE_URL` estiver preenchida, essa URL direta tem prioridade.
Caso contrario, o script consulta a API do GitHub, localiza o asset configurado
por `INMET_DATABASE_ASSET_NAME` e baixa o arquivo para `INMET_DATABASE_PATH`.
O download e feito primeiro para `data/processed/inmet_historical.duckdb.tmp`
e so depois o arquivo final e substituido.

Se o arquivo final ja existir, o script nao baixa novamente. Para forcar novo
download:

```powershell
python scripts\download_inmet_database.py --force
```

O hash SHA256 e validado quando `INMET_DATABASE_SHA256` estiver configurado.
O valor padrao corresponde ao asset `inmet_historical.duckdb` da release
`inmet-db-v1`.

Em repositorio privado, pode ser necessario configurar um token do GitHub:

```env
GITHUB_TOKEN=seu_token_aqui
```

Crie o token em GitHub -> Settings -> Developer settings -> Personal access
tokens, com permissao de leitura no repositorio. O token e lido pelo `.env` e
nao deve ser impresso no terminal nem versionado no Git.

Como alternativa ao download, gere a base localmente a partir dos ZIPs do INMET
com `scripts\build_inmet_duckdb.py`. O arquivo DuckDB nao e versionado no Git;
ele pode ser recriado localmente ou distribuido por GitHub Releases.

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

## Dataset Rotulado para Machine Learning

A etapa de dataset rotulado cria uma base supervisionada inicial a partir do
INMET DuckDB, que e a base historica oficial processada do projeto. A base
OpenWeather em Postgres continua separada: ela demonstra coleta continua e
armazenamento proprio, mas ainda nao e usada como fonte principal de treino.

Os rotulos iniciais sao gerados pelo `RiskClassifier` atual. Isso significa que
o dataset recebe `risk_level` e `event_type` a partir das mesmas regras tecnicas
explicaveis usadas no dashboard. Na etapa de dataset, o objetivo e gerar,
salvar e validar a base usada pelo aprendizado supervisionado.

O script usa as colunas meteorologicas do INMET, cria features de data e hora
(`year`, `month`, `day`, `hour`, `day_of_year`) e remove registros sem dados
minimos para classificacao. Quando a coluna `feels_like` nao existir no
historico, a temperatura e usada como fallback para manter o dataset completo e
explicavel.

Para gerar o dataset filtrando algumas estacoes:

```powershell
python scripts\build_ml_dataset.py --start-year 2020 --end-year 2026 --stations A001,A101,A312
```

Para gerar uma amostra pequena de teste:

```powershell
python scripts\build_ml_dataset.py --start-year 2025 --end-year 2026 --limit 10000
```

Por padrao, o arquivo e salvo em:

```text
data/processed/ml_training_dataset.parquet
```

Tambem e possivel gerar CSV:

```powershell
python scripts\build_ml_dataset.py --start-year 2025 --end-year 2026 --limit 10000 --format csv --output data/processed/ml_training_dataset.csv
```

O resumo exibido no terminal mostra total de registros, periodo, quantidade de
estacoes, distribuicao por `risk_level`, distribuicao por `event_type` e caminho
salvo. O dataset gerado nao e versionado no Git; arquivos `*.parquet` e `*.csv`
em `data/processed/` permanecem ignorados.

## Treinamento dos Modelos de Machine Learning

A etapa de treinamento usa o dataset rotulado gerado a partir do INMET DuckDB.
O alvo inicial e `risk_level`, com as classes `baixo`, `moderado`, `alto` e
`critico`. A coluna `event_type` permanece no dataset para analises futuras,
mas ainda nao e usada como alvo principal.

Os rotulos vieram do `RiskClassifier`, ou seja, de regras tecnicas explicaveis.
Isso cria uma base supervisionada inicial para comparar modelos, mantendo a
rastreabilidade entre regra, rotulo e resultado do treinamento.

Os modelos treinados nesta etapa sao:

- Baseline de classe majoritaria, como referencia minima de comparacao;
- Regressao Logistica, como modelo linear interpretavel;
- Random Forest, para capturar relacoes nao lineares simples;
- XGBoost, para avaliar um modelo de boosting supervisionado.

Como eventos severos e criticos tendem a ser raros, o dataset e desbalanceado.
Por isso, a acuracia sozinha nao e suficiente para avaliar o modelo: um modelo
que acerta muitos casos `baixo` pode ainda falhar nos eventos mais importantes.
As metricas `f1_macro` e `recall_macro` recebem destaque porque consideram as
classes de forma mais equilibrada e ajudam a enxergar desempenho em classes
minoritarias.

Fluxo recomendado:

```powershell
python scripts\build_ml_dataset.py --start-year 2020 --end-year 2026 --stations A001,A101,A312
python scripts\train_ml_models.py
```

O treinamento usa por padrao:

```text
data/processed/ml_training_dataset.parquet
```

O melhor modelo e salvo em:

```text
data/models/risk_level_model.joblib
```

Os metadados e relatorios sao salvos em:

```text
data/models/risk_level_model_metadata.json
data/reports/risk_level_training_report.json
data/reports/risk_level_confusion_matrix.csv
```

A selecao do melhor modelo usa `f1_macro` por padrao. Os diretorios
`data/models/` e `data/reports/`, os modelos `*.joblib` e os datasets gerados
nao sao versionados no Git.

## Avaliacao Robusta dos Modelos de Machine Learning

A avaliacao robusta compara o baseline de classe majoritaria com Regressao
Logistica, Random Forest e XGBoost para prever `risk_level`. O baseline
`baseline_most_frequent` usa `DummyClassifier(strategy="most_frequent")` e
serve como referencia minima: um modelo real precisa superar essa estrategia
simples para justificar sua utilidade.

Existem dois tipos de divisao treino/teste:

- `random`: embaralha os registros e separa uma parte para teste. E util para
  uma verificacao inicial, mas pode misturar anos parecidos entre treino e
  teste.
- `temporal`: treina com anos anteriores e testa com anos posteriores. Esse
  formato e mais realista para clima, pois simula o uso do passado para prever
  dados futuros.

A acuracia sozinha nao basta porque a base pode ter muitas linhas de risco
`baixo` e poucas linhas `alto` ou `critico`. Por isso o relatorio destaca
`f1_macro`, `recall_macro` e metricas por classe. O `f1_macro` compara as
classes de forma equilibrada, o `recall_macro` ajuda a observar falhas de
identificacao em classes minoritarias, e as metricas por classe mostram
`precision`, `recall`, `f1-score` e `support` para cada nivel de risco.

Limite metodologico importante: os rotulos `risk_level` sao derivados do
`RiskClassifier`. Assim, os modelos aprendem a reproduzir uma classificacao
tecnica baseada em regras explicaveis. As metricas desta etapa nao representam
validacao contra eventos reais oficiais de desastre.

Fluxo recomendado:

```powershell
python scripts/build_ml_dataset.py --start-year 2020 --end-year 2026 --stations A001,A101,A312
python scripts/train_ml_models.py
python scripts/evaluate_ml_models.py --split temporal --train-end-year 2024 --test-start-year 2025
```

Para executar avaliacao com split aleatorio:

```powershell
python scripts/evaluate_ml_models.py --split random
```

O relatorio robusto e salvo em:

```text
data/reports/risk_level_robust_evaluation_temporal_report.json
data/reports/risk_level_robust_confusion_matrix_temporal.csv
data/reports/risk_level_robust_evaluation_random_report.json
data/reports/risk_level_robust_confusion_matrix_random.csv
```

Esses relatorios sao gerados localmente e nao sao versionados. O diretorio
`data/reports/` permanece ignorado pelo Git.

## Camada de Machine Learning Supervisionado no Dashboard

O dashboard possui uma camada de Machine Learning supervisionado para predizer
`risk_level` a partir dos dados meteorologicos atuais. Ele nao usa IA
generativa e nao treina modelos durante a renderizacao da tela; apenas tenta
carregar estes arquivos locais:

```text
data/models/risk_level_model.joblib
data/models/risk_level_model_metadata.json
```

Para gerar a base, treinar o modelo, avaliar os candidatos e abrir o dashboard:

```powershell
python scripts/build_ml_dataset.py --start-year 2020 --end-year 2026 --stations A001,A101,A312
python scripts/train_ml_models.py
python scripts/evaluate_ml_models.py --split temporal --train-end-year 2024 --test-start-year 2025
streamlit run app/streamlit_app.py
```

Na tela, o fluxo foi simplificado para uma experiencia de consulta:

- antes da busca, o dashboard mostra um banner, o formulario e uma explicacao
  curta do fluxo;
- depois da busca, a primeira area exibida e `Resultado da analise`, com a
  predicao de Machine Learning supervisionado, o risco por regras, o modelo
  usado, a metrica de selecao e a quantidade de anomalias historicas;
- ao lado do resultado, a area `Por que esse resultado?` apresenta um resumo
  executivo com no maximo quatro frases: resultado, evidencia principal,
  divergencia entre ML e regras quando existir, e recomendacao curta;
- `Evidencias usadas` mostra cards compactos com temperatura, sensacao termica,
  umidade, chuva, vento, pressao, estacao INMET, periodo historico e principal
  anomalia;
- `Detalhes tecnicos` concentra informacoes mais pesadas em abas: desempenho
  dos modelos treinados, probabilidades por classe, features usadas pelo
  modelo, regras acionadas, estatisticas historicas e variaveis brutas.

O tema visual do dashboard e claro, com fundo branco e cards discretos, para
facilitar a apresentacao do TCC. Detalhes metodologicos, variaveis completas,
percentis, estacao INMET e cautelas ficam em abas ou expanders, evitando
poluir o resultado principal.

A comparacao dos demais modelos no dashboard e apenas de desempenho registrado
nos relatorios locais. A predicao atual usa somente o modelo principal salvo em
`data/models/risk_level_model.joblib`.

O dashboard prefere o relatorio temporal de avaliacao robusta:

```text
data/reports/risk_level_robust_evaluation_temporal_report.json
```

Se ele nao existir, tenta o relatorio random e depois o relatorio basico de
treinamento. Quando nenhum relatorio existe, a tela orienta gerar:

```powershell
python scripts/evaluate_ml_models.py --split temporal --train-end-year 2024 --test-start-year 2025
```

Quando o modelo nao existe, o dashboard continua funcionando. No deploy, ele
tenta baixar modelo, metadados e relatorio da GitHub Release configurada. Se o
download nao estiver configurado ou falhar, a tela orienta gerar o dataset e
treinar com `python scripts/train_ml_models.py`.

Cada ambiente precisa treinar o modelo localmente ou receber esses arquivos por
fora do repositorio. O modelo `*.joblib`, os metadados em `data/models/`, os
datasets em `data/processed/` e os relatorios em `data/reports/` nao sao
versionados no Git.

Limite metodologico: os rotulos usados no treinamento sao derivados do
`RiskClassifier`. A predicao ML representa uma reproducao aprendida dessa
classificacao tecnica por regras. Essa escolha torna o prototipo interpretavel
para a banca, mas as metricas e predicoes nao representam validacao contra
eventos reais oficiais de desastre.

## Deploy no Streamlit Community Cloud

O app esta preparado para rodar como demo no Streamlit Community Cloud usando:

```text
app/streamlit_app.py
```

Use preferencialmente a branch `main` para publicar a demo. Branches de feature
podem ser usadas para teste, mas a URL publica deve apontar para uma versao
revisada.

Configure os secrets no painel do Streamlit Cloud, sem commitar `.env`:

```toml
OPENWEATHER_API_KEY = "sua_chave_openweather"
GITHUB_TOKEN = "opcional_se_release_privada"
DATABASE_URL = "opcional_para_supabase_postgres"
INMET_DATABASE_PATH = "data/processed/inmet_historical.duckdb"
INMET_DATABASE_RELEASE_REPO = "HitokiriGD/sistema-alerta-climatico"
INMET_DATABASE_RELEASE_TAG = "inmet-db-v1"
INMET_DATABASE_ASSET_NAME = "inmet_historical.duckdb"
INMET_DATABASE_SHA256 = "sha256_do_asset"
ML_ARTIFACTS_RELEASE_REPO = "HitokiriGD/sistema-alerta-climatico"
ML_ARTIFACTS_RELEASE_TAG = "ml-artifacts-v1"
ML_MODEL_ASSET_NAME = "risk_level_model.joblib"
ML_MODEL_METADATA_ASSET_NAME = "risk_level_model_metadata.json"
ML_EVALUATION_REPORT_ASSET_NAME = "risk_level_robust_evaluation_temporal_report.json"
ML_MODEL_SHA256 = "sha256_do_joblib"
ML_MODEL_METADATA_SHA256 = "sha256_do_metadata_json"
ML_EVALUATION_REPORT_SHA256 = "sha256_do_report_json"
```

`GITHUB_TOKEN` so e necessario se o repositorio ou alguma release forem
privados. `DATABASE_URL` so e necessario para a coleta continua em
Supabase/Postgres; a consulta do dashboard nao deve imprimir esse valor.

No Streamlit Cloud, o arquivo `data/processed/inmet_historical.duckdb` nao fica
versionado no repositorio. Quando a analise historica for usada pela primeira
vez e o DuckDB ainda nao existir no ambiente, o dashboard tenta baixar
automaticamente o asset configurado em `INMET_DATABASE_RELEASE_REPO`,
`INMET_DATABASE_RELEASE_TAG`, `INMET_DATABASE_ASSET_NAME` e
`INMET_DATABASE_SHA256`. O hash SHA256 e validado quando configurado.

Como a release atual e publica, `GITHUB_TOKEN` e opcional. Se a release voltar
a ser privada, configure `GITHUB_TOKEN` nos secrets do Streamlit Cloud para
permitir leitura do asset sem expor o token no app. Se o download falhar, a
demo continua com dados atuais da OpenWeather, regras tecnicas e Machine
Learning quando disponivel, mas sem comparacao historica INMET.

Os artefatos de Machine Learning tambem nao ficam versionados. No primeiro uso,
quando algum arquivo estiver ausente, o dashboard tenta baixar da release:

```text
HitokiriGD/sistema-alerta-climatico | tag ml-artifacts-v1
```

Arquivos esperados:

- `data/models/risk_level_model.joblib`;
- `data/models/risk_level_model_metadata.json`;
- `data/reports/risk_level_robust_evaluation_temporal_report.json`.

O download usa os assets configurados por `ML_MODEL_ASSET_NAME`,
`ML_MODEL_METADATA_ASSET_NAME` e `ML_EVALUATION_REPORT_ASSET_NAME`. Os hashes
`ML_MODEL_SHA256`, `ML_MODEL_METADATA_SHA256` e
`ML_EVALUATION_REPORT_SHA256` sao validados quando configurados. Se o download
falhar, o app continua funcionando com classificacao tecnica por regras,
OpenWeather e historico INMET quando disponivel.

Arquivos grandes e sensiveis nao sao versionados:

- `.env`;
- `data/processed/`;
- `data/models/`;
- `data/reports/`;
- arquivos `*.duckdb`, `*.joblib`, `*.parquet`, `*.csv` e `*.zip`.

O dashboard abre em tres niveis:

1. Sem DuckDB, sem modelo e sem relatorio: a tela abre e orienta como baixar ou
   gerar os artefatos. Ao executar a analise, tenta baixar o DuckDB e os
   artefatos ML automaticamente das releases configuradas.
2. Com DuckDB e OpenWeather: consulta atual e comparacao historica funcionam.
3. Com DuckDB, modelo e relatorio: a demo completa exibe ML, historico e
   comparacao dos modelos.

Antes de publicar, valide localmente:

```powershell
python -m pytest
streamlit run app/streamlit_app.py
```

## Classificador por Regras

A etapa atual implementa um classificador inicial por regras em
`src/alerts/risk_classifier.py`. Ele usa os dados meteorologicos atuais
padronizados, vindos da `Entrada manual` ou da `OpenWeather`, para gerar uma
classificacao tecnica explicavel. Esses rotulos tambem servem como base
supervisionada inicial para o treinamento dos modelos.

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
modelagem; ele nao gera alerta atual sozinho nesta etapa. No dashboard, as
regras ficam nos detalhes tecnicos como explicabilidade da classificacao e
ajudam a justificar os rotulos aprendidos pelos modelos.

## Camada de Avisos e Recomendacoes

A comunicacao de risco do dashboard foi separada em uma camada propria em
`src/alerts/risk_messages.py`. Essa camada nao cria novas regras de
classificacao e nao altera o `RiskClassifier`; ela transforma o resultado ja
calculado em mensagens mais claras para apoio a decisao.

Para cada nivel (`baixo`, `moderado`, `alto` e `critico`), o sistema gera:

- titulo do aviso;
- descricao curta do risco potencial;
- nivel de atencao;
- recomendacao para usuario ou populacao;
- recomendacao para autoridades ou responsaveis;
- cautela metodologica.

As mensagens tambem consideram o tipo de evento principal, como
`calor_extremo`, `baixa_umidade`, `chuva_intensa`, `vento_forte`,
`frio_intenso`, `risco_incendio` e `sem_risco_relevante`. Assim, um caso de
baixa umidade orienta hidratacao e atencao respiratoria, enquanto um caso de
chuva intensa destaca acompanhamento de deslocamentos, alagamentos e areas
sensiveis.

No dashboard, o bloco `Resultado da analise` mostra o nivel de atencao, o aviso
principal e uma recomendacao curta. A aba `Avisos e recomendacoes`, em
`Detalhes tecnicos`, mostra as evidencias consideradas, recomendacao completa
para a populacao, recomendacao para responsaveis e a cautela metodologica.

Esses avisos sao preventivos e interpretativos. Eles usam dados atuais,
resultado do modelo quando disponivel, classificacao por regras e comparacao
historica como apoio a decisao, mas nao substituem alertas oficiais, protocolos
institucionais ou avaliacao de orgaos competentes.

## Comparacao Historica no Dashboard

O dashboard tambem exibe a secao `Comparacao com historico INMET`. Ela usa os
dados atuais carregados por `Entrada manual` ou `OpenWeather` e compara esses
valores com a serie historica de uma estacao INMET.

No modo `OpenWeather`, o fluxo principal parte de um formulario unico no topo
da tela. Informe cidade, pais, ano inicial e ano final do historico antes de
clicar em `Buscar dados e comparar com historico`. O dashboard so busca dados
atuais e historicos depois desse clique, usando exatamente o periodo escolhido.
Em seguida, o sistema tenta associar a estacao INMET mais adequada. Primeiro
ele usa latitude e longitude retornadas pela OpenWeather para escolher a estacao
mais proxima no catalogo local do INMET. Se as coordenadas nao estiverem
disponiveis, tenta encontrar uma estacao pelo nome da cidade. Depois carrega o
historico da estacao no intervalo selecionado e executa o `HistoricalAnalyzer`.

Na tela, o sistema prioriza o resultado principal da consulta:

- `Resultado da analise`, com a conclusao supervisionada e a explicacao curta;
- `Evidencias usadas`, com os valores meteorologicos e a referencia historica;
- `Detalhes tecnicos`, com regras acionadas, estatisticas historicas, features
  do modelo, probabilidades por classe e comparacao de desempenho dos modelos
  treinados quando houver relatorios locais.

A estacao manual continua disponivel em `Opcao avancada: alterar estacao INMET
manualmente`. Ela serve para corrigir a associacao automatica ou para escolher
outra estacao de referencia.

No modo `Entrada manual`, nao ha coordenadas automaticas da cidade. Por isso, o
dashboard solicita a estacao INMET de referencia para comparacao historica. O
historico e carregado automaticamente apos a selecao.

O app mostra nome da estacao, UF, codigo, altitude quando disponivel, distancia
aproximada ate a cidade atual quando a associacao usa coordenadas, periodo
carregado, quantidade de registros e fonte usada (`DuckDB` ou `ZIPs locais`).
Quando ha dados atuais e historico, o `HistoricalAnalyzer` calcula anomalias
estatisticas e o dashboard deixa os detalhes tecnicos em abas:

- anomalias historicas identificadas, com tipo, severidade, valor atual,
  referencia historica e justificativa;
- percentis e estatisticas historicas em expander tecnico;
- detalhes tecnicos da pressao atmosferica em um expander.

Essa comparacao historica complementa o diagnostico principal. Ela ajuda a
explicar se o dado atual esta fora do padrao observado para a estacao. A
predicao ML aparece no resultado principal e as regras ficam como
explicabilidade tecnica. Nenhuma dessas camadas substitui alertas oficiais.

A pressao segue o mesmo cuidado de referencial descrito acima: o historico do
INMET usa pressao ao nivel da estacao. Quando a OpenWeather fornece
`grnd_level`, o sistema usa esse valor como `pressure_station_hpa`; quando nao
fornece, o sistema estima a pressao ao nivel da estacao com a altitude da
estacao INMET. A tela de detalhes mostra a pressao ao nivel do mar, quando
existir, e a pressao de estacao usada na comparacao. Assim, a comparacao nao
usa diretamente a pressao ao nivel do mar contra a pressao historica da
estacao.

### Demonstracao do Dashboard

Roteiro curto para apresentacao do TCC:

1. Abrir o dashboard e apresentar o objetivo: dado atual por cidade,
   comparacao com historico INMET e Machine Learning supervisionado.
2. Explicar as fontes: OpenWeather como dado atual e INMET como base historica.
3. Informar uma cidade, por exemplo `Manaus`, selecionar o periodo historico,
   por exemplo `2025` a `2026`, e clicar em `Buscar dados e comparar com
   historico`.
4. Apresentar `Resultado da analise` como conclusao principal.
5. Explicar a coluna `Por que esse resultado?`, destacando concordancia ou
   divergencia entre ML supervisionado e regras tecnicas.
6. Mostrar `Evidencias usadas` para localizar os valores meteorologicos,
   estacao INMET e anomalia principal.
7. Abrir `Detalhes tecnicos` apenas se a banca pedir metricas dos modelos,
   probabilidades, features, regras, percentis ou variaveis brutas.
8. Reforcar a observacao metodologica: a predicao ML reproduz rotulos tecnicos
   derivados de regras e nao substitui alertas oficiais.

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

No dashboard, use o formulario principal:

- `Cidade`: cidade consultada na OpenWeather.
- `Pais`: codigo do pais, como `BR`.
- `Ano inicial do historico`: inicio do intervalo INMET.
- `Ano final do historico`: fim do intervalo INMET.
- `Fonte dos dados atuais`: `OpenWeather` por padrao ou `Entrada manual` como
  opcao secundaria.

O dashboard nao carrega historico antes do clique no botao. Se o usuario buscar
`Manaus` com periodo `2025` a `2026`, a consulta ao DuckDB ou aos ZIPs locais
usara esse intervalo. Se buscar `2019` a `2026`, o historico sera recarregado
com esse novo periodo. Se o ano inicial for maior que o ano final, o app mostra
um erro amigavel e nao executa a busca.

Na tela principal, os detalhes tecnicos ficam em expanders: variaveis brutas,
regras acionadas, percentis, pressao normalizada, registros historicos
carregados, features usadas na predicao ML e probabilidades por classe quando
disponiveis. Os dados historicos nao geram alerta climatico diretamente.

## Testes

```powershell
python -m pytest
```
