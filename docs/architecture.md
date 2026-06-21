# Arquitetura

O projeto segue uma arquitetura modular simples para apoiar o desenvolvimento
academico do TCC.

## Modulos

- `app`: interface Streamlit para visualizacao e entrada manual de dados.
- `src.config`: carregamento de variaveis de ambiente.
- `src.data`: clientes para fontes externas de dados meteorologicos.
- `src.processing`: limpeza e preparacao dos dados.
- `src.ml`: criacao, treinamento e avaliacao de modelos supervisionados.
- `src.alerts`: classificacao de risco e mensagens de alerta.

## Fluxo de Dados

1. Dados meteorologicos sao coletados ou informados manualmente.
2. Os dados sao normalizados para colunas padronizadas.
3. As variaveis sao tratadas antes do uso em modelos.
4. Modelos supervisionados classificam eventos ou riscos.
5. O dashboard apresenta os dados e o alerta gerado.
