# chute-certo

Predição de resultados do Brasileirão Série A usando machine learning.

## Como funciona

Dois workflows rodam automaticamente no GitHub Actions:

| Workflow | Frequência | O que faz |
|----------|-----------|-----------|
| `predict.yml` | Diário (5h UTC) | Baixa dados → avalia rodada anterior → prediz próxima rodada |
| `train.yml` | Semanal (segunda, 5h UTC) | Baixa dados → retreina e registra o modelo no MLflow |

As predições ficam salvas no [Turso](https://turso.tech) (SQLite hospedado) e o modelo é versionado no [DagsHub/MLflow](https://dagshub.com/igorkuivjogi/chute-certo).

## Modelo

- **Features**: média móvel dos últimos jogos (pontos, gols marcados, gols sofridos) para mandante e visitante
- **Algoritmo**: Regressão Logística com `StandardScaler`
- **Validação**: walk-forward cross-validation por rodada (sem vazamento temporal)
- **Saída**: probabilidades H / D / A + resultado previsto

## Estrutura

```
scripts/
  download_data.py     # Baixa fixtures da temporada via football-data.org
  train_model.py       # Treina, avalia com CV e registra modelo no MLflow
  predict_upcoming.py  # Prediz a próxima rodada e salva no Turso
  evaluate_round.py    # Preenche resultados reais após cada rodada
  serve.py             # Sobe a API FastAPI localmente

src/chute_certo/
  ingestion/           # Coleta e parsing de dados da API
  features/            # Engenharia de features
  training/            # CV walk-forward e treinamento
  predictions/         # Leitura/escrita no Turso
  serving/             # API FastAPI (/predict, /predictions, /health)
```

## Rodando localmente

```bash
cp .env.example .env  # preencher as chaves
uv sync
uv run python scripts/download_data.py --seasons 2023 2024 2025 2026
uv run python scripts/train_model.py
uv run python scripts/predict_upcoming.py 2026
uv run python scripts/serve.py  # API em localhost:8000

# Predição pontual
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_form_points":1.8,"home_form_scored":1.5,"home_form_conceded":1.0,"away_form_points":1.2,"away_form_scored":1.1,"away_form_conceded":1.4}'

# Todas as predições salvas
curl localhost:8000/predictions
```

## Variáveis de ambiente

Ver `.env.example`. Secrets necessários no GitHub Actions: `API_KEY_FOOTBALL_DATA`, `MLFLOW_TRACKING_PASSWORD`, `TURSO_AUTH_TOKEN` + variable `TURSO_URL`.
