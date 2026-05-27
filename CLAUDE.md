# CLAUDE.md

## Projeto

Predição de resultados do Brasileirão Série A. Python 3.12, `uv`, src layout.

## Decisões importantes

- **Sem vazamento temporal no CV**: o Brasileirão adia partidas com frequência. O split de treino usa `df["date"] < val_min_date`, não por número de rodada.
- **`compute_current_form` não usa `.shift(1)`**: é para predição futura, não treinamento — inclui o último jogo jogado.
- **`build_features` usa `.shift(1).rolling()`**: é para treinamento — exclui o jogo atual para evitar leakage.
- **Turso via HTTP**: usar `libsql://` como URL causa erro de WebSocket. Substituir por `https://` no cliente.
- **MLflow via `mlflow.set_tracking_uri()`**: não usar `dagshub.init()` — causa prompt interativo em CI.
- **Idempotência**: `has_predictions_for_round()` impede re-predição da mesma rodada em execuções diárias.

## Stack

- **Dados**: football-data.org (API v4, competição `BSA`)
- **Features**: 6 colunas de forma recente (pontos, gols marcados/sofridos) para mandante e visitante
- **Modelo**: `LogisticRegression` + `StandardScaler` via `make_models()` em `train.py`
- **Tracking**: MLflow no DagsHub (`brasileirao-predictor` no Model Registry)
- **DB predições**: Turso (SQLite hospedado, persiste entre runners efêmeros do GitHub Actions)
- **API**: FastAPI em `src/chute_certo/serving/api.py`

## Comandos úteis

```bash
uv run pytest                        # testes
uv run ruff check src/ scripts/      # lint
uv run ruff format src/ scripts/     # format
```

## Lint

`ruff` com regras E, F, I, W, UP, B. Line length 88. `pre-commit` configurado.
