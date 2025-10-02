# ---- Base Python + Poetry ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.3

# Dependências de sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# Poetry
RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

# ---- Camada de deps (cache mais eficiente) ----
COPY pyproject.toml ./
# Se você tiver poetry.lock, descomente a linha abaixo:
# COPY poetry.lock ./

# Instala deps no sistema (sem venv do Poetry)
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi

# ---- Código da aplicação ----
COPY src ./src
COPY app.py ./
COPY README.md ./

# Diretório para memória/cache do agente
RUN mkdir -p .cache

# Porta padrão (informativo; Railway injeta $PORT)
EXPOSE 8501

# IMPORTANTE: usar bash -c para expandir $PORT em runtime
CMD bash -c "poetry run streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT"
