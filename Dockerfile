FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data

RUN pip install --no-cache-dir ".[serve]"

EXPOSE 8000

CMD ["uvicorn", "ragbench.server:app", "--host", "0.0.0.0", "--port", "8000"]
