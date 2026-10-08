FROM python:3.12-slim
WORKDIR /workspace
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
RUN python -m pip install --no-cache-dir -e '.[dev]'
CMD ["python", "-m", "pytest", "-q"]
