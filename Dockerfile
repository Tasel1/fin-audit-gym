FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt requirements-dev.txt pyproject.toml /app/
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

COPY . /app
RUN pip install --no-cache-dir -e .

CMD ["pytest", "tests/", "-v"]
