FROM python:3.12-alpine

WORKDIR /app

COPY . .

RUN pip3 install poetry && poetry install --no-root

CMD ["poetry", "run", "python3", "src/main.py"]
