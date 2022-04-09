FROM python:3.7.7

RUN apt-get update && apt-get install -y git python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip -r requirements.txt

WORKDIR /workdir

EXPOSE 8501

COPY src /workdir/

RUN echo "cache bust"

ENTRYPOINT ["python", "server.py", "serve"]