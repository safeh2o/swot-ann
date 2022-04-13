FROM python:3.8

WORKDIR /app

COPY . /app

RUN ["python", "setup.py", "install"]
