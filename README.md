# API com Rede Neural para Previsão de Churn

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-black?logo=PyTorch)](https://pytorch.org)

## Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Documentação](#documentação)
- [Instalação Local](#instalação-local)
- 
- 
-

---

## Sobre o Projeto

Esta é a **Tech Challenge** da fase 2 - Big Data Architecture da Pós Tech em Engenharia de Machine Learning da FIAP.

O projeto se trata de uma API para inferência de modelos ML, com tema central: 
Rede Neural para Previsão de Notas em filmes
- 🔥**Multi-Layer Perceptron** com PyTorch para inferência

## Documentação
Consulte: 
- [Model Card](docs/ModelCard.md)

## Instalação Local

### Pré-requisitos

- Python 3.13+
- uv (como instalar [aqui](https://docs.astral.sh/uv/))
- Docker e Docker Compose
- Git

### Opção 1: Desenvolvimento Local (Python)

```bash
# Clone o repositório
git clone https://github.com/bluheart/fiap-tech-challenge-fase-2.git
cd fiap-tech-challenge-fase-2

# prepare o ambiente e dependencias
uv sync

# Ative o ambiente
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

dvc repro

#exemplo
python src/predict.py --ratings data/raw/ratings.csv --movies data/raw/movies.csv
```

### Opção 2: Docker Compose (Recomendado)

```bash
# Clone o repositório
git clone https://github.com/bluheart/fiap-tech-challenge-fase-2.git
cd fiap-tech-challenge-fase-2

# Suba toda a stack
docker-compose up -d

# script de predição, exemplo
docker exec ml-pipeline python src/predict.py --ratings data/raw/ratings.csv --movies data/raw/movies.csv

# shell interativa
docker exec -it ml-pipeline bash

# MLflow UI
docker exec -d ml-pipeline mlflow ui --host 0.0.0.0 --port 5000

# parar container
docker stop ml-pipeline

# remover container
docker rm ml-pipeline
```