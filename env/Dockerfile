# Usar uma imagem base oficial de Python leve
FROM python:3.11-slim

# Definir metadados de reprodutibilidade para o artigo
LABEL maintainer="Eduarda Pereira, Gonçalo Ferreira, Gonçalo Magalhães"
LABEL project="DriftSense-PM"
LABEL version="2.0"

# Instalar dependências do sistema necessárias para compilar pacotes se preciso
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Definir o diretório de trabalho dentro do contentor
WORKDIR /app

# Copiar os requisitos e instalar as bibliotecas Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar toda a estrutura do projeto para o contentor
COPY . .

# Comando padrão por defeito: executa a matriz fatorial completa da Fase 2
CMD ["python", "scripts/CWRU_dataset/master_script.py", "--repetitions", "30"]