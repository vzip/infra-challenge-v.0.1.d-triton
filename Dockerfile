# Использование PyTorch образа с поддержкой CUDA и cuDNN в качестве базового
# FROM pytorch/pytorch:latest
# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubi8
FROM nvcr.io/nvidia/tritonserver:23.01-pyt-python-py3

# Обновление pip
RUN pip install --upgrade pip

RUN pip install tritonclient[all]

# Set CUDA environment variables
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    make \
    gcc \
    procps \
    lsof \
    vim \
    supervisor \
    nginx \
    linux-headers-$(uname -r)

COPY default /etc/nginx/sites-available/
COPY default /etc/nginx/sites-enabled/

WORKDIR /solution

COPY requirements.txt .

RUN pip install -r requirements.txt

# Создание директории для хранения моделей
RUN mkdir /models

# Создание структуры директорий для каждой из моделей
RUN mkdir -p /models/worker1/1
RUN mkdir -p /models/worker2/1
RUN mkdir -p /models/worker3/1
RUN mkdir -p /models/worker4/1
RUN mkdir -p /models/worker5/1

# Копирование конфигурационных файлов моделей
COPY model_configs/worker1/config.pbtxt /models/worker1/
COPY model_configs/worker2/config.pbtxt /models/worker2/
COPY model_configs/worker3/config.pbtxt /models/worker3/
COPY model_configs/worker4/config.pbtxt /models/worker4/
COPY model_configs/worker5/config.pbtxt /models/worker5/


COPY /dev_parallel/ .

COPY /gunicorn_conf.py .

# Копируйте файл конфигурации supervisord в контейнер
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Используйте CMD для запуска supervisord
CMD ["/usr/bin/supervisord"]
