FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# upgrade pip
RUN pip install --upgrade pip

# install poetry
RUN pip install poetry

# disable virtualenv for poetry
COPY ./pyproject.toml /app/pyproject.toml
RUN poetry config virtualenvs.create false

# install dependencies
RUN poetry install

# copy contents of project into docker
COPY ./model/ /app/model/

COPY ./pyproject.toml /app/pyproject.toml
COPY ./models/ /app/models/

COPY ./autoencoder_mnist/util.py /app/autoencoder_mnist/util.py
COPY ./autoencoder_mnist/config.py /app/autoencoder_mnist/config.py

COPY ./autoencoder_mnist/core /app/autoencoder_mnist/core
COPY ./autoencoder_mnist/server /app/autoencoder_mnist/server
COPY ./autoencoder_mnist/model/ /app/autoencoder_mnist/model


# set path to our python api file
ENV MODULE_NAME="autoencoder_mnist.server.api"
ENV PYTHONPATH=/app/autoencoder_mnist/