FROM python:3.10-slim

WORKDIR /snake

# Install make wget gcc
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && \
    apt install -y \
    make wget gcc && \
    apt-get clean all && \
    rm -rf /var/lib/apt/lists/*

# Install Golang 
ARG GO_VERSION=1.18.9
RUN wget -nv https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz && \
    rm -rf go${GO_VERSION}.linux-amd64.tar.gz && \
    ln -s /usr/local/go/bin/go /usr/local/bin/go

# Setup Python dependencies
RUN pip install poetry
ADD poetry.lock pyproject.toml ./
RUN poetry add gymnasium matplotlib torch
RUN poetry install

# Compile the Golang battlesnake
ADD Makefile ./
ADD ./battlesnake ./battlesnake
RUN pwd && ls
RUN make compile

# Setup the pettingzoo env
ADD setup.py .
ADD pz_battlesnake ./pz_battlesnake
RUN poetry run python ./setup.py install

# Run the demo
ADD dqn_solo_demo.py .
CMD ["poetry", "run", "python", "dqn_solo_demo.py"]
