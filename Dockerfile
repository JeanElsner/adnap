FROM python:3.7-slim

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -q
RUN apt-get install --no-install-recommends --quiet --yes\
    build-essential libeigen3-dev libpoco-dev

COPY . .
RUN pip install -e .
ENV PANDA_MODEL_PATH ./libfrankamodel.linux_x64.so
