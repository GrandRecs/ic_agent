FROM continuumio/miniconda3:latest

WORKDIR /usr/src/llm
COPY . .

RUN conda install -y cuda -c nvidia && \
    apt update -y && \
    apt install --upgrade python3 python3-pip git -y && \
    python3 -m pip install pip && \
    pip install -r requirements.txt --no-cache-dir --no-build-isolation && \
    pip install flash-attn==2.5.6 --no-build-isolation

EXPOSE 50051
CMD python3 llmserver.py
