FROM ubuntu:latest

WORKDIR /usr/src/llm
COPY . .

RUN apt update -y && apt install --upgrade python3 python3-pip -y && python3 -m pip install pip && pip install -r requirements.txt --upgrade

EXPOSE 50051
CMD python3 llmserver.py
