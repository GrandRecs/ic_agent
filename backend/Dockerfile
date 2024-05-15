FROM ubuntu:22.04

WORKDIR /usr/src/app
COPY . .

RUN apt update -y &&\
    apt install --upgrade python3.10 python3-pip -y &&\
    python3 -m pip install pip &&\
    pip install -r requirements.txt --no-cache-dir --force-reinstall

EXPOSE 65500
CMD python3 api.py
