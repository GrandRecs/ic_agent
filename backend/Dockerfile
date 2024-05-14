FROM python:3.10

WORKDIR /usr/src/backend
COPY . .

RUN python3 -m pip install pip && pip install -r requirements.txt

EXPOSE 65500
CMD python3 api.py
