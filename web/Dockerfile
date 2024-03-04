FROM node:20.11.1-alpine3.19
WORKDIR /usr/src/app
COPY . .
RUN npm i && npm run build
EXPOSE 3000
CMD npm start
