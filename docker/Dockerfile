# FROM ubuntu:23.10
FROM python:3.9.17
# copy the whole code directory
COPY . /digits/
# RUN apt-get update
# RUN apt-get install -y python3 python3-pip
# RUN pip3 install -r /digits/requirements.txt

RUN pip3 install Flask

# need python
# no need for conda or venv
WORKDIR /digits

# VOLUME /digits/models
EXPOSE 5000
# requirements installation
# CMD ["python3","exp.py"]
ENV FLASK_APP=api/app.py

CMD ["flask","run", "--host=0.0.0.0"]


