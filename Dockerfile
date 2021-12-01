FROM tensorflow/tensorflow
RUN apt update & apt upgrade
RUN apt-get update & apt-get upgrade
RUN apt-get install zip unzip -y --fix-missing

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
RUN python3 training_model.py

RUN echo "create docker image successfully!"