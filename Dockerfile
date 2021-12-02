FROM tensorflow/tensorflow
RUN apt update -y & apt upgrade -y
RUN apt-get update -y & apt-get upgrade -y
RUN apt-get install zip unzip -y --fix-missing

COPY . .
RUN pip3 install -r requirements.txt

RUN python3 training_model.py

RUN echo "Docker image has been built!"
