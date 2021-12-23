FROM tensorflow/tensorflow
RUN apt update -y & apt upgrade -y
RUN apt-get update -y & apt-get upgrade -y
RUN apt-get install zip unzip -y --fix-missing

# copy src
COPY . .

# install requirements
RUN pip3 install -r requirements.txt

# train
RUN python3 -m src.train.training_model.py

RUN echo "Docker image has been built!"
