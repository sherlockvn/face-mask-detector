setup: # install requirements
	pip3 install -r requirements.txt
train: # start training model
	python3 -m src.train.training_model.py
run:  # run real-time detection
	python3 -m src.test.open_cam.py
