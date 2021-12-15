setup: # install requirements
	pip3 install -r requirements.txt
train: # start training model
	python3 src/train/training_model.py
run:  # run real-time detection
	python3 src/test/open_cam.py
