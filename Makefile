setup: # install requirements
	pip3 install -r requirements.txt
train: # start training model
	python3 training_model.py
run:  # run real-time detection
	python3 open_cam.py
