# Face mask detection application

## library needed:
 - tensorflow, numpy, cv2, keras, sci-ki learn,...

# set up enviroment in ubuntu
```
python3 setEnv.py setupEnv --yes --force-yes && mkvirtualenv dl4cv -p python3 && workon dl4cv && python3 setEnv.py setupVirEnv --yes --force-yes
```
# training the mask detector
train the network with datasets
```
python train_mask_detector.py --dataset dataset
```
# make face mask detection for the testing images 
```
python detect_mask_image.py --image examples/example_01.png
```

# detect mask in real-time
```
python detect_mask_video.py
```
