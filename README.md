## Bike tracker

## Getting started
### Install dependencies
#### Requirements
- PyTorch>=0.4.1
- torchvision>=0.2.1
- opencv-python>=3.4.2
- check requiremtns.txt
```
pip install -r requirements.txt
```
### Getting models and weights
- https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view (get craft_mlt_25k.pth and place it into bike_tracker/text-detection/weights)
- https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW (get TPS-ResNet-BiLSTM-Attn.pth and place it into bike_tracker/text-recognition/weights)
- https://pjreddie.com/media/files/yolov3.weights (get yolov3.weights and place it into bike_tracker/weights folder)
```
python load_weights.py
```

* Run project
``` (with python 3.7)
python webapp.py 
```


