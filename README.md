# components_yolov4

## Installation
### Installing from source

For normal training and evaluation we should install the package from source using colab environment.

```bash
git clone https://github.com/PhongPX1603/components_yolov4.git
cd components_yolov4/
pip install -U PyYAML
pip install -r requirements.txt
```

#### Download pretrained weights

| Model | Test_size | AP50 | cfg | weights |
| ---   | ---       |     ---  |  ---|   ---   |
| YOLOv4 | 640 |  74.7%    | [cfg](https://drive.google.com/file/d/1FWggnicui0lNfPb34nbYP2uZE_6BooIM/view?usp=sharing) | [weights](https://drive.google.com/file/d/11Cy4QNBRZhtGfRVoF5IKwnVxmUL5dlCk/view?usp=sharing) |
| YOLOv4-tiny | 640 |  62.4%    | [cfg](https://drive.google.com/file/d/1FaxNbf1iGsDx2FFo2Lr4xpDSdFocAs4e/view?usp=sharing) | [weights](https://drive.google.com/file/d/10_WhvSrYQaciyATPG5fQ7mdnM14MkSc5/view?usp=sharing) |


## Project Structure
```
    detect_electric_components
                    |
                    ├── cfg
                    |	  ├── yolov4-tiny.cfg
                    |     └── yolov4.cfg
                    |
                    ├── data
                    |	  ├── components.data  
                    |	  ├── components.names 
                    |	  ├── components.yaml 
                    |	  └── hyp.scratch.yaml     
                    |
                    ├── inference
                    |	  ├── output
                    |	  ├── config.yaml
                    |	  ├── detector.py
                    |	  ├── inference.py
                    |	  ├── model.py
                    |     └── util.py
                    |
                    ├── model
                    |	  ├── export.py
                    |     └── models.py
                    |
                    ├── utils
                    |
                    ├── train.py 
                    └── test.py
```

## Dataset
* Data more 200 "Electric Components" images.
* Data divided into 2 parts: train and valid folders.

| Name  | Train | Valid | Test | Label's Format |
| ---   | ---         |     ---      |  --- |   --- |
| Electric Components | 167 |  39    |  ---   | txt    |


## How to Run
### Clone github
* Run the script below to clone my github.
```
git clone https://github.com/PhongPX1603/detect_electric_components.git
```

### Training
* Dataset structure
```
dataset
    ├── train
    │   ├── img1.jpg
    │   ├── img1.txt
    |   ├── img2.jpg
    |   ├── img2.txt
    │   └── ...
    │   
    └── valid
        ├── img1.jpg
        ├── img1.txt
        ├── img2.jpg
        ├── img2.txt
        └── ...
```
* Change your direct of dataset folder in ```cfg/yolov4.yaml```
* Run the script below to train the model:
```python train.py --cfg cfg/yolov4.cfg --device 0 --batch-size 4 --img 640 --data components.yaml --name components_tiny```


### Test
* After train, we have weights files. Use best.pt weights file to test model.
* Run the script below to test the model:
```python test.py --img 640 --conf 0.001 --batch 8 --device 0 --data components.yaml --cfg cfg/yolov4.cfg --weights weights/best.pt```


## Inference
* You can use this script to make inferences on particular folder
* Result are saved at <output/img.jpg> if type inference is 'image' or <video-output.mp4> with 'video or webcam' type.
```
cd inference
python inference.py --type-inference 'image' --input-dir <image dir> --video-output <video_output.mp4>
                                     'video'             <video dir>
                                     'webcam'            0
```


## Contributor
*Xuan-Phong Pham*
