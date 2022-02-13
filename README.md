# components_yolov4

## Installation
### Installing from source

For normal training and evaluation we should install the package from source using colab environment.

```bash
git clone https://github.com/PhongPX1603/components_yolov4.git
cd components_yolov4/
pip install -U PyYAML
```

#### Download pretrained weights

| Model | Test_size | AP50 | Cfg | Weights |
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
