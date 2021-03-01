# edge-computing

## Preparation 
### Data and Models
[download](https://gowustl-my.sharepoint.com/:u:/g/personal/ruiqi_w_wustl_edu/ETrKQ8R5LdJJpVz8biftCCwBKV8Q4Qd3F5YnVsI4O6r9yg?e=yTovB4) the zip (containing `<imgs/>` and `<model/>`), extract and put the folders in the folder.

### Dependencies 
**tensorflow (tflite_runtime):**
* On supported systems, install tflite_runtime: [link](https://www.tensorflow.org/lite/guide/python)
* Otherwise, install the normal tensorflow.

**open-cv:**
```
pip install opencv-contrib-python
```
* [opencv-python](https://pypi.org/project/opencv-python/)

## Test
```
python TFLite_detection_video.py --labels ../data/coco_labels.txt --threshold 0.5 --modeldir model/ --graph detect.tflite   --video imgs/vid2/ILSVRC2017_val_00685000.mp4 
```

## Run with Edge Computing
On the edge server:
```
python receive_and_inference.py --modeldir model/ --graph detect.tflite 
```
On the embedded system:
```
python offload_and_display.py --labels ../data/coco_labels.txt --threshold 0.5 --modeldir model/ --graph detect.tflite   --video imgs/vid2/ILSVRC2017_val_00685000.mp4 
```
