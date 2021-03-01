
use this for testing.
python TFLite_detection_video.py --labels ../data/coco_labels.txt --threshold 0.5 --modeldir model/ --graph detect.tflite   --video imgs/vid2/ILSVRC2017_val_00685000.mp4 


edge:
python3 receive_and_inference.py --modeldir model/ --graph detect.tflite 
offload:
python offload_and_display.py --labels ../data/coco_labels.txt --threshold 0.5 --modeldir model/ --graph detect.tflite   --video imgs/vid2/ILSVRC2017_val_00685000.mp4 



ignore the following:
python TFLite_detection_video.py --modeldir model/ --video imgs/vid2/vdo.avi  --graph detect.tflite  --labels ../data/coco_labels.txt

python TFLite_detection_image.py --modeldir model/ --imagedir ./imgs/coco20  --graph detect.tflite  --labels ../data/coco_labels.txt 

python TFLite_detection_video.py --labels ../data/coco_labels.txt --threshold 0.5 --modeldir model/ --graph ssd_mobilenet_v1_1_metadata_1.tflite   --video imgs/vid2/ILSVRC2017_val_00430000.mp4 

python TFLite_detection_video.py --labels ../data/coco_labels.txt --threshold 0.5 --modeldir model/ --graph detect.tflite   --video imgs/vid2/ILSVRC2017_val_00430000.mp4 

tflite_convert  --graph_def_file=model/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb --output_file=tflite/detect.tflite  --output_format=TFLITE  --input_shapes=1,300,300,3  --input_arrays=normalized_input_image_tensor  --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  --inference_type=QUANTIZED_UINT8  --mean_values=128 --std_dev_values=127  --change_concat_input_ranges=false  --allow_custom_ops --enable_v1_converter

tflite_convert  --graph_def_file=model/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb --output_file=tflite/detect.tflite  --output_format=TFLITE    --input_arrays=input  --allow_custom_ops --enable_v1_converter

tflite_convert  --output_file=face_ssd.tflite  --graph_def_file=model/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb  --inference_type=QUANTIZED_UINT8  --input_shapes=1,320,320,3  --input_arrays normalized_input_image_tensor  --output_arrays "TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3"  --mean_values 128  --std_dev_values 128  --allow_custom_ops  --change_concat_input_ranges=false  --allow_nudging_weights_to_use_fast_gemm_kernel=true
