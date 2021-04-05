
from numpy import expand_dims
from keras.models import load_model, Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from yolov3 import decode_netout, correct_yolo_boxes, do_nms, get_boxes, draw_boxes


def get_output_size(model, layer_id, img):
    sub_model = keras.models.Model(inputs=model.input, outputs=model.layers[layer_id].output)
    output = sub_model.predict(img)
    size = 1
    for x in output.shape:
        size *= x
    return size


# load and prepare an image
def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height


def model_partition(model, layer_id):
    up_model = keras.models.Model(inputs=model.input, outputs=model.layers[layer_id].output)
    down_model = K.function([model.layers[layer_id + 1].input], [model.outputs])

    return up_model, down_model


def device_calculate(up_model, img):
    intermid_output = up_model.predict(img)

    return intermid_output


def edge_calculate(down_model, intermid_outout):
    yhat = down_model([intermid_outout])

    return yhat


def show_boxes(yhat):
    # define the anchors
    anchors = [[116, 90, 156, 198, 373, 326],
               [30, 61, 62, 45, 59, 119],
               [10, 13, 16, 30, 33, 23]]

    # define the probability threshold for detected objects
    class_threshold = 0.2

    # define the labels
    labels = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    # summarize the shape of the list of arrays
    print([a.shape for a in yhat])
    boxes = list()
    for i in range(len(yhat)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)

    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    # suppress non-maximal boxes
    do_nms(boxes, 0.1)

    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
    # summarize what we found
    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])

    # draw what we found
    draw_boxes(photo_filename, v_boxes, v_labels, v_scores)


if __name__ == "__main__":
    candidate_layers = [0, 4, 8, 12, 16, 20]
    model = load_model('model/yolov3-tiny.h5')
    # keras.utils.plot_model(yolov3, show_shapes=True, to_file='model/yolov3.png')
    # keras.utils.plot_model(yolov3_tiny, show_shapes=True, to_file='model/yolov3_tiny.png')

    photo_filename = 'imgs/coco20/000000291619.jpg'
    image, image_w, image_h = load_image_pixels(photo_filename, (416, 416))
    model = model
    size_list = []
    can_list = []
    for layer_id in range(len(model.layers)):
        size_list.append(get_output_size(model, layer_id, image))
    for layer_id in candidate_layers:
        can_list.append(get_output_size(model, layer_id, image))
    plt.figure()
    plt.bar(height=size_list, x=range(len(size_list)), color='c', edgecolor='k', alpha=0.65, label='Other Layer')
    plt.bar(height=can_list, x=candidate_layers, color='r', edgecolor='k', alpha=0.65, label='Pooling Layer')
    plt.xlabel('Layer Index')
    plt.ylabel('Data Volume')
    plt.legend()
    plt.show()

    # define the expected input shape for the model
    input_w, input_h = 416, 416
    image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))

    # with partition
    """------------------------------------------------------"""
    # make prediction
    up_model, down_model = model_partition(model, layer_id=12)
    mid_output = device_calculate(up_model, image)
    yhat_edge = edge_calculate(down_model, mid_output)[0]
    show_boxes(yhat_edge)
    """------------------------------------------------------"""

    # original
    yhat = model.predict(image)
    show_boxes(yhat)






