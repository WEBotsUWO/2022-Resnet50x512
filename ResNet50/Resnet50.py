import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import numpy as np
import cv2
from tensorflow.python.framework.ops import get_to_proto_function

class Resnet50:

    DATA_DIR = os.path.join(os.getcwd(), 'data')
    MODELS_DIR = os.path.join(DATA_DIR, 'models')
    MODELS_DIR = './data/models/'
    MODEL_DATE = '20200711'
    MODEL_NAME = 'centernet_resnet50_v2_512x512_coco17_tpu-8'
    MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
    PATH_TO_MODEL_TAR = os.path.join(MODELS_DIR, MODEL_TAR_FILENAME)
    PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
    PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
    LABEL_FILENAME = 'mscoco_label_map.pbtxt'
    PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
    PATH_TO_CFG='./data/models/centernet_resnet50_v2_512x512_coco17_tpu-8/pipeline.config'

    def __init__(self):
        self.configs=None
        self.model_config=None
        self.detection_model =None
        self.ckpt = None

    # Enable GPU dynamic memory allocation
    def gpuSet(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Load pipeline config and build a detection model
    def pipelineConfig(self):
        self.configs = config_util.get_configs_from_pipeline_file(self.PATH_TO_CFG)
        self.model_config = self.configs['model']
        self.detection_model = model_builder.build(model_config=self.model_config, is_training=False)

    # Restore checkpoint
    def setckpt(self):
        self.ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        self.ckpt.restore(os.path.join(self.PATH_TO_CKPT, 'ckpt-0')).expect_partial()

    ## CAT INDEX
    def catIndex(self):
        self.category_index = label_map_util.create_category_index_from_labelmap(self.PATH_TO_LABELS,
                                                                        use_display_name=True)
    def setup(self):
        self.gpuSet()
        self.pipelineConfig()
        self.setckpt()
        self.catIndex()
    
    def detect_fn(self,image):
        """Detect objects in image."""

        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

if __name__=="__main__":

    r = Resnet50()
    r.setup()

    ## RUN
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame from camera
        ret, image_np = cap.read()

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Things to try:
        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()

        # Convert image to grayscale
        # image_np = np.tile(
        #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = r.detect_fn(input_tensor)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
            detections['detection_scores'][0].numpy(),
            r.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

        # Display output
        cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()