# from google.colab import drive

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from segmentation_models import get_preprocessing
from segmentation_models.metrics import iou_score
from segmentation_models import Unet

BACKBONE = "densenet121"
preprocess_input = get_preprocessing(BACKBONE)

model = Unet(BACKBONE, input_shape=(128, 128, 3),
             classes=1,
             encoder_weights='imagenet',
             activation='sigmoid')

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=[iou_score])

filepath = "model_no_aug.h5"

model.load_weights('./'+filepath)


def predict(test_generator, data_len):
    predictions = model.predict(test_generator,
                                steps=data_len,
                                verbose=1)
