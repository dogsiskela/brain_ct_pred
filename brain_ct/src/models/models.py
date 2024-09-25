import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from segmentation_models import Unet
from segmentation_models.metrics import iou_score
from segmentation_models import get_preprocessing



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


def predict(x_data):
    predictions = model.predict(x_data, verbose=1)
    return predictions
