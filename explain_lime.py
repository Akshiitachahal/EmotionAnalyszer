import cv2
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from keras.models import load_model

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def explain_prediction(image, model_path='model/emotion_model.h5'):
    model = load_model(model_path)

    def predict_fn(images):
        images = np.array([cv2.resize(img, (48, 48))[:, :, 0] for img in images])
        images = images.reshape(images.shape[0], 48, 48, 1).astype("float32") / 255.0
        return model.predict(images)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image.astype('double'),
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.title("LIME Explanation")
    plt.axis('off')
    plt.show()
