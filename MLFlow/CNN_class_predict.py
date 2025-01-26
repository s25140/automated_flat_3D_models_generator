import sys
import os

from tensorflow import convert_to_tensor
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from local_package.ConvNN import preproces


def get_model(path):
    return load_model(path)

def predict(model, img_path):
    image = preproces(img_path)
    image = convert_to_tensor(image.reshape(1, 256, 256, 1))
    prediction = model.predict(image)
    return prediction[0][0]  # Return single value

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        print("Usage: python CNN_class_predict.py <path_to_image>")
        sys.exit(1)
    try:
        model = get_model("models/model_two_inputs_GPU.h5")
        prediction = predict(model, args[1])
        print(prediction)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)