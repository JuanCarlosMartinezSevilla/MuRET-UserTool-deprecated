import numpy as np
import cv2
from tensorflow.keras.models import load_model
model = load_model('./MuRETPackage/DocumentAnalysis/DocumentAnalysis.h5')
#model = load_model(f'/content/models/{modelName}/{modelName}.h5')
ImagesToPredict = []

def processImage(path_to_image, model):
    image = cv2.imread(path_to_image)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image_array = np.array(gray_img)
    width = model.get_config()["layers"][0]["config"]["batch_input_shape"][1]
    height = model.get_config()["layers"][0]["config"]["batch_input_shape"][2]
    gray_img_resized = cv2.resize(gray_image_array,(width,height), interpolation=cv2.INTER_LINEAR)
    gray_img_resized = gray_img_resized.astype(np.float32)
    gray_img_resized = (255. - gray_img_resized) / 255.
    gray_img_resized = np.expand_dims(gray_img_resized, axis=2)
    return gray_img_resized

if __name__ == '__main__':

    pred = processImage('./dataset/SRC/bardi_lauroohi_A.jpg', model)
    ImagesToPredict.append(pred)

    ImagesToPredict = np.array(ImagesToPredict)
    prediction = model.predict(ImagesToPredict)

    for pred, img in zip(prediction, ImagesToPredict):
        cv2.imshow('a',img*255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('a',pred*255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()