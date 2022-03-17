import numpy as np
import cv2
from tensorflow.keras.models import load_model


def processImage(path_to_image, model):
    image = cv2.imread(path_to_image)
    
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    gray_image_array = np.array(gray_img)
    width = model.get_config()["layers"][0]["config"]["batch_input_shape"][1]
    height = model.get_config()["layers"][0]["config"]["batch_input_shape"][2]
    #print(width, height)
    gray_img_resized = cv2.resize(gray_image_array,(width,height), interpolation=cv2.INTER_LINEAR)
    gray_img_resized = gray_img_resized.astype(np.float32)
    
    gray_img_resized = (255. - gray_img_resized) / 255.
    gray_img_resized = np.expand_dims(gray_img_resized, axis=2)
    return gray_img_resized

def run(model):
    #model = load_model(f'/content/models/{modelName}/{modelName}.h5')

    ImagesToPredict = []
    pred = processImage('./dataset/SRC/12698.JPG', model)
    ImagesToPredict.append(pred)

    ImagesToPredict = np.array(ImagesToPredict)
    prediction = model.predict(ImagesToPredict)

    for pred, img in zip(prediction, ImagesToPredict):
        background = img
        overlay = pred

        added_image = cv2.addWeighted(background,1,overlay,0.7,0)

        cv2.imshow('GTImage', added_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def getConfig(model):
    print(model.get_config())

if __name__ == '__main__':
    model = load_model('./MuRETPackage/DocumentAnalysis/DocumentAnalysis.h5')
    
    run(model)
    #getConfig(model)