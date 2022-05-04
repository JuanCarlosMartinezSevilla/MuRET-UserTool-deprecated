from SAE.autoencoder import getModel
from SAE.daConfig import DAConfig
import random
import json
import cv2
import numpy as np
import tensorflowjs as tfjs

def resizeImage(img):
    h = DAConfig.image_size
    return cv2.resize(img, (h, h), interpolation=cv2.INTER_LINEAR)

def dataGen(routes_dict):
    # Generador
    while True:
        gray_images = []
        binarized_images = []
        for _ in range(DAConfig.batch_size):
            #json | img
            json_path, img_path = random.choice(list(routes_dict.items()))

            with open(json_path) as json_file:
                data = json.load(json_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                orig_img = img
                img = np.array(img)
                img = np.zeros_like(img)

                if 'pages' in data:
                    for page in data['pages']:
                        if 'regions' in page:
                            for reg in page['regions']:
                                if reg['type'] == DAConfig.classes_to_predict:
                                    if 'bounding_box' in reg:
                                        top, left, bottom, right = reg['bounding_box']['fromY'], \
                                                                   reg['bounding_box']['fromX'], \
                                                                   reg['bounding_box']['toY'],   \
                                                                   reg['bounding_box']['toX']
                                        
                                        rf = DAConfig.reduction_factor
                                        top = int(top + ((bottom-top) * (rf/100)))
                                        bottom = int(bottom - ((bottom-top) * (rf/100)))

                                        img = cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 255) , -1 )
                    
                    img = resizeImage(img)
                    orig_img = resizeImage(orig_img)

                    img = img.astype(np.float32)
                    orig_img = orig_img.astype(np.float32)
                    

                    img /= 255
                    orig_img = (255. - orig_img) / 255.

                    img = np.expand_dims(img, axis=2)
                    orig_img = np.expand_dims(orig_img, axis=2)

                    binarized_images.append(img)
                    gray_images.append(orig_img)

        yield (np.array(gray_images), np.array(binarized_images))          

def main(routes_dict, args):
    
    epochs = DAConfig.epochs
    steps = len(routes_dict)//DAConfig.batch_size

    gen = dataGen(routes_dict)

    SAEmodel = getModel()

    print("\n--- Training process ---\n")

    SAEmodel.fit(
            gen,
            verbose=2,
            steps_per_epoch=steps,
            epochs=epochs)
        
    if args.h5:
        SAEmodel.save(f'{args.pkg_name}/document_analysis/document_analysis.h5')
    
    # Save model to use it with tensorflow.js
    tfjs.converters.save_keras_model(SAEmodel, f'{args.pkg_name}/document_analysis/tfjs/')


if __name__ == '__main__':
    prueba = {'dataset/json_agnostic_symbol_images5752039876947046400/3/292/13.jpg.json' : 'dataset/SRC/13.jpg' }
    dataGen(prueba)