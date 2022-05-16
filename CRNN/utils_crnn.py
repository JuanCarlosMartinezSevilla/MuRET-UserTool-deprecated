import numpy as np
import itertools
from CRNN.augmentation import ImageModificator
import cv2
import json

import os
import shutil

class UtilsCRNN():

    def augmentation(image):
        mod = ImageModificator({ImageModificator.Contrast,
                                ImageModificator.Rotation,
                                ImageModificator.EroDila})

        return mod.apply(image)

    def resize(image, height):
        # Normalizing images
        width = int(float(height * image.shape[1]) / image.shape[0])
        return cv2.resize(image, (width, height))

    def normalize(image):
        return (255. - image) / 255.

    def greedy_decoding_aux(prediction, i2w):
        out_best = np.argmax(prediction, axis=1)
        print(out_best)
        out_best = [k for k, g in itertools.groupby(out_best[0])]
        print(out_best)
        return [i2w[f"{s}"] for s in out_best if s != len(i2w)]

    def greedy_decoding(prediction, i2w):
        out_best = np.argmax(prediction, axis=1)
        out_best = [k for k, g in itertools.groupby(list(out_best))]
        return [i2w[s] for s in out_best if s != len(i2w)]

    def listFiles(extension, path):
        result = []
        files = os.listdir(path)
        for f in files:
            if extension in f:
                result.append(f)
        
        return result

    def levenshtein(a,b):
        "Computes the Levenshtein distance between a and b."
        n, m = len(a), len(b)

        if n > m:
            a,b = b,a
            n,m = m,n

        current = range(n+1)
        for i in range(1,m+1):
            previous, current = current, [i]+[0]*n
            for j in range(1,n+1):
                add, delete = previous[j]+1, current[j-1]+1
                change = previous[j-1]
                if a[j-1] != b[i-1]:
                    change = change + 1
                current[j] = min(add, delete, change)

        return current[n]

    def parse_lst_dict_ligatures(lst_path: dict):

        print("Tamaño dataset ligatures: ", len(lst_path))

        X = []
        Y = []
        vocabulary = set()
        
        if not lst_path == None:
            for key in lst_path:
                json_path = key
                page_path = lst_path[key]
                with open(json_path) as json_file:
                    data = json.load(json_file)
                    image = cv2.imread(page_path, cv2.IMREAD_COLOR)
                    if 'ligatures' in data:
                        for l in data['ligatures']:
                            if 'bounding_box' in l:
                                top, left, bottom, right = l['bounding_box']['fromY'], \
                                                           l['bounding_box']['fromX'], \
                                                           l['bounding_box']['toY'],   \
                                                           l['bounding_box']['toX']
                            if 'symbols' in l:
                                symbols = l['symbols']
                                if len(symbols) > 0:
                                    X.append(image[top:bottom, left:right])

                                    gt = ['{}:{}'.format(s['agnostic_symbol_type'], s["position_in_staff"])
                                        for s in symbols]

                                    Y.append(gt)
                                    vocabulary.update(gt)

        w2i = {symbol: idx for idx, symbol in enumerate(vocabulary)}
        i2w = {idx: symbol for idx, symbol in enumerate(vocabulary)}

        print("{} samples loaded with {}-sized vocabulary".format(len(X), len(w2i)))        
        return X, Y, w2i, i2w

    def parse_lst_dict(lst_path: dict):


        path_to_save_crops = "./dataset_crops/e2e_crops"

        X = 0

        if os.path.exists(path_to_save_crops):
            shutil.rmtree(path_to_save_crops)
            os.makedirs(path_to_save_crops)
        else:
            os.makedirs(path_to_save_crops)
            
        vocabulary = set()

        print("--- Cropping images ---")

        if not lst_path == None:
            for file_num ,key in enumerate(lst_path):
                json_path = key
                page_path = lst_path[key]

                with open(json_path) as json_file:
                    data = json.load(json_file)
                    image = cv2.imread(page_path, cv2.IMREAD_COLOR)
                    #print(idx, image.shape)
                    if 'pages' in data:
                        for page_num, page in enumerate(data['pages']):
                            if 'regions' in page:
                                for reg_number, region in enumerate(page['regions']):
                                    if region['type'] == 'staff' and 'symbols' in region:
                                        symbols = region['symbols']
    
                                        if len(symbols) > 0:
                                            top, left, bottom, right = region['bounding_box']['fromY'], \
                                                                       region['bounding_box']['fromX'], \
                                                                       region['bounding_box']['toY'],   \
                                                                       region['bounding_box']['toX']
    
                                            img_x = image[top:bottom, left:right]

                                            if img_x.shape[0] != 0 and img_x.shape[1] != 0:
                                                #X.append(img_x)
                                                X = X + 1
                                                cv2.imwrite(f"{path_to_save_crops}/crop_{file_num+1}.{page_num+1}.{reg_number+1}.png", img_x)
    
                                                gt = ['{}:{}'.format(s['agnostic_symbol_type'], s["position_in_staff"])
                                                    for s in symbols]

                                                aux_dict = {"info": gt}
                                                with open(f"{path_to_save_crops}/crop_{file_num+1}.{page_num+1}.{reg_number+1}.json", 'w') as fp:
                                                    json.dump(aux_dict, fp)
                                                
                                                #Y.append(gt)
                                                vocabulary.update(gt)

        w2i = {symbol: idx for idx, symbol in enumerate(vocabulary)}
        i2w = {idx: symbol for idx, symbol in enumerate(vocabulary)}

        print("{} samples loaded with {}-sized vocabulary\n".format(X, len(w2i)))
        
        return w2i, i2w

    def appendSymbols(images):
        Y = []
        for img in images:
            img = img.replace('png', 'json')

            with open(f'./dataset_crops/e2e_crops/{img}') as json_file:
                data = json.load(json_file)
            Y.append(data['info'])

        return Y

    def parse_lst(lst_path):
        X = []
        Y = []
        vocabulary = set()

        lines = open(lst_path, 'r').read().splitlines()
        for line in lines:
            page_path, json_path = line.split()

            with open(json_path) as json_file:
                data = json.load(json_file)
                image = cv2.imread(page_path, cv2.IMREAD_COLOR)
                for page in data['pages']:
                    if 'regions' in page:
                        for region in page['regions']:
                            if region['type'] == 'staff' and 'symbols' in region:
                                symbols = region['symbols']

                                if len(symbols) > 0:
                                    top, left, bottom, right = region['bounding_box']['fromY'], region['bounding_box'][
                                        'fromX'], \
                                                            region['bounding_box']['toY'], region['bounding_box']['toX']
                                    X.append(image[top:bottom, left:right])

                                    gt = ['{}:{}'.format(s['agnostic_symbol_type'], s["position_in_staff"])
                                        for s in symbols]
                                    Y.append(gt)
                                    vocabulary.update(gt)

        w2i = {symbol: idx for idx, symbol in enumerate(vocabulary)}
        i2w = {idx: symbol for idx, symbol in enumerate(vocabulary)}

        print("{} samples loaded with {}-sized vocabulary".format(len(X), len(w2i)))
        return X, Y, w2i, i2w

