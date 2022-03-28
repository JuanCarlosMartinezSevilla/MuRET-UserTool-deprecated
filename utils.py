import os
import tarfile
from tqdm import tqdm
import json
import urllib.request
from DataAugmentation.DataAugmentationGenerator import DataAugmentationGenerator
from SAE.SAE import SAE
from DataAugmentation.file_manager import FileManager
from CRNN.utils_crnn import UtilsCRNN
import cv2
from CRNN.experiment import main as CRNNMain
from SymbolClassifier.SymbolDataGenerator import SymbDG
import re

class Utils:

    def printCV2(X, Y, window_name, flag):
        for i, a in enumerate(X[:10]):
            if flag:
                print(Y[i])
            cv2.imshow(window_name, X[i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @staticmethod
    def callSymbClassifier(fileList, args):
        SymbDG.main(fileList, args)

    @staticmethod
    def callE2ELigatures(fileList, args):
                                # ligatures
        CRNNMain(fileList, True, args)

    @staticmethod
    def callE2E(fileList, args):
        CRNNMain(fileList, False, args)

    @staticmethod
    def callSAE(args):
        epochs = 10
        classes_to_predict = 'staff'
        image_size = 128
        batch_size = 8
        list_json_pathfiles = FileManager.listFilesRecursive('./dataset')
        routes_dict = FileManager.createRoutesDict(list_json_pathfiles)
        values = {"epochs":epochs, 
                  "classes": classes_to_predict,
                  "image_size": image_size,
                  "batch_size": batch_size,
                  "routes": routes_dict}
        generator = SAE.dataGen(routes_dict, batch_size, image_size, classes_to_predict)
        
        print("\n--- Training process ---\n")
        SAE.model(image_size, epochs, generator, len(routes_dict)//batch_size, args)

    @staticmethod
    def SymbolsHeightParse(lst_path: dict):
        X = []
        Y = []

        for key in lst_path:
            json_path = key
            page_path = lst_path[key]

            if not 'daugImage' in json_path:
                with open(json_path) as json_file:
                    data = json.load(json_file)
                    image = cv2.imread(page_path, cv2.IMREAD_COLOR)
                    for page in data['pages']:
                        if 'regions' in page:
                            for region in page['regions']:
                                if region['type'] == 'staff' and 'symbols' in region:
                                    symbols = region['symbols']

                                    if len(symbols) > 0:
                                        topR, leftR, bottomR, rightR = region['bounding_box']['fromY'], region['bounding_box']['fromX'], \
                                                                       region['bounding_box']['toY'], region['bounding_box']['toX']
                                        for s in symbols:
                                            if 'bounding_box' in s and 'position_in_staff' in s:
                                                top, left, bottom, right = s['bounding_box']['fromY'], s['bounding_box']['fromX'], \
                                                                           s['bounding_box']['toY'], s['bounding_box']['toX']
                                        
                                                X.append(image[topR:bottomR, left:right])
                                                Y.append(s['position_in_staff'])
                                            #if 'approximateX' in s and 'position_in_staff' in s:
                                            #    left, right = s['approximateX'], s['approximateX'] + 10
                                            #    X.append(image[topR:bottomR, left:right])
                                            #    Y.append(s['position_in_staff'])
                                                
        
        print("{} samples loaded".format(len(X)))
        return X, Y

    @staticmethod
    def createHeightDataset(fileList):

        X, Y = Utils.SymbolsHeightParse(fileList)

        Utils.printCV2(X, Y, 'Symbol height', True)

    @staticmethod
    def SymbolsParse(lst_path: dict):

        X = []
        Y = []

        for key in lst_path:
            json_path = key
            page_path = lst_path[key]

            if not 'daugImage' in json_path:
                
                with open(json_path) as json_file:
                    data = json.load(json_file)
                    image = cv2.imread(page_path, cv2.IMREAD_COLOR)
                    for page in data['pages']:
                        if 'regions' in page:
                            for region in page['regions']:
                                
                                if region['type'] == 'staff' and 'symbols' in region:
                                    symbols = region['symbols']

                                    if len(symbols) > 0:
                                        for s in symbols:
                                            if 'bounding_box' in s and 'agnostic_symbol_type' in s:
                                                #print("Reading symbols", json_path)
                                                top, left, bottom, right = s['bounding_box']['fromY'], s['bounding_box']['fromX'], \
                                                                        s['bounding_box']['toY'], s['bounding_box']['toX']
                                        
                                                X.append(image[top:bottom, left:right])
                                                Y.append(s['agnostic_symbol_type'])
                                                
        
        Ydiff = set(Y)
        print("{} samples loaded with {} different symbols".format(len(X), len(Ydiff)))
        return X, Y

    @staticmethod
    def createSymbolsDataset(fileList):
        X, Y = Utils.SymbolsParse(fileList)

        Utils.printCV2(X, Y, 'Symbol', True)
            
    @staticmethod
    def createStavesDataset(fileList):
        X, Y, w2i, i2w = UtilsCRNN.parse_lst_dict(fileList)
        #CRNNMain(fileList)
        
        Utils.printCV2(X, Y, 'Staff', False)

    @staticmethod
    def callDataAug(number_images):
        rotation = True, 
        vertical_resizing = True,
        rf = 0.2,
        window = 101,
        kernel = 0.2,
        angle = 3

        DataAugmentationGenerator.generateNewImageFromListByBoundingBoxesRandomSelectionAuto('./dataset', number_images, rotation, vertical_resizing, rf, window, kernel, angle)

    @staticmethod
    def getURLJSON(file, json_classes, path_to_save):

        with open(file, encoding="UTF-8") as f:
            json_read = json.load(f)

        erase = False
        if 'url' in json_read.keys():
            url = json_read['url']
            
            url_split = [True if 'localhost' in u else False for u in url.split('/')]
            
            if True in url_split:
                url = re.sub('(localhost:)\w+', 'muret.dlsi.ua.es/images', url)
                
            filename = json_read["filename"]
            urllib.request.urlretrieve(url, os.path.join(path_to_save, filename))
        else:
            erase = True
            
        if 'pages' in json_read:
            pages = json_read['pages']
            for p in pages:
                if 'regions' in p:
                    regions = p['regions']
                    for r in regions:
                        if r['type'] not in json_classes:
                            json_classes.append(r['type'])

        if erase:
            os.remove(file)
        return json_classes

    @staticmethod
    def getURLJSONLigatures(file, json_classes, path_to_save):

        with open(file, encoding="UTF-8") as f:
            json_read = json.load(f)

        url = orig_url = ""
        erase = False
        if 'url' in json_read.keys():
            url = json_read['url']
        else:
            erase = True
        if 'original' in json_read.keys():
            orig_url = json_read['original']
            
            if url == orig_url:
                filename = json_read["filename"]
                urllib.request.urlretrieve(orig_url, os.path.join(path_to_save, json_read["filename"]))
            elif url != "":
                urllib.request.urlretrieve(url, path_to_save)
            else:
                urllib.request.urlretrieve(orig_url, path_to_save)
            
        if 'pages' in json_read:
            pages = json_read['pages']
            for p in pages:
                if 'regions' in p:
                    regions = p['regions']
                    for r in regions:
                        if r['type'] not in json_classes:
                            json_classes.append(r['type'])
                    #print(json_classes)
            
            #print(f"File {json_read['filename']} saved in {path_to_save}")
        if erase:
            os.remove(file)
        return json_classes

    @staticmethod
    def readJSONGetImagesFromUrl(files, path):

        path_to_save = os.path.join(path, 'SRC')

        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
        
        print("\n---- Fetching images from URLs ----\n")
        print(f"Saving images in {path_to_save} \n")
        json_classes = []
        for f in tqdm(files):
            json_classes = Utils.getURLJSON(f, json_classes, path_to_save)
        print(f'\nImporting finished, images saved in: {path_to_save}')

    @staticmethod
    def decompressFile (aux_path, path ):  

        #tar_file = "./capitan.tgz"
        tar_file = aux_path
        
        print("\nExtracting from .tgz file \n")

        tar = tarfile.open(tar_file, mode="r:gz")
        
        members = tar.getmembers()

        for member in members:
            tar.extract(member, path=path)            
        tar.close()

        print(f"\nFiles extracted in {path} ...\n")
        return True
