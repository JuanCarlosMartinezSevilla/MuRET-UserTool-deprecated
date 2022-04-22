import pathlib
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import argparse
import numpy as np
import os
import shutil
import cv2


from CRNN.utils_crnn import UtilsCRNN as U
import json

from DataAugmentation.file_manager import FileManager
from SymbolClassifier.SymbolDataGenerator import SymbDG
from utils import Utils


def load_model_aux(path):
    model = load_model(path)
    return model

def load_data(path):
    # get list of files
    file_list = FileManager.listFilesRecursive(path)
    Utils.readJSONGetImagesFromUrl(file_list, 'testImages')
    file_list = FileManager.listFilesRecursive(path)
    routes_dict = FileManager.createRoutesDict(file_list)

    return routes_dict

def test_sc(model, routes_dict, selection, i2w):

    #img_height_g = 40 #img_height_p = 224 #img_width_g = 40 #img_width_p = 112

    X_g, X_p, Y_g, Y_p, _, _ = SymbDG.parse_files(routes_dict)
    

    #position classifier
    if selection:
        counter = 0
        for idx, img in enumerate(X_p):

            aux = SymbDG.resize(img, 224, 112)
            aux = np.expand_dims(aux, axis=-1)
            aux = np.expand_dims(aux, axis=0)
            pred = model.predict(aux)
            # Its onehot encoding
            best_cat = np.argmax(pred)
            
            cat_name = i2w[f'{best_cat}']
            

            #Utils.printCV2(img, f"Tag: {Y_p[idx]} | Pred: {cat_name}")
            if Y_p[idx] == cat_name:
                print(f"Tag: {Y_p[idx]} | Pred: {cat_name} ✓✓✓")
                counter += 1
            else:
                print(f"Tag: {Y_p[idx]} | Pred: {cat_name}")
        print(f"Number of good predictions: {counter} out of {len(X_p)} --> {counter/len(X_p)}")


    #symbol/glyph classifier
    else:
        counter = 0
        for idx, img in enumerate(X_g):
            aux = SymbDG.resize(img, 40, 40)
            aux = np.expand_dims(aux, axis=-1)
            aux = np.expand_dims(aux, axis=0)
            pred = model.predict(aux)
            print(pred)
            print(best_cat)
            input()
            # Its onehot encoding
            best_cat = np.argmax(pred)

            cat_name = i2w[f'{best_cat}']
            
            #Utils.printCV2(img, f"Tag: {Y_g[idx]} | Pred: {cat_name}")
            if Y_g[idx] == cat_name:
                print(f"Tag: {Y_g[idx]} | Pred: {cat_name} ✓✓✓")
                counter += 1
            else:
                print(f"Tag: {Y_g[idx]} | Pred: {cat_name}")
        print(f"Number of good predictions: {counter} out of {len(X_g)} --> {counter/len(X_g)}")

def test_e2e(model, routes_dict):
    print()

def test_e2el(model, routes_dict):
    print()

def test_da(model, routes_dict):
    print()

def load_dict(path):
    with open(path) as json_file:
        i2w = json.load(json_file)
    return i2w

def main(args):

    model = load_model_aux(args.model_path)
    routes_dict = load_data(args.data_path)    
   
    if args.doc_analysis:
        print('Testing document analysis model\n')
        test_da(model, routes_dict)
    if args.end_to_end:
        print('Testing e2e model\n')
        i2w = load_dict(args.dictionary)
        test_e2e(model, routes_dict, i2w)
    if args.end_to_end_ligatures:
        print('Testing e2e ligatures model\n')
        i2w = load_dict(args.dictionary)
        test_e2el(model, routes_dict, i2w)
    if args.symb_classifier_symb:
        print('Testing symbol classifier\n') 
        i2w = load_dict(args.dictionary)
        print(i2w)
        test_sc(model, routes_dict, False, i2w)
    if args.symb_classifier_pos:
        print('Testing position classifier\n')
        i2w = load_dict(args.dictionary)
        test_sc(model, routes_dict, True, i2w)

    shutil.rmtree('testImages/SRC')


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-data_path", "--data_path", required=True, type=pathlib.Path,
                            help="Path to folder with data to test.")
    parser.add_argument("-model_path", "--model_path", required=True, type=pathlib.Path,
                            help="Path to model's h5 file.")
    parser.add_argument('-da', '--doc_analysis', action='store_true',
                            help='Test a document analysis model.')
    parser.add_argument('-e2e', '--end_to_end', action='store_true',
                            help='Test an agnostic end to end model.')
    parser.add_argument('-e2el', '--end_to_end_ligatures', action='store_true',
                            help='Test an agnostic end to end model for ligatures recognition.')
    parser.add_argument('-scpos', '--symb_classifier_pos', action='store_true',
                            help='Test a position classifier model.')
    parser.add_argument('-scsymb', '--symb_classifier_symb', action='store_true',
                            help='Test a symbol classifier model.')
    parser.add_argument('-dict', '--dictionary', type=pathlib.Path,
                            help='Path to dictionary to decode.')

    return parser


if __name__ == '__main__':

    #746,117,777,215

    parser = argument_parser()
    args = parser.parse_args()

    print(args)

    main(args)
