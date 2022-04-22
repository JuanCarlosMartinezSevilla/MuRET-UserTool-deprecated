from CRNN.data import DataGenerator
from CRNN.model import get_model
from CRNN.evaluator import ModelEvaluator
from CRNN.utils_crnn import UtilsCRNN as U
from CRNN.config import Config
import argparse
import logging
from sklearn.model_selection import train_test_split
import tensorflowjs as tfjs
import json

from description import Description, End2EndDescription
import sys

def save_dicts(dg, ligatures, args):
    aux = ''
    if ligatures:
        aux = '_ligatures' 
    
    with open(f'{args.pkg_name}/agnostic_end2end{aux}/i2w.json', 'w') as fp:
            json.dump(dg.i2w, fp)

    with open(f'{args.pkg_name}/agnostic_end2end{aux}/w2i.json', 'w') as fp:
        json.dump(dg.w2i, fp)

def split_data(fileList):
    print(f"\n ■ Number of images in the dataset: {len(fileList)}")
    aux = []
    [aux.append(k) for k in fileList.keys()]
    train, test = train_test_split(aux, test_size=0.2)
    test, val = train_test_split(test, test_size=0.5)

    train_dict = {name: fileList[f'{name}'] for name in train}
    val_dict = {name: fileList[f'{name}'] for name in val}
    test_dict = {name: fileList[f'{name}'] for name in test}
    
    return train_dict, val_dict, test_dict

def main(fileList, ligatures, args):

    description = End2EndDescription('agnostic_end2end', None, None, None, None, fileList)
    batch_size = 8

    #   We make this first so we get all the dataset symbols, if we split the data first
    #   we can lose some
    dg = DataGenerator(dataset_list_path=fileList,
                       aug_factor=3, # seq (1 5)
                       batch_size=batch_size,
                       num_channels=3,
                       width_reduction=8, ligatures=ligatures)

    # Save dictionaries
    save_dicts(dg, ligatures, args)

    train_dict, val_dict, test_dict = split_data(fileList)

    print("\n=== Train data ===")
    
    dg = DataGenerator(dataset_list_path=train_dict,
                       aug_factor=3, # seq (1 5)
                       batch_size=batch_size,
                       num_channels=3,
                       width_reduction=8, ligatures=ligatures)

    

    model_tr, model_pr = get_model(vocabulary_size=len(dg.w2i))

    #X_val, Y_val, _, _ = U.parse_lst(args.validation)
    #fileList = dict(itertools.islice(fileList.items(), 4))
    print("\n=== Validation data ===")
    if ligatures:
        X_val, Y_val, _, _ = U.parse_lst_dict_ligatures(val_dict)
    else:
        X_val, Y_val, _, _ = U.parse_lst_dict(val_dict)
    #evaluator_val = ModelEvaluator([X_val, Y_val], aug_factor=args.aug_test)
    evaluator_val = ModelEvaluator([X_val, Y_val], aug_factor=0)

    #X_test, Y_test, _, _ = U.parse_lst(args.test)
    print("\n=== Test data ===")
    if ligatures:
        X_test, Y_test, _, _ = U.parse_lst_dict_ligatures(test_dict)
    else:
        X_test, Y_test, _, _ = U.parse_lst_dict(test_dict)
    #evaluator_test = ModelEvaluator([X_test, Y_test], aug_factor=args.aug_test)
    evaluator_test = ModelEvaluator([X_test, Y_test], aug_factor=0)

    #if args.model:
    #    best_ser_val = 100
    best_ser_val = 100
    epochs = 150

    description.model_epochs = epochs
    description.batch = batch_size
    description.input_h = Config.img_height
    description.i2w = dg.i2w
    description.w2i = dg.w2i
    description.save_description()
    

    for super_epoch in range(epochs):
        print("Epoch {}/{}".format(super_epoch, epochs))
        model_tr.fit(dg,
                     steps_per_epoch=100,
                     epochs=1,
                     verbose=2)

        print(f"\tEvaluating...\tBest SER val: {best_ser_val:.2f}")
        ser_val = evaluator_val.eval(model_pr, dg.i2w)
        ser_test = evaluator_test.eval(model_pr, dg.i2w)
        print("\tEpoch {}\t\tSER_val: {:.2f}\tSER_test: {:.2f}\n".format(super_epoch, ser_val, ser_test))

        #if args.model:
        if ser_val < best_ser_val:
            print("\tSER improved from {} to {} --> Saving model.".format(best_ser_val, ser_val))
            best_ser_val = ser_val
            #model_pr.save_weights("model_weights.h5")
            if ligatures:
                if args.h5:
                    model_pr.save(f'{args.pkg_name}/agnostic_end2end/agnostic_end2end_ligatures.h5')
                # Save model to use it with tensorflow.js
                EndToEndLigatures = model_pr
                tfjs.converters.save_keras_model(EndToEndLigatures, f'{args.pkg_name}/agnostic_end2end/')
            else:
                if args.h5:
                    model_pr.save(f'{args.pkg_name}/agnostic_end2end/agnostic_end2end.h5')
                EndToEnd = model_pr
                tfjs.converters.save_keras_model(EndToEnd, f'{args.pkg_name}/agnostic_end2end/tfjs/')


def build_argument_parser():

    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-tr', '--train', action='store', required=True,
                        help='List of training samples.')
    parser.add_argument('-val', '--validation', action='store', required=True,
                        help='List of validation samples.')
    parser.add_argument('-ts', '--test', action='store', required=True,
                        help='List of test samples.')
    parser.add_argument('-a', '--aug_train', action='store', required=True,
                        type=int,
                        help='Augmentation factor during training.')
    parser.add_argument('-b', '--aug_test', action='store', required=True,
                        type=int,
                        help='Augmentation factor during validation and test.')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Turn on DEBUG messages.')
    parser.add_argument('-m', '--model', action='store_true',
                        help='Turn on saving best validation model.')
    return parser


if __name__ == '__main__':

    parser = build_argument_parser()
    args = parser.parse_args()

    print(args)

    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
