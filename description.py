import json
from CRNN.config import Config
from SAE.daConfig import DAConfig

from SymbolClassifier.configuration import Configuration


def save_json(path, info, args):
    with open(f'{args.pkg_name}/{path}/description.json', 'w') as fp:
        json.dump(info, fp)


def end_to_end_description(args, train):
    info = {
        "name" : "agnostic_end2end",
        "img_h": Config.img_height,
        "channels": Config.num_channels,
        "batch_size": Config.batch_size,
        "epochs": Config.epochs,
        "files": train
    }
    save_json('agnostic_end2end', info, args)


def symbol_classifier_description(args, posFiles, glyphFiles):
    info_pos = {
        "name" : "agnostic_symbol_and_position_from_image_position",
        "img_h": Configuration.img_height_p,
        "img_w": Configuration.img_width_p,
        "batch_size": Configuration.batch_size,
        "epochs": Configuration.epochs,
        "files": posFiles
    }

    save_json('agnostic_symbol_and_position_from_image/position', info_pos, args)
    
    info_glyph = {
        "name" : "agnostic_symbol_and_position_from_image_glyph",
        "img_h": Configuration.img_height_g,
        "img_w": Configuration.img_width_g,
        "batch_size": Configuration.batch_size,
        "epochs": Configuration.epochs,
        "files": glyphFiles
    }
    
    save_json('agnostic_symbol_and_position_from_image/symbol', info_glyph, args)


def document_analysis_description(args):
    info = {
        "name" : "document_analysis",
        "img_h": DAConfig.image_size,
        "img_w": DAConfig.image_size,
        "reduction_factor": DAConfig.reduction_factor,
        "classes_to_predict": DAConfig.classes_to_predict,
        "batch_size": DAConfig.batch_size,
        "epochs": DAConfig.epochs
    }
    save_json('document_analysis', info, args)


if __name__ == "__main__":
    pass