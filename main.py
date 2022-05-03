from utils import Utils
from DataAugmentation.file_manager import FileManager
import shutil
import os
import argparse
import errno
from messages import Messages
import shutil
import pathlib


class Main:

    def create_package_tree(args):
        os.mkdir(args.pkg_name)
        if args.doc_analysis:
            os.makedirs(f'{args.pkg_name}/document_analysis/tfjs')
        if args.end_to_end:
            os.makedirs(f'{args.pkg_name}/agnostic_end2end/tfjs')
        if args.end_to_end_ligatures:
            os.makedirs(f'{args.pkg_name}/agnostic_end2end_ligatures/tfjs')
        if args.symb_classifier:
            os.makedirs(f'{args.pkg_name}/agnostic_symbol_and_position_from_image/symbol/tfjs')
            os.makedirs(f'{args.pkg_name}/agnostic_symbol_and_position_from_image/position/tfjs')
            

    def seeDir(args, path_to_download_images):

        aux_path = args.path

        # Check if there is a MuRETPackage already
        if os.path.exists(args.pkg_name):
            print(f'\nErasing existing MuRETPackage with name: {args.pkg_name}')
            shutil.rmtree(args.pkg_name)
            # Comparison
        Main.create_package_tree(args)

        # To not retrieve the images again
        if args.reload:

            if os.path.exists(path_to_download_images):
                shutil.rmtree(path_to_download_images)
                os.mkdir(path_to_download_images)
            else:
                os.mkdir(path_to_download_images)

            # if we have multiple datasets
            all_paths_checker = all(os.path.exists(aux) for aux in aux_path)

            if all_paths_checker:
                                # path of the dataset, path to extract images
                Utils.decompressFile(aux_path, path_to_download_images)
                fileList = FileManager.listFilesRecursive(path_to_download_images)
                Utils.readJSONGetImagesFromUrl(fileList, path_to_download_images)

            else:
                raise FileNotFoundError( errno.ENOENT, os.strerror(errno.ENOENT), aux_path.split('/')[-1])
            
        fileList = FileManager.listFilesRecursive(path_to_download_images)
        fileList = FileManager.createRoutesDict(fileList)
        return fileList

    def local_main(args):

        path_to_download_images = 'dataset'

        if not os.path.exists(path_to_download_images):
            os.mkdir(path_to_download_images)

        Messages.welcome()

        fileList = Main.seeDir(args, path_to_download_images)

        if args.doc_analysis:
            Messages.using_document()
            
            new_images = args.new_images
            
            Messages.new_images(new_images)
            
            Utils.callDataAug(new_images, path_to_download_images)
            Utils.callSAE(args)

        if args.end_to_end:
            #Launch E2E
            Messages.e2e()
            Utils.callE2E(fileList, args)
            
        
        if args.end_to_end_ligatures:
            Utils.callE2ELigatures(fileList, args)
            
        if args.symb_classifier:
            Messages.sc()
            Utils.callSymbClassifier(fileList, args)
            

        Utils.save_folder_compressed(args)
        Messages.end(args)
        

    def argument_parser():
        parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                            formatter_class=argparse.RawDescriptionHelpFormatter)

        parser.add_argument("-p", "--path", required=True, nargs='+', type=pathlib.Path,
                                help="Path to dataset .tgz file.")
        parser.add_argument("-pkg", "--pkg_name", required=True, type=str,
                                help="Generated package's name.")
        parser.add_argument('-da', '--doc_analysis', action='store_true',
                                help='Train a document analysis model.')
        parser.add_argument('-e2e', '--end_to_end', action='store_true',
                                help='Train an agnostic end to end model.')
        parser.add_argument('-e2el', '--end_to_end_ligatures', action='store_true',
                                help='Train an agnostic end to end model for ligatures recognition.')
        parser.add_argument('-sc', '--symb_classifier', action='store_true',
                                help='Train a symbol classifier model.')
        parser.add_argument('-rl', '--reload', action='store_true',
                                help='Reload dataset.')
        parser.add_argument('-ni', '--new_images', action='store', type=int, default=0,
                                help='Number of new images.')
        parser.add_argument('-h5', '--h5', action='store_true',
                                help='Save models in .h5 format.')

        return parser

if __name__ == '__main__':

    parser = Main.argument_parser()
    args = parser.parse_args()

    print(args)

    Main.local_main(args)
