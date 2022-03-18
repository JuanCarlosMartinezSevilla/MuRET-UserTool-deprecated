from utils import Utils
from DataAugmentation.file_manager import FileManager
import shutil
import os

class Main:

    def seeDir(args, aux_path):
        if not os.path.exists('./MuRETPackage'):
            os.mkdir('./MuRETPackage')

        if args[0] and os.path.exists(aux_path):
            shutil.rmtree(aux_path)

        if not os.path.exists(aux_path):
            os.mkdir(aux_path)  

        
            Utils.decompressFile()
            fileList = FileManager.listFilesRecursive(aux_path)
            #print(fileList)
            Utils.readJSONGetImagesFromUrl(fileList, True)
        
        fileList = FileManager.listFilesRecursive(aux_path)
        fileList = FileManager.createRoutesDict(fileList)
        return fileList


    def localMain(args):

        fileList = Main.seeDir(args, aux_path = './dataset')

        ## UNCOMMENT TO LAUNCH SAE
        #if args[1]:
            #Utils.callDataAug(1)
            #Utils.callSAE(fileList)

        if args[2]:
            #Utils.createStavesDataset(fileList)
            # Launch E2E
            Utils.callE2E(fileList)
        #
        #if args[3]:
        #    Utils.createSymbolsDataset(fileList)
        #    # Launch Symbol Classifier
        #
        #if args[4]:
        #    Utils.createHeightDataset(fileList)
        #    # Launch Height Classifier

    def ligatures(args):
        fileList = Main.seeDir(args, aux_path = './ligaturesDataset')
        Utils.callE2ELigatures(fileList)


if __name__ == '__main__':
    NewDatasetLoad = True
    DocumentAnalysis = True
    E2E = True
    SymbolAnalysis = True
    HeightAnalysis = True
    #
    args = [NewDatasetLoad, DocumentAnalysis, E2E, SymbolAnalysis, HeightAnalysis]
    #Main.localMain(args)
    Main.ligatures(args)
