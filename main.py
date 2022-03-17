from utils import Utils
from DataAugmentation.file_manager import FileManager
import shutil
import os

class Main:

    def localMain(args):

        if not os.path.exists('./MuRETPackage'):
            os.mkdir('./MuRETPackage')


        if not os.path.exists('./dataset') or args[0]:   

            if args[0] and os.path.exists('./dataset'):
                shutil.rmtree('./dataset')

            Utils.decompressFile()
            fileList = FileManager.listFilesRecursive('./dataset')
            #print(fileList)
            Utils.readJSONGetImagesFromUrl(fileList)
        
        fileList = FileManager.listFilesRecursive('./dataset')
        fileList = FileManager.createRoutesDict(fileList)

        

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



if __name__ == '__main__':
    NewDatasetLoad = True
    DocumentAnalysis = True
    E2E = True
    SymbolAnalysis = True
    HeightAnalysis = True

    args = [NewDatasetLoad, DocumentAnalysis, E2E, SymbolAnalysis, HeightAnalysis]
    Main.localMain(args)
