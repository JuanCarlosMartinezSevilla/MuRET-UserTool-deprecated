from utils import Utils
from DataAugmentation.file_manager import FileManager


import os

class Main:

    def localMain(args):

        if not os.path.exists('./MuRETPackage'):
            os.mkdir('./MuRETPackage')

        if not os.path.exists('./dataset'):    
            Utils.decompressFile()
            fileList = FileManager.listFilesRecursive('./dataset')
            #print(fileList)
            Utils.readJSONGetImagesFromUrl(fileList)
        
        fileList = FileManager.listFilesRecursive('./dataset')
        fileList = FileManager.createRoutesDict(fileList)

        

        ## UNCOMMENT TO LAUNCH SAE
        if args[0]:
            Utils.callDataAug(10)
            Utils.callSAE()

        if args[1]:
            Utils.createStavesDataset(fileList)
            # Launch E2E
        
        if args[2]:
            Utils.createSymbolsDataset(fileList)
            # Launch Symbol Classifier

        if args[3]:
            Utils.createHeightDataset(fileList)
            # Launch Height Classifier



if __name__ == '__main__':
    DocumentAnalysis = True
    E2E = True
    SymbolAnalysis = True
    HeightAnalysis = True

    args = [DocumentAnalysis, E2E, SymbolAnalysis, HeightAnalysis]
    Main.localMain(args)
