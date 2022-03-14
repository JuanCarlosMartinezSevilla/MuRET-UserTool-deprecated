from utils import Utils
from DataAugmentation.file_manager import FileManager

import os

class Main:

    def localMain():

        if not os.path.exists('./MuRETPackage'):
            os.mkdir('./MuRETPackage')

        if not os.path.exists('./dataset'):    
            Utils.decompressFile()
            fileList = FileManager.listFilesRecursive('./dataset')
            #print(fileList)
            Utils.readJSONGetImagesFromUrl(fileList)
        
        fileList = FileManager.listFilesRecursive('./dataset')
        fileList = FileManager.createRoutesDict(fileList)

        ## UNCOMMENT TO LAUNCH DATA AUG
        Utils.callDataAug()

        ## UNCOMMENT TO LAUNCH SAE
        #Utils.callSAE()

        #Utils.createStavesDataset(fileList)
        #Utils.createSymbolsDataset(fileList)
        #Utils.createHeightDataset(fileList)


if __name__ == '__main__':
    Main.localMain()
