from distutils import filelist
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

        Utils.callDataAug()

        Utils.callSAE()


        # Prepare data
        # SAE



if __name__ == '__main__':
    Main.localMain()
