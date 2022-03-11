from distutils import filelist
from utils import Utils
from DataAugmentation.file_manager import FileManager


class Main:

    def localMain():

        Utils.decompressFile()
        fileList = FileManager.listFilesRecursive('./dataset')
        #print(fileList)
        Utils.readJSONGetImagesFromUrl(fileList)


        # Prepare data
        # SAE



if __name__ == '__main__':
    Main.localMain()
