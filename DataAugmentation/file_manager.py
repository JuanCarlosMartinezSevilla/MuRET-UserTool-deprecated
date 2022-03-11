import cv2
import os, sys
from os import listdir
from os.path import isfile, join, exists
import numpy as np
import shutil


class FileManager:

    @staticmethod
    def createFolders(path_file):
        assert type(path_file) == str

        path_dir = FileManager.nameOfDirFromPath(path_file)

        if (path_dir != ""):
            if not os.path.exists(path_dir):
                os.makedirs(path_dir, 493);

    @staticmethod
    def deleteFolder(path_dir):
        assert type(path_dir) == str

        shutil.rmtree(path_dir, ignore_errors=True)

    #==============================================================================
    # 
    #==============================================================================
    @staticmethod
    def loadImage2 (path_dir, filename, with_color):
        assert type(with_color) == bool
        assert type(path_dir) == str
        assert type(filename) == str
        
        return cv2.convertScaleAbs(np.load(path_dir + '/' + filename, with_color))
    
    
    #==============================================================================
    # 
    #==============================================================================
    @staticmethod
    def loadImageFromPath (path_file, with_color):
        assert type(with_color) == bool
        assert type(path_file) == str
        
        img = np.load(path_file, with_color, allow_pickle=True)
        return np.asarray( img, dtype='uint8' )
    
    
    
    
    # =============================================================================
    #     
    # =============================================================================
    @staticmethod
    def existsFile(path_file):
        assert (type(path_file) is str)
        return os.path.exists(path_file)
    
    
    
    #==============================================================================
    #     
    #==============================================================================
    @staticmethod
    def saveString(content_string, path_file, close_file):
        assert type(content_string) == str
        assert type(path_file) == str
        assert type(close_file) == bool
        
        path_dir = FileManager.nameOfDirFromPath(path_file)

        if (path_dir != ""):
            if not os.path.exists(path_dir):
                os.makedirs(path_dir, 493);
                
        f = open(path_file,"w+")
        f.write(content_string)
        
        if (close_file == True):
            f.close()
        
    
    # =============================================================================
    #     
    # =============================================================================
    @staticmethod
    def appendString(content_string, path_file, close_file = True):
        assert type(content_string) == str
        assert type(path_file) == str
        
        path_dir = FileManager.nameOfDirFromPath(path_file)

        if (path_dir != ""):
            if not os.path.exists(path_dir):
                os.makedirs(path_dir, 493);
                
        f = open(path_file,"a")
        f.write(content_string)
        
        if close_file == True:
            f.close()
        
    
    # =============================================================================
    #     
    # =============================================================================
    @staticmethod
    def copyFile(path_file_orig, path_file_dest):
        assert type(path_file_orig) == str
        assert type(path_file_dest) == str
        
        path_dir_dest = FileManager.nameOfDirFromPath(path_file_dest)

        if (path_dir_dest != ""):
            if not os.path.exists(path_dir_dest):
                os.makedirs(path_dir_dest, 493)
                
        shutil.copyfile(path_file_orig, path_file_dest)        

    #==============================================================================
    #         
    #==============================================================================
    @staticmethod
    def readStringFile(path_file):
        assert type(path_file) == str

        f = open(path_file)
        
        content = f.read()
        f.close()
        
        assert type(content) == str

        return content
        
       
    #==============================================================================
    #         
    #==============================================================================
    @staticmethod
    def saveImage (image, path_dir, filename):
        assert 'numpy.ndarray' in str(type(image))
        assert type(path_dir) == str
        assert type(filename) == str
        if (path_dir == ""):
            path_file = filename
        else:
            path_file = path_dir + '/' + filename
            
            if not os.path.exists(path_dir):
                os.makedirs(path_dir, 493);
    
        path_file_png = FileManager.changeExtPathFileImageToPng(path_file)
        cv2.imwrite(path_file_png, image)
    

    # =============================================================================
    #     
    # =============================================================================
    @staticmethod
    def loadImage(path_file, with_color):
        assert (type(path_file) is str)
        
        type_reading = cv2.IMREAD_COLOR
        if (with_color == False):
            type_reading = cv2.IMREAD_GRAYSCALE
            
        return cv2.convertScaleAbs(cv2.imread(path_file, type_reading))
        
    # =============================================================================
    #     
    # =============================================================================
    @staticmethod
    def changeExtPathFileImageToPng(path_file):
        path, filename_ext = os.path.split(path_file)
        filename, ext = os.path.splitext(filename_ext)

        if path == '':
            return filename + ".png"
        else:
            return path + "/" + filename + ".png"
     
    
    
    # =============================================================================
    #     
    # =============================================================================
    @staticmethod
    def saveImageFullPath(image, path_file):
        assert 'numpy.ndarray' in str(type(image))
        assert type(path_file) == str
         
        path_dir = FileManager.nameOfDirFromPath(path_file)
        if path_dir != '' and not os.path.exists(path_dir):
            os.makedirs(path_dir, 493)
    
        path_file_png = FileManager.changeExtPathFileImageToPng(path_file)
        cv2.imwrite(path_file_png, image)

    
    #==============================================================================
    # 
    #==============================================================================
    @staticmethod
    def saveGTImage(gt_image, path_dir, filename):
        assert 'numpy.ndarray' in str(type(gt_image))
        assert type(path_dir) == str
        assert type(filename) == str
        
        if (path_dir == ""):
            path_file = filename
        else:
            path_file = path_dir + '/' + filename
            if not os.path.exists(path_dir):
                os.makedirs(path_dir, 493);
  
        gt_image.dump(path_file)
    
    # =============================================================================
    #     
    # =============================================================================

    @staticmethod
    def saveDictionary(dictionary, path_file):
        assert (type(dictionary) is dict)
        assert type(path_file) == str
         
        path_dir = FileManager.nameOfDirFromPath(path_file)
        
        if not os.path.exists(path_dir):
            os.makedirs(path_dir, 493)

        f = open(path_file,'w+')

        import json
        f.write(json.dumps(dictionary, indent=4))
        f.close()

    @staticmethod
    def loadDictionary(path_file):
        assert type(path_file) == str
        
        f = open(path_file,'w+')

        import json

        dictionary =  json.load(f.read())
        f.close()
        return dictionary

    # =============================================================================
    #     
    # =============================================================================
    
    @staticmethod
    def saveGTImage2(gt_image, path_file):
        assert 'numpy.ndarray' in str(type(gt_image))
        assert type(path_file) == str
        
        path_dir = FileManager.nameOfDirFromPath(path_file)
        
        [path_dir, filename] = FileManager.separateDirectoryAndFilename(path_file)
        
        if not os.path.exists(path_dir):
            os.makedirs(path_dir, 493)
  
        gt_image.dump(path_file)
    
    
    #==============================================================================
    # 
    #==============================================================================
    @staticmethod
    def listFiles (path_dir):
        assert type(path_dir) == str
        if (exists(path_dir) == False):
            path_dir = "../" + path_dir
        
        list_files = ([f for f in listdir(path_dir) if isfile(join(path_dir, f))])
        list_files.sort()
        return list_files


    @staticmethod
    def cleanListOnlyJSON(list_json_pathfiles):
        auxList = []

        for file in list_json_pathfiles:
            if ('json' or 'dict') in file.split('.')[-1]:
                auxList.append(file)

        return auxList

    @staticmethod
    def createRoutesDict(list_files, idx):
        idx = idx + 1
        routes_dict = {}
        for l in list_files:
            if 'json' in l.split('.')[-1]:
                img_name = l.split('/')[-1][:-5]
                #print(img_name)
                for img_route in list_files:
                    #TODO
                    for name in img_route.split('/'):
                        if img_name == name and 'json' not in img_route.split('.')[-1]:
                            routes_dict[l] = img_route
            elif 'dict' in l.split('.')[-1]:
                img_name = l.split('/')[-1][:-5]
                if f'daug_{idx}' in l.split('/'):
                    routes_dict[l] = f'/content/dataset/dataAugmentation/daug_{idx}/SRC/{img_name}.png'

        return routes_dict
        
    # =============================================================================
    # 
    # =============================================================================
    
    @staticmethod
    def listFilesRecursive(path_dir):
        
        try:
            listOfFile = os.listdir(path_dir)
        except Exception:
            pathdir_exec = os.path.dirname(os.path.abspath(__file__))
            path_dir = pathdir_exec + "/" + path_dir
            listOfFile = os.listdir(path_dir)

        list_files = list()
        
        for entry in listOfFile:
            fullPath = os.path.join(path_dir, entry)
            if os.path.isdir(fullPath):
                list_files_in_subfolder = FileManager.listFilesRecursive(fullPath)
                list_files = list_files + list_files_in_subfolder
            else:
                
                list_files.append(fullPath)
        
        list_files.sort() 

        return list_files


    #==============================================================================
    # 
    #==============================================================================
    @staticmethod
    def executeFunctionEachFileInDir(path_dir, ptr_function):
        assert type(path_dir) == str
        assert ptr_function is not None
        
        for f in FileManager.listFiles (path_dir):      
            ptr_function(f)
        
               
    #==============================================================================
    #         
    #==============================================================================
    @staticmethod
    def executeFunctionEachFileInDirExtra(path_dir, ptr_function, data_extra):
        assert type(path_dir) == str
        assert data_extra is not None
        assert ptr_function is not None
        
        for f in FileManager.listFiles (path_dir):        
            ptr_function(f, data_extra)
        
        
    #==============================================================================
    #         
    #==============================================================================
    @staticmethod
    def nameOfFileWithExtension(path_file):
        assert type(path_file) == str
        
        return os.path.basename(path_file)

        
    #==============================================================================
    #         
    #==============================================================================
    @staticmethod
    def nameOfDirFromPath(path_file):
        assert type(path_file) == str
        splits = path_file.split('/')

        dir_path = ''
        is_full_path = False
        for split in splits:
            if ".." in split:
                dir_path = ".."
            else:
                if "." not in split:
                    if dir_path == '':
                        if (is_full_path):
                            dir_path = dir_path + "/" + split
                        else:
                            dir_path = split
                            is_full_path = True
                    else:
                        dir_path = dir_path + "/" + split
    
        return dir_path
        
        
    #==============================================================================
    #         
    #==============================================================================
    @staticmethod
    def nameOfFile(path_file):
        assert type(path_file) == str
        assert path_file is not None
        return FileManager.nameOfFileWithExtension(path_file).split('.')[0]
        

    
    #==============================================================================
    #         
    #==============================================================================
    @staticmethod
    def makeDirsIfNeeded(path_dir):
        assert type(path_dir) == str
        if not os.path.exists(path_dir):
            os.makedirs(path_dir, 493);
            
         
    #==============================================================================
    #             
    #==============================================================================
    @staticmethod
    def separateDirectoryAndFilename(path_file):
        assert type(path_file) == str
    
        path_dir = FileManager.nameOfDirFromPath(path_file)       
        filename = FileManager.nameOfFileWithExtension(path_file)
    
        return [path_dir, filename]
   
    #==============================================================================
    #     
    #==============================================================================
    @staticmethod
    def deleteLastExtension(path_file):
        assert type(path_file) == str

        splits = path_file.split(".")
        
        path_file_without_ext_dat = ''
        
        number_dots = len(splits)

        for i in range(number_dots - 1):
            if '' == path_file_without_ext_dat:
                path_file_without_ext_dat = splits[i]
            else:
                path_file_without_ext_dat = path_file_without_ext_dat + "." + splits[i]
                        
        return path_file_without_ext_dat
    
    #==============================================================================
    #     
    #==============================================================================
    @staticmethod
    def getExtension(path_file):
        assert type(path_file) == str

        splits = path_file.split(".")
        
        path_file_without_ext_dat = ''
        
        number_dots = len(splits)

        return  splits[number_dots - 1]
        
   
    @staticmethod
    def separateNameAndExtension(path_file):
        
        name = FileManager.deleteLastExtension(path_file)
        ext = FileManager.getExtension(path_file)
        
        return [name, ext]
