import os
import tarfile
from tqdm import tqdm
import json
import urllib.request

class Utils:


    def getURLJSON(file, json_classes, path_to_save):

        with open(file) as f:
            json_read = json.load(f)

        url = orig_url = ""
        erase = False
        if 'url' in json_read.keys():
            url = json_read['url']
        else:
            erase = True
        if 'original' in json_read.keys():
            orig_url = json_read['original']
            if url == orig_url:
                filename = json_read["filename"]
                urllib.request.urlretrieve(orig_url, os.path.join(path_to_save, json_read["filename"]))
            elif url != "":
                urllib.request.urlretrieve(url, path_to_save)
            else:
                urllib.request.urlretrieve(orig_url, path_to_save)
            
        if 'pages' in json_read:
            pages = json_read['pages']
            for p in pages:
                if 'regions' in p:
                    regions = p['regions']
                    for r in regions:
                        if r['type'] not in json_classes:
                            json_classes.append(r['type'])
                    #print(json_classes)
            
            #print(f"File {json_read['filename']} saved in {path_to_save}")
        if erase:
            os.remove(file)
        return json_classes



    def readJSONGetImagesFromUrl(files):

        path_to_save = "./images"

        if not os.path.exists("./images"):
            os.mkdir("./images")
        
        print("\n---- Fetching images from URLs ----\n")
        print(f"Saving in {path_to_save} \n")
        
        json_classes = []
        for f in tqdm(files):
            json_classes = Utils.getURLJSON(f, json_classes, path_to_save)

    @staticmethod
    def decompressFile ():  

        tar_file = "./capitan.tgz"
        path = "./dataset"
        
        print("\nExtracting from .tgz file \n")

        tar = tarfile.open(tar_file, mode="r:gz")
        
        members = tar.getmembers()

        for member in tqdm(members):
            tar.extract(member, path=path)            
        tar.close()

        print(f"\nFiles extracted in {path} ...\n")
        return True