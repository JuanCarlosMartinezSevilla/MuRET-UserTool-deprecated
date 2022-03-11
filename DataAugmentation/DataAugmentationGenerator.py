from DataAugmentation.file_manager import FileManager
from DataAugmentation.MuretInterface import MuretInterface
import numpy as np
import cv2
import random
from skimage.filters import (threshold_sauvola)
import math
import skimage.transform as st
import json

#Generator of data augmentation images
class DataAugmentationGenerator:

    @staticmethod
    def rotate(image, angle, center=None, scale=1.0):
        image_shape = image.shape
        h = image_shape[0]
        w = image_shape[1]
        if len(image_shape) == 3:
            c = image_shape[2]
            borderValue = (255,)*c
        else:
            borderValue = 255
        
        if center is None:
            center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(
                            src=image.astype(np.float32), 
                            M=M, 
                            dsize=(w,h), 
                            borderValue=borderValue)
        return rotated, center


    @staticmethod
    def rotatePoint(origin, point, angle_grad):
        ox, oy = origin
        px, py = point

        angle_rad = math.radians(angle_grad)
        qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
        qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)
        return int(qx), int(qy)
        
    @staticmethod
    def applyDataAugmentationToRegion(src_region, fixed_angle=None):

        if fixed_angle is None:
            angle = random.randint(-1,1)
        else:
            angle = fixed_angle

        rotated_src_region, center = DataAugmentationGenerator.rotate(src_region, angle)

        return rotated_src_region, angle, center
    

    @staticmethod
    def obtainBinaryImageBySauvola(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        window = 101
        kernel = 0.2
        thresh_sauvola = threshold_sauvola(image = img_gray, window_size=window, k=kernel)
        binary_sauvola = (img_gray > thresh_sauvola)*255
        return binary_sauvola

    @staticmethod
    def extractCoordinatesFromBoundingBox(bbox_region):
        min_row_orig = bbox_region[0]
        max_row_orig = bbox_region[2]
        min_col_orig = bbox_region[1]
        max_col_orig = bbox_region[3]
        return min_row_orig, max_row_orig, min_col_orig, max_col_orig

    
    @staticmethod
    def countNumberItemsList(mylist):
        num_elements = 0

        for item in mylist:
            if type(item) is list:
                num_elements += DataAugmentationGenerator.countNumberItemsList(item)
            else:
                num_elements += 1
        return num_elements

    @staticmethod
    def countNumberBBoxes(dict_regions, key):
        number_elements = 0
        for item in dict_regions[key]:
            if (type(dict_regions[key][item]) is list):
                number_elements += DataAugmentationGenerator.countNumberItemsList(dict_regions[key][item])
            else:
                number_elements += 1

        return number_elements

    @staticmethod
    def generateNewImageRandomAuto(img_orig, json_orig, list_json_pathfiles, vertical_region_resize, uniform_rotate, routes_dict, window, kernel, angle, verbose=False):

        new_bbox_regions = {}
        regions_dicts = []
        new_coords = []
        new_img = np.copy(img_orig)
        gt_img = np.zeros((img_orig.shape[0], img_orig.shape[1]))
        
        dict_regions = MuretInterface.getAllBoxesByRegionName(list_json_pathfiles = list_json_pathfiles, considered_classes = None, verbose=verbose)
        bboxes_orig = MuretInterface.readBoundingBoxes(list_json_pathfiles = list([json_orig]), considered_classes = None, verbose=verbose)

        if verbose is True:
            number_staves = DataAugmentationGenerator.countNumberBBoxes(dict_regions, "staff")
            number_empty_staves = DataAugmentationGenerator.countNumberBBoxes(dict_regions, "empty_staff")
            number_lyrics = DataAugmentationGenerator.countNumberBBoxes(dict_regions, "lyrics")
            print("Number of pages: " + str(len(list_json_pathfiles)))
            print("Lyrics: " + str(number_lyrics))
            print("Staves: " + str(number_staves))
            print("Empty staves: " + str(number_empty_staves))
            print("Total staves: " + str(number_staves + number_empty_staves))

        if (uniform_rotate is True):
            uniform_angle = random.randint(-angle,angle)
        else:
            uniform_angle = None
        
        overlapped = 0
        assert(len(bboxes_orig) == 1)
        idx_region=0
        
        idx_region_staff = 1
        dict_keys_value = {}
        for key in bboxes_orig:
            bboxes_regions = bboxes_orig[key]

            idx_region_key = 1
            for key_region in bboxes_regions:
                if verbose is True:
                    print(key_region)

                if key_region not in dict_keys_value:
                    if key_region == "staff":
                        idx_region_staff = idx_region_key

                    if "staff" in key_region:
                        dict_keys_value[key_region] = idx_region_staff
                    else:
                        dict_keys_value[key_region] = idx_region_key
                    
                    idx_region_key += 1
                    

                for bbox_region in bboxes_regions[key_region]:
                    if verbose is True:
                        print (bbox_region)

                    #print("Key region before selectRandomRegion: ", key_region)
                    selected_patch, selected_region_key = MuretInterface.selectRandomRegion(key_region, dict_regions, routes_dict)

                    selected_patch_gray = cv2.cvtColor(selected_patch, cv2.COLOR_RGB2GRAY)
                    thresh_sauvola = threshold_sauvola(image = selected_patch_gray, window_size=101, k=0.2)
                    binary_sauvola = (selected_patch_gray > thresh_sauvola)*255

                    min_row_orig = bbox_region[0][0]
                    max_row_orig = bbox_region[0][2]
                    min_col_orig = bbox_region[0][1]
                    max_col_orig = bbox_region[0][3]
                    height_orig_bbox = max_row_orig - min_row_orig
                    width_orig_bbox = max_col_orig - min_col_orig
                    
                    min_row = bbox_region[0][0]
                    max_row = min(bbox_region[0][0] + binary_sauvola.shape[0], new_img.shape[0])
                    min_col = bbox_region[0][1]
                    max_col = min(bbox_region[0][1] + binary_sauvola.shape[1], new_img.shape[1])
                    height_new_bbox = max_row - min_row
                    width_new_bbox = max_col-min_col

                    max_row = min(max_row, max_row_orig)
                    max_col = min(max_col, max_col_orig)

                    if vertical_region_resize:
                        width_new_bbox = int((height_orig_bbox * selected_patch.shape[1]) / selected_patch.shape[0])

                        selected_patch = cv2.resize(selected_patch,(width_new_bbox, height_orig_bbox), interpolation=cv2.INTER_NEAREST)
                        binary_sauvola = cv2.resize(binary_sauvola,(width_new_bbox, height_orig_bbox), interpolation=cv2.INTER_NEAREST)

                        min_row = bbox_region[0][0]
                        max_row = min(bbox_region[0][0] + height_orig_bbox, new_img.shape[0])
                        min_col = bbox_region[0][1]
                        max_col = min(bbox_region[0][1] + width_orig_bbox, new_img.shape[1])
                        height_new_bbox = max_row - min_row
                        width_new_bbox = max_col-min_col
                    
                    selected_patch = selected_patch[0:height_new_bbox, 0:width_new_bbox]
                    selected_patch, angle, center = DataAugmentationGenerator.applyDataAugmentationToRegion(selected_patch, uniform_angle)

                    binary_sauvola_patch = binary_sauvola[0:height_new_bbox, 0:width_new_bbox]
                    binary_sauvola_patch, _ = DataAugmentationGenerator.rotate(binary_sauvola_patch, angle)

                    binary_sauvola_patch = (binary_sauvola_patch > 128)*255

                    idx_region+=1
                    binary_sauvola_patch = binary_sauvola_patch.astype(np.uint8)

                    coords = np.where(binary_sauvola_patch == 0)

                    center_in_full_img = (center[0] + min_col, center[1]+min_row)

                    min_col_new = center_in_full_img[0] - binary_sauvola_patch.shape[1]//2
                    min_row_new = center_in_full_img[1] - binary_sauvola_patch.shape[0]//2
                    max_col_new = center_in_full_img[0] + binary_sauvola_patch.shape[1]//2
                    max_row_new = center_in_full_img[1] + binary_sauvola_patch.shape[0]//2

                    min_row_new = max(0, min_row_new)
                    max_row_new = min(new_img.shape[0]-1, max_row_new)
                    min_col_new = max(0, min_col_new)
                    max_col_new = min(new_img.shape[1]-1, max_col_new)

                    labels = np.unique(gt_img[min_row_new:max_row_new, min_col_new:max_col_new])
                    
                    if (dict_keys_value[key_region] in labels): #Overlapping with other bounding box
                        if verbose is True:
                            print ("Overlapping!")
                        if (overlapped > 10):
                            break
                        overlapped+=1
                        continue

                    overlapped = 0
                    
                    for idx in range(0,len(coords[0])):
                        coord_x = coords[0][idx]
                        coord_y = coords[1][idx]

                        if (min_row_new + coord_x) >= 0 and (min_row_new + coord_x) < new_img.shape[0] and (min_col_new + coord_y) >= 0 and (min_col_new + coord_y) < new_img.shape[1]: 
                            new_img[min_row_new + coord_x, min_col_new + coord_y] = selected_patch[coord_x, coord_y]


                    gt_img[min_row_new:max_row_new, min_col_new:max_col_new] = dict_keys_value[key_region]

                    #print("Dict keys value: ", dict_keys_value[key_region])

                    new_bbox_region  = (min_row_new, min_col_new, max_row_new, max_col_new)

                    if key_region not in new_bbox_regions:
                        new_bbox_regions[key_region] = []

                    # Añadir las nuevas coords a la region antigua
                    #print("New coords: ", counter , new_bbox_region)

                    regions_dicts.append(selected_region_key)
                    new_coords.append(new_bbox_region)

                    new_bbox_regions[key_region].append(new_bbox_region)

        return new_img, new_bbox_regions, new_coords, regions_dicts

    
    @staticmethod
    def generateNewImageFromListByBoundingBoxesRandomSelectionAuto(json_dataset, number_new_images, uniform_rotate, vertical_region_resize, reduction_factor, window, kernel, angle, json_dirpath_out=None):

        list_json_pathfiles = FileManager.listFilesRecursive(json_dataset)
        routes_dict = FileManager.createRoutesDict(list_json_pathfiles)
        list_json_pathfiles = FileManager.cleanListOnlyJSON(list_json_pathfiles)

        if json_dirpath_out is None:
            json_dirpath_out = "./dataset/JSON/"
            src_dirpath_out = "./dataset/SRC/"

        for idx_new_image in range(number_new_images):
            with_regions = False
            while (with_regions == False):
                json_pathfile = random.choice(list_json_pathfiles)
                img_pathfile = routes_dict[json_pathfile]

                print("ID: " + str(idx_new_image + 1) + "-blur from: " + str(img_pathfile))
                img = FileManager.loadImage(img_pathfile, True)
                blur_img = MuretInterface.applyBlurring(img=img, factor = 2)
                new_img, new_bbox_regions, new_coords, regions_dicts = DataAugmentationGenerator.generateNewImageRandomAuto(blur_img, json_pathfile, list_json_pathfiles, 
                                                                                                                            vertical_region_resize, uniform_rotate, routes_dict, window, kernel, angle)

                if len(new_bbox_regions) > 0:
                    with_regions = True

            src_filepath_out = src_dirpath_out + 'daugImage' +str(idx_new_image + 1) + ".png"
            #gt_filepath_out = src_filepath_out.replace("/SRC/", "/GT/")
            json_filepath_out = src_filepath_out.replace("/SRC/", "/JSON/").replace(".png", ".dict")
            
            FileManager.saveImageFullPath(new_img, src_filepath_out)
            print("Saved data augmentation image in " + str(src_filepath_out) + " (from " + str(img_pathfile) + ")")
            
            src_filename = FileManager.nameOfFileWithExtension(src_filepath_out)

            json_muret = MuretInterface.generateJSONMURET(new_coords, regions_dicts, new_bbox_regions, json_pathfile, src_filename, new_img.shape)
            FileManager.saveString(str(json_muret), json_filepath_out, True)

    def printRectangle(path_to_img, fx, fy, tx, ty):
        img = cv2.imread(path_to_img)

                      # fromx fromy tox toy
        img = cv2.rectangle(img,(fx,fy),(tx,ty),(0,255,0),3)

        window_name = 'image'
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    def returnBBoxes(path_to_json, path_to_img):
    
        with open (path_to_json) as f:
            json_read = json.load(f)

        pages = json_read['pages']

        for pag in pages:
            regions = pag['regions']
            for reg in regions:
                bbox = reg['bounding_box']
                fromX = bbox['fromX']
                fromY = bbox['fromY']
                toX = bbox['toX']
                toY = bbox['toY']
                DataAugmentationGenerator.printRectangle(path_to_img, fromX, fromY, toX, toY )
                #print(bbox.keys())

    def seeBoxes(path_to_img, path_to_json):
                
        DataAugmentationGenerator.returnBBoxes(path_to_json, path_to_img)
