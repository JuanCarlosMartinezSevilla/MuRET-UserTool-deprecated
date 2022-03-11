import GTJSONReaderMuret as muret
from CustomJson import CustomJson
from file_manager import FileManager
import numpy as np
import cv2
import random
import json 


class MuretInterface:
    @staticmethod
    def countNumberSymbols(dictionary):
        total_symbols = 0
        if "pages" in dictionary:
            if len(dictionary["pages"]) == 1:
                page=0
                if "regions" in dictionary["pages"][page]:
                    for region in dictionary["pages"][page]["regions"]:
                        if "symbols" in region:
                            total_symbols+= len(region["symbols"])
            else:
                for page in dictionary["pages"]:
                    if "regions" in page:
                        for region in page["regions"]:
                            if "symbols" in region:
                                total_symbols+= len(region["symbols"])
                                
        return total_symbols

    @staticmethod
    def readBoundingBoxes(list_json_pathfiles, considered_classes, verbose=False):
        assert(type(list_json_pathfiles) is list)
        assert(considered_classes is None or type(considered_classes) is list)

        dict_bboxes = {}

        total_symbols = 0
        ### CORREGIR INTERVALO DE ARCHIVOS
        for json_pathfile in list_json_pathfiles:
            js = CustomJson()
            js.loadJson(json_pathfile)

            total_symbols += MuretInterface.countNumberSymbols(js.dictionary)

            gtjson = GTJSONReaderMuret()
            gtjson.load(js)

            #print(gtjson)

            bboxes = gtjson.getListBoundingBoxesPerClass(considered_classes)
            dict_bboxes[json_pathfile] = bboxes

        if verbose is True:
            print ("Total symbols: " + str(total_symbols))
        return dict_bboxes

    @staticmethod
    def getAllBoxesByRegionName(list_json_pathfiles, considered_classes = None, verbose=False):

        bboxes = MuretInterface.readBoundingBoxes(list_json_pathfiles = list_json_pathfiles, considered_classes = considered_classes, verbose=verbose)
        assert(type(bboxes) is dict)
        
        num_regions = 0
        dict_regions = {}
        idx = 0
        for key in bboxes:
            num_regions += len(bboxes[key])
            idx += 1

            for key_region in bboxes[key]:
                if key_region not in dict_regions:
                    dict_regions[key_region] = {}

                num_regions = len(bboxes[key][key_region])

                if len(bboxes[key][key_region]) > 0:
                    if key not in dict_regions[key_region]:
                        dict_regions[key_region][key] = []

                    dict_regions[key_region][key].append(bboxes[key][key_region])

        return dict_regions

    @staticmethod
    def applyBlurring(img, factor = 2):

        rows = img.shape[0]//factor
        cols = img.shape[1]//factor
        kernel = np.ones((rows, cols),np.float32)/(rows*cols)
        blur_img = cv2.filter2D(img,-1,kernel)
        return blur_img

    @staticmethod
    def selectRandomRegion(considered_class, dict_regions, routes_dict):
        candidate_images = dict_regions[considered_class]
        #print("Candidadtes: ", candidate_images)
        candidate_regions = random.choice(list(candidate_images.items()))
        #print("Region: ",candidate_regions[1][0])

        selected_region_key = random.choice(candidate_regions[1][0])
        #print("Selected region key: ", selected_region_key)
        # PATH to json
        selected_json = candidate_regions[0]
        #print("selected json: ", selected_json)
        selected_img_pathfile = routes_dict[selected_json]

        
        #print(selected_img_pathfile)
        selected_img = FileManager.loadImage(selected_img_pathfile, True)
        #print("Region key: ", selected_region_key[0], selected_region_key[1])
        selected_patch = selected_img[selected_region_key[0][0]:selected_region_key[0][2], selected_region_key[0][1]:selected_region_key[0][3]]
        #print("selected patch: ", selected_patch)

        return selected_patch, selected_region_key

        
    @staticmethod
    def generateGTImages(bboxes, shape, reduction_factor, margin=5):
        assert(len(shape) == 2)
        gt_ims = []
        
        key_region_id = 1
        for key_region in bboxes:
            gt_im = np.zeros(shape)
            
            for bbox in bboxes[key_region]:
                min_row = bbox[0]
                min_col = bbox[1]
                max_row = bbox[2]
                max_col = bbox[3]

                if reduction_factor is not None:
                    rows = max_row - min_row

                    reducted_rows = rows*(reduction_factor)
                    max_row -= int(reducted_rows)
                    min_row += int(reducted_rows)

                max_row = min(shape[0]-margin, max_row)
                min_row = max(margin, min_row)
                max_col = min(shape[1]-margin, max_col)
                min_col = max(margin, min_col)

                gt_im[min_row:max_row, min_col:max_col] = key_region_id

            gt_ims.append(gt_im)

        return gt_ims

    @staticmethod
    def splitGTImageByTypeRegion(gt_im, considered_classes):

        for considered_class in considered_classes:

            pass

        return
    
    @staticmethod
    def modifyWithNewCoords (new_coords, regions_dicts):
        
        new_regions_dicts = []

        #min_row_new, min_col_new, max_row_new, max_col_new
        #from y, from x, to y, to x

        for coords, region in zip(new_coords, regions_dicts):
            region[1]['bounding_box']['fromX'] = coords[1]
            region[1]['bounding_box']['toX'] = coords[3]
            region[1]['bounding_box']['fromY'] = coords[0]
            region[1]['bounding_box']['toY'] = coords[2]

        return regions_dicts

    @staticmethod
    def generateJSONMURET(new_coords, regions_dicts, new_bbox_regions, json_pathfile, src_filename, src_shape):

        #print(new_bbox_regions)

        new_regions_dicts = MuretInterface.modifyWithNewCoords(new_coords, regions_dicts)
        
        with open(json_pathfile) as json_file:
            json_data = json.load(json_file)


        bbox = {}
        bbox["bounding_box"] = {"fromX":0,"toX":src_shape[0],"fromY":0,"toY":src_shape[1]}
        json_data["pages"] = []
        json_data["pages"].append(bbox)


        idx_new_bbox = 0

        if len(json_data["pages"]) == 1:
            json_data["pages"][0]['regions'] = []
            json_data["filename"] = src_filename
            json_data["id"] = -1

            for region in regions_dicts:
                #print(region[1])
                json_data["pages"][0]['regions'].append(region[1])
                

            #for key_region in new_bbox_regions:    
            #    for bbox_region in new_bbox_regions[key_region]:
            #        bbox = {}
            #        fromX = bbox_region[1]
            #        toX = bbox_region[3]
            #        fromY = bbox_region[0]
            #        toY = bbox_region[2]
            #        bbox["bounding_box"] = {"fromX":fromX,"toX":toX,"fromY":fromY,"toY":toY}
            #        bbox["type"] = key_region

            #        json_data["pages"][0]['regions'].append(bbox)

            idx_new_bbox += 1

        
        return json.dumps(json_data, indent = 4)

