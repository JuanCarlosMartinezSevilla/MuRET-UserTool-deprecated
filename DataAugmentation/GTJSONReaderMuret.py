from enum import Enum
import numpy as np
import random

from DataAugmentation.CustomJson import CustomJson
from DataAugmentation.file_manager import FileManager
import json

class PropertyType(Enum):
    PROPERTY_TYPE_PAGES,\
    PROPERTY_TYPE_REGIONS,\
    PROPERTY_TYPE_REGION_TYPE,\
    PROPERTY_TYPE_BOUNDING_BOX,\
    PROPERTY_TYPE_BBOX_FROM_X,\
    PROPERTY_TYPE_BBOX_TO_X,\
    PROPERTY_TYPE_BBOX_FROM_Y,\
    PROPERTY_TYPE_BBOX_TO_Y,\
    PROPERTY_TYPE_STAFF_REGION,\
    PROPERTY_TYPE_SYMBOLS,\
    PROPERTY_TYPE_AGNOSTIC_SYMBOL,\
    PROPERTY_TYPE_ID,\
    PROPERTY_TYPE_APPROXIMATE_X,\
    PROPERTY_TYPE_POSITION_IN_STAFF\
    = range(14)

    def __str__(self):
        return property_type_keys[self]

    def __repr__(self):
        return self.__str__()

    
    
property_type_keys = {
                    PropertyType.PROPERTY_TYPE_PAGES:                          "pages",\
                    PropertyType.PROPERTY_TYPE_REGIONS:                        "regions",\
                    PropertyType.PROPERTY_TYPE_REGION_TYPE:                    "type",\
                    PropertyType.PROPERTY_TYPE_BOUNDING_BOX:                   "bounding_box",\
                    PropertyType.PROPERTY_TYPE_BBOX_FROM_X:                    "fromX",\
                    PropertyType.PROPERTY_TYPE_BBOX_TO_X:                      "toX",\
                    PropertyType.PROPERTY_TYPE_BBOX_FROM_Y:                    "fromY",\
                    PropertyType.PROPERTY_TYPE_BBOX_TO_Y:                      "toY",\
                    PropertyType.PROPERTY_TYPE_STAFF_REGION:                   "staff",\
                    PropertyType.PROPERTY_TYPE_SYMBOLS:                        "symbols",\
                    PropertyType.PROPERTY_TYPE_AGNOSTIC_SYMBOL:                "agnostic_symbol_type",\
                    PropertyType.PROPERTY_TYPE_ID:                             "id",\
                    PropertyType.PROPERTY_TYPE_APPROXIMATE_X:                  "approximateX",\
                    PropertyType.PROPERTY_TYPE_POSITION_IN_STAFF:              "position_in_staff"\
                    }


# =============================================================================
# Music symbol
# =============================================================================
class GTSymbol:
    name_label = ""
    position_in_staff = ""
    id = 0
    approximateX = 0
    info = {}

     
    def __i_integrity(self):
        assert(type(self.name_label) is str)
        assert(self.name_label != "")
        assert(type(self.position_in_staff) is str)
        assert(self.position_in_staff != "")
        assert(type(self.id) is int)
        assert(type(self.approximateX) is int)
        assert(self.approximateX >= 0)
        
    def __init__(self):
        self.name_label = ""
        self.position_in_staff = ""
        id = 0
        approximateX = 0
        info = {}

    def __str__(self):
        self.__i_integrity()
        #return self.name_label + ":" + self.position_in_staff + 'Dictionary ' + self.info
        return 'Dictionary ' + str(self.info)

    def __repr__(self):
        self.__i_integrity()
        #return self.__str__()
        return str(self.info)

    def fromDictionary(self, dictionary):
        assert (type(dictionary) is dict)
        
        self.name_label = dictionary[str(PropertyType.PROPERTY_TYPE_AGNOSTIC_SYMBOL)]
        self.position_in_staff = dictionary[str(PropertyType.PROPERTY_TYPE_POSITION_IN_STAFF)]
        #self.approximateX = dictionary[str(PropertyType.PROPERTY_TYPE_APPROXIMATE_X)]
        self.id = dictionary[str(PropertyType.PROPERTY_TYPE_ID)]
        self.info = dictionary
        self.__i_integrity()

    def getInfo(self):
        return self.info


# =============================================================================
# Music region
# =============================================================================
class GTRegion:
    name_label = ""
    coord_p1 = (0,0)
    coord_p2 = (0,0)
    symbols = []
    info = {}

    def __i_integrity(self):
        assert(type(self.name_label) is str)
        assert(self.name_label != "")
        assert(type(self.coord_p1) is tuple)
        assert(type(self.coord_p2) is tuple)
        assert(self.symbols is None or type(self.symbols) is list)
        assert(type(self.info) is dict)

    def __init__(self):
        self.name_label = ""
        self.coord_p1 = (0,0)
        self.coord_p2 = (0,0)
        self.info = {}

    def __str__(self):
        self.__i_integrity()
        #return self.name_label + ":" + "(" + str(self.coord_p1) + "->" + str(self.coord_p2) + ")"
        return str(self.info)
        
    def __repr__(self):
        self.__i_integrity()
        #return self.__str__()
        return str(self.info)

    def getInfo(self):
        return self.info

    def isNameInList(self, list_symbol_labels):
        self.__i_integrity()
        assert (type(list_symbol_labels) is list)
        return self.name_label in list_symbol_labels


    def getSRCSample(self, src_image):
        self.__i_integrity()
        if len(src_image.shape) == 3:
            sample = src_image[self.coord_p1[0]:self.coord_p2[0], self.coord_p1[1]:self.coord_p2[1], :]
        else:
            sample = src_image[self.coord_p1[0]:self.coord_p2[0], self.coord_p1[1]:self.coord_p2[1]]

        return sample
        

    def getNameSymbol(self):
        self.__i_integrity()
        return self.name_label


    def append_label_without_repetitions(self, list_symbol_labels):
        self.__i_integrity()
        assert (type(list_symbol_labels) is list)
        
        if self.name_label not in list_symbol_labels:
            list_symbol_labels.append(self.name_label)
        

    def fromDictionary_bounding_box(self, dictionary):
        assert (type(dictionary) is dict)
        
        fromX = int(dictionary[str(PropertyType.PROPERTY_TYPE_BBOX_FROM_X)])
        toX = int(dictionary[str(PropertyType.PROPERTY_TYPE_BBOX_TO_X)])
        
        fromY = int(dictionary[str(PropertyType.PROPERTY_TYPE_BBOX_FROM_Y)])
        toY = int(dictionary[str(PropertyType.PROPERTY_TYPE_BBOX_TO_Y)])
        
        self.coord_p1 = (fromY, fromX)
        self.coord_p2 = (toY, toX)

        

    def fromDictionary(self, dictionary):
        assert (type(dictionary) is dict)
        
        info_bbox = dictionary[str(PropertyType.PROPERTY_TYPE_BOUNDING_BOX)]
        
        self.fromDictionary_bounding_box(info_bbox)
        self.name_label = dictionary[str(PropertyType.PROPERTY_TYPE_REGION_TYPE)]

        if str(PropertyType.PROPERTY_TYPE_SYMBOLS) in dictionary:
            info_symbols = dictionary[str(PropertyType.PROPERTY_TYPE_SYMBOLS)]
            for info_symbol in info_symbols:
                symbol = GTSymbol()
                symbol.fromDictionary(info_symbol)
                symbol = symbol.getInfo()
                self.symbols.append(symbol)

        self.info = dictionary

        self.__i_integrity()


# =============================================================================
# Image page
# =============================================================================
class GTPage:
    coord_p1 = (0,0)
    coord_p2 = (0,0)

    regions = []


    def __i_integrity(self):
        assert(type(self.coord_p1) is tuple)
        assert(type(self.coord_p2) is tuple)
        assert(self.regions is None or type(self.regions) is list)
        

    def __init__(self):
        self.coord_p1 = (0,0)
        self.coord_p2 = (0,0)
        self.regions = []

    def hasRegions(self):
        self.__i_integrity()
        if (self.regions is None):
            return False
        else:
            assert(len(self.regions) > 0)
            return True

    def getBBoxPage(self, considered_classes):
        self.__i_integrity()

        list_bbox = []
        #print("Considered classes page: ", considered_classes)
        if considered_classes is None:
            list_bbox.append((self.coord_p1[0], self.coord_p1[1], self.coord_p2[0], self.coord_p2[1]))
        elif str(PropertyType.PROPERTY_TYPE_PAGES) in considered_classes:
            list_bbox.append((self.coord_p1[0], self.coord_p1[1], self.coord_p2[0], self.coord_p2[1]))
        
        #print("List bbox page: ", list_bbox)

        for region in self.regions:
            #print("Region in for loop: ", type(region.getInfo()))
            if considered_classes is None or region.isNameInList(considered_classes):
                list_bbox.append(((region.coord_p1[0], region.coord_p1[1], region.coord_p2[0], region.coord_p2[1]), region.getInfo()))
        
        #print("List bbox after for loop: ", list_bbox)

        return list_bbox


    def getListRegions(self, list_possible_region_names=None):
        self.__i_integrity()

        list_regions_considered = []
        for region in self.regions:
            if list_possible_region_names is None or region.isNameInList(list_possible_region_names):
                list_regions_considered.append(region)
                
                
        return list_regions_considered
        
    def getListRegionNames(self):
        self.__i_integrity()
        list_labels = []
        
        for region in self.regions:
            region.append_label_without_repetitions(list_labels)
        
        return list_labels
        
    def __str__(self):
        self.__i_integrity()
        return "Page(" + str(self.coord_p1) + "->" + str(self.coord_p2) + ")"

        
    def __repr__(self):
        self.__i_integrity()
        return self.__str__()


    def isNameInList(self, list_symbol_labels):
        self.__i_integrity()
        assert (type(list_symbol_labels) is list)
        return str(PropertyType.PROPERTY_TYPE_PAGES) in list_symbol_labels


    def getSRCSample(self, src_image, sample_size=None):
        self.__i_integrity()
        if len(src_image.shape) == 3:
            sample = src_image[self.coord_p1[0]:self.coord_p2[0], self.coord_p1[1]:self.coord_p2[1], :]
        else:
            sample = src_image[self.coord_p1[0]:self.coord_p2[0], self.coord_p1[1]:self.coord_p2[1]]
            
        return sample
        

    def fromDictionary_bounding_box(self, dictionary):
        assert (type(dictionary) is dict)
        
        fromX = int(dictionary[str(PropertyType.PROPERTY_TYPE_BBOX_FROM_X)])
        toX = int(dictionary[str(PropertyType.PROPERTY_TYPE_BBOX_TO_X)])
        
        fromY = int(dictionary[str(PropertyType.PROPERTY_TYPE_BBOX_FROM_Y)])
        toY = int(dictionary[str(PropertyType.PROPERTY_TYPE_BBOX_TO_Y)])
        
        self.coord_p1 = (fromY, fromX)
        self.coord_p2 = (toY, toX)
        
        
    def fromDictionary(self, dictionary):
        assert (type(dictionary) is dict)
        
        info_bbox = dictionary[str(PropertyType.PROPERTY_TYPE_BOUNDING_BOX)]
        self.fromDictionary_bounding_box(info_bbox)

        if (str(PropertyType.PROPERTY_TYPE_REGIONS) in dictionary):
            list_info_regions = dictionary[str(PropertyType.PROPERTY_TYPE_REGIONS)]
            for info_region in list_info_regions:
                region = GTRegion()
                region.fromDictionary(info_region)
                self.regions.append(region)
        else:
            self.regions = []

        self.__i_integrity()


    def addGT(self, gt_im, considered_classes, vertical_reduction_regions=0.):
        self.__i_integrity()

        if str(PropertyType.PROPERTY_TYPE_PAGES) in considered_classes:
            x_start = self.coord_p1[0]
            y_start = self.coord_p1[1]
            x_end = self.coord_p2[0]
            y_end = self.coord_p2[1]
            gt_im[x_start:x_end, y_start:y_end] = 1
        
        for region in self.regions:
            if region.name_label in considered_classes:
    
                x_start = region.coord_p1[0]
                y_start = region.coord_p1[1]
                x_end = region.coord_p2[0]
                y_end = region.coord_p2[1]
                if vertical_reduction_regions is not None and vertical_reduction_regions > 0.:
                    vertical_region_size = x_end - x_start
                    vertical_reduction_region_side = int((vertical_reduction_regions * vertical_region_size) // 2)
                    x_start += int(vertical_reduction_region_side)
                    x_end -= int(vertical_reduction_region_side)
                
                gt_im[x_start:x_end, y_start:y_end] = 1


class GTJSONReaderMuret:

    filename = None    
    pages = []

    def __i_integrity(self):
        assert(self.filename is not None)
        assert(type(self.filename) is str)

    def __init__(self):
        self.filename = None
        self.pages = []
        
    
    def hasRegions(self):
        self.__i_integrity()

        for page in self.pages:
            if (page.hasRegions()):
                return True
        
        return False
        

    def getFileName(self):
        self.__i_integrity()
        return self.filename


    def getListRegionNames(self):
        self.__i_integrity()
        list_labels = []

        for page in self.pages:
            list_region_names_in_page = page.getListRegionNames()
            for region_names_in_page in list_region_names_in_page:
                if region_names_in_page not in list_labels:
                    list_labels.append(region_names_in_page)
        
        return list_labels

    def getListRegions(self, list_possible_region_names=None):
        self.__i_integrity()
        list_labels = []

        for page in self.pages:
            list_region_names_in_page = page.getListRegions(list_possible_region_names)
            for region_names_in_page in list_region_names_in_page:
                if region_names_in_page not in list_labels:
                    list_labels.append(region_names_in_page)
        
        return list_labels
        
    def getListBoundingBoxes(self, considered_classes=None):
        self.__i_integrity()
        list_bbox = []

        for page in self.pages:
            list_bbox_page = page.getBBoxPage(considered_classes)
            for bbox_page in list_bbox_page:
                list_bbox.append(bbox_page)

        return list_bbox

    def getListBoundingBoxesPerClass(self, considered_classes):
        self.__i_integrity()
        dict_bbox = {}
        
        region_names = self.getListRegionNames()

        #print(considered_classes, region_names)

        if considered_classes is not None:
            region_names = considered_classes
            
        for region_name in region_names:
            dict_bbox[region_name] = []

        list_bbox_page = []
        for page in self.pages:
            #print(page)
            for region_name in region_names:
                list_bbox_page = page.getBBoxPage(list([region_name]))
                #print(list_bbox_page)
                for bbox_page in list_bbox_page:
                    dict_bbox[region_name].append(bbox_page)
                #print(dict_bbox)
        return dict_bbox

    def fromDictionary(self, dictionary):
        assert (type(dictionary) is dict)
        
        self.filename = dictionary["filename"]

        if str(PropertyType.PROPERTY_TYPE_PAGES) in dictionary:
            info_pages = dictionary[str(PropertyType.PROPERTY_TYPE_PAGES)]

            for info_page in info_pages:
                page = GTPage()
                page.fromDictionary(info_page)
                self.pages.append(page)
            
    def load (self, js):
        assert (isinstance(js, CustomJson))
        
        dictionary = js.dictionary
        
        self.fromDictionary(dictionary)
        self.__i_integrity()


    def generateGT(self, considered_classes, img_shape, vertical_reduction_regions = 0.):
        assert(type(considered_classes) is list)
        assert(type(img_shape) is tuple)
        assert(len(img_shape) == 2)

        gt_im = np.zeros(img_shape)

        for page in self.pages:
            page.addGT(gt_im, considered_classes, vertical_reduction_regions)

        return gt_im
