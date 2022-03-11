import collections
import json
from file_manager import FileManager


class CustomJson:
    
    dictionary = {}
    
    def __init__(self):
        self.dictionary = {}
        
    def addPair(self, key, value):
        self.dictionary[key] = value
        
    def deleteKey(self, key):
        self.dictionary.pop(key, None)
        
    def __getitem__(self, key):
        return self.dictionary.__getitem__(key)

    def __setitem__(self, key, data):
        self.dictionary.__setitem__(key, data)

    def __delitem__(self, key):
        self.dictionary.__delitem__(key)
        
    
    def saveJson(self, path_file):
        assert(type(path_file) is str)
        content = json.dumps(self.dictionary, sort_keys=True, indent=4, separators=(',', ': '))
        FileManager.saveString(content, path_file, True) #close_file
        
        
    
    def convert(self, data):
        try:
            basestring
        except NameError:
            basestring = str
            
        if isinstance(data, basestring):
            return str(data)
        if type(data) is dict:
            return data
        elif isinstance(data, collections.Mapping):
            return dict(map(self.convert, data.iteritems()))
        elif isinstance(data, collections.Iterable):
            return type(data)(map(self.convert, data))
        else:
            return data
        
        
    
    def loadJson(self, path_file):
        assert(type(path_file) is str)
        
        content = FileManager.readStringFile(path_file)
        
        self.dictionary = self.convert(json.loads(content))

    def printJson (self):
    
        print(json.dumps(self.dictionary))
    
    
    
    def listToDictionary(lst):
        assert(type(lst) is list)
        
        
        dictionary = {}
        count = 0
        for item in lst:
            if (type(item) is list):
                dictionary[count] = CustomJson.listToDictionary(item)
            elif type(item) is dict:
                dictionary[count] = item
            else:
                assert(type(item) is str)
                dictionary[count] = item
            
            count = count + 1
        
        return dictionary
        
    listToDictionary = staticmethod(listToDictionary)    

    
    def listFromDictionary(dct):
        assert(type(dct) is dict)
        
        lst = []

        for count in range(len(dct)):
            item = dct[str(count)]
            
            if (type(item) is dict):
                lst.append(CustomJson.listFromDictionary(item))
            else:
                assert(type(item) is str or type(item) is unicode)
                lst.append(str(item))
        
        return lst

    listFromDictionary = staticmethod(listFromDictionary)    
    
    def listFromDictionaryWithoutNumerate(dct):
        assert(type(dct) is dict)
        
        lst = []

        for item in dct:
            
            if (type(item) is dict):
                lst.append(CustomJson.listFromDictionary(item))
            else:
                assert(type(item) is str or type(item) is unicode)
                lst.append(str(item))
        
        return lst

    listFromDictionaryWithoutNumerate = staticmethod(listFromDictionaryWithoutNumerate)    
        
    