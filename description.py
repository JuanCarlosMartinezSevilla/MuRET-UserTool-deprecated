import json
from datetime import datetime

class Description:
    creation_date = datetime.now().isoformat()

    def __init__(self, name, epochs=None, height=None, width=None, batch=None, data=None):
        self.name = name
        self.model_epochs = epochs
        self.input_h = height
        self.input_w = width
        self.batch = batch
        self.all_data = data
    

class DocumentAnalysisDescription(Description):

    def __init__(self, name, epochs=None, height=None, width=None, batch=None, data=None, regions=None):
        super().__init__(name, epochs, height, width, batch, data)
        self.regions = regions


    def save_description(self):
        description = {
            "name": self.name,
            "epochs": self.model_epochs,
            "img_height": self.input_h,
            "img_width": self.input_w,
            "classes": self.regions,
            "data": self.all_data,
            "date": self.creation_date,
        }

        with open(f'./MuRETPackage/{self.name}/description.json', 'w') as fp:
            json.dump(description, fp, indent=4)

class End2EndDescription(Description):

    def __init__(self, name, epochs=None, height=None, width=None, batch=None, data=None):
        super().__init__(name, epochs, height, width, batch, data)
        self.w2i = None
        self.i2w = None

    def save_description(self):

        description = {
            "name": self.name,
            "epochs": self.model_epochs,
            "img_height": self.input_h,
            "img_width": self.input_w,
            "w2i": self.w2i,
            "i2w": self.i2w,
            "data": self.all_data,
            "date": self.creation_date,
        }

        with open(f'./MuRETPackage/{self.name}/description.json', 'w') as fp:
            json.dump(description, fp, indent=4)

class SymbolClassifierDescription(Description):

    def __init__(self, name, epochs=None, height=None, width=None, batch=None, data=None):
        super().__init__(name, epochs, height, width, batch, data)
        self.input_h_2 = None
        self.input_w_2 = None
        self.w2i_g = None
        self.w2i_p = None
        self.i2w_g = None
        self.i2w_p = None


    def save_description(self):
        description = {
            "name": self.name,
            "epochs": self.model_epochs,
            "img_height": self.input_h,
            "img_width": self.input_w,
            "img_height_2": self.input_h_2,
            "img_width_2": self.input_w_2,
            "w2i_glyphs": self.w2i_g,
            "i2w_glyphs": self.i2w_g,
            "w2i_positions": self.w2i_p,
            "i2w_positions": self.i2w_p,
            "data": self.all_data,
            "date": self.creation_date,
        }

        with open(f'./MuRETPackage/{self.name}/description.json', 'w') as fp:
            json.dump(description, fp, indent=4)
