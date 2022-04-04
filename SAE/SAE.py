from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
import json
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import tensorflowjs as tfjs

class SAE:
        # Function | Reads JSON files from path and calls "process_muret_json" function
    def read_json_file_from_dir (path_to_file, df, routes_dict, classes):

        df, loop = SAE.process_muret_json(path_to_file, df, routes_dict, classes) # CALL
        #df = get_staves(df)
        return df, loop

    # Function | Gets data from JSON
    def process_muret_json (path, df, routes_dict, classes):
    
        with open (path) as f:
            json_read = json.load(f)

        filename = routes_dict[path]
        loop = False

        if 'pages' in json_read.keys():
            # Esto es una lista
            pages = json_read['pages']
            num_pages = len(pages)
	        
            
            # array | each element all single pag coords
            # (if we have an image with 2 pages, we will have 2 components in the array)

            all_pages_coords = []
            # array | same for regions
            all_regions_coords = []

            for index, pag in enumerate(pages):

                pag_coords = []
                # Pages bounding boxes
                if 'regions' in pag.keys():
                    pag_coords.append(pag['bounding_box']['fromX'])
                    pag_coords.append(pag['bounding_box']['toX'])
                    pag_coords.append(pag['bounding_box']['fromY'])
                    pag_coords.append(pag['bounding_box']['toY'])
                    regions = pag['regions']
                    #print(filename, pag_coords)

                    staves = []
                    # Staves bounding boxes
                    for reg in regions:
                        if reg['type'] in classes:
                            if 'bounding_box' in reg.keys():
                                bb_staff = reg['bounding_box']
                                staves.append(bb_staff)
                                #symbols_staff = reg['symbols']

                    all_pages_coords.append(pag_coords)
                    all_regions_coords.append(staves)
                else:
                    loop = True

            if loop == False:
                df.loc[df.shape[0]] = [filename, num_pages, all_pages_coords, all_regions_coords]
        else:
            loop = True
        return df, loop


    # Function | Plots images. Real and binarized overlayed
    def plot_images(img_name, gray_image_array, binarized_image):
        plt.figure(figsize=(7, 7))
        plt.title('Filename: ' + str(img_name) + " Image shape: " + str(gray_image_array.shape))
        plt.imshow(gray_image_array, interpolation='nearest', cmap='gray')
        plt.imshow(binarized_image, interpolation='nearest', alpha=0.5)
        plt.show()

    # Function | Trims top and botton margin by a reduction factor (rf)
    def trim_coords(box, rf):

        fromY = box['fromY']
        toY = box['toY']

        height = toY - fromY
        fromY_trimmed = int(fromY + (height * (rf/100)))
        toY_trimmed = int(toY - (height * (rf/100)))

        return box['fromX'], box['toX'], fromY_trimmed, toY_trimmed

    # Function | Creates a binarized image from the JSON coordinates
    def binarize_image(image, index, rf, df):
        # Get values of DF column
        for page_boxes in df['Staves'].iloc[index]:
            # Get boxes of a page
            for box in page_boxes:
                # To trim rf = reduction factor percentage in height the boxes
                fromX, toX, fromY, toY = SAE.trim_coords(box, rf)
                # Iterate row of the image matrix
                for index_row, row in enumerate(image):
                    # Selection of rows where the box is
                    if index_row >= fromY and index_row <= toY:
                        # Iterate lane and change values when in range of the box
                        for index_col, elem in enumerate(row):
                            # Range of the box
                            if index_col >= fromX and index_col <= toX :
                              row[index_col] = 1

        return image

    # Function | Reads images and creates its binarized one
    #image_gen(path_to_img, path_to_json, num_img)
    def image_gen(routes_dict, num_img, img_tam, classes):

        GrayImages = []
        BinarizedImages = []
        width = height = img_tam
        # Reduction factor (in %)
        rf = 20

        df = pd.DataFrame()
        df['Filename'] = None
        df['Pages_per_image'] = None
        df['Pag_coords'] = None
        df['Staves'] = None

        for i in range(0, num_img):
            loop = True
            while loop:
                json_path = random.choice(list(routes_dict))
                df, loop = SAE.read_json_file_from_dir (json_path, df, routes_dict, classes)
                #print(json_path)


        for index, img_name in enumerate(df['Filename']):
            #print(img_name)
            image = cv2.imread(img_name)
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            gray_image_array = np.array(gray_img)

            # call to function

            binarized_image_array = SAE.binarize_image(np.zeros_like(gray_image_array), index, rf, df)

            # Uncomment this if you want to plot the results
            #SAE.plot_images(img_name, gray_image_array, binarized_image_array)

            binarized_image_array *= 255

            bin_img_resized = cv2.resize(binarized_image_array,(width,height), interpolation=cv2.INTER_LINEAR)
            gray_img_resized = cv2.resize(gray_image_array,(width,height), interpolation=cv2.INTER_LINEAR)

            gray_img_resized = gray_img_resized.astype(np.float32)
            bin_img_resized = bin_img_resized.astype(np.float32)

            gray_img_resized = (255. - gray_img_resized) / 255.
            bin_img_resized /= 255

            gray_img_resized = np.expand_dims(gray_img_resized, axis=2)
            bin_img_resized = np.expand_dims(bin_img_resized, axis=2)

            GrayImages.append(gray_img_resized)
            BinarizedImages.append(bin_img_resized)

        

            #background = GrayImages[0]
            #overlay = BinarizedImages[0]*255
            #added_image = cv2.addWeighted(background,1,overlay,0.5,0)
            #cv2.imshow('GTImage', added_image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

        return np.array(GrayImages), np.array(BinarizedImages)

    def dataGen(routes_dict, batch_size, img_tam, classes_to_predict):
        while True:
            #print(routes_dict)
            yield(SAE.image_gen(routes_dict, batch_size, img_tam, classes_to_predict))

    def model(img_tam, epochs, generator, steps, args):
        input_img = Input(shape=(img_tam, img_tam, 1))

        x = input_img
        
        filters = [128, 128, 128, 128, 128, 128]
        kernel = (5, 5)
        pool = (2, 2)

        for idx, f in enumerate(filters):
            if idx <= 2:
                x = Conv2D(f, kernel, activation='relu', padding='same')(x)
                x = MaxPooling2D(pool)(x)
            else:
                x = Conv2D(f, kernel, activation='relu', padding='same')(x)
                x = UpSampling2D(pool)(x)

        decoded = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x)

        SAEmodel = Model(input_img, decoded)
        SAEmodel.compile(optimizer='adam', loss = 'binary_crossentropy')

        SAEmodel.fit(
            generator,
            verbose=1,
            steps_per_epoch=steps,
            epochs=epochs)
        
        if args.h5:
            SAEmodel.save(f'./MuRETPackage/document_analysis/document_analysis.h5')
        
        # Save model to use it with tensorflow.js
        tfjs.converters.save_keras_model(SAEmodel, './MuRETPackage/document_analysis/tfjs/')
