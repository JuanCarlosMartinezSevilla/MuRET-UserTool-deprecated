import cv2


def funct():
    path = "dataset/SRC/daugImage1.png"


    #"bounding_box": {
    #    "fromX": 219,
    #    "toX": 312,
    #    "fromY": 264,
    #    "toY": 388
    #},

    #top, left, bottom, right = region['bounding_box']['fromY'], region['bounding_box']['fromX'], region['bounding_box']['toY'], region['bounding_box']['toX']

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    top = 157
    left = 1582
    bottom = 342
    right = 1658


    window_name = 'image'
    cv2.imshow(window_name, image[top:bottom, left:right])
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == '__main__':
    funct()