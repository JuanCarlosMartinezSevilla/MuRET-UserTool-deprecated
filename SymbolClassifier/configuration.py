class Configuration:
    img_height_g = 40
    img_height_p = 224
    img_width_g = 40
    img_width_p = 112
    batch_size = 32
    epochs = 15

    # Model
    filters = [32, 32, 64, 64]
    kernel = (3 , 3)
    pooling_kernel = (2,2)