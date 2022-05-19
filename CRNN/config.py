
class Config:
    img_height = 256
    num_channels = 3
    batch_size = 8
    aug_factor = 3
    width_reduction = 8
    epochs = 150
    #steps_per_epoch = 100
    steps_per_epoch = 100

    conv_filters = [16, 32, 64]