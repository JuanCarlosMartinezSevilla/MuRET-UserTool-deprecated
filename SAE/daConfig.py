class DAConfig:
    batch_size = 8
    image_size = 40
    classes_to_predict = 'staff'
    reduction_factor = 20
    epochs = 10
    
    # Model
    filters = [128, 128, 128, 128, 128, 128]
    kernel = (5, 5)
    pool = (2, 2)
