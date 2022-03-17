import utils as U
from CRNN.config import Config
import numpy as np

class ModelEvaluator:

    def __init__(self, set, aug_factor):
        self.X, self.Y = set
        self.aug_factor = aug_factor

    def eval(self, model, i2w):
        acc_ed = 0
        acc_len = 0
        acc_count = 0

        for idx in range(len(self.X)):

            # aug factor == 0 means no augmentation at all
            if self.aug_factor == 0:
                sample_image = self.X[idx]
                sample_image = U.normalize(sample_image)
                sample_image = U.resize(sample_image, Config.img_height)

                batch_sample = np.zeros(
                    shape=[1,Config.img_height, sample_image.shape[1], Config.num_channels],
                    dtype=np.float32)

                batch_sample[0] = sample_image
                prediction = model.predict(batch_sample)[0]

            else:
                # Aumentar, comprobar y elegir
                raise NotImplementedError

            h = U.greedy_decoding(prediction, i2w)

            acc_ed += U.levenshtein(h, self.Y[idx])
            acc_len += len(self.Y[idx])
            acc_count += 1

        return 100.0*acc_ed/acc_len
