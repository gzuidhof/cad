import time

class Params():
    def __init__(self):
        self.MODEL_ID = str(int(time.time()))

        # Domain
        self.PIXELS = 32
        self.CHANNELS = 3
        self.N_CLASSES = 10

        # Network and learning parameters
        self.START_LEARNING_RATE = 0.02
        self.MOMENTUM = 0.9
        self.BATCH_NORMALIZATION = True

        self.REGULARIZATION = True
        self.REGULARIZATION_WEIGHT = 0.1
        self.UPDATEFUNCTION = "RMSPROP"
        self.DEEPER = True
        self.EVEN_DEEPER = True

        #Did not improve results in N=1 test
        self.HISTOGRAM_EQUALIZATION = False
        self.CLAHE = False #adaptive equalization

        # Augmentation
        self.AUGMENT = True
        self.COLOR_AUGMENTATION = True
        self.AUGMENTATION_PARAMS = {
            'zoom_range': (0.9, 1.1),
            'rotation_range':(-12,12),
            'translation_range': (-3,3),
            'do_flip': True,
            'hue_range': (-0.1, 0.1),
            'saturation_range': (-0.25, 0.25),
            'value_range': (-0.25, 0.25)
        }

params = Params()
