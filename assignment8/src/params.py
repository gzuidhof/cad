import time

class Params():
    def __init__(self):

        self.PIXELS = 32

        self.START_LEARNING_RATE = 0.01
        self.MOMENTUM = 0.9

        self.CHANNELS = 3
        self.N_CLASSES = 10

        self.AUGMENT = True
        self.COLOR_AUGMENTATION = True
        self.NETWORK_INPUT_TYPE = 'RGB'

        self.MODEL_ID = str(int(time.time()))

        self.AUGMENTATION_PARAMS = {
            'zoom_range': (1.0, 1.0),
            'rotation_range':(0,0),
            #'translation_range': (-3, 3),
            'translation_range': (0,0),
            'do_flip': True,
            'hue_range': (-0.1, 0.1),
            'saturation_range': (-0.1, 0.1),
            'value_range': (-0.1, 0.1)
        }

params = Params()
