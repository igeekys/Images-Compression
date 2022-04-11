from pprint import pprint

class HyperParams() :
    def __init__(self, verbose):
        # Hard params and magic numbers
        # self.n_labels    = 257
        self.top_k       = 20
        self.image_h     = 224
        self.image_w     = 224
        self.image_c     = 3
        self.mscam       = 0
        self.mscam_softmax=0 # after mscam
        self.normalize   = 0 # default
        self.finetune    = 0 # need train or not
        if verbose:
            pprint(self.__dict__)



