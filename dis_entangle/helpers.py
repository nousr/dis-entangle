"""Helpers for dis-entangle"""

from dis_entangle.data_loader_cache import normalize

class GOSNormalize():
    '''
    Normalize the Image using torch.transforms
    '''

    def __init__(self, mean=None, std=None):
        self.mean = mean if mean is not None else [0.485,0.456,0.406]
        self.std = std if std is not None else [0.229,0.224,0.225]

    def __call__(self,image):
        image = normalize(image,self.mean,self.std)
        return image

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
