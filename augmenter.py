'''
Small module with adapter class for albumentations
'''
import warnings
import numpy as np
import PIL
import torch
from albumentations import Compose

class Augmenter():

    '''
    Image augmentation class to integrate albumentations. Takes list of albumentations
    transforms, probability of applying any augmentation and the type of the target.
    Objects can be called on torch.Tensor PIL.Image.Image and ndarray and converts
    if necessary.
    '''
    def __init__(self, list_of_transforms=[], p=.5, target_type="mask"): #TODO: target_type=None?
        '''
        Instantiate albumentations augmenter.

        :param list_of_transforms: (list) list of albumentations objects to apply
        :param p: (float) probability of applying augmentations form list
        :param target_type: (str) type of target format. Possible values are "mask",
                                  "bbox" and "keypoints"
        '''
        if target_type not in ["mask", "bbox", "keypoints"]: # TODO: is keypoints correct?
            raise TypeError("Augmenter.__init__: target_type not recognized")

        self.transform = Compose(list_of_transforms, p=p)
        self.target_type = target_type

    def __call__(self, image, target=None):
        '''
        Call operator.

        :param image: (ndarray or torch.Tensor or PIL.Image.Image) image to apply augmentations
        :param target: (ndarray or torch.Tensor or PIL.Image.Image) target of image (default: None)

        :rtype (ndarray) or (ndarray, ndarray) returns changed image and optionally the target
        '''
        if target is None:
            return self.__single_transform(image)
        return self.__dual_transform(image, target)

    def __single_transform(self, image):
        '''
        Perform augmentations only on image

        :param image: (ndarray or torch.Tensor or PIL.Image.Image) image to apply augmentations

        :rtype (ndarray) changed image
        '''
        if not isinstance(image, np.ndarray):
            warnings.warn("Augmenter.__call__: expect ndarray, conversion might take time")
            image = to_ndarray(image)

        # this should be a raise or transpose once we know how to determine channel-ordering
        if image.shape[0] < image.shape[-1]:
            warnings.warn("Augmenter.__call__: expect channels-last ordering")

        data = {"image": image}
        augmented = self.transform(**data)
        return augmented["image"]

    def __dual_transform(self, image, target):
        '''
        Perform augmentations only on image

        :param image: (ndarray or torch.Tensor or PIL.Image.Image) image to apply augmentations
        :param target: (ndarray or torch.Tensor or PIL.Image.Image) target of image


        :rtype (ndarray, ndarray) changed image and target
        '''
        if not isinstance(image, np.ndarray):
            warnings.warn("Augmenter.__call__: expect ndarray, conversion might take time")
            image = to_ndarray(image)

        if not isinstance(target, np.ndarray):
            warnings.warn("Augmenter.__call__: expect ndarray, conversion might take time")
            target = to_ndarray(target)

        # this should be a raise or transpose once we know how to determine channel-ordering
        if image.shape[0] < image.shape[-1] or target.shape[0] < target.shape[-1]:
            warnings.warn("Augmenter.__call__: expect channels-last ordering")

        data = {"image": image, self.target_type: target}
        augmented = self.transform(**data)
        return augmented["image"], augmented["mask"]

#TODO move this to separate file?
def to_ndarray(image):
    '''
    Convert torch.Tensor or PIL.Image.Image to ndarray.

    :param image: (torch.Tensor or PIL.Image.Image) image to convert to ndarray

    :rtype (ndarray): image as ndarray
    '''
    if isinstance(image, torch.Tensor):
        return image.numpy()
    if isinstance(image, PIL.Image.Image):
        return np.array(image)
    raise TypeError("to_ndarray: expect torch.Tensor or PIL.Image.Image")
