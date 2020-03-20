# Borrowed from: https://github.com/yjxiong/tsn-pytorch/blob/master/transforms.py

import torchvision
import random
from PIL import Image
import PIL
import numpy as np
import numbers
import torch
import torchvision.transforms.functional as F


class GroupResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0.5, contrast=0.8, saturation=0.5, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        if np.random.rand()>= 0.5:
            if isinstance(clip[0], np.ndarray):
                raise TypeError(
                    'Color jitter not yet implemented for numpy arrays')
            elif isinstance(clip[0], PIL.Image.Image):
                brightness, contrast, saturation, hue = self.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)

                # Create img transform function sequence
                img_transforms = []
                if brightness is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
                if saturation is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
                if hue is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
                if contrast is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
                random.shuffle(img_transforms)

                # Apply to all images
                jittered_clip = []
                for img in clip:
                    for func in img_transforms:
                        jittered_img = func(img)
                    jittered_clip.append(jittered_img)

            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                'but got list of {0}'.format(type(clip[0])))
            return jittered_clip
        else:
            return clip


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert (img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    def __call__(self, img_group):
        if random.random() < 0.5:
            img_group = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
        return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):  # (T, 3, 224, 224)
        for b in range(tensor.size(0)):
            for t, m, s in zip(tensor[b], self.mean, self.std):
                t.sub_(m).div_(s)
        return tensor


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        if np.random.rand() >= 0.5:
            for b in range(tensor.size(0)):
                noise = torch.randn(tensor.size())
                for t, m, s in zip(tensor[b], self.mean, self.std):
                    t + noise * self.std + self.mean
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class LoopPad(object):

    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, tensor):
        length = tensor.size(0)

        if length == self.max_len:
            return tensor

        # repeat the clip as many times as is necessary
        n_pad = self.max_len - length
        pad = [tensor] * (n_pad // length)
        if n_pad % length > 0:
            pad += [tensor[0:n_pad % length]]

        tensor = torch.cat([tensor] + pad, 0)
        return tensor


# NOTE: Returns [0-255] rather than torchvision's [0-1]
class ToTensor(object):
    def __init__(self):
        self.worker = lambda x: F.to_tensor(x) * 255

    def __call__(self, img_group):
        img_group = [self.worker(img) for img in img_group]
        return torch.stack(img_group, 0)
