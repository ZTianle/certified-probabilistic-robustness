import os
import shutil
import time

import torch
import torch.utils.data
from . import imagenet_models as models
from . import imagenet_models, cifar_models
from torchvision import transforms, datasets
ch = torch

from .tools import constants
from . import data_augmentation as da
from . import loaders
from . import cifar_models

from .tools.helpers import get_label_mapping

###
# Datasets: (all subclassed from dataset)
# In order:
## ImageNet
## Restricted Imagenet 
## Other Datasets:
## - CIFAR
## - CINIC
## - A2B (orange2apple, horse2zebra, etc)
###

class DataSet(object):
    '''
    Base class for representing a dataset. Meant to be subclassed, with
    subclasses implementing the `get_model` function. 
    '''

    def __init__(self, ds_name, data_path, **kwargs):
        '''
        Args:
            ds_name (str) : string identifier for the dataset
            data_path (str) : path to the dataset 
            num_classes (int) : *required kwarg*, the number of classes in
                the dataset
            mean (ch.tensor) : *required kwarg*, the mean to normalize the
                dataset with (e.g.  :samp:`ch.tensor([0.4914, 0.4822,
                0.4465])` for CIFAR-10)
            std (ch.tensor) : *required kwarg*, the standard deviation to
                normalize the dataset with (e.g. :samp:`ch.tensor([0.2023,
                0.1994, 0.2010])` for CIFAR-10)
            custom_class (type) : *required kwarg*, a
                :samp:`torchvision.models` class corresponding to the
                dataset, if it exists (otherwise :samp:`None`)
            label_mapping (dict[int,str]) : *required kwarg*, a dictionary
                mapping from class numbers to human-interpretable class
                names (can be :samp:`None`)
            transform_train (torchvision.transforms) : *required kwarg*, 
                transforms to apply to the training images from the
                dataset
            transform_test (torchvision.transforms) : *required kwarg*,
                transforms to apply to the validation images from the
                dataset
        '''
        required_args = ['num_classes', 'mean', 'std', 'custom_class',
            'label_mapping', 'transform_train', 'transform_test']
        assert set(kwargs.keys()) == set(required_args), "Missing required args"
        self.ds_name = ds_name
        self.data_path = data_path
        self.__dict__.update(kwargs)

    def get_model(self, arch, pretrained):
        '''
        Should be overriden by subclasses. Also, you will probably never
        need to call this function, and should instead by using
        `model_utils.make_and_restore_model </source/robustness.model_utils.html>`_.

        Args:
            arch (str) : name of architecture 
            pretrained (bool): whether to try to load torchvision 
                pretrained checkpoint

        Returns:
            A model with the given architecture that works for each
            dataset (e.g. with the right input/output dimensions).
        '''

        raise NotImplementedError

    def make_loaders(self, workers, batch_size, data_aug=True, test_subset=None, test_subset_size=1000, subset=None, 
                     subset_start=0, subset_type='rand', val_batch_size=None,
                     only_val=False):
        '''
        Args:
            workers (int) : number of workers for data fetching (*required*).
                batch_size (int) : batch size for the data loaders (*required*).
            data_aug (bool) : whether or not to do train data augmentation.
            subset (None|int) : if given, the returned training data loader
                will only use a subset of the training data; this should be a
                number specifying the number of training data points to use.
            subset_start (int) : only used if `subset` is not None; this specifies the
                starting index of the subset.
            subset_type ("rand"|"first"|"last") : only used if `subset is
                not `None`; "rand" selects the subset randomly, "first"
                uses the first `subset` images of the training data, and
                "last" uses the last `subset` images of the training data.
            val_batch_size (None|int) : if not `None`, specifies a
                different batch size for the validation set loader.
            only_val (bool) : If `True`, returns `None` in place of the
                training data loader

        Returns:
            A training loader and validation loader according to the
            parameters given. These are standard PyTorch data loaders, and
            thus can just be used via:

            >>> train_loader, val_loader = ds.make_loaders(workers=8, batch_size=128)
            >>> for im, lab in train_loader:
            >>>     # Do stuff...
        '''
        transforms = (self.transform_train, self.transform_test)
        return loaders.make_loaders(workers=workers,
                                    batch_size=batch_size,
                                    transforms=transforms,
                                    data_path=self.data_path,
                                    data_aug=data_aug,
                                    dataset=self.ds_name,
                                    label_mapping=self.label_mapping,
                                    custom_class=self.custom_class,
                                    val_batch_size=val_batch_size,
                                    test_subset=test_subset,
                                    test_subset_size=test_subset_size,
                                    subset=subset,
                                    subset_start=subset_start,
                                    subset_type=subset_type,
                                    only_val=only_val)

class ImageNet(DataSet):
    '''
    ImageNet Dataset [DDS+09]_.

    Requires ImageNet in ImageFolder-readable format. 
    ImageNet can be downloaded from http://www.image-net.org. See
    `here <https://pytorch.org/docs/master/torchvision/datasets.html#torchvision.datasets.ImageFolder>`_
    for more information about the format.

    .. [DDS+09] Deng, J., Dong, W., Socher, R., Li, L., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. 2009 IEEE Conference on Computer Vision and Pattern Recognition, 248-255.
    '''
    def __init__(self, data_path, aug_method=None, **kwargs):
        if aug_method == "RandAugment":
            TRAIN_TRANSFORMS = da.TRAIN_TRANSFORMS_IMAGENET_RANDAUGMENT
        elif aug_method == "TrivialAugment":
            TRAIN_TRANSFORMS = da.TRAIN_TRANSFORMS_IMAGENET_TRIVIALAUGMENT
        elif aug_method == "AugMix":
            TRAIN_TRANSFORMS = da.TRAIN_TRANSFORMS_IMAGENET_AUGMIX
        else:
            TRAIN_TRANSFORMS = da.TRAIN_TRANSFORMS_IMAGENET

        ds_kwargs = {
            'num_classes': 1000,
            'mean': ch.tensor([0.485, 0.456, 0.406]),
            'std': ch.tensor([0.229, 0.224, 0.225]),
            'custom_class': None,
            'label_mapping': None,
            'transform_train': TRAIN_TRANSFORMS,
            'transform_test': da.TEST_TRANSFORMS_IMAGENET
        }
        super(ImageNet, self).__init__('imagenet', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        """
        """
        return models.__dict__[arch](num_classes=self.num_classes, 
                                        pretrained=pretrained)

class ImageNetNoCrop(ImageNet):
    def __init__(self, data_path, aug_method=None,**kwargs):
        ds_kwargs = {
            'num_classes': 1000,
            'mean': ch.tensor([0.485, 0.456, 0.406]),
            'std': ch.tensor([0.229, 0.224, 0.225]),
            'custom_class': None,
            'label_mapping': None,
            'transform_train': da.TRAIN_TRANSFORMS_IMAGENET_NOCROP,
            'transform_test': da.TEST_TRANSFORMS_IMAGENET
        }

        super(ImageNet, self).__init__('imagenet', data_path, **ds_kwargs)

class RestrictedImageNet(DataSet):
    '''
    RestrictedImagenet Dataset [TSE+19]_

    A subset of ImageNet with the following labels:

    * Dog (classes 151-268)
    * Cat (classes 281-285)
    * Frog (classes 30-32)
    * Turtle (classes 33-37)
    * Bird (classes 80-100)
    * Monkey (classes 365-382)
    * Fish (classes 389-397)
    * Crab (classes 118-121)
    * Insect (classes 300-319)

    To initialize, just provide the path to the full ImageNet dataset
    (no special formatting required).

    .. [TSE+19] Tsipras, D., Santurkar, S., Engstrom, L., Turner, A., &
        Madry, A. (2019). Robustness May Be at Odds with Accuracy. ICLR
        2019.
    '''
    def __init__(self, data_path, aug_method=None, **kwargs):
        ds_name = 'restricted_imagenet'

        if aug_method == "RandAugment":
            TRAIN_TRANSFORMS = da.TRAIN_TRANSFORMS_IMAGENET_RANDAUGMENT
        elif aug_method == "TrivialAugment":
            TRAIN_TRANSFORMS = da.TRAIN_TRANSFORMS_IMAGENET_TRIVIALAUGMENT
        elif aug_method == "AugMix":
            TRAIN_TRANSFORMS = da.TRAIN_TRANSFORMS_IMAGENET_AUGMIX
        else:
            TRAIN_TRANSFORMS = da.TRAIN_TRANSFORMS_IMAGENET
            
        ds_kwargs = {
            'num_classes': len(constants.RESTRICTED_IMAGNET_RANGES),
            'mean': ch.tensor([0.4717, 0.4499, 0.3837]), 
            'std': ch.tensor([0.2600, 0.2516, 0.2575]),
            'custom_class': None,
            'label_mapping': get_label_mapping(ds_name,
                constants.RESTRICTED_IMAGNET_RANGES),
            'transform_train': TRAIN_TRANSFORMS,
            'transform_test': da.TEST_TRANSFORMS_IMAGENET
        }
        super(RestrictedImageNet, self).__init__(ds_name,
                data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        """
        """
        if pretrained:
            raise ValueError("Dataset doesn't support pytorch_pretrained")
        return models.__dict__[arch](num_classes=self.num_classes)

class CustomImageNet(DataSet):
    '''
    CustomImagenet Dataset 
    A subset of ImageNet with the user-specified labels
    To initialize, just provide the path to the full ImageNet dataset
    along with a list of lists of wnids to be grouped together
    (no special formatting required).
    '''
    def __init__(self, data_path, custom_grouping, aug_method=None,**kwargs):
        """
        """
        ds_name = 'custom_imagenet'

        if aug_method == "RandAugment":
            TRAIN_TRANSFORMS = da.TRAIN_TRANSFORMS_IMAGENET_RANDAUGMENT
        elif aug_method == "TrivialAugment":
            TRAIN_TRANSFORMS = da.TRAIN_TRANSFORMS_IMAGENET_TRIVIALAUGMENT
        elif aug_method == "AugMix":
            TRAIN_TRANSFORMS = da.TRAIN_TRANSFORMS_IMAGENET_AUGMIX
        else:
            TRAIN_TRANSFORMS = da.TRAIN_TRANSFORMS_IMAGENET
            
        ds_kwargs = {
            'num_classes': len(custom_grouping),
            'mean': ch.tensor([0.4717, 0.4499, 0.3837]), 
            'std': ch.tensor([0.2600, 0.2516, 0.2575]),
            'custom_class': None,
            'label_mapping': get_label_mapping(ds_name,
                custom_grouping),
            'transform_train': TRAIN_TRANSFORMS,
            'transform_test': da.TEST_TRANSFORMS_IMAGENET
        }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super(CustomImageNet, self).__init__(ds_name,
                data_path, **ds_kwargs)

class ImageNet100(CustomImageNet):
    def __init__(self, data_path, **kwargs):
        """
        """
        ds_name = 'imagenet100'
        super(ImageNet100, self).__init__(ds_name,
            data_path=data_path,
            custom_grouping=[[label] for label in range(0, 1000, 10)],
            **kwargs,)

    def get_model(self, arch, pretrained):
        """
        """
        if pretrained:
            raise ValueError("Dataset doesn't support pytorch_pretrained")
        return imagenet_models.__dict__[arch](num_classes=self.num_classes)

class CIFAR(DataSet):
    """
    CIFAR-10 dataset [Kri09]_.

    A dataset with 50k training images and 10k testing images, with the
    following classes:

    * Airplane
    * Automobile
    * Bird
    * Cat
    * Deer
    * Dog
    * Frog
    * Horse
    * Ship
    * Truck

    .. [Kri09] Krizhevsky, A (2009). Learning Multiple Layers of Features
        from Tiny Images. Technical Report.
    """
    def __init__(self, data_path='/tmp/', aug_method=None, **kwargs):
        if aug_method == "RandAugment":
            TRAIN_TRANSFORMS = da.TRAIN_TRANSFORMS_DEFAULT_RANDAUGMENT(32)
        elif aug_method == "TrivialAugment":
            TRAIN_TRANSFORMS = da.TRAIN_TRANSFORMS_DEFAULT_TRIVIALAUGMENT(32)
        elif aug_method == "AugMix":
            TRAIN_TRANSFORMS = da.TRAIN_TRANSFORMS_DEFAULT_AUGMIX(32)
        else:
            TRAIN_TRANSFORMS = da.TRAIN_TRANSFORMS_DEFAULT(32)

        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0.4914, 0.4822, 0.4465]),
            'std': ch.tensor([0.2023, 0.1994, 0.2010]),
            'custom_class': datasets.CIFAR10,
            'label_mapping': None, 
            'transform_train': TRAIN_TRANSFORMS,
            'transform_test': da.TEST_TRANSFORMS_DEFAULT(32)
        }
        super(CIFAR, self).__init__('cifar', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        """
        """
        if pretrained:
            raise ValueError('CIFAR does not support pytorch_pretrained=True')
        return cifar_models.__dict__[arch](num_classes=self.num_classes)

class CIFAR100(DataSet):
    """
    CIFAR-100 dataset [Kri09]_.

    A dataset with 50k training images and 10k testing images, with 100 classes:

    .. [Kri09] Krizhevsky, A (2009). Learning Multiple Layers of Features
        from Tiny Images. Technical Report.
    """
    def __init__(self, data_path='/tmp/', aug_method=None, **kwargs):

        if aug_method == "RandAugment":
            TRAIN_TRANSFORMS = da.TRAIN_TRANSFORMS_DEFAULT_RANDAUGMENT(32)
        elif aug_method == "TrivialAugment":
            TRAIN_TRANSFORMS = da.TRAIN_TRANSFORMS_DEFAULT_TRIVIALAUGMENT(32)
        elif aug_method == "AugMix":
            TRAIN_TRANSFORMS = da.TRAIN_TRANSFORMS_DEFAULT_AUGMIX(32)
        else:
            TRAIN_TRANSFORMS = da.TRAIN_TRANSFORMS_DEFAULT(32)

        ds_kwargs = {
            'num_classes': 100,
            'mean': ch.tensor([0.5071, 0.4867, 0.4408]),
            'std': ch.tensor([0.2675, 0.2565, 0.2761]),
            'custom_class': datasets.CIFAR100,
            'label_mapping': None, 
            'transform_train': TRAIN_TRANSFORMS,
            'transform_test': da.TEST_TRANSFORMS_DEFAULT(32)
        }
        super(CIFAR100, self).__init__('cifar100', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        """
        """
        if pretrained:
            raise ValueError('CIFAR does not support pytorch_pretrained=True')
        return cifar_models.__dict__[arch](num_classes=self.num_classes)

class CINIC(DataSet):
    """
    CINIC-10 dataset [DCA+18]_.

    A dataset with the same classes as CIFAR-10, but with downscaled images
    from various matching ImageNet classes added in to increase the size of
    the dataset.

    .. [DCA+18] Darlow L.N., Crowley E.J., Antoniou A., and A.J. Storkey
        (2018) CINIC-10 is not ImageNet or CIFAR-10. Report
        EDI-INF-ANC-1802 (arXiv:1810.03505)
    """
    def __init__(self, data_path, **kwargs):
        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0.47889522, 0.47227842, 0.43047404]),
            'std': ch.tensor([0.24205776, 0.23828046, 0.25874835]),
            'custom_class': None,
            'label_mapping': None,
            'transform_train': da.TRAIN_TRANSFORMS_DEFAULT(32),
            'transform_test': da.TEST_TRANSFORMS_DEFAULT(32)
        }
        super(CINIC, self).__init__('cinic', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        """
        """
        if pretrained:
            raise ValueError('CINIC does not support pytorch_pretrained=True')
        return cifar_models.__dict__[arch](num_classes=self.num_classes)

class A2B(DataSet):
    """
    A-to-B datasets [ZPI+17]_

    A general class for image-to-image translation dataset. Currently
    supported are:

    * Horse <-> Zebra
    * Apple <-> Orange
    * Summer <-> Winter

    .. [ZPI+17] Zhu, J., Park, T., Isola, P., & Efros, A.A. (2017).
        Unpaired Image-to-Image Translation Using Cycle-Consistent
        Adversarial Networks. 2017 IEEE International Conference on
        Computer Vision (ICCV), 2242-2251.
    """
    def __init__(self, data_path, **kwargs):
        _, ds_name = os.path.split(data_path)
        valid_names = ['horse2zebra', 'apple2orange', 'summer2winter_yosemite']
        assert ds_name in valid_names, \
                f"path must end in one of {valid_names}, not {ds_name}"
        ds_kwargs = {
            'num_classes': 2,
            'mean': ch.tensor([0.5, 0.5, 0.5]),
            'std': ch.tensor([0.5, 0.5, 0.5]),
            'transform_train': da.TRAIN_TRANSFORMS_IMAGENET,
            'transform_test': da.TEST_TRANSFORMS_IMAGENET
        }
        super(A2B, self).__init__(ds_name, data_path, **ds_kwargs)

    def get_model(self, arch, pretrained=False):
        """
        """
        if pretrained:
            raise ValueError('A2B does not support pytorch_pretrained=True')
        return models.__dict__[arch](num_classes=self.num_classes)

class MNIST(DataSet):
    def __init__(self, data_path='/tmp/', **kwargs):
        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0., 0., 0.]),
            'std': ch.tensor([1., 1., 1.]),
            'custom_class': datasets.MNIST,
            'label_mapping': None, 
            'transform_train': transforms.ToTensor(), # TODO
            'transform_test': transforms.ToTensor() # TODO
        }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super(MNIST, self).__init__('mnist', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        if pretrained:
            raise ValueError('MNIST does not support pytorch_pretrained=True')
        return cifar_models.__dict__[arch](num_classes=self.num_classes)

DATASETS = {
    'imagenet': ImageNet,
    'imagenet_nocrop': ImageNetNoCrop,
    'imagenet100': ImageNet100,
    'restricted_imagenet': RestrictedImageNet,
    'cifar': CIFAR,
    'cifar100': CIFAR100,
    'cinic': CINIC,
    'a2b': A2B,
    'mnist': MNIST
}
'''
Dictionary of datasets. A dataset class can be accessed as:

>>> import robustness.datasets
>>> ds = datasets.DATASETS['cifar']('/path/to/cifar')
'''
