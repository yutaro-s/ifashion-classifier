import json
from pathlib import Path
import torch
from torchvision.datasets import VisionDataset
from PIL import Image


class iFashionAttribute(VisionDataset):
    """`iFashion Attributes <https://github.com/visipedia/imat_fashion_comp>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    def __init__(self, root, annFile, transform=None, num_label=228):
        super(iFashionAttribute, self).__init__(root=root, transform=transform)
        self.info = self.__load_annotations__(annFile)
        self.num_label = num_label
        self.list_id = list(sorted(self.info.keys()))

    def __load_annotations__(self, annFile):
        # load json
        with Path(annFile).open(mode='r') as f:
            info = json.load(f)
        return info

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (sample, target)
        """
        img_id = self.list_id[index]

        # load image
        filename = self.info[img_id]['filename']
        img_path = Path(self.root, filename)
        sample = Image.open(img_path, mode='r').convert('RGB')

        # load label
        label = self.info[img_id]['label']
        target = self.__pad__(label)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __pad__(self, label):
        target = torch.zeros(self.num_label)
        target.scatter_(0, torch.LongTensor(label), 1)
        return target

    def __len__(self):
        return len(self.list_id)
