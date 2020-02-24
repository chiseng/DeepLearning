import torch
import numpy as np
from PIL import Image
import tarfile
from torch.utils.data import Dataset
import os
from torchvision import transforms
import matplotlib as pyplot

labels = "ILSVRC2012_bbox_val_v3.tgz"
images = "imagenet2500.tar"
image_folder = "imagespart"
label_folder = "val"
def open_tar(fname):
    try:
        if fname.endswith("tgz"):
            tar = tarfile.open(fname,"r:gz")
        else:
            tar = tarfile.open(fname,"r:tar")
        for item in tar:
            tar.extract(item)
    except FileNotFoundError:
        return "File not found"

class DataLoader(Dataset):
    def __init__(self, image_folder, label_folder,transform=None):
        self.labels = [name for name in os.listdir(label_folder)]
        self.images = [name for name in os.listdir(image_folder)]
        self.i_dir = image_folder
        self.l_dir = label_folder
        self.transform = transforms.CenterCrop((224,224))

    def __len__(self):
        return len(self.images)

    def resize(self,image,size):
        width, height = image.shape
        rescale = size/min(width,height)
        resized = (np.ceil(width*rescale),np.ceil(height*rescale))
        image.resize(resized)
        return image


    #getitem to load image? Then how about labels?
    def __getitem__(self, idx):
        label = self.labels[idx]
        image_name = self.images[idx]
        open_image = Image.open(os.path.join(self.i_dir,image_name))
        if self.transform is not None:
            resized = self.resize(open_image,224)
            image = self.transform(resized)
        open_label = open(os.path.join(self.l_dir,label)).read()
        sample = (image,open_label)

        return sample

#comment out after run
if not os.path.isdir(image_folder):
    open_tar(images)
if not os.path.isdir(label_folder):
    open_tar(labels)
sample = DataLoader(image_folder,label_folder)

