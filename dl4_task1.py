import torch
import numpy as np
from PIL import Image
import tarfile
from torch.utils.data import Dataset
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.models as models

root_dir = "/content/drive/"
labels = os.path.join(root_dir,"ILSVRC2012_bbox_val_v3.tgz")
images = os.path.join(root_dir,"imagenet2500.tar")
image_folder = os.path.join(root_dir,"imagespart")
label_folder = os.path.join(root_dir,"val")
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

def test_image(image):
    img = mpimg.imread(image)
    imgplot = plt.imshow(img)
    plt.show()

def test():
    resnet = models.resnet18(pretrained=True)

#comment out after run
def main():
    if not os.path.isdir(os.path.join(root_dir,image_folder)):
        open_tar(images)
    if not os.path.isdir(os.path.join(root_dir,label_folder)):
        open_tar(labels)
    sample = DataLoader(image_folder,label_folder)


if __name__=="__main__":
    main()
