import os
from PIL import Image
from scipy.io import loadmat
import numpy as np
import torch

from torchvision import transforms

class PASCALDataset(object):
    def __init__(self, root):
        self.root = root
        self.transforms = transforms.Compose([transforms.ToTensor()])
        # load all image files
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        image_number = img_path.split('/')[-1].split('.')[0]
        bbox_path = os.path.join(self.root, "BBox", image_number + '_bbox.mat')
        label_path = os.path.join(self.root, "Labels", image_number + '_label.mat')

        img = Image.open(img_path).convert("RGB")
        boxes = loadmat(bbox_path)
        boxes = boxes['bboxes']		
        labels = loadmat(label_path)
        labels = labels['labels'].flatten()	
        num_of_objects = len(labels)
        #print(labels.shape)

        labels = torch.as_tensor(labels , dtype=torch.int64)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_of_objects,), dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img= self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
