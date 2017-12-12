import sys
sys.path.append('.')

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms
from torchvision import datasets

import os
import argparse


parser = argparse.ArgumentParser(description='Feature extractor for image literals')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

args = parser.parse_args()


class CustomImageFolder(datasets.ImageFolder):

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, path


# Load up pretrained Resnet-18
model = models.resnet18(pretrained=True)

# Chop last fc layer
model = nn.Sequential(*list(model.children())[:-1])

# Data loader
img_dir = os.path.join(args.data)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

img_loader = torch.utils.data.DataLoader(
    CustomImageFolder(img_dir, transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=False)

model.eval()

imgs = [int(x[0][-9:-4])-1 for x in img_loader.dataset.imgs]
features = np.zeros([np.max(imgs)+1, 512], np.float32)

print(features.shape)

for i, (input, path) in enumerate(img_loader):
    input_var = Variable(input, volatile=True)
    output = model(input_var).view(-1, 512).data.numpy()

    # From filename to index: minus 1
    idxs = [int(x[-9:-4])-1 for x in path]

    # Populate array
    features[idxs, :] = output

    print(i)

# Save features
features = np.vstack(features)
print('Saving features of size {}'.format(features.shape))
np.save('data/ml-100k/bin/image_literals.npy', features)
