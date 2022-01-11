import torch 
import os
# from Caffe2Pytorch.caffe2pth.caffenet import *
import cv2
from torchviz import make_dot
from PIL import Image
from torchvision import transforms

class MyDataset():
    def __init__(self, images):
        self.images = images
        self.crop = transforms.FiveCrop(size=10)
        
    def __getitem__(self, index):
        image = self.images[index]
        # image dim: [3, 100, 100]
        # Crop to smaller image patches
        crops = self.crop(image)
        crops = torch.stack([transforms.ToTensor()(crop) for crop in self.images])
        # Crops dim: [5, 3, 10, 10]
        
        return crops
        
    def __len__(self):
        return len(self.images)


def testCaffe(img, model):
    # a = Image.open(imgPath)
    a = img.resize((224,224), Image.BILINEAR)
    dataset = MyDataset([a])

    # model = CaffeNet('models/ResNet_50/ResNet_50_test.prototxt')
    # model.load_state_dict(torch.load('test.pt'))
    model = model.eval()


    with torch.no_grad():
        processing = model(dataset[0])
        output = processing["pool5"]
        
    
        
        # ---- L2-norm Feature ------
        ff = output.data.cpu()
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        
        numpy = output.cpu().detach().numpy()
        numpy = numpy.flatten()
        return numpy