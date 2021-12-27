import h5py
import PIL.Image as Image
import numpy as np
import os
import glob
import scipy
from image import *
from model import CANNet
import torch
import torch.nn as nn
from torch.autograd import Variable
from matplotlib import cm
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
import cv2
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
])

# the folder contains all the test images
img_folder = '.\\archive\\ShanghaiTech\\part_B\\test_data\\images'

img_paths = [f'{img_folder}\\IMG_2.jpg']

# for img_path in glob.glob(os.path.join(img_folder, '*.jpg')):
#     img_paths.append(img_path)

model = CANNet()

model = model.cuda()

checkpoint = torch.load('_model_best.pth.tar')

model.load_state_dict(checkpoint['state_dict'])
# model.backend.register_forward_hook(lambda m, input, output: print(output.shape))
model.eval()

pred = []
gt = []


def plotDensity(density, plot_path):
    '''
    @density: np array of corresponding density map
    @plot_path: path to save the plot
    '''
    density = density * 255.0
    #print("SHAPE: ",density.shape)
    # plot with overlay
    colormap_i = cm.jet(density)[:, :, 0:3]

    overlay_i = colormap_i

    new_map = overlay_i.copy()
    new_map[:, :, 0] = overlay_i[:, :, 2]
    new_map[:, :, 2] = overlay_i[:, :, 0]
    #print(new_map)
    cv2.imshow('xx', new_map)
    cv2.waitKey(1000)
    cv2.imwrite(plot_path, new_map * 255)


for i in range(len(img_paths)):
    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    img = img.unsqueeze(0)
    h, w = img.shape[2:4]
    h_d = h // 2
    w_d = w // 2
    img_1 = Variable(img[:, :, :h_d, :w_d].cuda())
    img_2 = Variable(img[:, :, :h_d, w_d:].cuda())
    img_3 = Variable(img[:, :, h_d:, :w_d].cuda())
    img_4 = Variable(img[:, :, h_d:, w_d:].cuda())

    density_1 = model(img_1).data.cpu().numpy()
    density_2 = model(img_2).data.cpu().numpy()
    density_3 = model(img_3).data.cpu().numpy()
    density_4 = model(img_4).data.cpu().numpy()




    pure_name = os.path.splitext(os.path.basename(img_paths[i]))[0]
    gt_file = h5py.File(img_paths[i].replace('.jpg', '.h5').replace('images', 'ground-truth-h5'), 'r')
    groundtruth = np.asarray(gt_file['density'])
    pred_sum = density_1.sum() + density_2.sum() + density_3.sum() + density_4.sum()

    pred.append(pred_sum)
    gt.append(np.sum(groundtruth))


mae = mean_absolute_error(pred, gt)
rmse = np.sqrt(mean_squared_error(pred, gt))

print('MAE: ', mae)
print('RMSE: ', rmse)
