import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import os
import numpy as np
import torch.nn.init
import random
import glob
import tqdm
from src import center

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='A Spatial Guided Self-supervised Clustering Network for Medical Image Segmentation')
parser.add_argument('--nChannel', metavar='N', default=100, type=int, 
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=50, type=int, 
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int, 
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, 
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int, 
                    help='number of convolutional layers')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image folder path', required=True)
parser.add_argument('--stepsize_ce', metavar='CE', default=1, type=float,
                    help='step size for cross entropy loss', required=False)
parser.add_argument('--stepsize_ss', metavar='SS', default=5, type=float, 
                    help='step size for sparse spatial loss')
parser.add_argument('--center', action='store_true', default=False, 
                    help='use context-based consistency loss')
args = parser.parse_args()

# Convolutional Segmentation Network
class CSNet(nn.Module):
    def __init__(self,input_dim):
        super(CSNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(args.nChannel) )
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

def get_variance(s_map, x_c, y_c):

    h,w = s_map.shape
    x_map, y_map = center.get_coordinate_tensors(h,w)
    x_map = torch.transpose(x_map,1,0)
    y_map = torch.transpose(y_map,1,0)

    v_x_map = (x_map - x_c) * (x_map - x_c)
    v_y_map = (y_map - y_c) * (y_map - y_c)

    v_x = (s_map * v_x_map).sum()
    v_y = (s_map * v_y_map).sum()
    return v_x, v_y


# load image
img_list = sorted(glob.glob(args.input+ '/test/*'))
if not os.path.exists(os.path.join(args.input, 'result/')):
    os.mkdir(os.path.join(args.input, 'result/'))
print('Testing '+str(len(img_list))+' images.')

# for each image
for img_file in tqdm.tqdm(img_list):

    im = cv2.imread(img_file)
    print(args.input)
    data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )
    if use_cuda:
        data = data.cuda()
    data = Variable(data)

    # Load the model
    model = CSNet(data.size(1))
    if use_cuda:
        model.cuda()
    model.train()

    # Cross-Entropy loss definition
    loss_ce = torch.nn.CrossEntropyLoss()

    # Sparse Spatial loss definition
    loss_mpy = torch.nn.L1Loss(size_average = True)
    loss_mpz = torch.nn.L1Loss(size_average = True)

    mpy_target = torch.zeros(im.shape[0]-1, im.shape[1], args.nChannel)
    mpz_target = torch.zeros(im.shape[0], im.shape[1]-1, args.nChannel)
    
    if use_cuda:
        mpy_target = mpy_target.cuda()
        mpz_target = mpz_target.cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    label_colours = np.random.randint(255,size=(100,3))
    
    # Training
    for batch_idx in range(args.maxIter):
        # forwarding
        optimizer.zero_grad()
        output = model(data)[0]
        
        # Context-based consistency loss
        if args.center:
            C,H,W = np.shape(output)
            closs = 0
            epsilon = 1e-3
            centers_all = center.get_centers(output)
            for c in range(C):
                output_map = output[c,:,:] + epsilon
                k = output_map.sum()
                output_map_pdf = output_map/k
                x_c, y_c = centers_all[c]
                v_x, v_y = get_variance(output_map_pdf, x_c, y_c)
                loss_per_output = (v_x + v_y)
                closs = loss_per_output + closs


        output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
        
        #Reshaping it to row, col, channel
        outputmp = output.reshape( (im.shape[0], im.shape[1], args.nChannel) )

        # Calculate the vertical and horizontal differens of the segmentation map
        mpy = outputmp[1:, :, :] - outputmp[0:-1, :, :]
        mpz = outputmp[:, 1:, :] - outputmp[:, 0:-1, :]


        lmpy = loss_mpy(mpy,mpy_target)
        lmpz = loss_mpz(mpz,mpz_target)

        ignore, target = torch.max( output, 1 )
        img_target = target.data.cpu().numpy()
        nLabels = len(np.unique(img_target))

        if args.center:
            loss = args.stepsize_ce * loss_ce(output, target) + args.stepsize_ss * (lmpy + lmpz) + closs
            loss.backward()
        else:
            loss = args.stepsize_ce * loss_ce(output, target) + args.stepsize_ss * (lmpy + lmpz)
            loss.backward()
        optimizer.step()

        print (batch_idx, '/', args.maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())
        if nLabels <= args.minLabels:
            print ("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
            break

    # save output image
    output = model(data)[0]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    
    ignore, target = torch.max(output, 1)

    img_target = target.data.cpu().numpy()
    img_target_rgb = np.array([label_colours[ c % args.nChannel ] for c in img_target])
    img_target_rgb = img_target_rgb.reshape(im.shape).astype( np.uint8 )

    imgpath =  os.path.join(args.input, 'result/') + os.path.basename(img_file)
    print(imgpath)
    cv2.imwrite( os.path.join(args.input, 'result/') + os.path.basename(img_file), img_target_rgb )



