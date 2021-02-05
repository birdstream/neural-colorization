import torch
import torch.nn as nn
import argparse
from torch.autograd import Variable
import torchvision.models as models
import os
from torch.utils import data
from model import generator
import numpy as np
from PIL import Image, ImageStat
from skimage.color import rgb2yuv,yuv2rgb,rgb2hsv
import cv2
import torchvision.transforms as transforms
import sys
from matplotlib import pyplot as plt
from PIL import ImageFilter
import adabound as optim
import curses
from unet import UNet

def keypress(stdscr):
    stdscr.nodelay(True)
    return stdscr.getch()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a GAN based model")
    parser.add_argument("-d",
                        "--training_dir",
                        type=str,
                        required=True,
                        help="Training directory (folder contains all images)")
    parser.add_argument("-t",
                        "--test_image",
                        type=str,
                        default=None,
                        help="Test image location")
    parser.add_argument("-t2",
                        "--test_image2",
                        type=str,
                        default=None,
                        help="Test image 2 location")
    parser.add_argument("-c",
                        "--checkpoint_location",
                        type=str,
                        required=True,
                        help="Place to save checkpoints")
    parser.add_argument("-e",
                        "--epoch",
                        type=int,
                        default=120,
                        help="Epoches to run training")
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="which GPU to use?")
    parser.add_argument("-b",
                        "--batch_size",
                        type=int,
                        default=20,
                        help="batch size")
    parser.add_argument("-w",
                        "--num_workers",
                        type=int,
                        default=6,
                        help="Number of workers to fetch data")
    parser.add_argument("-p",
                        "--pixel_loss_weights",
                        type=float,
                        default=1000.0,
                        help="Pixel-wise loss weights")
    parser.add_argument("--g_every",
                        type=int,
                        default=1,
                        help="Training generator every k iteration")
    parser.add_argument("--g_lr",
                        type=float,
                        default=1e-4,
                        help="learning rate for generator")
    parser.add_argument("--d_lr",
                        type=float,
                        default=1e-4,
                        help="learning rate for discriminator")
    parser.add_argument("-i",
                        "--checkpoint_every",
                        type=int,
                        default=100,
                        help="Save checkpoint every k iteration (checkpoints for same epoch will overwrite)")
    parser.add_argument("--d_init",
                        type=str,
                        default=None,
                        help="Init weights for discriminator")
    parser.add_argument("--g_init",
                        type=str,
                        default=None,
                        help="Init weights for generator")
    parser.add_argument("--d_disable",
                        action="store_true",
                        help="Disable training of discriminator")
    parser.add_argument("--g_disable",
                        action="store_true",
                        help="Disable training of generator")
    parser.add_argument("--label_fake",
                        type=float,
                        default=0.0,
                        help="Set fake label value")
    parser.add_argument("--label_real",
                        type=float,
                        default=1,
                        help="Set real label value")
    parser.add_argument("--smooth",
                        type=float,
                        default=0,
                        help="Set smooth value")
    parser.add_argument("--images",
                        type=int,
                        default=0,
                        help="Set no. of images to train in one epoch")
    parser.add_argument("--runs",
                        type=int,
                        default=0,
                        help="Set no. of runs")
    parser.add_argument("--flip",
                        type=float,
                        default=0,
                        help="Chance of flipping labels for D (0.0-1.0)")
    parser.add_argument("--res",
                        type=int,
                        default=224,
                        help="Resolution")
    parser.add_argument("--train-part",
                        action="store_true",
                        help="Train only part of network")
    parser.add_argument("--nogan",
                        action="store_true",
                        help="Dont use GAN")
    args = parser.parse_args()
    return args

args = parse_args()
if not os.path.exists(os.path.join(args.checkpoint_location,'weights')):
    os.makedirs(os.path.join(args.checkpoint_location,'weights'))

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
# define data generator
class img_data(data.Dataset):
    def __init__(self, path):
        files = (os.listdir(path))
        self.files = [os.path.join(path,x) for x in files[:args.images]]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert('RGB')
        min_size = min(img.size)
        transform = transforms.Compose([
            transforms.CenterCrop(min_size),
            transforms.Resize(args.res),
            transforms.RandomHorizontalFlip()
#            transforms.ToTensor(),
#            AddGaussianNoise(0.,0.1),
#            transforms.ToPILImage()
            ])
        img = transform(img)
#        img = np.array(transform(img))
#        if np.median(rgb2hsv(img)[...,1]) < 0.1:
#            img = Image.new('RGB',(args.res, args.res),color = 0)
        yuv = rgb2yuv(img)
        y = yuv[...,0]-0.5 + np.random.normal(0,1)
        u_t = yuv[...,1] / 0.43601035
        v_t = yuv[...,2] / 0.61497538
        return torch.Tensor(np.expand_dims(y,axis=0)),torch.Tensor(np.stack([u_t,v_t],axis=0))



# Define G, same as torch version
#
G = UNet(1, 2).cuda(args.gpu)

#if args.train_part:
#    for param in G.parameters():
#        param.requires_grad = False
#    for param in G[15:].parameters():
#        param.requires_grad = True

# define D
if not args.nogan:
    D = models.resnet18(pretrained=False)

    if args.train_part:
        for param in D.parameters():
            param.requires_grad = False
    D.fc = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())
    D.avgpool = nn.AdaptiveAvgPool2d(2)
    D = D.cuda(args.gpu)
#trainset = img_data(args.training_dir)
#params = {'batch_size': args.batch_size,
#          'shuffle': True,
#          'num_workers': args.num_workers}
#training_generator = data.DataLoader(trainset, **params)
if args.test_image is not None:
    test_img = Image.open(args.test_image)
    test_img.thumbnail((args.res,args.res))
    W, H = test_img.size
    test_img.convert("RGB")
    test_yuv = rgb2yuv(test_img)
    test_inf = test_yuv[...,0].reshape(1,1,H,W)
    test_var = Variable(torch.Tensor(test_inf-0.5)).cuda(args.gpu)
if args.test_image2 is not None:
    test_img2 = Image.open(args.test_image2).convert('RGB')
    test_yuv2 = rgb2yuv(test_img2)
    test_inf2 = test_yuv2[...,0].reshape(1,1,args.res,args.res)
    test_var2 = Variable(torch.Tensor(test_inf2-0.5)).cuda(args.gpu)
if args.d_init is not None:
    D.load_state_dict(torch.load(args.d_init,map_location='cuda:0'))
if args.g_init is not None:
    G.load_state_dict(torch.load(args.g_init,map_location='cuda:0'))
plt.figure('Loss')
plt.figure('Train')
# save test image for beginning
if args.test_image is not None:
    test_res = G(test_var)
    uv=test_res.cpu().detach().numpy()
    uv[:,0,:,:] *= 0.43601035
    uv[:,1,:,:] *= 0.61497538
    test_yuv = np.concatenate([test_inf,uv],axis=1).reshape(3,H,W)
    test_rgb = yuv2rgb(test_yuv.transpose(1,2,0))
    im = plt.imshow(test_rgb.clip(min=0,max=1))
    plt.xlabel('Fake label smoothing:' + str(args.label_fake) + '\n Real label smoothing:' + str(args.label_real))
    cv2.imwrite(os.path.join(args.checkpoint_location,'test_init.jpg'),(test_rgb.clip(min=0,max=1)*256)[:,:,[2,1,0]])
if args.test_image2 is not None:
    test_res2 = G(test_var2)
    uv2=test_res2.cpu().detach().numpy()
    uv2[:,0,:,:] *= 0.43601035
    uv2[:,1,:,:] *= 0.61497538
    test_yuv2 = np.concatenate([test_inf2,uv2],axis=1).reshape(3,args.res,args.res)
    test_rgb2 = yuv2rgb(test_yuv2.transpose(1,2,0))
    im = plt.imshow(np.concatenate([test_rgb.clip(min=0,max=1),test_rgb2.clip(min=0,max=1)]))

plt.show(block=False)
plt.pause(1)


i = 0
g_loss_gan_history = []
g_loss_history = []
d_loss_history = []
d_loss = 0
g_loss_gan = 0
adversarial_loss = nn.BCELoss()
optimizer_G = optim.AdaBound(G.parameters(),betas=(0.1,0.999), lr=args.g_lr)
if not args.nogan:
    optimizer_D = optim.AdaBound(D.parameters(),betas=(0.1,0.999), lr=args.d_lr)
Loss = nn.MSELoss()
rng = np.random.default_rng()
torch.backends.cudnn.benchmark = True
for run in range(args.runs):
    trainset = img_data(args.training_dir)
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers,
              'pin_memory': True}
    training_generator = data.DataLoader(trainset, **params)
    for epoch in range(args.epoch):
        for y, uv, in training_generator:
            yvar = Variable(y).cuda(args.gpu)
            uvgen = G(yvar)
            uvvar = Variable(uv).cuda(args.gpu)
            if not args.nogan:
                gen_imgs = torch.cat([yvar.detach(),uvgen],dim=1)
                real_imgs = torch.cat([yvar,uvvar],dim=1)
                valid = Variable(torch.Tensor(y.size(0), 1).fill_(args.label_real), requires_grad=False).cuda(args.gpu)
                fake = Variable(torch.Tensor(y.size(0), 1).fill_(args.label_fake), requires_grad=False).cuda(args.gpu)
            if i%args.g_every==0 and not args.g_disable:
                optimizer_G.zero_grad(set_to_none=True)                
                if not args.nogan:
                    g_loss_gan = adversarial_loss(D(gen_imgs), valid) / args.pixel_loss_weights
                g_loss = g_loss_gan + Loss(uvgen, uvvar)
                if not args.nogan:
                    g_loss_gan_history.append(g_loss_gan.item())
                g_loss_history.append(g_loss.item())
                g_loss.backward()
                optimizer_G.step()

                #D.load_state_dict(D_rollback)
                #gen_train=True

            else:
                g_loss = 0
                g_loss_gan = 0


            if not args.d_disable:
                optimizer_D.zero_grad(set_to_none=True)
                # Measure discriminator's ability to classify real from generated samples
                if rng.random() > args.flip:
                    real_loss = adversarial_loss(D(real_imgs), valid - args.smooth)
                    fake_loss = adversarial_loss(D((gen_imgs).detach()), fake)
                else:
                    real_loss = adversarial_loss(D(real_imgs), fake)
                    fake_loss = adversarial_loss(D((gen_imgs).detach()), valid - args.smooth)
                d_loss = (real_loss + fake_loss) / 2
                d_loss_history.append(d_loss.item())
                d_loss.backward()
                optimizer_D.step()
                #if gen_train==True:
                #    D_rollback=D.state_dict()
                #    gen_train=False
            else:
                d_loss = 0
            key=curses.wrapper(keypress)
            if args.test_image is not None and (i%args.checkpoint_every==0 or not key ==-1):
                test_res = G(test_var)
                uv=test_res.cpu().detach().numpy()
                uv[:,0,:,:] *= 0.43601035
                uv[:,1,:,:] *= 0.61497538
                test_yuv = np.concatenate([test_inf,uv],axis=1).reshape(3,H,W)
                test_rgb = yuv2rgb(test_yuv.transpose(1,2,0))
                im.set_data(test_rgb.clip(min=0,max=1))
                if args.test_image2 is not None:
                    test_res2 = G(test_var2)
                    uv2=test_res2.cpu().detach().numpy()
                    uv2[:,0,:,:] *= 0.43601035
                    uv2[:,1,:,:] *= 0.61497538
                    test_yuv2 = np.concatenate([test_inf2,uv2],axis=1).reshape(3,512,512)
                    test_rgb2 = yuv2rgb(test_yuv2.transpose(1,2,0))
                    im.set_data(np.concatenate([test_rgb.clip(min=0,max=1),test_rgb2.clip(min=0,max=1)]))

                plt.gcf().canvas.start_event_loop(0.001)
                plt.savefig((os.path.join(args.checkpoint_location,'test_epoch_'+str(epoch)+'_iter_'+str(i)+'.jpg')))

            i+=1

            plt.gcf().canvas.start_event_loop(0.001)

            print ("Epoch: % 4d: Iter: % 6d [D loss: % 10.5f] [G total loss: % 10.5f] [G GAN loss: % 10.5f] \r" % (epoch, i, d_loss, g_loss, g_loss_gan), end='')
            if i%args.checkpoint_every==0 or not key == -1:
                print ("\n", end='')
                if not args.nogan:
                    torch.save(D.state_dict(), os.path.join(args.checkpoint_location,'weights','D'+str(epoch)+'.pth'))
                torch.save(G.state_dict(), os.path.join(args.checkpoint_location,'weights','G'+str(epoch)+'.pth'))
                torch.save(G.state_dict(), os.path.join('model.pth'))
                plt.figure('Loss')
                plt.gcf().clear()
                plt.subplot(311)
                plt.plot(g_loss_history)
                plt.xlabel('Generator g_loss')
                plt.subplot(312)
                plt.plot(g_loss_gan_history)
                plt.xlabel('Generator g_loss_gan')
                plt.subplot(313)
                plt.plot(d_loss_history)
                plt.xlabel('Discriminator d_loss')
                plt.gcf().canvas.draw_idle()
                plt.gcf().canvas.start_event_loop(0.001)
                plt.figure('Train')
    if not args.nogan:
        torch.save(D.state_dict(), os.path.join(args.checkpoint_location,'D_final.pth'))
    torch.save(G.state_dict(), os.path.join(args.checkpoint_location,'G_final.pth'))

