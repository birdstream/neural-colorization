import torch
import torch.nn as nn
import argparse
from torch.autograd import Variable
import torchvision.models as models
import os
from torch.utils import data
from model import generator
import numpy as np
from PIL import Image
from skimage.color import rgb2yuv,yuv2rgb
import cv2
import torchvision.transforms as transforms
import sys
from matplotlib import pyplot as plt
from PIL import ImageFilter
from adam_lrd import Adam_LRD



def parse_args():
    parser = argparse.ArgumentParser(description="Train a GAN based model")
    parser.add_argument("-d",
                        "--training_dir",
                        type=str,
                        required=True,
                        help="Training directory (folder contains all 256*256 images)")
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
    parser.add_argument("--d_noise",
                        type=float,
                        default=0.01,
                        help="Discriminator input noise factor")
    parser.add_argument("--d_disable",
                        action="store_true",
                        help="Disable training of discriminator")
    parser.add_argument("--g_disable",
                        action="store_true",
                        help="Disable training of generator")
    args = parser.parse_args()
    return args

# define data generator
class img_data(data.Dataset):
    def __init__(self, path):
        files = (os.listdir(path))
        self.files = [os.path.join(path,x) for x in files]
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert("RGB")
        #transform = transforms.RandomAffine(0,(0.2,0.2),(2,1),resample=Image.BICUBIC)
        transform = transforms.Compose([
            #transforms.RandomRotation(20,resample=Image.BICUBIC),
            transforms.RandomResizedCrop((128,128)),
            transforms.RandomHorizontalFlip()
            ])
        img = transform(img)
        #img = img.filter(ImageFilter.GaussianBlur(np.random.rand()*2))
        yuv = rgb2yuv(img)
        y = yuv[...,0]-0.5
        u_t = yuv[...,1] / 0.43601035
        v_t = yuv[...,2] / 0.61497538
        return torch.Tensor(np.expand_dims(y,axis=0)),torch.Tensor(np.stack([u_t,v_t],axis=0))

args = parse_args()
if not os.path.exists(os.path.join(args.checkpoint_location,'weights')):
    os.makedirs(os.path.join(args.checkpoint_location,'weights'))

# Define G, same as torch version
G = generator().cuda(args.gpu)

# define D
D = models.resnet18(pretrained=False)
#for param in D.parameters():
#    param.requires_grad = False
D.fc = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())
D.avgpool = nn.AdaptiveAvgPool2d(2)
D = D.cuda(args.gpu)
trainset = img_data(args.training_dir)
params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': args.num_workers}
training_generator = data.DataLoader(trainset, **params)
if args.test_image is not None:
    test_img = Image.open(args.test_image).convert('RGB').resize((512,512))
    test_yuv = rgb2yuv(test_img)
    test_inf = test_yuv[...,0].reshape(1,1,512,512)
    test_var = Variable(torch.Tensor(test_inf-0.5)).cuda(args.gpu)
if args.test_image2 is not None:
    test_img2 = Image.open(args.test_image2).convert('RGB').resize((512,512))
    test_yuv2 = rgb2yuv(test_img2)
    test_inf2 = test_yuv2[...,0].reshape(1,1,512,512)
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
    test_yuv = np.concatenate([test_inf,uv],axis=1).reshape(3,512,512)
    test_rgb = yuv2rgb(test_yuv.transpose(1,2,0))
    im = plt.imshow(test_rgb.clip(min=0,max=1))
    #plt.show(block=False)
    #plt.pause(1)
    cv2.imwrite(os.path.join(args.checkpoint_location,'test_init.jpg'),(test_rgb.clip(min=0,max=1)*256)[:,:,[2,1,0]])
if args.test_image2 is not None:
    test_res2 = G(test_var2)
    uv2=test_res2.cpu().detach().numpy()
    uv2[:,0,:,:] *= 0.43601035
    uv2[:,1,:,:] *= 0.61497538
    test_yuv2 = np.concatenate([test_inf2,uv2],axis=1).reshape(3,512,512)
    test_rgb2 = yuv2rgb(test_yuv2.transpose(1,2,0))
    im = plt.imshow(np.concatenate([test_rgb.clip(min=0,max=1),test_rgb2.clip(min=0,max=1)]))

plt.show(block=False)
plt.pause(1)


i=0; dls=0
i_n = args.d_noise
g_loss_gan_history=[]
g_loss_history=[]
d_loss_history=[]
g_onoff=[]
d_onoff=[]
d_loss = 0
adversarial_loss = torch.nn.BCELoss()
optimizer_G = Adam_LRD(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999),dropout=0.5)
optimizer_D = Adam_LRD(D.parameters(), lr=args.d_lr, betas=(0.5, 0.999),dropout=0.5)
for epoch in range(args.epoch):
    for y, uv in training_generator:
        # Adversarial ground truths
        valid = Variable(torch.Tensor(y.size(0), 1).fill_(1.0), requires_grad=False).cuda(args.gpu)
        fake = Variable(torch.Tensor(y.size(0), 1).fill_(0.0), requires_grad=False).cuda(args.gpu)

        yvar = Variable(y).cuda(args.gpu)
        uvvar = Variable(uv).cuda(args.gpu)
        real_imgs = torch.cat([yvar,uvvar],dim=1)
        #real_imgs = real_imgs + ((0.1-np.clip(i*0.00001, 0, 0.1))**0.5) * torch.randn(256,256).cuda(args.gpu)

        optimizer_G.zero_grad()
        uvgen = G(yvar)
        # Generate a batch of images
        gen_imgs = torch.cat([yvar.detach(),uvgen],dim=1)
        #gen_imgs = gen_imgs + ((0.1-np.clip(i*0.00001, 0, 0.1))**0.5) * torch.randn(256,256).cuda(args.gpu)

        # Loss measures generator's ability to fool the discriminator
        g_loss_gan = adversarial_loss(D(gen_imgs), valid)
        g_loss = g_loss_gan + args.pixel_loss_weights * torch.mean((uvvar-uvgen)**2)
        g_loss_gan_history.append(g_loss_gan)
        g_loss_history.append(g_loss)
        if i%args.g_every==0 and not args.g_disable and not (g_loss_gan < 0.05 or d_loss > 1.5):
        #if g_loss_gan.item() > gh:
            g_loss.backward()
            optimizer_G.step()
            g_onoff.append(0)
        else:
            g_onoff.append(1)
            g_temp=g_loss.item()
            g_loss-=g_loss
            g_loss.backward()
            optimizer_G.step()
            g_loss=g_temp
        #    gh+=.1*gh
        #else:
        #    gh-=.1*gh

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        if not args.d_disable and not d_loss < 0.2:
            real_loss = adversarial_loss(D(real_imgs + i_n**0.5 * torch.randn(128,128).cuda(args.gpu)), valid)
            fake_loss = adversarial_loss(D((gen_imgs + i_n**0.5 * torch.randn(128,128).cuda(args.gpu)).detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            d_onoff.append(0)
        else:
            real_loss = adversarial_loss(D(real_imgs), valid)
            fake_loss = adversarial_loss(D((gen_imgs).detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_onoff.append(1)
            d_temp=d_loss.item()
            d_loss-=d_loss
            d_loss.backward()
            optimizer_D.step()
            d_loss=d_temp
        d_loss_history.append(d_loss)
        if args.test_image is not None and i%args.checkpoint_every==0:
            test_res = G(test_var)
            uv=test_res.cpu().detach().numpy()
            uv[:,0,:,:] *= 0.43601035
            uv[:,1,:,:] *= 0.61497538
            test_yuv = np.concatenate([test_inf,uv],axis=1).reshape(3,512,512)
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
            #plt.draw()
            #plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.start_event_loop(0.001)
            plt.savefig((os.path.join(args.checkpoint_location,'test_epoch_'+str(epoch)+'_iter_'+str(i)+'.jpg')))

        i+=1
        plt.gcf().canvas.start_event_loop(0.001)
        print ("Epoch: % 4d: Iter: % 6d [D loss: % 10.5f] [G total loss: % 10.5f] [G GAN loss: % 10.5f] [D Noise: % 1.5f] \r" % (epoch, i, d_loss, g_loss, g_loss_gan, i_n), end='')
        if i%args.checkpoint_every==0:
            print ("\n", end='')

            torch.save(D.state_dict(), os.path.join(args.checkpoint_location,'weights','D'+str(epoch)+'.pth'))
            torch.save(G.state_dict(), os.path.join(args.checkpoint_location,'weights','G'+str(epoch)+'.pth'))
            plt.figure('Loss')
            #plt.xlim(0, len(g_loss_history))
            plt.gcf().clear()
            plt.subplot(311)
            plt.plot(g_loss_history)
            plt.xlabel('Generator g_loss')
            plt.subplot(312)
            plt.plot(g_loss_gan_history)
            plt.plot(g_onoff)
            plt.xlabel('Generator g_loss_gan')
            plt.subplot(313)
            plt.plot(d_loss_history)
            plt.plot(d_onoff)
            plt.xlabel('Discriminator d_loss')
            plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.start_event_loop(0.001)
            plt.figure('Train')
            #if args.test_image is not None:
                #test_res = G(test_var)
                #uv=test_res.cpu().detach().numpy()
                #uv[:,0,:,:] *= 0.436
                #uv[:,1,:,:] *= 0.615
                #test_yuv = np.concatenate([test_inf,uv],axis=1).reshape(3,256,256)
                #test_rgb = yuv2rgb(test_yuv.transpose(1,2,0))
                #cv2.imwrite(os.path.join(args.checkpoint_location,'test_epoch_'+str(epoch)+'_iter_'+str(i)+'.jpg'),(test_rgb.clip(min=0,max=1)*256)[:,:,[2,1,0]])
torch.save(D.state_dict(), os.path.join(args.checkpoint_location,'D_final.pth'))
torch.save(G.state_dict(), os.path.join(args.checkpoint_location,'G_final.pth'))
