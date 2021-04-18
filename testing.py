from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt

manualSeed = 999
workers = 2
batch_size = 64
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 100
lr = 0.0002
beta1 = 0.5
ngpu = 1 #for DL laptop
wcm = 0.0 #weight mean initialization for convolutional layers
wbm = 1.0 #weight mean initialization for batch norm layers
ws = 0.02 #weight stf initilalzation for all layers
bias = 0 #for weight initialization
maxR = 1.2
minR = 0.7
maxF = 0.3
minF = 0.0

class Generator(nn.Module):
  def __init__(self, ngpu):
    super(Generator, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
      nn.Conv2d(nc, ngf, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf) x 32 x 32
      nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 2),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*2) x 16 x 16
      nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 4),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*4) x 8 x 8
      nn.Conv2d(ndf * 4, ngf * 8, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 8),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*8) x 4 x 4
      nn.Conv2d(ngf * 8, nz, 4, 1, 0, bias=False),
      nn.ReLU(True),
      # input is Z, going into a convolution
      nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
      nn.BatchNorm2d(ngf * 8),
      nn.ReLU(True),
      # state size. (ngf*8) x 4 x 4
      nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 4),
      nn.ReLU(True),
      # state size. (ngf*4) x 8 x 8
      nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 2),
      nn.ReLU(True),
      # state size. (ngf*2) x 16 x 16
      nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf),
      nn.ReLU(True),
      # state size. (ngf) x 32 x 32
      nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
      nn.Tanh()
      # state size. (nc) x 64 x 64
    )

  def forward(self, input):
    return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
      super(Discriminator, self).__init__()
      self.ngpu = ngpu
      self.main = nn.Sequential(
        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 32 x 32
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 16 x 16
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 8 x 8
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*8) x 4 x 4
        nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        nn.Sigmoid()
      )

    def forward(self, input):
      return self.main(input)


def main():
  ## Load Checkpoint
  device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

  netG = Generator(ngpu).to(device)
  netD = Discriminator(ngpu).to(device)

  optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
  optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

  checkpoint = torch.load("./checkpoints/8_4000_train4") #0_0_train4")

  loss_G = checkpoint['loss_generator']
  loss_D = checkpoint['loss_discriminator']
  epoch = checkpoint['epoch']
  D_x = checkpoint['D_x']
  D_G_z = checkpoint['D_G_z']

  netG.load_state_dict(checkpoint['generator_state_dict'])
  optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])

  netD.load_state_dict(checkpoint['discriminator_state_dict'])
  optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])

  ## Run once with training and one with testing (w/in sample and outside sample)
  #train_data = torch.load('./dataloader/train_train4')
  test_data = torch.load('./dataloaders/test_train4')

  fake = netG(next(iter(test_data))[0]).detach().cpu()
  img_list = vutils.make_grid(fake, padding=2, normalize=True)

  ctr = 0
  total = 0
  real_image_list = []
  fake_image_list =[]

  for i, (real, sparse) in enumerate(test_data, 0):
    real_cpu = real.to(device)
    fake = netG(sparse)

    real_image_list.append(vutils.make_grid(real, padding=2, normalize=True))
    fake_image_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    ctr += 1

  ctr = 0
  for i in real_image_list:
    fig = plt.figure(figsize=(15, 15))
    imgplot = plt.imshow(np.transpose(i, (1, 2, 0)))
    plt.savefig('testing_real_end/' + str(ctr))
    plt.close()
    ctr += 1

  ctr = 0
  for i in fake_image_list:
    fig = plt.figure(figsize=(15, 15))
    imgplot = plt.imshow(np.transpose(i, (1, 2, 0)))
    plt.savefig('testing_sparse_end/' + str(ctr))
    plt.close()
    ctr += 1

  return

if __name__ == '__main__':
  main()

