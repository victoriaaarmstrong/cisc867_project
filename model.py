from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt

from data_set import get_images

manualSeed = 999
workers = 6
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


def weights_init(m):
  """
  takes initialized model m and reinitializes all convolutional, convolutional-transpose and batch norm layers
  """
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, wcm, ws)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, wbm, ws)
    nn.init.constant_(m.bias.data, bias)


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

def train(name):
  device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

  netG = Generator(ngpu).to(device)
  netD = Discriminator(ngpu).to(device)

  netG.apply(weights_init)
  netD.apply(weights_init)

  print("Net G:")
  print(netG)
  print("Net D:")
  print(netD)

  #device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

  dataloader_train, dataloader_test = get_images(batch_size, image_size, workers)

  torch.save(dataloader_test, './dataloaders/test_')
  torch.save(dataloader_train, './dataloaders/train_')

  criterion = nn.BCELoss()

  optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
  optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

  img_list = []
  G_losses = []
  D_losses = []

  iters = 0
  checkpoint_num = 0
  for epoch in range(num_epochs):
    for i, (real, sparse) in enumerate(dataloader_train, 0):
      ## Update D Network
      ## Train with all real batch
      netD.zero_grad()
      ## Format batch
      real_cpu = real.to(device)
      b_size = real_cpu.size(0)
      label = (maxR-minR)*torch.rand((b_size,)) + minR
      ## Forward pass real batch through D
      output = netD(real_cpu).view(-1)
      ## Calculate loss on all-real batch
      errD_real = criterion(output, label)
      ## Calculate gradients for D in backward pass
      errD_real.backward()
      D_x = output.mean().item()

      ##Train with all fake batch

      ## Generate fake image batch with G
      fake = netG(sparse)
      label = (maxF-minF)*torch.rand((b_size,)) + minF
      ## Classify all fake batch with D
      output = netD(fake.detach()).view(-1)
      ## Calculate D's loss on the all-fake batch
      errD_fake = criterion(output, label)
      ## Calculate the gradients for this batch
      errD_fake.backward()
      D_G_z1 = output.mean().item()
      ## Add the gradients from the all-real and all-fake batches
      errD = errD_real + errD_fake
      ## Update D
      optimizerD.step()

      ## Update G Network
      netD.zero_grad()
      label = (maxR - minR) * torch.rand((b_size,)) + minR
      ## Since we just updated D, perform another forward pass of all-fake batch through D
      output = netD(fake).view(-1)
      ## Calculate G's loss based on this output
      errG = criterion(output, label)
      ## Calculate gradients for F
      errG.backward()
      D_G_z2 = output.mean().item()
      ## Update G
      optimizerG.step()

      ## Output training stats
      if i % 50 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader_train),
                       errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        ## Save losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

      ## Save some checkpoints and images to see how training is going
      if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader_train)-1)):
        torch.save({'epoch': epoch,
                    'iterations': iters,
                    'discriminator_state_dict': netD.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'generator_state_dict': netG.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'loss_discriminator': errD.item(),
                    'loss_generator': errG.item(),
                    'D_x': D_x,
                    'D_G_z': (D_G_z1 / D_G_z2),
                    }, './checkpoints/' + str(checkpoint_num) + '_' + str(iters) + '_' + name)
        checkpoint_num += 1

        with torch.no_grad():
          fake = netG(next(iter(dataloader_test))[0]).detach().cpu()
          img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        fig = plt.figure(figsize=(15, 15))
        plt.axis("off")
        imgplot = plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
        plt.savefig('./training_images/' + str(epoch) + "_" + str(iters) + '_' + name)
        plt.close()

      iters += 1

  ## Print losses
  plt.figure(figsize=(10, 5))
  plt.title("Generator and Discriminator Loss During Training")
  plt.plot(G_losses, label="G")
  plt.plot(D_losses, label="D")
  plt.xlabel("iterations")
  plt.ylabel("Loss")
  plt.legend()
  plt.show()

  return
