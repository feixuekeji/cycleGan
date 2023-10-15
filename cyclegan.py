import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import AvgPool2d
from torch.nn import UpsamplingNearest2d

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="m", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--save_image_path", type=str, default="images", help="name of the dataset")
parser.add_argument("--save_model_path", type=str, default="saved_models", help="path of saved_models")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument("--downsample_loss", type=float, default=5.0, help="identity loss weight")
parser.add_argument("--upsample_loss", type=float, default=5.0, help="identity loss weight")
parser.add_argument('--lambda_identity', type=float, default=-1.0,
                    help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = G_BA(real_B)

    ###
    real_C = Variable(imgs["C"].type(Tensor))

    # Resize real_A and fake_A to match the size of real_B and fake_B
    real_A = F.interpolate(real_A, size=(256, 256), mode='bilinear', align_corners=False)
    fake_A = F.interpolate(fake_A, size=(256, 256), mode='bilinear', align_corners=False)

    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)

    real_C = make_grid(real_C, nrow=5, normalize=True)

    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_C, real_B, fake_A), 1)
    save_image(image_grid, "%s/%s/cycle_%s.png" % (opt.save_image_path, opt.dataset_name, batches_done), normalize=False)


if __name__ == '__main__':

    opt = parser.parse_args()
    print(opt)

    # Create sample and checkpoint directories
    os.makedirs("%s/%s" % (opt.save_image_path, opt.dataset_name), exist_ok=True)
    os.makedirs("%s/%s" % (opt.save_model_path, opt.dataset_name), exist_ok=True)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    cuda = torch.cuda.is_available()

    input_shape = (opt.channels, opt.img_height, opt.img_width)

    # Initialize generator and discriminator
    G_AB = GeneratorA(input_shape, opt.n_residual_blocks)
    G_BA = GeneratorB(input_shape, opt.n_residual_blocks)
    D_A = Discriminator(opt.channels)
    D_B = Discriminator(opt.channels)

    if cuda:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        G_AB.load_state_dict(torch.load("%s/%s/G_AB_%d.pth" % (opt.save_model_path, opt.dataset_name, opt.epoch)))
        G_BA.load_state_dict(torch.load("%s/%s/G_BA_%d.pth" % (opt.save_model_path, opt.dataset_name, opt.epoch)))
        D_A.load_state_dict(torch.load("%s/%s/D_A_%d.pth" % (opt.save_model_path, opt.dataset_name, opt.epoch)))
        D_B.load_state_dict(torch.load("%s/%s/D_B_%d.pth" % (opt.save_model_path, opt.dataset_name, opt.epoch)))
    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Image transformations
    transforms_ = [
        # transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
        # transforms.RandomCrop((opt.img_height, opt.img_width)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    # Training data loader
    dataloader = DataLoader(
        ImageDataset("data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )
    # Test data loader
    val_dataloader = DataLoader(
        ImageDataset("data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
        batch_size=5,
        shuffle=True,
        num_workers=1,
    )

    # ----------
    #  Training
    # ----------

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            # Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))

            # Adversarial ground truths
            # valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
            # fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()
            G_BA.train()
            optimizer_G.zero_grad()

            # Identity loss
            loss_identity = 0
            if opt.lambda_identity > 0:
                loss_id_A = criterion_identity(G_BA(real_A), real_A)
                loss_id_B = criterion_identity(G_AB(real_B), real_B)
                loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            # print(fake_B.size())
            fake_DB = D_B(fake_B)
            size_DB = fake_DB.size()
            label_B_true = torch.tensor(np.ones(size_DB), dtype=torch.float32, device='cuda', requires_grad=False)
            label_B_false = torch.tensor(np.zeros(size_DB), dtype=torch.float32, device='cuda', requires_grad=False)
            loss_GAN_AB = criterion_GAN(fake_DB, label_B_true)

            fake_A = G_BA(real_B)
            fake_DA = D_A(fake_A)
            size_DA = fake_DA.size()
            label_A_true = torch.tensor(np.ones(size_DA), dtype=torch.float32, device='cuda', requires_grad=False)
            label_A_false = torch.tensor(np.zeros(size_DA), dtype=torch.float32, device='cuda', requires_grad=False)
            loss_GAN_BA = criterion_GAN(fake_DA, label_A_true)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            if opt.downsample_loss > 0:
                # print('use the downsampleloss')
                m = AvgPool2d(2, stride=2)
                downsample_fake_B = m(fake_B)
                loss_down = criterion_cycle(downsample_fake_B, real_A)
            else:
                loss_down = 0

            if opt.upsample_loss > 0:
                # print('use the upsampleloss')
                n = UpsamplingNearest2d(scale_factor=2)
                upsample_fake_A = n(fake_A)
                loss_up = criterion_cycle(upsample_fake_A, real_B)

            else:
                loss_up = 0

            loss_up_down = (loss_up + loss_down) / 2

            # Total loss
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity + loss_up_down

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_A.zero_grad()

            # Real loss

            loss_real = criterion_GAN(D_A(real_A), label_A_true)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), label_A_false)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B(real_B), label_B_true)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), label_B_false)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_up_down.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if opt.checkpoint_interval > 0 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), "%s/%s/G_AB_%d.pth" % (opt.save_model_path, opt.dataset_name, epoch))
            torch.save(G_BA.state_dict(), "%s/%s/G_BA_%d.pth" % (opt.save_model_path, opt.dataset_name, epoch))
            torch.save(D_A.state_dict(), "%s/%s/D_A_%d.pth" % (opt.save_model_path, opt.dataset_name, epoch))
            torch.save(D_B.state_dict(), "%s/%s/D_B_%d.pth" % (opt.save_model_path, opt.dataset_name, epoch))
