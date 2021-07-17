"""
Created on July 15 10:39:07 2020

@author: Panpan
"""

import copy
from tqdm import tqdm

import torch
from torch import nn, optim
import torch.backends.cudnn as cudnn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils import tensorboard

from utilis.DataHelper import *
from utilis.loss import dice_bce_loss
from utilis.utilis import AverageMeter, ImageEval_, adjust_learning_rate

# Chose models
from networks.U_Net import UNet # U-Net
from networks.Attention_UNet import Att_UNet  # Attention U-Net
from networks.AB_UNet import AB_UNet # AB-UNet


def train():

    # CUDA
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TensorBoard
    tb = tensorboard.SummaryWriter()

    # Normalization
    x_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
         ])
    y_transform = transforms.ToTensor()

    ###############################################################################
    # Main parameters
    ###############################################################################

    # Initialization
    starting_lr = 1e-3
    batch_size = 32
    epochs = 150

    img_root = r'F:\CPP\DATASATS_Version4.0\Train-02\img'
    label_root = r'F:\CPP\DATASATS_Version4.0\Train-02\label'
    img_root_eval = r'F:\CPP\DATASATS_Version4.0\Eval-02\img'
    label_root_eval = r'F:\CPP\DATASATS_Version4.0\Eval-02\label'

    Model_PATH = 'model/'

    CHECKPOINT_PATH = 'checkpoints'
    checkpoint_flag = False
    CHECKPOINT_DIR = 'checkpoints/UNetRev5-loss20.tar'

    # Model
    model = AB_UNet(1, 1)

    # GPU Parallel
    if torch.cuda.device_count() > 1:
        print('Let us use GPU!')
        model = nn.DataParallel(model, device_ids=[0, 1, 2])
    model = model.to(device)

    # Loss function
    criterion = dice_bce_loss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), starting_lr)

    # Train dataset
    train_set = TrainDataset(img_root,
                             label_root,
                             transform=x_transform,
                             target_transform=y_transform)
    train_dataloader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=8)
    # Evaluation dataset
    eval_set = EvalDataset(img_root_eval,
                           label_root_eval,
                           transform=x_transform,
                           target_transform=y_transform)
    eval_dataloader = DataLoader(dataset=eval_set,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=4)

    dataiter = iter(train_dataloader)
    images, labels = dataiter.next()

    # TensorBoard: Draw model graph
    tb.add_graph(model, images.to(device))
    tb.close()

    ###############################################################################
    # Load checkpoint
    ###############################################################################

    if checkpoint_flag:
        checkpoint = torch.load(CHECKPOINT_DIR)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print("restore from checkpoint...")
    else:
        start_epoch = 0

    ###############################################################################
    # Start training
    ###############################################################################

    for epoch in range(start_epoch, epochs):

        print('>>>>>>>>>>>>> Start training：Epoch {}/{} >>>>>>>>>>>>>>>'.format(epoch + 1, epochs))
        model.train()
        epoch_loss = 0
        step = 0

        # Adjust learning rate according to schedule
        adjust_learning_rate(starting_lr, optimizer, epoch)

        with tqdm(total=(len(train_set) - len(train_set) % batch_size)) as t:
            t.set_description('Epoch {}/{}'.format(epoch + 1, epochs))

            for inputs, labels in train_dataloader:
                step += 1

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                loss = criterion(labels, outputs)

                # backward
                loss.backward()
                optimizer.step()

                # calculate epoch_loss
                epoch_loss += loss.item()

                # average loss in each epoch
                avg_epoch_loss = epoch_loss / step

                t.set_postfix(Epoch_Loss='{:.6f}'.format(avg_epoch_loss))
                t.update(len(inputs))

                # print loss value every 20 batches
                if (step + 1) % 20 == 0:
                    tb.add_scalar('Loss', avg_epoch_loss, epoch)
                    print("step %d loss:%0.3f" % (step + 1, avg_epoch_loss))

        # Save model to checkpoint
        # save model weights every 10 batches
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(checkpoint, os.path.join(CHECKPOINT_PATH, 'UNetRev-0.25{}.tar'.format(epoch + 1)))

        ###############################################################################
        # Evaluation
        ###############################################################################

        model.eval()

        best_ssim = 0.0
        epoch_val_loss = AverageMeter()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()

        for inputs, labels in eval_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                predicts = model(inputs)
                val_loss = criterion(labels, predicts)

                # Calculate SSIM and PSNR
                predicts = predicts.clamp(0.1, 1.0)
                ImageEval = ImageEval_(predicts, labels, channel=1)
                ssim = ImageEval.cal_ssim()
                psnr = ImageEval.cal_psnr()

            # Update
            epoch_psnr.update(psnr, len(inputs))
            epoch_ssim.update(ssim, len(inputs))
            epoch_val_loss.update(val_loss, len(inputs))

            tb.add_scalar('VAl Loss', epoch_val_loss.avg, epoch)
            tb.add_scalar('PSNR', epoch_psnr.avg, epoch)
            tb.add_scalar('SSIM', epoch_ssim.avg, epoch)

        if epoch_ssim.avg > best_ssim:
            best_epoch = epoch + 1
            best_ssim = epoch_ssim.avg
            best_weights = copy.deepcopy(model.state_dict())

            # Print best epoch and save best weights
            print('best epoch: {}, ssim: {:.2f}'.format(best_epoch, best_ssim))
            torch.save(best_weights, os.path.join(Model_PATH, 'best_wights.pth'))

        print('***************')


if __name__ == '__main__':
    print("Start Training")
    train()
    print('Training Finished')
