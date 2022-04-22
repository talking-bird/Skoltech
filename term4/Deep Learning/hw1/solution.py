# Don't erase the template code, except "Your code here" comments.

import subprocess
import sys

# List any extra packages you need here
PACKAGES_TO_INSTALL = ["gdown==4.4.0", "tensorboard"]
subprocess.check_call([sys.executable, "-m", "pip", "install"] + PACKAGES_TO_INSTALL)
subprocess.check_call([sys.executable, "-m","pip", "install","albumentations==0.4.6"])
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import make_grid
from torchvision import transforms as T
from torchvision import models, datasets

from torch.utils.tensorboard import SummaryWriter

from albumentations import (
	Compose,
    HorizontalFlip,
    Normalize,
    RandomCrop,
    PadIfNeeded,
    RGBShift,
    Rotate
)
from albumentations.pytorch import ToTensor
######### For dataloader ######### 
def albumentations_transforms(p=1.0, is_train=False):
	# Mean and standard deviation of train dataset
	mean = np.array([0.4914, 0.4822, 0.4465])
	std = np.array([0.2023, 0.1994, 0.2010])
	transforms_list = []
	# Use data aug only for train data
	if is_train:
		transforms_list.extend([
			PadIfNeeded(min_height=72, min_width=72, p=1.0),
			RandomCrop(height=64, width=64, p=1.0),
			HorizontalFlip(p=0.25),
			Rotate(limit=15, p=0.25),
			RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.25),
			#CoarseDropout(max_holes=1, max_height=32, max_width=32, min_height=8,
						#min_width=8, fill_value=mean*255.0, p=0.5),
		])
	transforms_list.extend([
		Normalize(
			mean=mean,
			std=std,
			max_pixel_value=255.0,
			p=1.0
		),
		ToTensor()
	])
	data_transforms = Compose(transforms_list, p=p)
	return lambda img: data_transforms(image=np.array(img))["image"]
	
def _transforms():
	# Data Transformations
	train_transform = albumentations_transforms(p=1.0, is_train=True)
	test_transform = albumentations_transforms(p=1.0, is_train=False)
	return train_transform, test_transform
################################## 

def get_dataloader(path, kind):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val', the dataloader should be deterministic.

    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train' or 'val'

    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
    """
    
    if kind=='train':
        transform = _transforms()[0]
    elif kind=='val':
        transform = _transforms()[1]
    
    # Read image files to pytorch dataset using ImageFolder, a generic data 
    # loader where images are in format root/label/filename
    # See https://pytorch.org/vision/stable/datasets.html
    dataset = datasets.ImageFolder(path+kind+'/', transform=transform)
    
    # Wrap image dataset (defined above) in dataloader 
    batch_size = 64
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            num_workers=2,
                            shuffle=(kind=="train")
                            )
    
    return dataloader

def get_model():
    """
    Create neural net object, initialize it with raw weights, upload it to GPU.

    return:
    model:
        `torch.nn.Module`
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # return models.resnet101(pretrained=False, progress=True, num_classes=200).to(device)
    
    model = models.resnet18(pretrained=False, progress=True, num_classes=200)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, 200)
    )
    return model.to(device)

def get_optimizer(model):
    """
    Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.

    return:
    optimizer:
        `torch.optim.Optimizer`
    """
    # lr=1e-3
    # weight_decay = 1e-4
    # return optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    return optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def predict(model, batch):
    """
    model:
        `torch.nn.Module`
        The neural net, as defined by `get_model()`.
    batch:
        unspecified
        A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
        (with same preprocessing and device).

    return:
    prediction:
        `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
        The scores of each input image to belong to each of the dataset classes.
        Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
        belong to `j`-th class.
        These scores can be 0..1 probabilities, but for better numerical stability
        they can also be raw class scores after the last (usually linear) layer,
        i.e. BEFORE softmax.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch = batch.to(device)
    prediction = model(batch)
    assert prediction.shape == torch.Size([len(batch), 200]), f'{prediction.shape}'
    assert prediction.dtype == torch.float32, f'{prediction.dtype.item()}'
    return prediction
    

def validate(dataloader, model):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.

    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    running_correct = 0
    total = 0
    running_loss = 0.
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader: 
            # images = images.to(device)  #it's already done in predict
            labels = labels.to(device)
            outputs = predict(model, images)
            loss = criterion(outputs, labels)
            # convert output probabilities to predicted class
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            running_correct += (predicted == labels).sum().item()
            running_loss += loss.item()  
    mean_val_accuracy = running_correct / total
    mean_val_loss = running_loss / total
    model.train()
    return mean_val_accuracy, mean_val_loss
        
        
        

def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.

    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.
    """
    writer = SummaryWriter('runs/TinyImageNet1')
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    num_epochs = 30
    
    n_total_steps = len(train_dataloader)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.02, steps_per_epoch=n_total_steps,
                            epochs=num_epochs, div_factor=10, final_div_factor=10,
                            pct_start=10/num_epochs)
    model.train()
    for epoch in range(num_epochs):
        running_correct = 0
        running_loss = 0.0
        for i, (images, labels)  in enumerate(train_dataloader):

            # images = images.to(device) #it's already done in predict
            labels = labels.to(device)

            #forward pass
            outputs = predict(model, images)
            loss = criterion(outputs, labels)
            
            #backward and optimize 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            _, predicted = torch.max(outputs.data, 1)
            running_loss += loss.item()
            running_correct += (predicted == labels).sum().item()
            
            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
                ############## TENSORBOARD ########################
                #training
                writer.add_scalar('loss/training', running_loss / 100, epoch * n_total_steps + i+1)
                running_accuracy = running_correct / 100 / predicted.size(0)
                writer.add_scalar('accuracy/training', running_accuracy, epoch * n_total_steps + i+1)
                running_correct = 0
                running_loss = 0.0
                #validation
                val_accuracy, val_loss = validate(val_dataloader, model)
                writer.add_scalar('loss/validation', val_loss, epoch*n_total_steps +i+1)
                writer.add_scalar('accuracy/validation', val_accuracy, epoch*n_total_steps +i+1)
                ###################################################
        #create a checkpoint
        if (epoch+1) % 2 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict()
            }
            FILE = f'checkpoint_{epoch}.pth'
            torch.save(checkpoint, FILE)
    print('finished training')
    writer.close()


def load_weights(model, checkpoint_path):
    """
    Initialize `model`'s weights from `checkpoint_path` file.

    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])

def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    md5_checksum = "19f4ea79dcb7e47a2f05030ff056d457"
    google_drive_link = "https://drive.google.com/file/d/1keL6Z-0oBFYXhxU9tcb1UnOw1uPVF-Ni/view?usp=sharing"

    return md5_checksum, google_drive_link
