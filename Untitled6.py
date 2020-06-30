#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import Adam


import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import timeit


# In[ ]:


import os
cwd = os.getcwd() 
print(cwd)


# In[2]:


def load_data():
    npzfile = np.load('array.npz')
    X = np.array(npzfile['arr_0'])
    y = pd.read_csv("/home/adm2/python-scripts/notebooks/data/y.csv")
    y=y.drop('Unnamed: 0',1)
    

    return X, y

def load_and_transform_data():
    X, y = load_data() 
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    X  = torch.from_numpy(X)
    y  = torch.from_numpy(y)
    
    return X, y

X, y = load_and_transform_data()
X.shape


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

(X_train.shape, y_train.shape), (X_val.shape, y_val.shape)


# In[4]:


X_train,X_val = X_train.type(torch.FloatTensor), X_val.type(torch.FloatTensor)
y_train,y_val = y_train.type(torch.FloatTensor), y_val.type(torch.FloatTensor)


# In[5]:


print(y_train.dtype)
print(X.shape)
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)


# In[6]:


X_val = torch.flatten(X_val)
X_train = torch.flatten(X_train)


# In[6]:


def image_generator(file, label_file, batch_size = 16):
    new_bs=0
    batch_input  = []
    batch_output = []
    for i in range len(X_train):
        x = file[new_bs:(new_bs+batchsize)]
        y = label_file[new_bs:(new_bs+batchsize)]
        new_bs += batch_size
        batch_input += [ x ]
        batch_output += [ y ]
    

yield( batch_x, batch_y)


# In[7]:


for inputs, labels in image_generator(X_val, y_val, batch_size = 16):
    inputs =   
    labels =   


# In[14]:


def image_generator(file, label_file, batch_size = 16):
    new_bs=0
    while True:
        batch_paths  = file[new_bs:(new_bs+batchsize)]
        new_bs += batch_size
        batch_input  = []
        batch_output = []
        # Read in each input, perform preprocessing and get labels
        for input_path in batch_paths:
            x = input_path
            y = input_path

#         ip = preprocess_input(image=x)
            batch_input += [ x ]
            batch_output += [ y ]
        # Return a tuple of (input, output) to feed the network
        batch_x = torch.tensor( batch_input )
        batch_y = torch.tensor( batch_output )
#         print(len(batch_x))


        yield( batch_x, batch_y)


# In[ ]:


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


# In[15]:


dataloaders = {
    'train':(image_generator(X_train, y_train, batch_size = 16)
            ),
    'val':( image_generator(X_val, y_val, batch_size = 16),
          )}


# In[ ]:


for inputs,labels in dataloaders[phase]:


# In[16]:



def train_model(model, criterion, optimizer, num_epochs):
    since = timeit.timeit()

    #best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch +1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs,labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                #labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    print(outputs.shape)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            '''#For graph generation
            if phase == "train":
                train_loss.append(running_loss/dataset_sizes[phase])
                train_acc.append(running_corrects.double() / dataset_sizes[phase])
                epoch_counter_train.append(epoch)
            if phase == "val":
                val_loss.append(running_loss/ dataset_sizes[phase])
                val_acc.append(running_corrects.double() / dataset_sizes[phase])
                epoch_counter_val.append(epoch)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]'''
               
            #for printing        
            if phase == "train":    
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == "val":    
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                #best_model_wts = copy.deepcopy(model.state_dict())

            print()

    time_elapsed = timeit.timeit() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[19]:


y_val.shape


# In[ ]:





# In[ ]:





# In[7]:


dataloaders = {
    'train':
        torch.utils.data.DataLoader(
            image_generator(X_train, y_train, batch_size = 16),
            batch_size=16,
            shuffle=True,
            num_workers=4),
    'val':
        torch.utils.data.DataLoader(
            image_generator(X_val, y_val, batch_size = 16),
            batch_size=16,
            shuffle=False,
            num_workers=4)}

dataset_sizes = {'train': len(X_train), 
                 'val': len(X_val)
                }


# In[8]:


def train_model(model, criterion, optimizer, num_epochs):
    since = timeit.timeit()

    #best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch +1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs,labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                #labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    print(outputs.shape)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            '''#For graph generation
            if phase == "train":
                train_loss.append(running_loss/dataset_sizes[phase])
                train_acc.append(running_corrects.double() / dataset_sizes[phase])
                epoch_counter_train.append(epoch)
            if phase == "val":
                val_loss.append(running_loss/ dataset_sizes[phase])
                val_acc.append(running_corrects.double() / dataset_sizes[phase])
                epoch_counter_val.append(epoch)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]'''
               
            #for printing        
            if phase == "train":    
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == "val":    
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                #best_model_wts = copy.deepcopy(model.state_dict())

            print()

    time_elapsed = timeit.timeit() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[17]:


#from torchsummary import summary
device = torch.device("cpu")
start = timeit.timeit()

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True).to(device)

for param in model.parameters():
    param.requires_grad = False


model.fc = nn.Sequential(
    
    nn.Linear(2048, 1000),
    nn.ReLU(inplace=True),
    nn.Linear(1000, 136)).to(device)
model.eval()


# In[18]:




#model.summary()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())

model = train_model(model, criterion, optimizer, num_epochs=4) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




