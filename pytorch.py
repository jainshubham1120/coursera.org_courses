{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from PIL import Image\n",
    "%matplotlib inline\n",
    "\n",
    "import torch\n",
    "from torchvision import datasets, models, transforms\n",
    "import torch.nn as nn\n",
    "from torch.nn import functional as F\n",
    "import torch.optim as optim\n",
    "from torch.nn import Linear, CrossEntropyLoss\n",
    "from torch.optim import Adam\n",
    "\n",
    "\n",
    "import pandas as pd\n",
    "from sklearn import preprocessing\n",
    "from sklearn.model_selection import train_test_split\n",
    "import timeit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "cwd = os.getcwd() \n",
    "print(cwd)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/adm2/.virtualenvs/qbic/lib/python3.6/site-packages/sklearn/utils/validation.py:73: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  return f(**kwargs)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "torch.Size([44412, 3, 224, 224])"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def load_data():\n",
    "    npzfile = np.load('array.npz')\n",
    "    X = np.array(npzfile['arr_0'])\n",
    "    y = pd.read_csv(\"/home/adm2/python-scripts/notebooks/data/y.csv\")\n",
    "    y=y.drop('Unnamed: 0',1)\n",
    "    \n",
    "\n",
    "    return X, y\n",
    "\n",
    "def load_and_transform_data():\n",
    "    X, y = load_data() \n",
    "    le = preprocessing.LabelEncoder()\n",
    "    le.fit(y)\n",
    "    y = le.transform(y)\n",
    "    X  = torch.from_numpy(X)\n",
    "    y  = torch.from_numpy(y)\n",
    "    \n",
    "    return X, y\n",
    "\n",
    "X, y = load_and_transform_data()\n",
    "X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((torch.Size([26646, 3, 224, 224]), torch.Size([26646])),\n",
       " (torch.Size([8883, 3, 224, 224]), torch.Size([8883])))"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size=0.2, random_state=42)\n",
    "X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)\n",
    "\n",
    "(X_train.shape, y_train.shape), (X_val.shape, y_val.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_val = X_train.type(torch.FloatTensor), X_val.type(torch.FloatTensor)\n",
    "y_train,y_val = y_train.type(torch.FloatTensor), y_val.type(torch.FloatTensor)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.float32\n",
      "torch.Size([44412, 3, 224, 224])\n",
      "torch.Size([26646, 3, 224, 224])\n",
      "torch.Size([8883, 3, 224, 224])\n",
      "torch.Size([26646])\n"
     ]
    }
   ],
   "source": [
    "print(y_train.dtype)\n",
    "print(X.shape)\n",
    "print(X_train.shape)\n",
    "print(X_val.shape)\n",
    "print(y_train.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_val = torch.flatten(X_val)\n",
    "X_train = torch.flatten(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([16, 3, 224, 224])\n"
     ]
    }
   ],
   "source": [
    "def image_generator(file, label_file, batch_size = 16):\n",
    "    new_bs=0\n",
    "    batch_input  = []\n",
    "    batch_output = []\n",
    "    for i in range len(X_train):\n",
    "        x = file[new_bs:(new_bs+batchsize)]\n",
    "        y = label_file[new_bs:(new_bs+batchsize)]\n",
    "        new_bs += batch_size\n",
    "        batch_input += [ x ]\n",
    "        batch_output += [ y ]\n",
    "    \n",
    "\n",
    "yield( batch_x, batch_y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "26646"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "for inputs, labels in image_generator(X_val, y_val, batch_size = 16):\n",
    "    inputs =   \n",
    "    labels =   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "def image_generator(file, label_file, batch_size = 16):\n",
    "    new_bs=0\n",
    "    while True:\n",
    "        batch_paths  = file[new_bs:(new_bs+batchsize)]\n",
    "        new_bs += batch_size\n",
    "        batch_input  = []\n",
    "        batch_output = []\n",
    "        # Read in each input, perform preprocessing and get labels\n",
    "        for input_path in batch_paths:\n",
    "            x = input_path\n",
    "            y = input_path\n",
    "\n",
    "#         ip = preprocess_input(image=x)\n",
    "            batch_input += [ x ]\n",
    "            batch_output += [ y ]\n",
    "        # Return a tuple of (input, output) to feed the network\n",
    "        batch_x = torch.tensor( batch_input )\n",
    "        batch_y = torch.tensor( batch_output )\n",
    "#         print(len(batch_x))\n",
    "\n",
    "\n",
    "        yield( batch_x, batch_y)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class FaceLandmarksDataset(Dataset):\n",
    "    \"\"\"Face Landmarks dataset.\"\"\"\n",
    "\n",
    "    def __init__(self, csv_file, root_dir, transform=None):\n",
    "        \"\"\"\n",
    "        Args:\n",
    "            csv_file (string): Path to the csv file with annotations.\n",
    "            root_dir (string): Directory with all the images.\n",
    "            transform (callable, optional): Optional transform to be applied\n",
    "                on a sample.\n",
    "        \"\"\"\n",
    "        self.landmarks_frame = pd.read_csv(csv_file)\n",
    "        self.root_dir = root_dir\n",
    "        self.transform = transform\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.landmarks_frame)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        if torch.is_tensor(idx):\n",
    "            idx = idx.tolist()\n",
    "\n",
    "        img_name = os.path.join(self.root_dir,\n",
    "                                self.landmarks_frame.iloc[idx, 0])\n",
    "        image = io.imread(img_name)\n",
    "        landmarks = self.landmarks_frame.iloc[idx, 1:]\n",
    "        landmarks = np.array([landmarks])\n",
    "        landmarks = landmarks.astype('float').reshape(-1, 2)\n",
    "        sample = {'image': image, 'landmarks': landmarks}\n",
    "\n",
    "        if self.transform:\n",
    "            sample = self.transform(sample)\n",
    "\n",
    "        return sample"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataloaders = {\n",
    "    'train':(image_generator(X_train, y_train, batch_size = 16)\n",
    "            ),\n",
    "    'val':( image_generator(X_val, y_val, batch_size = 16),\n",
    "          )}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for inputs,labels in dataloaders[phase]:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def train_model(model, criterion, optimizer, num_epochs):\n",
    "    since = timeit.timeit()\n",
    "\n",
    "    #best_model_wts = copy.deepcopy(model.state_dict())\n",
    "    best_acc = 0.0\n",
    "\n",
    "    for epoch in range(num_epochs):\n",
    "        print('Epoch {}/{}'.format(epoch +1, num_epochs))\n",
    "        print('-' * 10)\n",
    "\n",
    "        # Each epoch has a training and validation phase\n",
    "        for phase in ['train', 'val']:\n",
    "            if phase == 'train':\n",
    "                #scheduler.step()\n",
    "                model.train()  # Set model to training mode\n",
    "            else:\n",
    "                model.eval()   # Set model to evaluate mode\n",
    "\n",
    "            running_loss = 0.0\n",
    "            running_corrects = 0\n",
    "\n",
    "            # Iterate over data.\n",
    "            for inputs,labels in dataloaders[phase]:\n",
    "                inputs = inputs.to(device)\n",
    "                labels = labels.to(device)\n",
    "                #labels = labels.to(device)\n",
    "\n",
    "                # zero the parameter gradients\n",
    "                optimizer.zero_grad()\n",
    "\n",
    "                # forward\n",
    "                # track history if only in train\n",
    "                with torch.set_grad_enabled(phase == 'train'):\n",
    "                    outputs = model(inputs)\n",
    "                    print(outputs.shape)\n",
    "                    _, preds = torch.max(outputs, 1)\n",
    "                    loss = criterion(outputs, labels)\n",
    "\n",
    "                    # backward + optimize only if in training phase\n",
    "                    if phase == 'train':\n",
    "                        loss.backward()\n",
    "                        optimizer.step()\n",
    "\n",
    "                # statistics\n",
    "                running_loss += loss.item() * inputs.size(0)\n",
    "                running_corrects += torch.sum(preds == labels.data)\n",
    "\n",
    "            '''#For graph generation\n",
    "            if phase == \"train\":\n",
    "                train_loss.append(running_loss/dataset_sizes[phase])\n",
    "                train_acc.append(running_corrects.double() / dataset_sizes[phase])\n",
    "                epoch_counter_train.append(epoch)\n",
    "            if phase == \"val\":\n",
    "                val_loss.append(running_loss/ dataset_sizes[phase])\n",
    "                val_acc.append(running_corrects.double() / dataset_sizes[phase])\n",
    "                epoch_counter_val.append(epoch)\n",
    "\n",
    "            epoch_loss = running_loss / dataset_sizes[phase]\n",
    "            epoch_acc = running_corrects.double() / dataset_sizes[phase]'''\n",
    "               \n",
    "            #for printing        \n",
    "            if phase == \"train\":    \n",
    "                epoch_loss = running_loss / dataset_sizes[phase]\n",
    "                epoch_acc = running_corrects.double() / dataset_sizes[phase]\n",
    "            if phase == \"val\":    \n",
    "                epoch_loss = running_loss / dataset_sizes[phase]\n",
    "                epoch_acc = running_corrects.double() / dataset_sizes[phase]\n",
    "            \n",
    "            \n",
    "            print('{} Loss: {:.4f} Acc: {:.4f}'.format(\n",
    "                phase, epoch_loss, epoch_acc))\n",
    "\n",
    "            # deep copy the best model\n",
    "            if phase == 'val' and epoch_acc > best_acc:\n",
    "                best_acc = epoch_acc\n",
    "                #best_model_wts = copy.deepcopy(model.state_dict())\n",
    "\n",
    "            print()\n",
    "\n",
    "    time_elapsed = timeit.timeit() - since\n",
    "    print('Training complete in {:.0f}m {:.0f}s'.format(\n",
    "        time_elapsed // 60, time_elapsed % 60))\n",
    "    print('Best val Acc: {:4f}'.format(best_acc))\n",
    "\n",
    "    # load best model weights\n",
    "    model.load_state_dict(best_model_wts)\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([8883])"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_val.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataloaders = {\n",
    "    'train':\n",
    "        torch.utils.data.DataLoader(\n",
    "            image_generator(X_train, y_train, batch_size = 16),\n",
    "            batch_size=16,\n",
    "            shuffle=True,\n",
    "            num_workers=4),\n",
    "    'val':\n",
    "        torch.utils.data.DataLoader(\n",
    "            image_generator(X_val, y_val, batch_size = 16),\n",
    "            batch_size=16,\n",
    "            shuffle=False,\n",
    "            num_workers=4)}\n",
    "\n",
    "dataset_sizes = {'train': len(X_train), \n",
    "                 'val': len(X_val)\n",
    "                }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "def train_model(model, criterion, optimizer, num_epochs):\n",
    "    since = timeit.timeit()\n",
    "\n",
    "    #best_model_wts = copy.deepcopy(model.state_dict())\n",
    "    best_acc = 0.0\n",
    "\n",
    "    for epoch in range(num_epochs):\n",
    "        print('Epoch {}/{}'.format(epoch +1, num_epochs))\n",
    "        print('-' * 10)\n",
    "\n",
    "        # Each epoch has a training and validation phase\n",
    "        for phase in ['train', 'val']:\n",
    "            if phase == 'train':\n",
    "                #scheduler.step()\n",
    "                model.train()  # Set model to training mode\n",
    "            else:\n",
    "                model.eval()   # Set model to evaluate mode\n",
    "\n",
    "            running_loss = 0.0\n",
    "            running_corrects = 0\n",
    "\n",
    "            # Iterate over data.\n",
    "            for inputs,labels in dataloaders[phase]:\n",
    "                inputs = inputs.to(device)\n",
    "                labels = labels.to(device)\n",
    "                #labels = labels.to(device)\n",
    "\n",
    "                # zero the parameter gradients\n",
    "                optimizer.zero_grad()\n",
    "\n",
    "                # forward\n",
    "                # track history if only in train\n",
    "                with torch.set_grad_enabled(phase == 'train'):\n",
    "                    outputs = model(inputs)\n",
    "                    print(outputs.shape)\n",
    "                    _, preds = torch.max(outputs, 1)\n",
    "                    loss = criterion(outputs, labels)\n",
    "\n",
    "                    # backward + optimize only if in training phase\n",
    "                    if phase == 'train':\n",
    "                        loss.backward()\n",
    "                        optimizer.step()\n",
    "\n",
    "                # statistics\n",
    "                running_loss += loss.item() * inputs.size(0)\n",
    "                running_corrects += torch.sum(preds == labels.data)\n",
    "\n",
    "            '''#For graph generation\n",
    "            if phase == \"train\":\n",
    "                train_loss.append(running_loss/dataset_sizes[phase])\n",
    "                train_acc.append(running_corrects.double() / dataset_sizes[phase])\n",
    "                epoch_counter_train.append(epoch)\n",
    "            if phase == \"val\":\n",
    "                val_loss.append(running_loss/ dataset_sizes[phase])\n",
    "                val_acc.append(running_corrects.double() / dataset_sizes[phase])\n",
    "                epoch_counter_val.append(epoch)\n",
    "\n",
    "            epoch_loss = running_loss / dataset_sizes[phase]\n",
    "            epoch_acc = running_corrects.double() / dataset_sizes[phase]'''\n",
    "               \n",
    "            #for printing        \n",
    "            if phase == \"train\":    \n",
    "                epoch_loss = running_loss / dataset_sizes[phase]\n",
    "                epoch_acc = running_corrects.double() / dataset_sizes[phase]\n",
    "            if phase == \"val\":    \n",
    "                epoch_loss = running_loss / dataset_sizes[phase]\n",
    "                epoch_acc = running_corrects.double() / dataset_sizes[phase]\n",
    "            \n",
    "            \n",
    "            print('{} Loss: {:.4f} Acc: {:.4f}'.format(\n",
    "                phase, epoch_loss, epoch_acc))\n",
    "\n",
    "            # deep copy the best model\n",
    "            if phase == 'val' and epoch_acc > best_acc:\n",
    "                best_acc = epoch_acc\n",
    "                #best_model_wts = copy.deepcopy(model.state_dict())\n",
    "\n",
    "            print()\n",
    "\n",
    "    time_elapsed = timeit.timeit() - since\n",
    "    print('Training complete in {:.0f}m {:.0f}s'.format(\n",
    "        time_elapsed // 60, time_elapsed % 60))\n",
    "    print('Best val Acc: {:4f}'.format(best_acc))\n",
    "\n",
    "    # load best model weights\n",
    "    model.load_state_dict(best_model_wts)\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using cache found in /home/adm2/.cache/torch/hub/pytorch_vision_v0.6.0\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "ResNet(\n",
       "  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)\n",
       "  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "  (relu): ReLU(inplace=True)\n",
       "  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)\n",
       "  (layer1): Sequential(\n",
       "    (0): Bottleneck(\n",
       "      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU(inplace=True)\n",
       "      (downsample): Sequential(\n",
       "        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      )\n",
       "    )\n",
       "    (1): Bottleneck(\n",
       "      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU(inplace=True)\n",
       "    )\n",
       "    (2): Bottleneck(\n",
       "      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU(inplace=True)\n",
       "    )\n",
       "  )\n",
       "  (layer2): Sequential(\n",
       "    (0): Bottleneck(\n",
       "      (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU(inplace=True)\n",
       "      (downsample): Sequential(\n",
       "        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)\n",
       "        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      )\n",
       "    )\n",
       "    (1): Bottleneck(\n",
       "      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU(inplace=True)\n",
       "    )\n",
       "    (2): Bottleneck(\n",
       "      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU(inplace=True)\n",
       "    )\n",
       "    (3): Bottleneck(\n",
       "      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU(inplace=True)\n",
       "    )\n",
       "  )\n",
       "  (layer3): Sequential(\n",
       "    (0): Bottleneck(\n",
       "      (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU(inplace=True)\n",
       "      (downsample): Sequential(\n",
       "        (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)\n",
       "        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      )\n",
       "    )\n",
       "    (1): Bottleneck(\n",
       "      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU(inplace=True)\n",
       "    )\n",
       "    (2): Bottleneck(\n",
       "      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU(inplace=True)\n",
       "    )\n",
       "    (3): Bottleneck(\n",
       "      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU(inplace=True)\n",
       "    )\n",
       "    (4): Bottleneck(\n",
       "      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU(inplace=True)\n",
       "    )\n",
       "    (5): Bottleneck(\n",
       "      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU(inplace=True)\n",
       "    )\n",
       "  )\n",
       "  (layer4): Sequential(\n",
       "    (0): Bottleneck(\n",
       "      (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU(inplace=True)\n",
       "      (downsample): Sequential(\n",
       "        (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)\n",
       "        (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      )\n",
       "    )\n",
       "    (1): Bottleneck(\n",
       "      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU(inplace=True)\n",
       "    )\n",
       "    (2): Bottleneck(\n",
       "      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
       "      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU(inplace=True)\n",
       "    )\n",
       "  )\n",
       "  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))\n",
       "  (fc): Sequential(\n",
       "    (0): Linear(in_features=2048, out_features=1000, bias=True)\n",
       "    (1): ReLU(inplace=True)\n",
       "    (2): Linear(in_features=1000, out_features=136, bias=True)\n",
       "  )\n",
       ")"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#from torchsummary import summary\n",
    "device = torch.device(\"cpu\")\n",
    "start = timeit.timeit()\n",
    "\n",
    "model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True).to(device)\n",
    "\n",
    "for param in model.parameters():\n",
    "    param.requires_grad = False\n",
    "\n",
    "\n",
    "model.fc = nn.Sequential(\n",
    "    \n",
    "    nn.Linear(2048, 1000),\n",
    "    nn.ReLU(inplace=True),\n",
    "    nn.Linear(1000, 136)).to(device)\n",
    "model.eval()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/4\n",
      "----------\n"
     ]
    },
    {
     "ename": "RuntimeError",
     "evalue": "Expected 4-dimensional input for 4-dimensional weight [64, 3, 7, 7], but got 1-dimensional input of size [16] instead",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mRuntimeError\u001b[0m                              Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-18-6f430c2b81d8>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0moptimizer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0moptim\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mAdam\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfc\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mparameters\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 6\u001b[0;31m \u001b[0mmodel\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtrain_model\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcriterion\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0moptimizer\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnum_epochs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m<ipython-input-16-3f6dc316fe84>\u001b[0m in \u001b[0;36mtrain_model\u001b[0;34m(model, criterion, optimizer, num_epochs)\u001b[0m\n\u001b[1;32m     32\u001b[0m                 \u001b[0;31m# track history if only in train\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     33\u001b[0m                 \u001b[0;32mwith\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mset_grad_enabled\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mphase\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m'train'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 34\u001b[0;31m                     \u001b[0moutputs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0minputs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     35\u001b[0m                     \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0moutputs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     36\u001b[0m                     \u001b[0m_\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mpreds\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmax\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0moutputs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.virtualenvs/qbic/lib/python3.6/site-packages/torch/nn/modules/module.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, *input, **kwargs)\u001b[0m\n\u001b[1;32m    548\u001b[0m             \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_slow_forward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    549\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 550\u001b[0;31m             \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mforward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    551\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0mhook\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_forward_hooks\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    552\u001b[0m             \u001b[0mhook_result\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mhook\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mresult\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.virtualenvs/qbic/lib/python3.6/site-packages/torchvision/models/resnet.py\u001b[0m in \u001b[0;36mforward\u001b[0;34m(self, x)\u001b[0m\n\u001b[1;32m    218\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    219\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mforward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 220\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_forward_impl\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    221\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    222\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.virtualenvs/qbic/lib/python3.6/site-packages/torchvision/models/resnet.py\u001b[0m in \u001b[0;36m_forward_impl\u001b[0;34m(self, x)\u001b[0m\n\u001b[1;32m    201\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_forward_impl\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    202\u001b[0m         \u001b[0;31m# See note [TorchScript super()]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 203\u001b[0;31m         \u001b[0mx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconv1\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    204\u001b[0m         \u001b[0mx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbn1\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    205\u001b[0m         \u001b[0mx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrelu\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.virtualenvs/qbic/lib/python3.6/site-packages/torch/nn/modules/module.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, *input, **kwargs)\u001b[0m\n\u001b[1;32m    548\u001b[0m             \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_slow_forward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    549\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 550\u001b[0;31m             \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mforward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    551\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0mhook\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_forward_hooks\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    552\u001b[0m             \u001b[0mhook_result\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mhook\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mresult\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.virtualenvs/qbic/lib/python3.6/site-packages/torch/nn/modules/conv.py\u001b[0m in \u001b[0;36mforward\u001b[0;34m(self, input)\u001b[0m\n\u001b[1;32m    351\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    352\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mforward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minput\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 353\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_conv_forward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mweight\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    354\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    355\u001b[0m \u001b[0;32mclass\u001b[0m \u001b[0mConv3d\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0m_ConvNd\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.virtualenvs/qbic/lib/python3.6/site-packages/torch/nn/modules/conv.py\u001b[0m in \u001b[0;36m_conv_forward\u001b[0;34m(self, input, weight)\u001b[0m\n\u001b[1;32m    348\u001b[0m                             _pair(0), self.dilation, self.groups)\n\u001b[1;32m    349\u001b[0m         return F.conv2d(input, weight, self.bias, self.stride,\n\u001b[0;32m--> 350\u001b[0;31m                         self.padding, self.dilation, self.groups)\n\u001b[0m\u001b[1;32m    351\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    352\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mforward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minput\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mRuntimeError\u001b[0m: Expected 4-dimensional input for 4-dimensional weight [64, 3, 7, 7], but got 1-dimensional input of size [16] instead"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "#model.summary()\n",
    "\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = optim.Adam(model.fc.parameters())\n",
    "\n",
    "model = train_model(model, criterion, optimizer, num_epochs=4) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "                    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "qbic",
   "language": "python",
   "name": "qbic"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
