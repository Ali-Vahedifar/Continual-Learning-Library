#!/usr/bin/env python
# coding: utf-8

# In[1]:


###GEM


# In[2]:



import argparse
import os
import numpy as np
import pickle
import torch
import subprocess
import sys

# Paths and URLs for CIFAR-100 and MNIST datasets
cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
cifar_path = "cifar-100-python.tar.gz"
mnist_url = "https://s3.amazonaws.com/img-datasets/mnist.npz"
mnist_path = "mnist.npz"

# Download CIFAR-100 dataset if not already present
if not os.path.exists(cifar_path):
    print("Downloading CIFAR-100 dataset...")
    subprocess.call(f"wget {cifar_url}", shell=True)

# Extract CIFAR-100 if not already extracted
if not os.path.exists('cifar-100-python'):
    print("Extracting CIFAR-100 dataset...")
    subprocess.call(f"tar xzfv {cifar_path}", shell=True)

# Download MNIST dataset if not already present
if not os.path.exists(mnist_path):
    print("Downloading MNIST dataset...")
    subprocess.call(f"wget {mnist_url}", shell=True)

# Function to unpickle CIFAR files
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Load CIFAR-100 dataset
cifar100_train = unpickle('cifar-100-python/train')
cifar100_test = unpickle('cifar-100-python/test')

# Process CIFAR-100 data
x_tr = torch.from_numpy(cifar100_train[b'data'])
y_tr = torch.LongTensor(cifar100_train[b'fine_labels'])
x_te = torch.from_numpy(cifar100_test[b'data'])
y_te = torch.LongTensor(cifar100_test[b'fine_labels'])

# Save processed CIFAR-100 dataset
torch.save((x_tr, y_tr, x_te, y_te), 'cifar100.pt')

# Load MNIST dataset
f = np.load(mnist_path)
x_tr_mnist = torch.from_numpy(f['x_train'])
y_tr_mnist = torch.from_numpy(f['y_train']).long()
x_te_mnist = torch.from_numpy(f['x_test'])
y_te_mnist = torch.from_numpy(f['y_test']).long()
f.close()

# Save processed MNIST dataset
torch.save((x_tr_mnist, y_tr_mnist), 'mnist_train.pt')
torch.save((x_te_mnist, y_te_mnist), 'mnist_test.pt')

# Argument parsing setup for task split
parser = argparse.ArgumentParser()
parser.add_argument('--i', default='cifar100.pt', help='input directory')
parser.add_argument('--o', default='cifar100_tasks.pt', help='output file')
parser.add_argument('--n_tasks', default=10, type=int, help='number of tasks')
parser.add_argument('--seed', default=0, type=int, help='random seed')

# Handle Jupyter's extra arguments
if 'ipykernel' in sys.modules:
    args = parser.parse_args([])
else:
    args = parser.parse_args()

# Set the random seed for reproducibility
torch.manual_seed(args.seed)

# Initialize lists for training and test tasks
tasks_tr = []
tasks_te = []

# Load data and preprocess
x_tr, y_tr, x_te, y_te = torch.load(os.path.join(args.i))
x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0
x_te = x_te.float().view(x_te.size(0), -1) / 255.0

# Split data into tasks
cpt = int(100 / args.n_tasks)
for t in range(args.n_tasks):
    c1 = t * cpt
    c2 = (t + 1) * cpt
    i_tr = ((y_tr >= c1) & (y_tr < c2)).nonzero().view(-1)
    i_te = ((y_te >= c1) & (y_te < c2)).nonzero().view(-1)
    tasks_tr.append([(c1, c2), x_tr[i_tr].clone(), y_tr[i_tr].clone()])
    tasks_te.append([(c1, c2), x_te[i_te].clone(), y_te[i_te].clone()])

# Save the processed tasks
torch.save([tasks_tr, tasks_te], args.o)


# In[3]:



import math
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(nclasses, nf=20):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)


# In[4]:


import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import random
import time
import uuid

# Auxiliary GEM functions for handling gradients
import quadprog  # Required for GEM's gradient projection

def store_grad(pp, grads, grad_dims, tid):
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1

def overwrite_grad(pp, newgrad, grad_dims):
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1

def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))

# GEM Model Definition
class GEM(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(GEM, self).__init__()
        self.margin = args.memory_strength
        self.net = ResNet18(n_outputs)  # Assuming ResNet18 is defined for CIFAR-100 tasks
        self.ce = nn.CrossEntropyLoss()
        self.opt = optim.SGD(self.parameters(), lr=args.lr)
        
        self.n_memories = args.n_memories
        self.memory_data = torch.FloatTensor(n_tasks, self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        self.gpu = args.cuda

        # Allocate gradient storage
        self.grad_dims = [param.data.numel() for param in self.parameters()]
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()
            self.grads = self.grads.cuda()

        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        self.nc_per_task = n_outputs // n_tasks

    def forward(self, x, t):
        output = self.net(x)
        offset1, offset2 = t * self.nc_per_task, (t + 1) * self.nc_per_task
        output[:, :offset1].data.fill_(-10e10)
        output[:, offset2:].data.fill_(-10e10)
        return output

    def observe(self, x, t, y):
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

        bsz = y.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[t, self.mem_cnt:endcnt].copy_(x[:effbsz])
        self.memory_labs[t, self.mem_cnt:endcnt].copy_(y[:effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        if len(self.observed_tasks) > 1:
            for past_task in self.observed_tasks[:-1]:
                self.zero_grad()
                ptloss = self.ce(self.forward(self.memory_data[past_task], past_task),
                                 self.memory_labs[past_task])
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims, past_task)

        self.zero_grad()
        loss = self.ce(self.forward(x, t), y)
        loss.backward()

        if len(self.observed_tasks) > 1:
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu else torch.LongTensor(self.observed_tasks[:-1])
            dotp = torch.mm(self.grads[:, t].unsqueeze(0), self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, t].unsqueeze(1), self.grads.index_select(1, indx), self.margin)
                overwrite_grad(self.parameters, self.grads[:, t], self.grad_dims)
        self.opt.step()

# CIFAR-100 Task Loader and Evaluation
def load_cifar100_tasks(args):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform)
    test_data = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform)

    classes_per_task = 100 // args.n_tasks
    train_tasks, test_tasks = [], []
    
    for task in range(args.n_tasks):
        task_classes = range(task * classes_per_task, (task + 1) * classes_per_task)
        train_idx = [i for i, target in enumerate(train_data.targets) if target in task_classes]
        test_idx = [i for i, target in enumerate(test_data.targets) if target in task_classes]
        
        train_tasks.append(DataLoader(Subset(train_data, train_idx), batch_size=args.batch_size, shuffle=True))
        test_tasks.append(DataLoader(Subset(test_data, test_idx), batch_size=args.batch_size, shuffle=False))
    
    return train_tasks, test_tasks

# Training and Evaluation Process
def train_gem_on_cifar100(args):
    device = torch.device("cuda:2" if args.cuda else "cpu")
    train_tasks, test_tasks = load_cifar100_tasks(args)
    model = GEM(32 * 32 * 3, 100, args.n_tasks, args).to(device)  # CIFAR-100 has 32x32x3 inputs and 100 classes

    if args.cuda:
        model.cuda()

    task_accuracies = []
    for task_id, (train_loader, test_loader) in enumerate(zip(train_tasks, test_tasks)):
        model.train()
        for epoch in range(args.n_epochs):
            for x, y in train_loader:
                x = x.view(x.size(0), -1)
                if args.cuda:
                    x, y = x.cuda(), y.cuda()
                model.observe(x, task_id, y)

        # Evaluate on all observed tasks
        model.eval()
        accuracies = []
        for test_id, test_loader in enumerate(test_tasks[:task_id + 1]):
            correct, total = 0, 0
            for x, y in test_loader:
                x = x.view(x.size(0), -1)
                if args.cuda:
                    x, y = x.cuda(), y.cuda()
                with torch.no_grad():
                    output = model(x, test_id)
                    _, pred = output.max(1)
                    correct += pred.eq(y).sum().item()
                    total += y.size(0)
            accuracy = 100 * correct / total
            accuracies.append(accuracy)
            print(f"Task {task_id+1}, Test on Task {test_id+1}, Accuracy: {accuracy:.2f}%")

        task_accuracies.append(accuracies)

    return task_accuracies

# Set arguments
class Args:
    data_path = './data'
    model = 'gem'
    n_tasks = 20
    n_memories = 200
    memory_strength = 0.5
    n_epochs = 1
    batch_size = 10
    lr = 1e-3
    cuda = torch.cuda.is_available()

args = Args()
task_accuracies = train_gem_on_cifar100(args)


# In[ ]:




