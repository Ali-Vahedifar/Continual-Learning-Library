import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import relu, avg_pool2d
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pickle
import numpy as np
import os


# Define ResNet Backbone
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf=20):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf)
        self.bn1 = nn.BatchNorm2d(nf)
        self.layer1 = self._make_layer(block, nf, num_blocks[0], stride=1)
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
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes, nf=20):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, nf)


# Learning without Forgetting Model
class Lwf(nn.Module):
    def __init__(self, backbone, num_classes, args):
        super(Lwf, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.args = args
        self.old_net = None
        self.current_task = 0
        self.device = torch.device("cuda:2" if args.cuda else "cpu")
        self.softmax_temp = args.softmax_temp

        self.criterion = nn.CrossEntropyLoss()
        self.opt = optim.SGD(self.parameters(), lr=args.lr, momentum=0.9)

    def forward(self, x):
        return self.backbone(x)

    def observe(self, inputs, labels, logits=None):
        self.opt.zero_grad()
        outputs = self(inputs)
        loss = self.criterion(outputs[:, :self.num_classes * (self.current_task + 1)], labels)
        if logits is not None:
            softmax = torch.nn.functional.softmax
            old_logits = softmax(logits[:, :self.num_classes * self.current_task] / self.softmax_temp, dim=1)
            new_logits = softmax(outputs[:, :self.num_classes * self.current_task] / self.softmax_temp, dim=1)
            kl_loss = -torch.mean(torch.sum(old_logits * torch.log(new_logits), dim=1))
            loss += self.args.alpha * kl_loss

        loss.backward()
        self.opt.step()
        return loss.item()

    def begin_task(self):
        if self.old_net:
            self.old_net.load_state_dict(self.state_dict())

    def end_task(self):
        self.old_net = ResNet18(self.num_classes).to(self.device)
        self.old_net.load_state_dict(self.state_dict())
        self.old_net.eval()


# Training CIFAR-100 with LwF
def train_lwf_on_cifar100(args):
    # Prepare CIFAR-100 dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform)
    test_data = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform)

    classes_per_task = 100 // args.n_tasks
    train_loaders, test_loaders = [], []

    for task in range(args.n_tasks):
        task_classes = range(task * classes_per_task, (task + 1) * classes_per_task)
        train_idx = [i for i, target in enumerate(train_data.targets) if target in task_classes]
        test_idx = [i for i, target in enumerate(test_data.targets) if target in task_classes]

        train_loaders.append(DataLoader(Subset(train_data, train_idx), batch_size=args.batch_size, shuffle=True))
        test_loaders.append(DataLoader(Subset(test_data, test_idx), batch_size=args.batch_size, shuffle=False))

    # Initialize LwF model
    backbone = ResNet18(args.n_tasks * classes_per_task).to(args.device)
    model = Lwf(backbone, classes_per_task, args).to(args.device)

    # Training and evaluation
    for task_id, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):
        model.current_task = task_id

        # Train
        for epoch in range(args.n_epochs):
            model.train()
            for inputs, labels in tqdm(train_loader, desc=f"Task {task_id + 1}, Epoch {epoch + 1}"):
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                logits = None
                if model.old_net:
                    with torch.no_grad():
                        logits = model.old_net(inputs)
                loss = model.observe(inputs, labels, logits)

        # Evaluate
        model.eval()
        for test_id in range(task_id + 1):
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in test_loaders[test_id]:
                    inputs, labels = inputs.to(args.device), labels.to(args.device)
                    outputs = model(inputs)
                    _, preds = outputs.max(1)
                    correct += preds.eq(labels).sum().item()
                    total += labels.size(0)
            accuracy = 100.0 * correct / total
            print(f"Task {task_id + 1}, Test on Task {test_id + 1}: Accuracy = {accuracy:.2f}%")


# Arguments
class Args:
    data_path = "./data"
    n_tasks = 10
    n_epochs = 1
    batch_size = 64
    lr = 0.01
    alpha = 0.5
    softmax_temp = 2.0
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:2" if cuda else "cpu")


# Run the training
args = Args()
train_lwf_on_cifar100(args)
