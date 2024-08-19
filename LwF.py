#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install scikit-learn


# In[2]:


pip install --upgrade pip


# In[3]:


get_ipython().system('git clone https://github.com/Mattdl/CLsurvey.git')


# In[4]:


get_ipython().system('cd CLsurvey')
get_ipython().system('ls src')


# In[5]:


ls /tf/CLsurvey/CLsurvey/src/utilities


# In[6]:


from CLsurvey.src.utilities import utils


# In[7]:


import sys
sys.path.append('/tf/CLsurvey/CLsurvey/src')

from methods.LwF.AlexNet_LwF import AlexNet_LwF
import methods.Finetune.train_SGD as SGD_Training
import utilities.utils as utils
import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


# In[8]:


#LWF method

def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=45):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    print('lr is ' + str(lr))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
def Rdistillation_loss(y, teacher_scores, T, scale):
    p_y = F.softmax(y)
    p_y = p_y.pow(1 / T)
    sumpy = p_y.sum(1)
    sumpy = sumpy.view(sumpy.size(0), 1)
    p_y = p_y.div(sumpy.repeat(1, scale))
    p_teacher_scores = F.softmax(teacher_scores)
    p_teacher_scores = p_teacher_scores.pow(1 / T)
    p_t_sum = p_teacher_scores.sum(1)
    p_t_sum = p_t_sum.view(p_t_sum.size(0), 1)
    p_teacher_scores = p_teacher_scores.div(p_t_sum.repeat(1, scale))
    loss = -p_teacher_scores * torch.log(p_y)
    loss = loss.sum(1)

    loss = loss.sum(0) / loss.size(0)
    return loss
def distillation_loss(y, teacher_scores, T, scale):
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
       scale is required as kl_div normalizes by nelements and not batch size.
    """

    maxy, xx = y.max(1)
    maxy = maxy.view(y.size(0), 1)
    norm_y = y - maxy.repeat(1, scale)
    ysafe = norm_y / T
    exsafe = torch.exp(ysafe)
    sumex = exsafe.sum(1)
    ######Tscores
    maxT, xx = teacher_scores.max(1)
    maxT = maxT.view(maxT.size(0), 1)
    teacher_scores = teacher_scores - maxT.repeat(1, scale)
    p_teacher_scores = F.softmax(teacher_scores)
    p_teacher_scores = p_teacher_scores.pow(1 / T)
    p_t_sum = p_teacher_scores.sum(1)
    p_t_sum = p_t_sum.view(p_t_sum.size(0), 1)
    p_teacher_scores = p_teacher_scores.div(p_t_sum.repeat(1, scale))

    loss = torch.sum(torch.log(sumex) - torch.sum(p_teacher_scores * ysafe, 1))

    loss = loss / teacher_scores.size(0)
    return loss

def set_lr(optimizer, lr, count):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    continue_training = True
    if count > 10:
        continue_training = False
        print("training terminated")
    if count == 5:
        lr = lr * 0.1
        print('lr is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return optimizer, lr, continue_training


def terminate_protocol(since, best_acc):
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
def train_model_lwf(model, original_model, criterion, optimizer, lr, dset_loaders, dset_sizes, use_gpu, num_epochs,
                    exp_dir='./', resume='', temperature=2, saving_freq=5, reg_lambda=1):
    print('dictoinary length' + str(len(dset_loaders)))
    # set orginal model to eval mode
    original_model.eval()

    since = time.time()
    val_beat_counts = 0  # Counter for the number of epochs without improvement in validation accuracy.
    best_model = model  # Stores the best-performing model.
    best_acc = 0.0  # Stores the best validation accuracy.
    mem_snapshotted = False  # Ensures memory usage is only logged once.
    preprocessing_time = 0  # Tracks the time spent on preprocessing during training.


    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        lr = checkpoint['lr']
        print("lr is ", lr)
        val_beat_counts = checkpoint['val_beat_counts']
        print('load')
        optimizer.load_state_dict(checkpoint['optimizer'])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        start_epoch = 0
        print("=> no checkpoint found at '{}'".format(resume))

    print(str(start_epoch))

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer, lr, continue_training = set_lr(optimizer, lr, count=val_beat_counts)
                if not continue_training:
                    terminate_protocol(since, best_acc)
                    utils.save_preprocessing_time(exp_dir, preprocessing_time)
                    return model, best_acc
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                start_preprocess_time = time.time()
                # get the inputs
                inputs, labels = data
                # ==========
                if phase == 'train':
                    original_inputs = inputs.clone()

                # wrap them in Variable
                if use_gpu:
                    if phase == 'train':
                        original_inputs = original_inputs.cuda()
                        original_inputs = Variable(original_inputs, requires_grad=False)
                    inputs, labels = Variable(inputs.cuda()),                                      Variable(labels.cuda())
                else:
                    if phase == 'train':
                        original_inputs = Variable(original_inputs, requires_grad=False)
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                model.zero_grad()
                original_model.zero_grad()
                # forward
                # tasks_outputs and target_logits are lists of outputs for each task in the previous model and current model
                orginal_logits = original_model(original_inputs)
                # Move to same GPU as current model.
                target_logits = [Variable(item.data, requires_grad=False)
                                 for item in orginal_logits]
                del orginal_logits
                scale = [item.size(-1) for item in target_logits]
                tasks_outputs = model(inputs)
                _, preds = torch.max(tasks_outputs[-1].data, 1)
                task_loss = criterion(tasks_outputs[-1], labels)

                # Compute distillation loss.
                dist_loss = 0
                # Apply distillation loss to all old tasks.
                if phase == 'train':
                    for idx in range(len(target_logits)):
                        dist_loss += distillation_loss(tasks_outputs[idx], target_logits[idx], temperature, scale[idx])
                    # backward + optimize only if in training phase

                total_loss = reg_lambda * dist_loss + task_loss
                preprocessing_time += time.time() - start_preprocess_time

                if phase == 'train':
                    total_loss.backward()
                    optimizer.step()

                if not mem_snapshotted:
                    utils.save_cuda_mem_req(exp_dir)
                    mem_snapshotted = True

                # statistics
                running_loss += task_loss.data.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    del tasks_outputs, labels, inputs, task_loss, preds
                    best_acc = epoch_acc
                    torch.save(model, os.path.join(exp_dir, 'best_model.pth.tar'))
                    val_beat_counts = 0
                else:
                    val_beat_counts += 1

        if epoch % saving_freq == 0:
            epoch_file_name = exp_dir + '/' + 'epoch' + '.pth.tar'
            save_checkpoint({
                'epoch': epoch + 1,
                'lr': lr,
                'val_beat_counts': val_beat_counts,
                'epoch_acc': epoch_acc,
                'best_acc': best_acc,
                'arch': 'alexnet',
                'model': model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, epoch_file_name)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    terminate_protocol(since, best_acc)
    utils.save_preprocessing_time(exp_dir, preprocessing_time)
    return model, best_acc


# In[9]:


get_ipython().system('pip install tqdm')


# In[10]:


#dataset.py
import os
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch

import utilities.utils as utils
import data.tinyimgnet_dataprep as dataprep_tiny
import data.inaturalist_dataprep as dataprep_inat
import data.recogseq_dataprep as dataprep_recogseq


def parse(ds_name):
    """Parse arg string to actual object."""
    if ds_name == InaturalistDataset.argname:
        return InaturalistDataset()
    elif ds_name == InaturalistDatasetUnrelToRel.argname:
        return InaturalistDatasetUnrelToRel()
    elif ds_name == InaturalistDatasetRelToUnrel.argname:
        return InaturalistDatasetRelToUnrel()

    elif ds_name == TinyImgnetDataset.argname:
        return TinyImgnetDataset()
    elif ds_name == TinyImgnetDatasetHardToEasy.argname:
        return TinyImgnetDatasetHardToEasy()
    elif ds_name == TinyImgnetDatasetEasyToHard.argname:
        return TinyImgnetDatasetEasyToHard()

    elif ds_name == ObjRecog8TaskSequence.argname:
        return ObjRecog8TaskSequence()

    elif ds_name == LongTinyImgnetDataset.argname:  # Supplemental
        return LongTinyImgnetDataset()

    else:
        raise NotImplementedError("Dataset not parseable: ", ds_name)


def get_nc_per_task(dataset):
    return [len(classes_for_task) for classes_for_task in dataset.classes_per_task.values()]


class CustomDataset(metaclass=ABCMeta):
    """
    Abstract properties/methods that can be used regardless of which subclass the instance is.
    """

    @property
    @abstractmethod
    def name(self): pass

    @property
    @abstractmethod
    def argname(self): pass

    @property
    @abstractmethod
    def test_results_dir(self): pass

    @property
    @abstractmethod
    def train_exp_results_dir(self): pass

    @property
    @abstractmethod
    def task_count(self): pass

    @property
    @abstractmethod
    def classes_per_task(self): pass

    @property
    @abstractmethod
    def input_size(self): pass

    @abstractmethod
    def get_task_dataset_path(self, task_name, rnd_transform):
        pass

    @abstractmethod
    def get_taskname(self, task_index):
        pass


class InaturalistDataset(CustomDataset):
    """
    iNaturalist dataset.
    - Raw/NoTransform: The ImageFolder has a transform operator that only resizes (e.g. no RandomHorizontalFlip,...)
    """

    name = 'iNaturalist'
    argname = 'inat'
    test_results_dir = 'inaturalist'
    train_exp_results_dir = 'inaturalist'
    task_count = 10
    classes_per_task = OrderedDict()
    input_size = (224, 224)

    def __init__(self, ordering=None, create=True, overwrite=False):
        config = utils.get_parsed_config()
        self.dataset_root = os.path.join(utils.read_from_config(config, 'ds_root_path'), 'inaturalist', 'train_val2018')

        self.unordered_tasks = ['Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca',
                                'Plantae', 'Reptilia'] if ordering is None else ordering
        print("TASK ORDER: ", self.unordered_tasks)

        # Only the train part of the original iNaturalist is used
        self.transformed_dataset_file = 'imgfolder_trainvaltest_rndtrans.pth.tar'
        self.raw_dataset_file = 'imgfolder_trainvaltest.pth.tar'

        self.joint_root = self.dataset_root
        self.joint_training_file = 'imgfolder_joint.pth.tar'

        if create:
            # Download the training/validation dataset of iNaturalist
            dataprep_inat.download_dset(os.path.dirname(self.dataset_root))

            # Divide it into our own training/validation/test splits
            dataprep_inat.prepare_inat_trainval(os.path.dirname(self.dataset_root), outfile=self.raw_dataset_file,
                                                # TRAINONLY_trainvaltest_dataset.pth.tar
                                                rnd_transform=False, overwrite=overwrite)
            dataprep_inat.prepare_inat_trainval(os.path.dirname(self.dataset_root),
                                                outfile=self.transformed_dataset_file,
                                                rnd_transform=True, overwrite=overwrite)
            dataprep_inat.prepare_JOINT_dataset(os.path.dirname(self.dataset_root), outfile=self.joint_training_file,
                                                overwrite=overwrite)
        self.min_class_count = 100
        self.random_chances = []

        # Init classes per task
        self.count_total_classes = 0
        print("Task Training-Samples Validation-Samples Classes Random-chance")
        for task_name in self.unordered_tasks:
            dataset_path = self.get_task_dataset_path(task_name=task_name)
            dsets = torch.load(dataset_path)
            dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
            dset_classes = dsets['train'].classes
            del dsets
            self.classes_per_task[task_name] = dset_classes
            self.count_total_classes += len(dset_classes)
            rnd_chance = '%.3f' % (1. / len(dset_classes))
            self.random_chances.append(rnd_chance)
            print("{} {} {} {} {}".format(str(task_name), dset_sizes['train'], dset_sizes['val'], dset_sizes['test'],
                                          len(dset_classes), rnd_chance))
        print("RANDOM CHANCES: ", ", ".join(self.random_chances))
        print("TOTAL CLASSES COUNT = ", self.count_total_classes)

    def get_task_dataset_path(self, task_name=None, rnd_transform=False):
        # JOINT
        if task_name is None:
            return os.path.join(self.joint_root, self.joint_training_file)

        # PER TASK
        if rnd_transform:
            filename = self.transformed_dataset_file
        else:
            filename = self.raw_dataset_file
        return os.path.join(self.dataset_root, task_name, filename)

    def get_taskname(self, task_count):
        """e.g. Translation of 'Task 1' to the actual name of the first task."""
        if task_count < 1 or task_count > self.task_count:
            raise ValueError('[INATURALIST] TASK COUNT EXCEEDED: count = ', task_count)
        return self.unordered_tasks[task_count - 1]


class InaturalistDatasetRelToUnrel(InaturalistDataset):
    """
    Inaturalsit with diff ordering: from related to unrelated.
    Aves is the largest and taken as init task,
    then each task with highest avg relatedness to all previous tasks is picked.
    """
    task_ordering = ['Aves', 'Mammalia', 'Reptilia', 'Amphibia', 'Animalia', 'Fungi', 'Mollusca', 'Arachnida',
                     'Insecta', 'Plantae']

    suffix = 'ORDERED-rel-to-unrel'
    name = InaturalistDataset.name + ' ' + suffix
    argname = 'inatrelunrel'
    test_results_dir = '_'.join([InaturalistDataset.test_results_dir, suffix])
    train_exp_results_dir = '_'.join([InaturalistDataset.train_exp_results_dir, suffix])

    def __init__(self):
        super().__init__(ordering=self.task_ordering)
        print("INATURALIST ORDERING = ", self.suffix)


class InaturalistDatasetUnrelToRel(InaturalistDataset):
    """
    Inaturalsit with diff ordering: from unrelated to related.
    Starting with biggest: Aves, then based on expert gate: pick most unrelated to all previous tasks (avg).
    """
    task_ordering = ['Aves', 'Fungi', 'Insecta', 'Mollusca', 'Plantae', 'Reptilia', 'Arachnida', 'Mammalia', 'Animalia',
                     'Amphibia']
    suffix = 'ORDERED-unrel-to-rel'
    name = InaturalistDataset.name + ' ' + suffix
    argname = 'inatunrelrel'
    test_results_dir = '_'.join([InaturalistDataset.test_results_dir, suffix])
    train_exp_results_dir = '_'.join([InaturalistDataset.train_exp_results_dir, suffix])

    def __init__(self):
        super().__init__(ordering=self.task_ordering)
        print("INATURALIST ORDERING = ", self.suffix)


class TinyImgnetDataset(CustomDataset):
    name = 'Tiny Imagenet'
    argname = 'tiny'
    test_results_dir = 'tiny_imagenet'
    train_exp_results_dir = 'tiny_imgnet'
    def_task_count, task_count = 10, 10
    classes_per_task = OrderedDict()
    tinyimgnet_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    input_size = (64, 64)

    def __init__(self, crop=False, create=True, task_count=10, dataset_root=None, overwrite=False):
        config = utils.get_parsed_config()

        self.dataset_root = dataset_root if dataset_root else os.path.join(
            utils.read_from_config(config, 'ds_root_path'), 'tiny-imagenet', 'tiny-imagenet-200')
        print("Dataset root = {}".format(self.dataset_root))
        self.crop = crop
        self.task_count = task_count

        self.transformed_dataset_file = 'imgfolder_trainvaltest_rndtrans.pth.tar'
        self.raw_dataset_file = 'imgfolder_trainvaltest.pth.tar'
        self.joint_dataset_file = 'imgfolder_trainvaltest_joint.pth.tar'

        if create:
            dataprep_tiny.download_dset(os.path.dirname(self.dataset_root))
            dataprep_tiny.prepare_dataset(self, self.dataset_root, task_count=self.task_count, survey_order=True,
                                          overwrite=overwrite)
        # Dataset with bare 64x64, no 56x56 crop
        if not crop:
            self.dataset_root = os.path.join(self.dataset_root, 'no_crop')

        # Version with how many tasks
        self.tasks_subdir = "{}tasks".format(task_count)
        if task_count != self.def_task_count:
            self.test_results_dir += self.tasks_subdir
            self.train_exp_results_dir += self.tasks_subdir

        for task_name in range(1, self.task_count + 1):
            dsets = torch.load(self.get_task_dataset_path(str(task_name)))
            dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
            dset_classes = dsets['train'].classes
            self.classes_per_task[str(task_name)] = dset_classes
            print("Task {}: dset_sizes = {}, #classes = {}".format(str(task_name), dset_sizes, len(dset_classes)))

    def get_task_dataset_path(self, task_name=None, rnd_transform=False):
        if task_name is None:  # JOINT
            return os.path.join(self.dataset_root, self.joint_dataset_file)

        filename = self.transformed_dataset_file if rnd_transform else self.raw_dataset_file
        return os.path.join(self.dataset_root, self.tasks_subdir, task_name, filename)

    def get_taskname(self, task_index):
        return str(task_index)


class LongTinyImgnetDataset(TinyImgnetDataset):
    """Tiny Imagenet split in 40 tasks. (Supplemental exps)"""
    suffix = 'LONG'
    name = 'Tiny Imagenet ' + suffix
    argname = 'longtiny'
    task_count = 40

    def __init__(self, crop=False, create=True, task_count=None, overwrite=False):
        task_count = task_count if task_count else self.task_count
        super().__init__(crop=crop, create=create, task_count=task_count, overwrite=overwrite)


class DifLongTinyImgnetDataset(LongTinyImgnetDataset):
    """ LongTinyImagnet + SVHN task (Supplemental exps)"""
    argname = 'diflongtiny'
    task_count = 41

    def __init__(self, crop=False, create=True, overwrite=False):
        # First 40 tasks
        super().__init__(crop=crop, create=create, task_count=40, overwrite=overwrite)
        # Last task
        self.prepare_extratask(overwrite)

        # Overwrite attributes
        self.task_count = 41
        dsets = torch.load(self.get_task_dataset_path(str(41)))
        dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
        dset_classes = dsets['train'].classes
        self.classes_per_task[str(41)] = dset_classes
        print("Task {}: dset_sizes = {}, #classes = {}".format(str(41), dset_sizes, len(dset_classes)))

    def get_task_dataset_path(self, task_name=None, rnd_transform=False):
        if task_name == super().get_taskname(41):
            return self.outpath
        elif task_name is None:  # Joint
            return None
        else:
            return super().get_task_dataset_path(task_name, rnd_transform)

    def prepare_extratask(self, overwrite):
        from torchvision import transforms
        exp_root = '/path/to/datasets/object_recog_8task_seq'
        dataset_filename = 'dataset_64x64_nornd.pth.tar'
        tr = {x: transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) for x in ['train', 'val', 'test']}
        classes = [str(i) for i in range(1, 11)]
        self.outpath = dataprep_recogseq.prepare_dataset(exp_root, exp_root, dataset_filename, 'Pytorch_SVHN_dataset',
                                                         data_transforms=tr, classes=classes, overwrite=overwrite)
        print("prepared all datasets")


class TinyImgnetDatasetHardToEasy(TinyImgnetDataset):
    """
    CREATE LINK: .../tiny-imagenet-200/no_crop/ORDERED-hard-to-easy/1 -> .../tiny-imagenet-200/no_crop/5
    CREATE LINK: .../tiny-imagenet-200/no_crop/ORDERED-hard-to-easy/2 -> .../tiny-imagenet-200/no_crop/7
    CREATE LINK: .../tiny-imagenet-200/no_crop/ORDERED-hard-to-easy/3 -> .../tiny-imagenet-200/no_crop/10
    CREATE LINK: .../tiny-imagenet-200/no_crop/ORDERED-hard-to-easy/4 -> .../tiny-imagenet-200/no_crop/2
    CREATE LINK: .../tiny-imagenet-200/no_crop/ORDERED-hard-to-easy/5 -> .../tiny-imagenet-200/no_crop/9
    CREATE LINK: .../tiny-imagenet-200/no_crop/ORDERED-hard-to-easy/6 -> .../tiny-imagenet-200/no_crop/8
    CREATE LINK: .../tiny-imagenet-200/no_crop/ORDERED-hard-to-easy/7 -> .../tiny-imagenet-200/no_crop/6
    CREATE LINK: .../tiny-imagenet-200/no_crop/ORDERED-hard-to-easy/8 -> .../tiny-imagenet-200/no_crop/4
    CREATE LINK: .../tiny-imagenet-200/no_crop/ORDERED-hard-to-easy/9 -> .../tiny-imagenet-200/no_crop/3
    CREATE LINK: .../tiny-imagenet-200/no_crop/ORDERED-hard-to-easy/10 -> .../tiny-imagenet-200/no_crop/1
    """
    task_ordering = [5, 7, 10, 2, 9, 8, 6, 4, 3, 1]
    suffix = 'ORDERED-hard-to-easy'
    name = 'Tiny Imagenet ' + suffix
    argname = 'tinyhardeasy'
    test_results_dir = 'tiny_imagenet_' + suffix
    train_exp_results_dir = 'tiny_imgnet_' + suffix

    def __init__(self, crop=False, create=False):
        super().__init__(crop=crop, create=create)
        self.original_dataset_root = self.dataset_root
        self.dataset_root = os.path.join(self.original_dataset_root, self.suffix)
        utils.create_dir(self.dataset_root)
        print(self.dataset_root)

        # Create symbolic links if non-existing
        for task in range(1, self.task_count + 1):
            src_taskdir = os.path.join(self.original_dataset_root, str(self.task_ordering[task - 1]))
            dst_tasklink = os.path.join(self.dataset_root, str(task))
            if not os.path.exists(dst_tasklink):
                os.symlink(src_taskdir, dst_tasklink)
                print("CREATE LINK: {} -> {}".format(dst_tasklink, src_taskdir))
            else:
                print("EXISTING LINK: {} -> {}".format(dst_tasklink, src_taskdir))


class TinyImgnetDatasetEasyToHard(TinyImgnetDataset):
    task_ordering = list(reversed([5, 7, 10, 2, 9, 8, 6, 4, 3, 1]))
    suffix = 'ORDERED-easy-to-hard'
    name = 'Tiny Imagenet ' + suffix
    argname = 'tinyeasyhard'
    test_results_dir = 'tiny_imagenet_' + suffix
    train_exp_results_dir = 'tiny_imgnet_' + suffix

    def __init__(self, crop=False, create=False):
        super().__init__(crop=crop, create=create)
        self.original_dataset_root = self.dataset_root
        self.dataset_root = os.path.join(self.original_dataset_root, self.suffix)
        utils.create_dir(self.dataset_root)
        print(self.dataset_root)

        # Create symbolic links if non-existing
        for task in range(1, self.task_count + 1):
            src_taskdir = os.path.join(self.original_dataset_root, str(self.task_ordering[task - 1]))
            dst_tasklink = os.path.join(self.dataset_root, str(task))
            if not os.path.exists(dst_tasklink):
                os.symlink(src_taskdir, dst_tasklink)
                print("CREATE LINK: {} -> {}".format(dst_tasklink, src_taskdir))
            else:
                print("EXISTING LINK: {} -> {}".format(dst_tasklink, src_taskdir))


class TaskDataset(object):

    def __init__(self, name, imagefolder_path, raw_dataset_path=None, dset_sizes=None, dset_classes=None):
        self.name = name
        self.imagefolder_path = imagefolder_path
        self.raw_dataset_path = raw_dataset_path
        self.dset_sizes = dset_sizes
        self.dset_classes = dset_classes

    def init_size_labels(self, classes_per_task):
        dsets = torch.load(self.imagefolder_path)
        dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
        dset_classes = dsets['train'].classes

        self.dset_sizes = dset_sizes
        self.dset_classes = dset_classes
        classes_per_task[self.name] = dset_classes


class ObjRecog8TaskSequence(CustomDataset):
    """
    Preparation script in rercogseq_dataprep.py (not automated).
    (ImageNet) → Flower → Scenes → Birds → Cars → Aircraft → Actions → Letters → SVHN

    Details:
    Pretrained model on ImageNet
    Task flowers: dset_sizes = {'train': 2040, 'val': 3074, 'test': 3075}, #classes = 102
    Task scenes: dset_sizes = {'train': 5360, 'val': 670, 'test': 670}, #classes = 67
    Task birds: dset_sizes = {'train': 5994, 'val': 2897, 'test': 2897}, #classes = 200
    Task cars: dset_sizes = {'train': 8144, 'val': 4020, 'test': 4021}, #classes = 196
    Task aircraft: dset_sizes = {'train': 6666, 'val': 1666, 'test': 1667}, #classes = 100
    Task actions: dset_sizes = {'train': 3102, 'val': 1554, 'test': 1554}, #classes = 11
    Task letters: dset_sizes = {'train': 6850, 'val': 580, 'test': 570}, #classes = 52
    Task svhn: dset_sizes = {'train': 73257, 'val': 13016, 'test': 13016}, #classes = 11
    """

    name = 'obj_recog_8task_seq'
    argname = 'obj8'
    test_results_dir = 'obj_recog_8task_seq'
    train_exp_results_dir = 'obj_recog_8task_seq'
    task_count = 8
    classes_per_task = OrderedDict()
    input_size = (224, 224)  # For AlexNet

    def __init__(self, crop=False):
        config = utils.get_parsed_config()

        assert not crop, ""
        self.crop = crop
        self.dataset_root = os.path.join(
            utils.read_from_config(config, 'ds_root_path'), 'object_recog_8task_seq')

        # Add Tasks ordered
        dataset_filename = 'dataset.pth.tar'
        self.ordered_tasks = []
        self.ordered_tasks.append(
            TaskDataset('flowers', os.path.join(self.dataset_root, 'Pytorch_Flowers', dataset_filename)))
        self.ordered_tasks.append(
            TaskDataset('scenes', os.path.join(self.dataset_root, 'Pytorch_Scenes', dataset_filename)))
        self.ordered_tasks.append(
            TaskDataset('birds', os.path.join(self.dataset_root, 'Pytorch_CUB11', dataset_filename)))
        self.ordered_tasks.append(
            TaskDataset('cars', os.path.join(self.dataset_root, 'Pytorch_Cars_dataset', dataset_filename)))
        self.ordered_tasks.append(
            TaskDataset('aircraft', os.path.join(self.dataset_root, 'Pytorch_AirCraft_dataset', dataset_filename)))
        self.ordered_tasks.append(
            TaskDataset('actions', os.path.join(self.dataset_root, 'Pytorch_Actions_dataset', dataset_filename)))
        self.ordered_tasks.append(
            TaskDataset('letters', os.path.join(self.dataset_root, 'Pytorch_Letters_dataset', dataset_filename)))
        self.ordered_tasks.append(
            TaskDataset('svhn', os.path.join(self.dataset_root, 'Pytorch_SVHN_dataset', dataset_filename)))

        # Init classes per task
        for task in self.ordered_tasks:
            task.init_size_labels(self.classes_per_task)
            print("{} {} {} {} {}".format(str(task.name), len(task.dset_classes), task.dset_sizes['train'],
                                          task.dset_sizes['val'], task.dset_sizes['test'],
                                          ))
        print("[{}] Initialized".format(self.name))

    def get_task_dataset_path(self, task_name=None, rnd_transform=True):
        if task_name is None:  # JOINT
            print("No JOINT dataset defined!")
            return None
        else:
            filename = [task.imagefolder_path for task in self.ordered_tasks if task_name == task.name]
            assert len(filename) == 1

        return os.path.join(self.dataset_root, task_name, filename[0])

    def get_taskname(self, task_index):
        """
        e.g. Translation of 'Task 1' to the actual name of the first task.
        :param task_index:
        :return:
        """
        if task_index < 1 or task_index > self.task_count:
            raise ValueError('[' + self.name + '] TASK INDEX EXCEEDED: idx = ', task_index)
        return self.ordered_tasks[task_index - 1].name


# In[13]:


#imgfolder.py
import bisect
import os
import os.path

from PIL import Image
import numpy as np
import copy
from itertools import accumulate

import torch
import torch.utils.data as data
from torchvision import datasets

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def make_dataset(dir, class_to_idx, file_list):
    images = []
    # print('here')
    dir = os.path.expanduser(dir)
    set_files = [line.rstrip('\n') for line in open(file_list)]
    for target in sorted(os.listdir(dir)):
        # print(target)
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    dir_file = target + '/' + fname
                    # print(dir_file)
                    if dir_file in set_files:
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
    return images


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolderTrainVal(datasets.ImageFolder):
    def __init__(self, root, files_list, transform=None, target_transform=None,
                 loader=default_loader, classes=None, class_to_idx=None, imgs=None):
        """
        :param root: root path of the dataset
        :param files_list: list of filenames to include in this dataset
        :param classes: classes to include, based on subdirs of root if None
        :param class_to_idx: overwrite class to idx mapping
        :param imgs: list of image paths (under root)
        """
        if classes is None:
            assert class_to_idx is None
            classes, class_to_idx = find_classes(root)
        elif class_to_idx is None:
            class_to_idx = {classes[i]: i for i in range(len(classes))}
        print("Creating Imgfolder with root: {}".format(root))
        imgs = make_dataset(root, class_to_idx, files_list) if imgs is None else imgs
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: {}\nSupported image extensions are: {}".
                                format(root, ",".join(IMG_EXTENSIONS))))
        self.root = root
        self.samples = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader


class ImageFolder_Subset(ImageFolderTrainVal):
    """
    Wrapper of ImageFolderTrainVal, subsetting based on indices.
    """

    def __init__(self, dataset, indices):
        self.__dict__ = copy.deepcopy(dataset).__dict__
        self.indices = indices  # Extra

    def __getitem__(self, idx):
        return super().__getitem__(self.indices[idx])  # Only return from subset

    def __len__(self):
        return len(self.indices)


class ImageFolder_Subset_ClassIncremental(ImageFolder_Subset):
    """
    ClassIncremental to only choose samples of specific label.
    Need to subclass in order to retain compatibility with saved ImageFolder_Subset objects.
    (Can't add new attributes...)
    """

    def __init__(self, imgfolder_subset, target_idx):
        """
        Subsets an ImageFolder_Subset object for only the target idx.
        :param imgfolder_subset: ImageFolder_Subset object
        :param target_idx: target int output idx
        """
        if not isinstance(imgfolder_subset, ImageFolder_Subset):
            print("Not a subset={}".format(imgfolder_subset))
            imagefolder_subset = random_split(imgfolder_subset, [len(imgfolder_subset)])[0]
            print("A subset={}".format(imagefolder_subset))

        # Creation of this object shouldn't interfere with original object
        imgfolder_subset = copy.deepcopy(imgfolder_subset)

        # Change ds classes here, to avoid any misuse
        imgfolder_subset.class_to_idx = {label: idx for label, idx in imgfolder_subset.class_to_idx.items()
                                         if idx == target_idx}
        assert len(imgfolder_subset.class_to_idx) == 1
        imgfolder_subset.classes = next(iter(imgfolder_subset.class_to_idx))

        # (path, FC_idx) => from (path, class_to_idx[class]) pairs
        orig_samples = np.asarray(imgfolder_subset.samples)
        subset_samples = orig_samples[imgfolder_subset.indices.numpy()]
        print("SUBSETTING 1 CLASS FROM DSET WITH SIZE: ", subset_samples.shape[0])

        # Filter these samples to only those with certain label
        label_idxs = np.where(subset_samples[:, 1] == str(target_idx))[0]  # indices row
        print("#SAMPLES WITH LABEL {}: {}".format(target_idx, label_idxs.shape[0]))

        # Filter the corresponding indices
        final_indices = imgfolder_subset.indices[label_idxs]

        # Sanity check
        # is first label equal to all others
        is_all_same_label = str(target_idx) == orig_samples[final_indices, 1]
        assert np.all(is_all_same_label)

        # Make a ImageFolder of the whole
        super().__init__(imgfolder_subset, final_indices)


class ImageFolder_Subset_PathRetriever(ImageFolder_Subset):
    """
    Wrapper for Imagefolder_Subset: Also returns path of the images.
    """

    def __init__(self, imagefolder_subset):
        if not isinstance(imagefolder_subset, ImageFolder_Subset):
            print("Transforming into Subset Wrapper={}".format(imagefolder_subset))
            imagefolder_subset = random_split(imagefolder_subset, [len(imagefolder_subset)])[0]
        super().__init__(imagefolder_subset, imagefolder_subset.indices)

    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolder_Subset_PathRetriever, self).__getitem__(index)
        # the image file path
        path = self.samples[self.indices[index]][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))

        return tuple_with_path


class ImagePathlist(data.Dataset):
    """
    Adapted from: https://github.com/pytorch/vision/issues/81
    Load images from a list with paths (no labels).
    """

    def __init__(self, imlist, targetlist=None, root='', transform=None, loader=default_loader):
        self.imlist = imlist
        self.targetlist = targetlist
        self.root = root
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]

        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        if self.targetlist is not None:
            target = self.targetlist[index]
            return img, target
        else:
            return img

    def __len__(self):
        return len(self.imlist)


def random_split(dataset, lengths):
    """
    Creates ImageFolder_Subset subsets from the dataset, by altering the indices.
    :param dataset:
    :param lengths:
    :return: array of ImageFolder_Subset objects
    """
    assert sum(lengths) == len(dataset)
    indices = torch.randperm(sum(lengths))
    return [ImageFolder_Subset(dataset, indices[offset - length:offset]) for offset, length in
            zip(accumulate(lengths), lengths)]


class ConcatDatasetDynamicLabels(torch.utils.data.ConcatDataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
        the output labels are shifted by the dataset index which differs from the pytorch implementation that return the original labels
    """

    def __init__(self, datasets, classes_len):
        """
        :param datasets: List of Imagefolders
        :param classes_len: List of class lengths for each imagefolder
        """
        super(ConcatDatasetDynamicLabels, self).__init__(datasets)
        self.cumulative_classes_len = list(accumulate(classes_len))

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
            img, label = self.datasets[dataset_idx][sample_idx]
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            img, label = self.datasets[dataset_idx][sample_idx]
            label = label + self.cumulative_classes_len[dataset_idx - 1]  # Shift Labels
        return img, label


# In[15]:


#imagenet dataprep.py Download TinyImageNet from: http://cs231n.stanford.edu/tiny-imagenet-200.zip
"""
Download TinyImageNet from: http://cs231n.stanford.edu/tiny-imagenet-200.zip
"""

import os
import torch
import shutil
import subprocess

from torchvision import transforms

import utilities.utils as utils
from data.imgfolder import random_split, ImageFolderTrainVal


def download_dset(path):
    utils.create_dir(path)

    if not os.path.exists(os.path.join(path, 'tiny-imagenet-200.zip')):
        subprocess.call(
            "wget -P {} http://cs231n.stanford.edu/tiny-imagenet-200.zip".format(path),
            shell=True)
        print("Succesfully downloaded TinyImgnet dataset.")
    else:
        print("Already downloaded TinyImgnet dataset in {}".format(path))

    if not os.path.exists(os.path.join(path, 'tiny-imagenet-200')):
        subprocess.call(
            "unzip {} -d {}".format(os.path.join(path, 'tiny-imagenet-200.zip'), path),
            shell=True)
        print("Succesfully extracted TinyImgnet dataset.")
    else:
        print("Already extracted TinyImgnet dataset in {}".format(os.path.join(path, 'tiny-imagenet-200')))


def create_training_classes_file(root_path):
    """
    training dir is ImageFolder like structure.
    Gather all classnames in 1 file for later use.
    Ordering may differ from original classes.txt in project!
    :return:
    """
    with open(os.path.join(root_path, 'classes.txt'), 'w') as classes_file:
        for class_dir in utils.get_immediate_subdirectories(os.path.join(root_path, 'train')):
            classes_file.write(class_dir + "\n")


def preprocess_val(root_path):
    """
    Uses val_annotations.txt to construct ImageFolder like structure.
    Images in 'image' folder are moved into class-folder.
    :return:
    """
    val_path = os.path.join(root_path, 'val')
    annotation_path = os.path.join(val_path, 'val_annotations.txt')

    lines = [line.rstrip('\n') for line in open(annotation_path)]
    for line in lines:
        subs = line.split('\t')
        imagename = subs[0]
        dirname = subs[1]
        this_class_dir = os.path.join(val_path, dirname, 'images')
        if not os.path.isdir(this_class_dir):
            os.makedirs(this_class_dir)

        utils.attempt_move(os.path.join(val_path, 'images', imagename), this_class_dir)


def divide_into_tasks(root_path, task_count=10):
    """
    Divides total subset data into task classes (into dirs "task_x").
    :return:
    """
    print("Be patient: dividing into tasks...")
    nb_classes_task = 200 // task_count
    assert 200 % nb_classes_task == 0, "total 200 classes must be divisible by nb classes per task"

    file_path = os.path.join(root_path, "classes.txt")
    lines = [line.rstrip('\n') for line in open(file_path)]
    assert len(lines) == 200, "Should have 200 classes, but {} lines in classes.txt".format(len(lines))
    subsets = ['train', 'val']
    img_paths = {t: {s: [] for s in subsets + ['classes', 'class_to_idx']} for t in range(1, task_count + 1)}

    for subset in subsets:
        task = 1
        for initial_class in (range(0, len(lines), nb_classes_task)):
            classes = lines[initial_class:initial_class + nb_classes_task]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            if len(img_paths[task]['classes']) == 0:
                img_paths[task]['classes'].extend(classes)
            img_paths[task]['class_to_idx'] = class_to_idx

            # Make subset dataset dir for each task
            for class_index in range(initial_class, initial_class + nb_classes_task):
                target = lines[class_index]
                src_path = os.path.join(root_path, subset, target, 'images')
                imgs = [(os.path.join(src_path, f), class_to_idx[target]) for f in os.listdir(src_path)
                        if os.path.isfile(os.path.join(src_path, f))]  # (label_idx, path)
                img_paths[task][subset].extend(imgs)
            task = task + 1
    return img_paths


def create_train_test_val_imagefolders(img_paths, root, normalize, include_rnd_transform, no_crop):
    # TRAIN
    pre_transf = None
    if include_rnd_transform:
        if no_crop:
            pre_transf = transforms.RandomHorizontalFlip()
        else:
            pre_transf = transforms.Compose([
                transforms.RandomResizedCrop(56),  # Crop
                transforms.RandomHorizontalFlip(), ])
    else:  # No rnd transform
        if not no_crop:
            pre_transf = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(56),  # Crop
            ])
    sufx_transf = [transforms.ToTensor(), normalize, ]
    train_transf = transforms.Compose([pre_transf] + sufx_transf) if pre_transf else transforms.Compose(sufx_transf)
    train_dataset = ImageFolderTrainVal(root, None, transform=train_transf, classes=img_paths['classes'],
                                        class_to_idx=img_paths['class_to_idx'], imgs=img_paths['train'])

    # Validation
    pre_transf_val = None
    sufx_transf_val = [transforms.ToTensor(), normalize, ]
    if not no_crop:
        pre_transf_val = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(56), ])
    val_transf = transforms.Compose([pre_transf_val] + sufx_transf_val) if pre_transf_val         else transforms.Compose(sufx_transf_val)
    test_dataset = ImageFolderTrainVal(root, None, transform=val_transf, classes=img_paths['classes'],
                                       class_to_idx=img_paths['class_to_idx'], imgs=img_paths['val'])

    # Validation set of TinyImgnet is used for testing dataset,
    # Training data set is split into train and validation.
    dsets = {}
    dsets['train'] = train_dataset
    dsets['test'] = test_dataset

    # Split original TinyImgnet trainset into our train and val sets
    dset_trainval = random_split(dsets['train'],
                                 [round(len(dsets['train']) * (0.8)), round(len(dsets['train']) * (0.2))])
    dsets['train'] = dset_trainval[0]
    dsets['val'] = dset_trainval[1]
    dsets['val'].transform = val_transf  # Same transform val/test
    print("Created Dataset:{}".format(dsets))
    return dsets


def create_train_val_test_imagefolder_dict(dataset_root, img_paths, task_count, outfile, no_crop=True, transform=False):
    """
    Makes specific wrapper dictionary with the 3 ImageFolder objects we will use for training, validation and evaluation.
    """
    # Data loading code
    if no_crop:
        out_dir = os.path.join(dataset_root, "no_crop", "{}tasks".format(task_count))
    else:
        out_dir = os.path.join(dataset_root, "{}tasks".format(task_count))

    for task in range(1, task_count + 1):
        print("\nTASK ", task)

        # Tiny Imgnet total values from pytorch
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        dsets = create_train_test_val_imagefolders(img_paths[task], dataset_root, normalize, transform, no_crop)
        utils.create_dir(os.path.join(out_dir, str(task)))
        torch.save(dsets, os.path.join(out_dir, str(task), outfile))
        print("SIZES: train={}, val={}, test={}".format(len(dsets['train']), len(dsets['val']),
                                                        len(dsets['test'])))
        print("Saved dictionary format of train/val/test dataset Imagefolders.")


def create_train_val_test_imagefolder_dict_joint(dataset_root, img_paths, outfile, no_crop=True):
    """
    For JOINT training: All 10 tasks in 1 data folder.
    Makes specific wrapper dictionary with the 3 ImageFolder objects we will use for training, validation and evaluation.
    """
    # Data loading code
    if no_crop:
        out_dir = os.path.join(dataset_root, "no_crop")
    else:
        out_dir = dataset_root

    # Tiny Imgnet total values from pytorch
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dsets = create_train_test_val_imagefolders(img_paths[1], dataset_root, normalize, True, no_crop=no_crop)

    ################ SAVE ##################
    utils.create_dir(out_dir)
    torch.save(dsets, os.path.join(out_dir, outfile))
    print("JOINT SIZES: train={}, val={}, test={}".format(len(dsets['train']), len(dsets['val']),
                                                          len(dsets['test'])))
    print("JOINT: Saved dictionary format of train/val/test dataset Imagefolders.")


def prepare_dataset(dset, target_path, survey_order=True, joint=True, task_count=10, overwrite=False):
    """
    Main datapreparation code for Tiny Imagenet.
    First download the set and set target_path to unzipped download path.
    See README dataprep.

    :param target_path: Path to Tiny Imagenet dataset
    :param survey_order: Use the original survey ordering of the labels to divide in tasks
    :param joint: Prepare the joint dataset
    """
    print("Preparing dataset")
    if not os.path.isdir(target_path):
        raise Exception("TINYIMGNET PATH IS NON EXISTING DIR: ", target_path)

    if os.path.isdir(os.path.join(target_path, 'train')):
        if survey_order:
            shutil.copyfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), "tinyimgnet_classes.txt"),
                            os.path.join(target_path, 'classes.txt'))
        else:
            create_training_classes_file(target_path)
    else:
        print("Already cleaned up original train")

    if not os.path.isfile(os.path.join(target_path, 'VAL_PREPROCESS.TOKEN')):
        preprocess_val(target_path)
        torch.save({}, os.path.join(target_path, 'VAL_PREPROCESS.TOKEN'))
    else:
        print("Already cleaned up original val")

    # Make different subset dataset for each task
    if not os.path.isfile(os.path.join(target_path, "DIV.TOKEN")) or overwrite:
        print("PREPARING DATASET: DIVIDING INTO {} TASKS".format(task_count))
        img_paths = divide_into_tasks(target_path, task_count=task_count)
        torch.save({}, os.path.join(target_path, 'DIV.TOKEN'))
    else:
        print("Already divided into tasks")

    if not os.path.isfile(os.path.join(target_path, "IMGFOLDER.TOKEN")) or overwrite:
        print("PREPARING DATASET: IMAGEFOLDER GENERATION")
        create_train_val_test_imagefolder_dict(target_path, img_paths, task_count, dset.raw_dataset_file,
                                               no_crop=True, transform=False)
        create_train_val_test_imagefolder_dict(target_path, img_paths, task_count, dset.transformed_dataset_file,
                                               no_crop=True, transform=True)
        torch.save({}, os.path.join(target_path, 'IMGFOLDER.TOKEN'))
    else:
        print("Task imgfolders already present.")

    if joint:
        if not os.path.isfile(os.path.join(target_path, "IMGFOLDER_JOINT.TOKEN")) or overwrite:
            print("PREPARING JOINT DATASET: IMAGEFOLDER GENERATION")
            img_paths = divide_into_tasks(target_path, task_count=1)
            # Create joint
            create_train_val_test_imagefolder_dict_joint(target_path, img_paths, dset.joint_dataset_file, no_crop=True)
            torch.save({}, os.path.join(target_path, 'IMGFOLDER_JOINT.TOKEN'))
        else:
            print("Joint imgfolders already present.")

    print("PREPARED DATASET")


# In[16]:


#model net.py
import os
from abc import ABCMeta, abstractmethod

import torch
from torchvision import models

import models.VGGSlim as VGGcreator
import utilities.utils


########################################
# PARSING
########################################

def parse_model_name(models_root_path, model_name, input_size):
    """
    Parses model name into model type object.
    :param model_name: e.g. small_VGG9_cl_512_512
    :param input_size: Size of the input: (w , h)
    :return: the actual model type (not pytorch model)
    """
    pretrained = "pretrained" in model_name
    if "alexnet" in model_name:
        base_model = AlexNet(models_root_path, pretrained=pretrained, create=True)
    elif SmallVGG9.vgg_config in model_name:
        base_model = SmallVGG9(models_root_path, input_size, model_name, create=True)
    elif WideVGG9.vgg_config in model_name:
        base_model = WideVGG9(models_root_path, input_size, model_name, create=True)
    elif DeepVGG22.vgg_config in model_name:
        base_model = DeepVGG22(models_root_path, input_size, model_name, create=True)
    elif BaseVGG9.vgg_config in model_name:
        base_model = BaseVGG9(models_root_path, input_size, model_name, create=True)
    else:
        raise NotImplementedError("MODEL NOT IMPLEMENTED YET: ", model_name)

    return base_model


def get_init_modelname(args):
    """
    The model_name of the first-task model in SI.
    Needs different 1st task model if using regularization: e.g. L2, dropout, BN, dropout+BN
    """
    name = ["e={}".format(args.num_epochs),
            "bs={}".format(args.batch_size),
            "lr={}".format(sorted(args.lr_grid))]
    if args.weight_decay != 0:
        name.append("{}={}".format(ModelRegularization.weight_decay, args.weight_decay))
    if ModelRegularization.batchnorm in args.model_name:
        name.append(ModelRegularization.batchnorm)
    if ModelRegularization.dropout in args.model_name:
        name.append(ModelRegularization.dropout)
    return '_'.join(name)


def extract_modelname_val(seg, tr_exp_dir):
    seg_found = [tr_seg.split('=')[-1] for tr_seg in tr_exp_dir.split('_') if seg == tr_seg.split('=')[0]]
    if len(seg_found) == 1:
        return seg_found[0]
    elif len(seg_found) > 1:
        raise Exception("Ambiguity in exp name: {}".format(seg_found))
    else:
        return None


class ModelRegularization(object):
    vanilla = 'vanilla'
    weight_decay = 'L2'
    dropout = 'DROP'
    batchnorm = 'BN'


########################################
# MODELS
########################################

class Model(metaclass=ABCMeta):
    @property
    @abstractmethod
    def last_layer_idx(self):
        """ Used in data-based methods LWF/EBLL to know where heads start."""
        pass

    @abstractmethod
    def name(self): pass

    @abstractmethod
    def path(self): pass


############################################################
############################################################
# AlexNet
############################################################
############################################################
class AlexNet(Model):
    last_layer_idx = 6

    def __init__(self, models_root_path, pretrained=True, create=False):
        if not os.path.exists(os.path.dirname(models_root_path)):
            raise Exception("MODEL ROOT PATH FOR ALEXNET DOES NOT EXIST: ", models_root_path)

        name = ["alexnet"]
        if pretrained:
            name.append("pretrained_imgnet")
        else:
            name.append("scratch")
        self.name = '_'.join(name)
        self.path = os.path.join(models_root_path,
                                 self.name + ".pth.tar")  # In training scripts: AlexNet pretrained on Imgnet when empty

        if not os.path.exists(self.path):
            if create:
                torch.save(models.alexnet(pretrained=pretrained), self.path)
                print("SAVED NEW ALEXNET MODEL (name=", self.name, ") to ", self.path)
            else:
                raise Exception("Not creating non-existing model: ", self.name)
        else:
            print("STARTING FROM EXISTING ALEXNET MODEL (name=", self.name, ") to ", self.path)

    def name(self):
        return self.name

    def path(self):
        return self.path


############################################################
############################################################
# VGG MODELS
############################################################
############################################################
class VGGModel(Model):
    """
    VGG based models.
    base_vgg9_cl_512_512_DROP_BN
    """
    last_layer_idx = 4  # vgg_classifier_last_layer_idx
    pooling_layers = 4  # in all our models 4 max pooling layers with stride 2

    def __init__(self, models_root_path, input_size, model_name, vgg_config, overwrite_mode=False, create=False):
        if not os.path.exists(os.path.dirname(models_root_path)):
            raise Exception("MODEL ROOT PATH FOR ", model_name, " DOES NOT EXIST: ", models_root_path)

        self.name = model_name
        self.final_featmap_count = VGGcreator.cfg[vgg_config][-2]
        parent_path = os.path.join(models_root_path,
                                   "customVGG_input={}x{}".format(str(input_size[0]), str(input_size[1])))
        self.path = os.path.join(parent_path, self.name + ".pth.tar")

        # After classifier name
        dropout = ModelRegularization.dropout in model_name.split("_")
        batch_norm = ModelRegularization.batchnorm in model_name.split("_")

        if dropout:
            self.last_layer_idx = 6

        if overwrite_mode or not os.path.exists(self.path):
            classifier = parse_classifier_name(model_name)

            last_featmap_size = (
                int(input_size[0] / 2 ** self.pooling_layers), int(input_size[1] / 2 ** self.pooling_layers))
            print("CREATING MODEL, with FC classifier size {}*{}*{}".format(self.final_featmap_count,
                                                                            last_featmap_size[0],
                                                                            last_featmap_size[1]))
            if create:
                utilities.utils.create_dir(parent_path)
                make_VGGmodel(last_featmap_size, vgg_config, self.path, classifier, self.final_featmap_count,
                              batch_norm, dropout)
                print("CREATED MODEL:")
                print(view_saved_model(self.path))
            else:
                raise Exception("Not creating non-existing model: ", self.name)
        else:
            print("MODEL ", model_name, " already exist in path = ", self.path)

    def name(self):
        return self.name

    def path(self):
        return self.path


class SmallVGG9(VGGModel):
    vgg_config = "small_VGG9"
    def_classifier_suffix = "_cl_128_128"

    def __init__(self, models_root_path, input_size, model_name=(vgg_config + def_classifier_suffix),
                 overwrite_mode=False, create=False):
        """
        :param model_name: defined in main script, e.g. small_VGG9_cl_128_128
        :param overwrite_mode: Overwrite if model already exists
        """
        super().__init__(models_root_path, input_size, model_name, vgg_config=self.vgg_config,
                         overwrite_mode=overwrite_mode, create=create)


class BaseVGG9(VGGModel):
    vgg_config = "base_VGG9"
    def_classifier_suffix = "_cl_512_512"

    def __init__(self, models_root_path, input_size, model_name=(vgg_config + def_classifier_suffix),
                 overwrite_mode=False, create=False):
        """
        :param model_name: defined in main script, e.g. base_VGG9_cl_512_512
        :param overwrite_mode: Overwrite if model already exists
        """
        super().__init__(models_root_path, input_size, model_name, vgg_config=self.vgg_config,
                         overwrite_mode=overwrite_mode, create=create)


class WideVGG9(VGGModel):
    vgg_config = "wide_VGG9"
    def_classifier_suffix = "_cl_512_512"

    def __init__(self, models_root_path, input_size, model_name=(vgg_config + def_classifier_suffix),
                 overwrite_mode=False, create=False):
        """
        :param model_name: defined in main script, e.g. base_vgg9_cl_512_512
        :param overwrite_mode: Overwrite if model already exists
        """
        super().__init__(models_root_path, input_size, model_name, vgg_config=self.vgg_config,
                         overwrite_mode=overwrite_mode, create=create)


class DeepVGG22(VGGModel):
    vgg_config = "deep_VGG22"
    def_classifier_suffix = "_cl_512_512"

    def __init__(self, models_root_path, input_size, model_name=(vgg_config + def_classifier_suffix),
                 overwrite_mode=False, create=False):
        """
        :param model_name: defined in main script, e.g. base_vgg9_cl_512_512
        :param overwrite_mode: Overwrite if model already exists
        """
        super().__init__(models_root_path, input_size, model_name, vgg_config=self.vgg_config,
                         overwrite_mode=overwrite_mode, create=create)


############################################################
# FUNCTIONS
############################################################
def make_VGGmodel(last_featmap_size, name, path, classifier, final_featmap_count, batch_norm, dropout):
    """
    Creates custom VGG model with specified classifier array.

    :param last_featmap_size: (w , h ) tupple showing last feature map size.
    :param name: custom VGG config name for feature extraction
    :param path:
    :param classifier: array of length 2, with sizes of 2 FC layers
    :param final_featmap_count: amount of feat maps in the last non-pooling layer. Used to calc classifier input.
    :return:
    """
    # Create and save the model in data root path
    model = VGGcreator.VGGSlim(config=name, num_classes=20,
                               classifier_inputdim=final_featmap_count * last_featmap_size[0] * last_featmap_size[1],
                               classifier_dim1=int(classifier[0]),
                               classifier_dim2=int(classifier[1]),
                               batch_norm=batch_norm,
                               dropout=dropout)
    torch.save(model, path)
    print("SAVED NEW MODEL (name=", name, ", classifier=", classifier, ") to ", path)


def parse_classifier_name(model_name, classifier_layers=3):
    """
    Takes in model name (e.g. base_vgg9_cl_512_512_BN), and returns classifier sizes: [512,512]
    :param model_name:
    :return:
    """
    return model_name[model_name.index("cl_"):].split("_")[1:classifier_layers]


def get_vgg_classifier_postfix(classifier):
    return "_cl_" + '_'.join(str(classifier))


def save_model_to_path(self, model):
    torch.save(model, self.path)


def print_module_composition(vgg_config_name):
    """
    Prints the amount of weights and biase parameters in the feat extractor.
    Formatted in a per module basis.
    :param vgg_config_name:
    :return:
    """
    vgg_config = VGGcreator.cfg[vgg_config_name]

    # Print Weights
    weight_str = []
    weight_str.append("(" + str(VGGcreator.conv_kernel_size) + "*" + str(VGGcreator.conv_kernel_size) + ") * {(")
    bias_str = []

    weightlist = vgg_config
    weightlist.insert(0, VGGcreator.img_input_channels)

    for idx in range(1, len(weightlist)):
        if 'M' == weightlist[idx]:
            weight_str.append(")")
            if idx != len(weightlist) - 1:
                weight_str.append(" + (")

        else:
            prev = str(weightlist[idx - 1])
            if prev == "M":
                prev = str(weightlist[idx - 2])
            elif idx > 1:
                weight_str.append(" + ")
            current_layer_size = str(weightlist[idx])
            weight_str.append(prev + "*" + current_layer_size)
            bias_str.append(current_layer_size)

    weight_str.append("}")

    print("=> Weights = ", "".join(weight_str))
    print("=> Biases = ", " + ".join(bias_str))


def count_parameters(model_type, loaded_model=None, print_module=True):
    """
    Returns the number of trainable parameters in the model.

    :param model:
    :return:
    """
    if loaded_model is None:
        model = torch.load(model_type.path)
    else:
        model = loaded_model
    classifier = model.classifier
    feat = model.features

    classifier_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    feat_params = sum(p.numel() for p in feat.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 10, "MODEL ", model_type.name, "=" * 10)
    # '{:,}'.format(1234567890.001)
    print('%12s  %12s  %12s' % ('Feat', 'Classifier', 'TOTAL'))
    print('%12s  %12s  %12s' % (
        '{:,}'.format(feat_params), '{:,}'.format(classifier_params), '{:,}'.format(total_params)))
    if print_module and hasattr(model_type, 'vgg_config'):
        print_module_composition(model_type.vgg_config)


def view_saved_model(path):
    """
    View model architecture of a saved model, by specifiying the path.
    :param path:
    :return:
    """
    print(torch.load(path))


# In[17]:


# model VGGslim.py
import torch.nn as nn
import torch
import torchvision

#############################
# Static params: Config
#############################
conv_kernel_size = 3
img_input_channels = 3
cfg = {
    '19normal': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    '16normal': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '11normal': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],

    # models TinyImgnet
    'small_VGG9': [64, 'M', 64, 'M', 64, 64, 'M', 128, 128, 'M'],  # 334,016 feat params,
    'base_VGG9': [64, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M'],  # 1.145.408 feat params
    'wide_VGG9': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],  # 4.500.864 feat params
    'deep_VGG22': [64, 'M', 64, 64, 64, 64, 64, 64, 'M', 128, 128, 128, 128, 128, 128, 'M',
                   256, 256, 256, 256, 256, 256, 'M'],  # 4.280.704 feat params
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = img_input_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=conv_kernel_size, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGSlim(torchvision.models.VGG):
    """
    Creates VGG feature extractor from config and custom classifier.
    """

    def __init__(self, config='11Slim', num_classes=50, init_weights=True,
                 classifier_inputdim=512 * 7 * 7, classifier_dim1=512, classifier_dim2=512, batch_norm=False,
                 dropout=False):
        features = make_layers(cfg[config], batch_norm=batch_norm)
        super(VGGSlim, self).__init__(features)

        if hasattr(self, 'avgpool'):  # Compat Pytorch>1.0.0
            self.avgpool = torch.nn.Identity()

        if dropout:  # Same as in Pytorch default: print(models.vgg11_bn())
            self.classifier = nn.Sequential(
                nn.Linear(classifier_inputdim, classifier_dim1),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(classifier_dim1, classifier_dim2),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(classifier_dim2, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(classifier_inputdim, classifier_dim1),
                nn.ReLU(True),
                nn.Linear(classifier_dim1, classifier_dim2),
                nn.ReLU(True),
                nn.Linear(classifier_dim2, num_classes),
            )
        if init_weights:
            self._initialize_weights()


# In[18]:


#LWF method



def fine_tune_SGD_LwF(dataset_path, previous_task_model_path, init_model_path='', exp_dir='', batch_size=200,
                      num_epochs=100, lr=0.0004, init_freeze=1, pretrained=True, weight_decay=0, last_layer_name=6,
                      saving_freq=5, reg_lambda=1):
    print('lr is ' + str(lr))

    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                   shuffle=True, num_workers=8, pin_memory=True)
                    for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes

    use_gpu = torch.cuda.is_available()
    resume = os.path.join(exp_dir, 'epoch.pth.tar')

    if os.path.isfile(resume):
        checkpoint = torch.load(resume)
        model_ft = checkpoint['model']
        previous_model = torch.load(previous_task_model_path)
        if not (type(previous_model) is AlexNet_LwF):
            previous_model = AlexNet_LwF(previous_model, last_layer_name=last_layer_name)
        original_model = copy.deepcopy(previous_model)
        del checkpoint
        del previous_model
    else:
        model_ft = torch.load(previous_task_model_path)

        if not (type(model_ft) is AlexNet_LwF):
            last_layer_index = (len(model_ft.classifier._modules) - 1)
            model_ft = AlexNet_LwF(model_ft, last_layer_name=last_layer_index)
            num_ftrs = model_ft.model.classifier[last_layer_index].in_features
            model_ft.num_ftrs = num_ftrs

        original_model = copy.deepcopy(model_ft)

        if not init_freeze:

            model_ft.model.classifier.add_module(str(len(model_ft.model.classifier._modules)),
                                                 nn.Linear(model_ft.num_ftrs, len(dset_classes)))
        else:

            init_model = torch.load(init_model_path)
            model_ft.model.classifier.add_module(str(len(model_ft.model.classifier._modules)),
                                                 init_model.classifier[6])
            del init_model
            # do something else
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

    if not hasattr(model_ft, 'reg_params'):
        model_ft.reg_params = {}
    model_ft.reg_params['reg_lambda'] = reg_lambda

    if use_gpu:
        model_ft = model_ft.cuda()
        original_model = original_model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr, momentum=0.9, weight_decay=weight_decay)

    model_ft = train_model_lwf(model_ft, original_model, criterion, optimizer_ft, lr, dset_loaders, dset_sizes, use_gpu,
                               num_epochs, exp_dir, resume,
                               saving_freq=saving_freq,
                               reg_lambda=reg_lambda)

    return model_ft


def fine_tune_freeze(dataset_path, model_path, exp_dir, batch_size=100, num_epochs=100, lr=0.0004):
    print('lr is ' + str(lr))

    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                   shuffle=True, num_workers=8, pin_memory=True)
                    for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes

    use_gpu = torch.cuda.is_available()
    resume = os.path.join(exp_dir, 'epoch.pth.tar')
    if os.path.isfile(resume):
        checkpoint = torch.load(resume)
        model_ft = checkpoint['model']

    model_ft = torch.load(model_path)
    if type(model_ft) is AlexNet_LwF:
        model_ft = model_ft.module
        last_layer_index = str(len(model_ft.classifier._modules) - 1)
        num_ftrs = model_ft.classifier[last_layer_index].in_features
        keep_poping = True
        while keep_poping:
            x = model_ft.classifier._modules.popitem()
            if x[0] == last_layer_index:
                keep_poping = False
    else:
        last_layer_index = str(len(model_ft.classifier._modules) - 1)
        num_ftrs = model_ft.classifier[last_layer_index].in_features

    model_ft.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, len(dset_classes))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if use_gpu:
        model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.classifier._modules[last_layer_index].parameters(), lr, momentum=0.9)
    model_ft = SGD_Training.train_model(model_ft, criterion, optimizer_ft, lr, dset_loaders, dset_sizes, use_gpu,
                                        num_epochs, exp_dir, resume)
    return model_ft


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


# In[19]:


get_ipython().system('pip install torchnet')


# In[20]:


from abc import ABC, abstractmethod
from enum import Enum, auto
from collections import OrderedDict
import os
import time
import warnings
import itertools
import copy

import torch
from torch.autograd import Variable

import utilities.utils
import data.dataset as dataset_utils
import models.net as models
from data.imgfolder import ConcatDatasetDynamicLabels
from models.net import ModelRegularization

import framework.inference as test_network

import methods.EWC.main_EWC as trainEWC
import methods.SI.main_SI as trainSI
import methods.MAS.main_MAS as trainMAS
import methods.LwF.main_LWF as trainLWF
import methods.EBLL.Finetune_SGD_EBLL as trainEBLL
import methods.packnet.main as trainPacknet
import methods.rehearsal.main_rehearsal as trainRehearsal
import methods.HAT.run as trainHAT
import methods.IMM.main_L2transfer as trainIMM
import methods.IMM.merge as mergeIMM
import methods.Finetune.main_SGD as trainFT


# PARSING
def parse(method_name):
    """Parse arg string to actual object."""
    # Exact
    if method_name == YourMethod.name:  # Parsing Your Method name as argument
        return YourMethod()

    elif method_name == EWC.name:
        return EWC()
    elif method_name == MAS.name:
        return MAS()
    elif method_name == SI.name:
        return SI()

    elif method_name == EBLL.name:
        return EBLL()
    elif method_name == LWF.name:
        return LWF()

    elif method_name == GEM.name:
        return GEM()
    elif method_name == ICARL.name:
        return ICARL()

    elif method_name == PackNet.name:
        return PackNet()
    elif method_name == HAT.name:
        return HAT()

    elif method_name == Finetune.name:
        return Finetune()
    elif method_name == FinetuneRehearsalFullMem.name:
        return FinetuneRehearsalFullMem()
    elif method_name == FinetuneRehearsalPartialMem.name:
        return FinetuneRehearsalPartialMem()

    elif method_name == Joint.name:
        return Joint()

    # Modes
    elif IMM.name in method_name:  # modeIMM,meanIMM
        mode = method_name.replace('_', '').replace(IMM.name, '').strip()
        return IMM(mode)
    else:
        raise NotImplementedError("Method not yet parseable")


class Method(ABC):
    @property
    @abstractmethod
    def name(self): pass

    @property
    @abstractmethod
    def eval_name(self): pass

    @property
    @abstractmethod
    def category(self): pass

    @property
    @abstractmethod
    def extra_hyperparams_count(self): pass

    @property
    @abstractmethod
    def hyperparams(self): pass

    @classmethod
    def __subclasshook__(cls, C):
        return False

    @abstractmethod
    def get_output(self, images, args): pass

    @staticmethod
    @abstractmethod
    def inference_eval(args, manager): pass


class Category(Enum):
    MODEL_BASED = auto()
    DATA_BASED = auto()
    MASK_BASED = auto()
    BASELINE = auto()
    REHEARSAL_BASED = auto()

    def __eq__(self, other):
        """Compare by equality rather than identity."""
        return self.name == other.name and self.value == other.value


####################################################
################ YOUR METHOD #######################
class YourMethod(Method):
    name = "YourMethodName"
    eval_name = name
    category = Category.REHEARSAL_BASED  # Change to your method
    extra_hyperparams_count = 1
    hyperparams = OrderedDict({'stability_related_hyperparam': 1})  # Hyperparams to decay
    static_hyperparams = OrderedDict({'hyperparams_not_to_decay': 1024})  # Hyperparams not to decay (e.g. buffer size)
    wrap_first_task_model = False  # Start SI model/ wrap a scratch model in a custom model

    @staticmethod
    def train_args_overwrite(args):
        """
        Overwrite whatever arguments for your method.
        :return: Nothing
        """
        # e.g. args.starting_task_count = 1 #(joint)
        pass

    # PREPROCESS: MAXIMAL PLASTICITY SEARCH
    def grid_prestep(self, args, manager):
        """Processing before starting first phase. e.g. PackNet modeldump for first task."""
        pass

    # MAXIMAL PLASTICITY SEARCH
    @staticmethod
    def grid_train(args, manager, lr):
        """
        Train for finetuning gridsearch learning rate.
        :return: best model, best accuracy
        """
        return Finetune.grid_train(args, manager, lr)  # Or your own FT-related access point

    # POSTPROCESS: 1st phase
    @staticmethod
    def grid_poststep(args, manager):
        """ Postprocessing after max plasticity search."""
        Finetune.grid_poststep(args, manager)

    # STABILITY DECAY
    def train(self, args, manager, hyperparams):
        """
        Train for stability decay iteration.
        :param args/manager: paths and flags, see other methods and main pipeline.
        :param hyperparams: current hyperparams to use for your method.
        :return: best model and accuracy
        """
        print("Your Method: Training")
        return {}, 100

    # POSTPROCESS 2nd phase
    def poststep(self, args, manager):
        """
        Define some postprocessing after the two framework phases. (e.g. iCaRL define exemplars this task)
        :return: Nothing
        """
        pass

    # INFERENCE ACCESS POINT
    @staticmethod
    def inference_eval(args, manager):
        """
        Loads and defines models and heads for evaluation.
        :param args/manager: paths etc.
        :return: accuracy
        """
        return Finetune.inference_eval(args, manager)

    # INFERENCE
    def get_output(self, images, args):
        """
        Get the output for your method. (e.g. iCaRL first selects subset of the single-head).
        :param images: input images
        :return: the network outputs
        """
        # offset1, offset2 = args.model.compute_offsets(args.current_head_idx, args.model.cum_nc_per_task)  # iCaRL
        # outputs = args.model(Variable(images), args.current_head_idx)[:, offset1: offset2]
        return args.model(Variable(images))

    ###################################################
    ###### OPTIONALS = Only define when required ######
    ###################################################

    # OPTIONAL: DATASET MERGING (JOINT): DEFINE DSET LIST
    # @staticmethod
    # def grid_datafetch(args, dataset):
    #     """ Only define for list of datasets to append (see Joint)."""
    #     max_task = dataset.task_count  # Include all datasets in the list
    #     current_task_dataset_path = [dataset.get_task_dataset_path(
    #         task_name=dataset.get_taskname(ds_task_counter), rnd_transform=False)
    #         for ds_task_counter in range(1, max_task + 1)]  # Merge current task dataset with all prev task ones
    #     print("Running JOINT for task ", args.task_name, " on datasets: ", current_task_dataset_path)
    #     return current_task_dataset_path

    # OPTIONAL: DATASET MERGING (JOINT): DEFINE IMGFOLDER
    # @staticmethod
    # def compose_dataset(dataset_path, batch_size):
    #     return Finetune.compose_dataset(dataset_path, batch_size)


##################################################
################ Functions #######################
# Defaults
def get_output_def(model, heads, images, current_head_idx, final_layer_idx):
    head = heads[current_head_idx]
    model.classifier._modules[final_layer_idx] = head  # Change head
    model.eval()
    outputs = model(Variable(images))
    return outputs


def set_hyperparams(method, hyperparams, static_params=False):
    """ Parse hyperparameter string using ';' for hyperparameter list value, single value floats using ','.
        e.g. 0.5,300 -> sets hyperparam1=0.5, hyperparam2=300.0
        e.g. 0.1,0.2;5.2,300 -> sets hyperparam1=[0.1, 0.2], hyperparam2=[5.2, 300.0]
    """
    assert isinstance(hyperparams, str)
    leave_default = lambda x: x == 'def' or x == ''
    hyperparam_vals = []
    split_lists = [x.strip() for x in hyperparams.split(';') if len(x) > 0]
    for split_list in split_lists:
        split_params = [float(x) for x in split_list.split(',') if not leave_default(x)]
        split_params = split_params[0] if len(split_params) == 1 else split_params
        if len(split_lists) == 1:
            hyperparam_vals = split_params
        else:
            hyperparam_vals.append(split_params)

    if static_params:
        if not hasattr(method, 'static_hyperparams'):
            print("No static hyperparams to set.")
            return
        target = method.static_hyperparams
    else:
        target = method.hyperparams

    for hyperparam_idx, (hyperparam_key, def_val) in enumerate(target.items()):
        if hyperparam_idx < len(hyperparam_vals):
            arg_val = hyperparam_vals[hyperparam_idx]
            if leave_default(arg_val):
                continue
            target[hyperparam_key] = arg_val
            print("Set value {}={}".format(hyperparam_key, target[hyperparam_key]))
        else:
            print("Retaining default value {}={}".format(hyperparam_key, def_val))

    method.init_hyperparams = copy.deepcopy(target)  # Backup starting hyperparams
    print("INIT HYPERPARAMETERS: {}".format(target))


#####################################################
################ SOTA Methods #######################

# REHEARSAL
class GEM(Method):
    name = "GEM"
    eval_name = name
    category = Category.REHEARSAL_BASED
    extra_hyperparams_count = 1
    hyperparams = OrderedDict({'margin': 1})
    static_hyperparams = OrderedDict({'mem_per_task': 1024})
    wrap_first_task_model = True

    def train(self, args, manager, hyperparams):
        print("Rehearsal: GEM")
        return _rehearsal_accespoint(args, manager, hyperparams['margin'], self.static_hyperparams['mem_per_task'],
                                     'gem')

    def get_output(self, images, args):
        offset1, offset2 = args.model.compute_offsets(args.current_head_idx,
                                                      args.model.cum_nc_per_task)  # No shared head
        outputs = args.model(Variable(images), args.current_head_idx)[:, offset1: offset2]
        return outputs

    def poststep(self, args, manager):
        """ GEM only needs to collect exemplars for the first SI model. """
        if args.task_counter > 1:
            return

        print("POSTPROCESS PIPELINE")
        start_time = time.time()
        save_path = manager.best_model_path  # Save wrapped SI model in first task best_model_path
        prev_model_path = manager.previous_task_model_path

        if os.path.exists(save_path):
            print("SKIPPING POSTPROCESS: ALREADY DONE")
        else:
            _rehearsal_accespoint(args, manager, self.hyperparams['margin'], self.static_hyperparams['mem_per_task'],
                                  'gem', save_path, prev_model_path,
                                  postprocess=args.task_counter == 1)

        args.postprocess_time = time.time() - start_time
        manager.best_model_path = save_path  # New best model (will be used for next task)

    def grid_train(self, args, manager, lr):
        args.lr = lr
        return _rehearsal_accespoint(args, manager, 0, self.static_hyperparams['mem_per_task'], 'gem',
                                     save_path=manager.gridsearch_exp_dir, finetune=True)

    @staticmethod
    def inference_eval(args, manager):
        return FinetuneRehearsalFullMem.inference_eval(args, manager)


class ICARL(Method):
    name = "ICARL"
    eval_name = name
    category = Category.REHEARSAL_BASED
    extra_hyperparams_count = 1
    hyperparams = OrderedDict({'lambda': 10})
    static_hyperparams = OrderedDict({'mem_per_task': 1024})
    wrap_first_task_model = True

    def train(self, args, manager, hyperparams):
        print("Rehearsal: ICARL")
        return _rehearsal_accespoint(args, manager, hyperparams['lambda'], self.static_hyperparams['mem_per_task'],
                                     'icarl')

    def get_output(self, images, args):
        offset1, offset2 = args.model.compute_offsets(args.current_head_idx,
                                                      args.model.cum_nc_per_task)  # No shared head
        outputs = args.model(Variable(images), args.current_head_idx, args=args)
        outputs = outputs[:, offset1: offset2]
        return outputs

    def poststep(self, args, manager):
        """ iCARL always needs this step to collect the exemplars. """
        print("POSTPROCESS PIPELINE")
        start_time = time.time()
        if args.task_counter == 1:
            save_path = manager.best_model_path  # Save wrapped SI model in first task best_model_path for iCarl
            prev_model_path = manager.previous_task_model_path  # SI common model first task (shared)
        else:
            save_path = os.path.join(manager.heuristic_exp_dir, 'best_model_postprocessed.pth.tar')
            prev_model_path = manager.best_model_path

        if os.path.exists(save_path):
            print("SKIPPING POSTPROCESS: ALREADY DONE")
        else:
            _rehearsal_accespoint(args, manager,
                                  self.hyperparams['lambda'], self.static_hyperparams['mem_per_task'], 'icarl',
                                  save_path, prev_model_path, postprocess=True)

        args.postprocess_time = time.time() - start_time
        manager.best_model_path = save_path  # New best model (will be used for next task)

    def grid_train(self, args, manager, lr):
        args.lr = lr
        return _rehearsal_accespoint(args, manager, 0, self.static_hyperparams['mem_per_task'], 'icarl',
                                     save_path=manager.gridsearch_exp_dir, finetune=True)

    @staticmethod
    def inference_eval(args, manager):
        return FinetuneRehearsalFullMem.inference_eval(args, manager)


def _rehearsal_accespoint(args, manager, memory_strength, mem_per_task, method_arg,
                          save_path=None, prev_model_path=None, finetune=False, postprocess=False):
    nc_per_task = dataset_utils.get_nc_per_task(manager.dataset)
    total_outputs = sum(nc_per_task)
    print("nc_per_task = {}, TOTAL OUTPUTS = {}".format(nc_per_task, total_outputs))

    save_path = manager.heuristic_exp_dir if save_path is None else save_path
    prev_model_path = manager.previous_task_model_path if prev_model_path is None else prev_model_path

    manager.overwrite_args = {
        'weight_decay': args.weight_decay,
        'task_name': args.task_name,
        'task_count': args.task_counter,
        'prev_model_path': prev_model_path,
        'save_path': save_path,
        'n_outputs': total_outputs,
        'method': method_arg,
        'n_memories': mem_per_task,
        'n_epochs': args.num_epochs,
        'memory_strength': memory_strength,
        'cuda': True,
        'dataset_path': manager.current_task_dataset_path,
        'n_tasks': manager.dataset.task_count,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'finetune': finetune,  # FT mode for iCarl/GEM
        'is_scratch_model': args.task_counter == 1,
        'postprocess': postprocess,
    }
    model, task_lr_acc = trainRehearsal.main(manager.overwrite_args, nc_per_task)
    return model, task_lr_acc


# MASK BASED
class PackNet(Method):
    name = "packnet"
    eval_name = name
    category = Category.MASK_BASED
    extra_hyperparams_count = 1
    hyperparams = OrderedDict({'prune_perc_per_layer': 0.9})
    grid_chkpt = True
    start_scratch = True

    def __init__(self):
        self.pruned_savename = None

    @staticmethod
    def get_dataset_name(task_name):
        return 'survey_TASK_' + task_name

    def train_init(self, args, manager):
        self.pruned_savename = os.path.join(manager.heuristic_exp_dir, 'best_model_PRUNED')

    def train(self, args, manager, hyperparams):
        prune_lr = args.lr * 0.1  # FT LR, order 10 lower

        print("PACKNET PRUNE PHASE")
        manager.overwrite_args = {
            'weight_decay': args.weight_decay,
            'train_path': manager.current_task_dataset_path,
            'test_path': manager.current_task_dataset_path,
            'mode': 'prune',
            'dataset': self.get_dataset_name(args.task_name),
            'loadname': manager.best_finetuned_model_path,  # Load FT trained model
            'post_prune_epochs': 10,
            'prune_perc_per_layer': hyperparams['prune_perc_per_layer'],
            'lr': prune_lr,
            'finetune_epochs': args.num_epochs,
            'cuda': True,
            'save_prefix': self.pruned_savename,  # exp path filename
            'train_bn': args.train_bn,
            'saving_freq': args.saving_freq,
            'current_dataset_idx': args.task_counter,
        }
        task_lr_acc = trainPacknet.main(manager.overwrite_args)
        return None, task_lr_acc

    def get_output(self, images, args):
        return get_output_def(args.model, args.heads, images, args.current_head_idx, args.final_layer_idx)

    def init_next_task(self, manager):
        assert self.pruned_savename is not None
        if os.path.exists(self.pruned_savename + "_final.pth.tar"):
            manager.previous_task_model_path = self.pruned_savename + "_final.pth.tar"
        elif os.path.exists(self.pruned_savename + "_postprune.pth.tar"):
            warnings.warn("Final file not found(no final file saved if finetune gives no improvement)! Using postprune")
            manager.previous_task_model_path = self.pruned_savename + "_postprune.pth.tar"
        else:
            raise Exception("Previous task pruned model final/postprune non-existing: {}".format(self.pruned_savename))

    def grid_prestep(self, args, manager):
        """ Make modeldump. """
        hyperparams = {}
        manager.dataset_name = self.get_dataset_name(args.task_name)
        manager.disable_pruning_mask = False

        # Make init dump of Wrapper Model object
        if args.task_counter == 1:
            init_wrapper_model_name = os.path.join(
                manager.ft_parent_exp_dir, manager.base_model.name + '_INIT_WRAPPED.pth')

            if not os.path.exists(init_wrapper_model_name):
                if isinstance(manager.base_model, models.AlexNet):
                    arch = 'alexnet'
                else:
                    arch = 'VGGslim_nopretrain'

                print("PACKNET INIT DUMP PHASE")
                overwrite_args = {
                    'arch': arch,
                    'init_dump': True,
                    'cuda': True,
                    'loadname': manager.previous_task_model_path,  # Raw model path
                    'save_prefix': init_wrapper_model_name,  # exp path filename
                    'last_layer_idx': manager.base_model.last_layer_idx,  # classifier last layer idx
                    'current_dataset_idx': args.task_counter,
                }
                hyperparams['pre_phase'] = overwrite_args
                trainPacknet.main(overwrite_args)
            else:
                "PACKNET MODEL DUMP ALREADY EXISTS"

            # Update to wrapper Model path
            manager.previous_task_model_path = init_wrapper_model_name
            manager.disable_pruning_mask = True  # Because packnet assume pretrained

    def grid_train(self, args, manager, lr):
        print("PACKNET TRAIN PHASE")
        ft_savename = os.path.join(manager.gridsearch_exp_dir, 'best_model')
        overwrite_args = {
            'weight_decay': args.weight_decay,
            'disable_pruning_mask': manager.disable_pruning_mask,
            'train_path': manager.current_task_dataset_path,
            'test_path': manager.current_task_dataset_path,
            'mode': 'finetune',
            'dataset': manager.dataset_name,
            'num_outputs': len(manager.dataset.classes_per_task[args.task_name]),
            'loadname': manager.previous_task_model_path,  # Model path
            'lr': lr,
            'finetune_epochs': args.num_epochs,
            'cuda': True,
            'save_prefix': ft_savename,  # exp path filename    # TODO, now only dir, not best_model.pth
            'batch_size': 200,  # batch_size try
            'train_bn': args.train_bn,
            'saving_freq': args.saving_freq,
            'current_dataset_idx': args.task_counter,
        }
        acc = trainPacknet.main(overwrite_args)
        return None, acc

    def grid_poststep(self, args, manager):
        manager.best_finetuned_model_path = os.path.join(manager.best_exp_grid_node_dirname, 'best_model.pth.tar')

    @staticmethod
    def train_args_overwrite(args):
        args.train_bn = True if ModelRegularization.batchnorm in args.model_name else False  # train BN params
        print("TRAINING BN PARAMS = ", str(args.train_bn))

    @staticmethod
    def inference_eval(args, manager):
        """ Inference for testing."""
        task_name = manager.dataset.get_taskname(args.eval_dset_idx + 1)
        overwrite_args = {
            'train_path': args.dset_path,
            'test_path': args.dset_path,
            'mode': 'eval',
            'dataset': PackNet.get_dataset_name(task_name),
            'loadname': args.eval_model_path,  # Load model
            'cuda': True,
            'batch_size': args.batch_size,
            'current_dataset_idx': args.eval_dset_idx + 1
        }
        accuracy = trainPacknet.main(overwrite_args)
        return accuracy


class Pathnet(Method):
    name = "pathnet"
    eval_name = name
    category = Category.MASK_BASED
    extra_hyperparams_count = 3  # M,N, gen
    hyperparams = OrderedDict({'N': 3})  # Typically 3,4 defined in paper
    static_hyperparams = OrderedDict({'M': 20, 'generations': 35})  # Allows 2 epochs training per time
    start_scratch = True

    # Do grid generations: [7,35,70]
    def grid_train(self, args, manager, lr):
        args.lr = lr
        parameter = list(self.hyperparams.values()) + list(self.static_hyperparams.values())
        return _modular_accespoint(args, manager, parameter, 'pathnet',
                                   save_path=manager.gridsearch_exp_dir, finetune=True)

    def train(self, args, manager, hyperparams):
        assert args.decaying_factor == 1
        parameter = list(hyperparams.values()) + list(self.static_hyperparams.values())
        return _modular_accespoint(args, manager, parameter, 'pathnet')

    def get_output(self, images, args):
        head = args.heads[args.current_head_idx]
        args.model.classifier = torch.nn.ModuleList()
        args.model.classifier.append(head)  # Change head
        args.model.eval()

        logits = args.model.forward(images, args.task_idx)
        return logits

    @staticmethod
    def decay_operator(a, decaying_factor):
        """ For N, we want it to increment instead of decay, with b >=1"""
        assert decaying_factor == 1
        return int(a + decaying_factor)

    @staticmethod
    def inference_eval(args, manager):
        return Finetune.inference_eval(args, manager)


class HAT(Method):
    name = "HAT"
    eval_name = name
    category = Category.MASK_BASED
    extra_hyperparams_count = 2  # s,c
    hyperparams = OrderedDict({'smax': 800, 'c': 2.5})  # Paper ranges: smax=[25,800], c=[0.1,2.5] but optimal 0.75
    start_scratch = True

    def grid_train(self, args, manager, lr):
        args.lr = lr
        return _modular_accespoint(args, manager, list(self.hyperparams.values()), 'hat',
                                   save_path=manager.gridsearch_exp_dir, finetune=True)

    def train(self, args, manager, hyperparams):
        return _modular_accespoint(args, manager, list(hyperparams.values()), 'hat')

    def get_output(self, images, args):
        head = args.heads[args.current_head_idx]
        args.model.classifier = torch.nn.ModuleList()
        args.model.classifier.append(head)  # Change head
        args.model.eval()

        logits, masks = args.model.forward(args.task_idx, images, s=args.model.smax)
        return logits

    @staticmethod
    def inference_eval(args, manager):
        return Finetune.inference_eval(args, manager)


def _modular_accespoint(args, manager, parameter, method_arg, save_path=None, prev_model_path=None, finetune=False):
    nc_per_task = dataset_utils.get_nc_per_task(manager.dataset)
    total_outputs = sum(nc_per_task)
    print("nc_per_task = {}, TOTAL OUTPUTS = {}".format(nc_per_task, total_outputs))

    save_path = manager.heuristic_exp_dir if save_path is None else save_path
    prev_model_path = manager.previous_task_model_path if prev_model_path is None else prev_model_path

    manager.overwrite_args = {
        'weight_decay': args.weight_decay,
        'task_name': args.task_name,
        'task_count': args.task_counter,
        'prev_model_path': prev_model_path,
        'model_name': args.model_name,
        'output': save_path,
        'nepochs': args.num_epochs,
        'parameter': parameter,  # CL hyperparam
        'cuda': True,
        'dataset_path': manager.current_task_dataset_path,
        'dataset': manager.dataset,
        'n_tasks': manager.dataset.task_count,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'is_scratch_model': args.task_counter == 1,
        'approach': method_arg,
        'nc_per_task': nc_per_task,
        'finetune_mode': finetune,
        'save_freq': args.saving_freq,
    }
    model, task_lr_acc = trainHAT.main(manager.overwrite_args)
    return model, task_lr_acc


class EWC(Method):
    name = "EWC"
    eval_name = name
    category = Category.MODEL_BASED
    extra_hyperparams_count = 1
    hyperparams = OrderedDict({'lambda': 400})

    @staticmethod
    def grid_train(args, manager, lr):
        return Finetune.grid_train(args, manager, lr)

    def train(self, args, manager, hyperparams):
        return trainEWC.fine_tune_EWC_acuumelation(dataset_path=manager.current_task_dataset_path,
                                                   previous_task_model_path=manager.previous_task_model_path,
                                                   exp_dir=manager.heuristic_exp_dir,
                                                   data_dir=args.data_dir,
                                                   reg_sets=manager.reg_sets,
                                                   reg_lambda=hyperparams['lambda'],
                                                   batch_size=args.batch_size,
                                                   num_epochs=args.num_epochs,
                                                   lr=args.lr,
                                                   weight_decay=args.weight_decay,
                                                   saving_freq=args.saving_freq)

    def get_output(self, images, args):
        return get_output_def(args.model, args.heads, images, args.current_head_idx, args.final_layer_idx)

    @staticmethod
    def inference_eval(args, manager):
        return Finetune.inference_eval(args, manager)


class SI(Method):
    name = "SI"
    eval_name = name
    category = Category.MODEL_BASED
    extra_hyperparams_count = 1
    hyperparams = OrderedDict({'lambda': 400})

    # start_scratch = True  # Reference model other methods, should run in basemodel_dump mode

    @staticmethod
    def grid_train(args, manager, lr):
        return Finetune.grid_train(args, manager, lr)

    def train(self, args, manager, hyperparams):
        return trainSI.fine_tune_elastic(dataset_path=manager.current_task_dataset_path,
                                         num_epochs=args.num_epochs,
                                         exp_dir=manager.heuristic_exp_dir,
                                         model_path=manager.previous_task_model_path,
                                         reg_lambda=hyperparams['lambda'],
                                         batch_size=args.batch_size, lr=args.lr, init_freeze=0,
                                         weight_decay=args.weight_decay,
                                         saving_freq=args.saving_freq)

    def get_output(self, images, args):
        return get_output_def(args.model, args.heads, images, args.current_head_idx, args.final_layer_idx)

    @staticmethod
    def inference_eval(args, manager):
        return Finetune.inference_eval(args, manager)


class MAS(Method):
    name = "MAS"
    eval_name = name
    category = Category.MODEL_BASED
    extra_hyperparams_count = 1
    hyperparams = OrderedDict({'lambda': 3})

    @staticmethod
    def grid_train(args, manager, lr):
        return Finetune.grid_train(args, manager, lr)

    def train(self, args, manager, hyperparams):
        return trainMAS.fine_tune_objective_based_acuumelation(
            dataset_path=manager.current_task_dataset_path,
            previous_task_model_path=manager.previous_task_model_path,
            init_model_path=args.init_model_path,
            exp_dir=manager.heuristic_exp_dir,
            data_dir=args.data_dir, reg_sets=manager.reg_sets,
            reg_lambda=hyperparams['lambda'],
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            lr=args.lr, norm='L2', b1=False,
            saving_freq=args.saving_freq,
        )

    def get_output(self, images, args):
        return get_output_def(args.model, args.heads, images, args.current_head_idx, args.final_layer_idx)

    @staticmethod
    def inference_eval(args, manager):
        return Finetune.inference_eval(args, manager)


class IMM(Method):
    name = "IMM"  # Training name
    eval_name = name  # Altered in init
    modes = ['mean', 'mode']
    category = Category.MODEL_BASED
    extra_hyperparams_count = 1
    hyperparams = OrderedDict({'lambda': 0.01})
    grid_chkpt = True
    no_framework = True  # Outlier method (see paper)

    def __init__(self, mode='mode'):
        if mode not in self.modes:
            raise Exception("NO EXISTING IMM MODE: '{}'".format(mode))

        # Only difference is in testing, in training mode/mean IMM are the same
        self.mode = mode  # Set the IMM mode (mean and mode), this is only required after training.
        self.eval_name = self.name + "_" + self.mode

    def set_mode(self, mode):
        """
        Set the IMM mode (mean and mode), this is only required after training.
        :param mode:
        :return:
        """
        if mode not in self.modes:
            raise Exception("TRY TO SET NON EXISTING IMM MODE: ", mode)
        self.mode = mode
        self.eval_name = self.name + "_" + self.mode

    def grid_train(self, args, manager, lr):
        return trainIMM.fine_tune_l2transfer(dataset_path=manager.current_task_dataset_path,
                                             model_path=manager.previous_task_model_path,
                                             exp_dir=manager.gridsearch_exp_dir,
                                             reg_lambda=self.hyperparams['lambda'],
                                             batch_size=args.batch_size,
                                             num_epochs=args.num_epochs,
                                             lr=lr,
                                             weight_decay=args.weight_decay,
                                             saving_freq=args.saving_freq,
                                             )

    def get_output(self, images, args):
        return get_output_def(args.model, args.heads, images, args.current_head_idx, args.final_layer_idx)

    @staticmethod
    def grid_poststep(args, manager):
        manager.previous_task_model_path = os.path.join(manager.best_exp_grid_node_dirname, 'best_model.pth.tar')
        print("SINGLE_MODEL MODE: Set previous task model to ", manager.previous_task_model_path)
        Finetune.grid_poststep_symlink(args, manager)

    def eval_model_preprocessing(self, args):
        """ Merging step before evaluation. """
        print("IMM preprocessing: '{}' mode".format(self.mode))
        models_path = mergeIMM.preprocess_merge_IMM(self, args.models_path, args.datasets_path, args.batch_size,
                                                    overwrite=True)
        return models_path

    @staticmethod
    def inference_eval(args, manager):
        return Finetune.inference_eval(args, manager)


class EBLL(Method):
    name = "EBLL"
    eval_name = name
    category = Category.DATA_BASED
    extra_hyperparams_count = 2
    hyperparams = OrderedDict({'reg_lambda': 10, 'ebll_reg_alpha': 1, })
    static_hyperparams = OrderedDict({'autoencoder_lr': [0.01], 'autoencoder_epochs': 50,  # Paper defaults
                                      "encoder_alphas": [1e-1, 1e-2], "encoder_dims": [100, 300]})  # Grid

    @staticmethod
    def grid_train(args, manager, lr):
        return Finetune.grid_train(args, manager, lr)

    def prestep(self, args, manager):
        print("-" * 40)
        print("AUTOENCODER PHASE: for prev task ", args.task_counter - 1)
        manager.autoencoder_model_path = self._autoencoder_grid(args, manager)
        print("AUTOENCODER PHASE DONE")
        print("-" * 40)

    def _autoencoder_grid(self, args, manager):
        """Gridsearch for an autoencoder for the task corresponding with given task counter."""
        autoencoder_parent_exp_dir = os.path.join(manager.parent_exp_dir, 'task_' + str(args.task_counter - 1),
                                                  'ENCODER_TRAINING')

        # CHECKPOINT
        processed_hyperparams = {'header': ('dim', 'alpha', 'lr')}
        grid_checkpoint_file = os.path.join(autoencoder_parent_exp_dir, 'grid_checkpoint.pth')
        if os.path.exists(grid_checkpoint_file):
            checkpoint = torch.load(grid_checkpoint_file)
            processed_hyperparams = checkpoint
            print("STARTING FROM CHECKPOINT: ", checkpoint)

        # GRID
        best_autoencoder_path = None
        best_autoencoder_acc = 0
        for hyperparam_it in list(itertools.product(self.static_hyperparams['encoder_dims'],
                                                    self.static_hyperparams['encoder_alphas'],
                                                    self.static_hyperparams['autoencoder_lr']
                                                    )):
            encoder_dim, alpha, lr = hyperparam_it
            exp_out_name = "dim={}_alpha={}_lr={}".format(str(encoder_dim), str(alpha), lr)
            autoencoder_exp_dir = os.path.join(autoencoder_parent_exp_dir, exp_out_name)
            print("\n AUTOENCODER SETUP: {}".format(exp_out_name))
            print("Batch size={}, Epochs={}, LR={}, alpha={}, dim={}".format(
                args.batch_size,
                self.static_hyperparams['autoencoder_epochs'],
                lr,
                alpha, encoder_dim))

            if hyperparam_it in processed_hyperparams:
                acc = processed_hyperparams[hyperparam_it]
                print("ALREADY DONE: SKIPPING {}, acc = {}".format(exp_out_name, str(acc)))
            else:
                utilities.utils.create_dir(autoencoder_exp_dir, print_description="AUTOENCODER OUTPUT")

                # autoencoder trained on the previous task dataset
                start_time = time.time()
                _, acc = trainEBLL.fine_tune_Adam_Autoencoder(dataset_path=args.previous_task_dataset_path,
                                                              previous_task_model_path=manager.previous_task_model_path,
                                                              exp_dir=autoencoder_exp_dir,
                                                              batch_size=args.batch_size,
                                                              num_epochs=self.static_hyperparams['autoencoder_epochs'],
                                                              lr=lr,
                                                              alpha=alpha,
                                                              last_layer_name=args.classifier_heads_starting_idx,
                                                              auto_dim=encoder_dim)
                args.presteps_elapsed_time += time.time() - start_time

                processed_hyperparams[hyperparam_it] = acc
                torch.save(processed_hyperparams, grid_checkpoint_file)
                print("Saved to checkpoint")

            print("autoencoder acc={}".format(str(acc)))
            if acc > best_autoencoder_acc:
                utilities.utils.rm_dir(best_autoencoder_path, content_only=False)  # Cleanup
                print("{}(new) > {}(old), New best path: {}".format(str(acc), str(best_autoencoder_acc),
                                                                    autoencoder_exp_dir))
                best_autoencoder_acc = acc
                best_autoencoder_path = autoencoder_exp_dir
            else:
                utilities.utils.rm_dir(autoencoder_exp_dir, content_only=False)  # Cleanup

        if best_autoencoder_acc < 0.40:
            print(
                "[WARNING] Auto-encoder grid not sufficient: max attainable acc = {}".format(str(best_autoencoder_acc)))
        return os.path.join(best_autoencoder_path, 'best_model.pth.tar')

    def train(self, args, manager, hyperparams):
        return trainEBLL.fine_tune_SGD_EBLL(dataset_path=manager.current_task_dataset_path,
                                            previous_task_model_path=manager.previous_task_model_path,
                                            autoencoder_model_path=manager.autoencoder_model_path,
                                            init_model_path=args.init_model_path,
                                            exp_dir=manager.heuristic_exp_dir,
                                            batch_size=args.batch_size,
                                            num_epochs=args.num_epochs,
                                            lr=args.lr,
                                            init_freeze=0,
                                            reg_alpha=hyperparams['ebll_reg_alpha'],
                                            weight_decay=args.weight_decay,
                                            saving_freq=args.saving_freq,
                                            reg_lambda=hyperparams['reg_lambda'])

    def get_output(self, images, args):
        try:
            outputs, _ = args.model(Variable(images))  # disgard autoencoder output codes
        except:
            outputs = args.model(Variable(images))  # SI init model
        if isinstance(outputs, list):
            outputs = outputs[args.current_head_idx]
        return outputs.data

    @staticmethod
    def inference_eval(args, manager):
        """ Inference for testing."""
        return LWF.inference_eval(args, manager)


class LWF(Method):
    name = "LWF"
    eval_name = name
    category = Category.DATA_BASED
    extra_hyperparams_count = 1
    hyperparams = OrderedDict({'lambda': 10})

    def __init__(self, warmup_step=False):
        self.warmup_step = warmup_step

    @staticmethod
    def grid_train(args, manager, lr):
        return Finetune.grid_train(args, manager, lr)

    def train(self, args, manager, hyperparams):
        # LWF PRE-STEP: WARM-UP (Train only classifier)
        if manager.method.warmup_step:
            print("LWF WARMUP STEP")
            warmup_exp_dir = os.path.join(manager.parent_exp_dir, 'task_' + str(args.task_counter), 'HEAD_TRAINING')
            trainLWF.fine_tune_freeze(dataset_path=manager.current_task_dataset_path,
                                      model_path=args.previous_task_model_path,
                                      exp_dir=warmup_exp_dir, batch_size=args.batch_size,
                                      num_epochs=int(args.num_epochs / 2),
                                      lr=args.lr)
            args.init_model_path = warmup_exp_dir
            print("LWF WARMUP STEP DONE")
        return trainLWF.fine_tune_SGD_LwF(dataset_path=manager.current_task_dataset_path,
                                          previous_task_model_path=manager.previous_task_model_path,
                                          init_model_path=args.init_model_path,
                                          exp_dir=manager.heuristic_exp_dir,
                                          batch_size=args.batch_size,
                                          num_epochs=args.num_epochs, lr=args.lr, init_freeze=0,
                                          weight_decay=args.weight_decay,
                                          last_layer_name=args.classifier_heads_starting_idx,
                                          saving_freq=args.saving_freq,
                                          reg_lambda=hyperparams['lambda'])

    def get_output(self, images, args):
        outputs = args.model(Variable(images))
        if isinstance(outputs, list):
            outputs = outputs[args.current_head_idx]
        return outputs.data

    @staticmethod
    def inference_eval(args, manager):
        """ Inference for testing."""
        if args.trained_model_idx > 0:
            return FinetuneRehearsalFullMem.inference_eval(args, manager)
        else:  # First is SI model
            return Finetune.inference_eval(args, manager)


##################################################
################ BASELINES #######################
class Finetune(Method):
    name = "finetuning"
    eval_name = name
    category = Category.BASELINE
    extra_hyperparams_count = 0
    hyperparams = {}
    grid_chkpt = True
    start_scratch = True

    def get_output(self, images, args):
        return get_output_def(args.model, args.heads, images, args.current_head_idx, args.final_layer_idx)

    @staticmethod
    def grid_train(args, manager, lr):
        dataset_path = manager.current_task_dataset_path
        print('lr is ' + str(lr))
        print("DATASETS: ", dataset_path)

        if not isinstance(dataset_path, list):  # If single path string
            dataset_path = [dataset_path]

        dset_dataloader, cumsum_dset_sizes, dset_classes = Finetune.compose_dataset(dataset_path, args.batch_size)
        return trainFT.fine_tune_SGD(dset_dataloader, cumsum_dset_sizes, dset_classes,
                                     model_path=manager.previous_task_model_path,
                                     exp_dir=manager.gridsearch_exp_dir,
                                     num_epochs=args.num_epochs, lr=lr,
                                     weight_decay=args.weight_decay,
                                     enable_resume=True,  # Only resume when models saved
                                     save_models_mode=True,
                                     replace_last_classifier_layer=True,
                                     freq=args.saving_freq,
                                     )

    @staticmethod
    def grid_poststep(args, manager):
        manager.previous_task_model_path = os.path.join(manager.best_exp_grid_node_dirname, 'best_model.pth.tar')
        print("SINGLE_MODEL MODE: Set previous task model to ", manager.previous_task_model_path)
        Finetune.grid_poststep_symlink(args, manager)

    @staticmethod
    def grid_poststep_symlink(args, manager):
        """ Create symbolic link to best model in gridsearch. """
        exp_dir = os.path.join(manager.parent_exp_dir, 'task_' + str(args.task_counter), 'TASK_TRAINING')
        if os.path.exists(exp_dir):
            os.unlink(exp_dir)
        print("Symlink best LR: ", utilities.utils.get_relative_path(manager.best_exp_grid_node_dirname, segments=2))
        os.symlink(utilities.utils.get_relative_path(manager.best_exp_grid_node_dirname, segments=2), exp_dir)

    @staticmethod
    def compose_dataset(dataset_path, batch_size):
        """Append all datasets in list, return single dataloader"""
        dset_imgfolders = {x: [] for x in ['train', 'val']}
        dset_classes = {x: [] for x in ['train', 'val']}
        dset_sizes = {x: [] for x in ['train', 'val']}
        for dset_count in range(0, len(dataset_path)):
            dset_wrapper = torch.load(dataset_path[dset_count])

            for mode in ['train', 'val']:
                dset_imgfolders[mode].append(dset_wrapper[mode])
                dset_classes[mode].append(dset_wrapper[mode].classes)
                dset_sizes[mode].append(len(dset_wrapper[mode]))

        cumsum_dset_sizes = {mode: sum(dset_sizes[mode]) for mode in dset_sizes}
        classes_len = {mode: [len(ds) for ds in dset_classes[mode]] for mode in dset_classes}
        dset_dataloader = {x: torch.utils.data.DataLoader(
            ConcatDatasetDynamicLabels(dset_imgfolders[x], classes_len[x]),
            batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
            for x in ['train', 'val']}  # Concat into 1 dataset
        print("dset_classes: {}, dset_sizes: {}".format(dset_classes, cumsum_dset_sizes))
        return dset_dataloader, cumsum_dset_sizes, dset_classes

    @staticmethod
    def inference_eval(args, manager):
        """ Inference for testing."""
        model = torch.load(args.eval_model_path)

        if isinstance(model, dict):
            model = model['model']

        # Check layer idx correct for current model
        head_layer_idx = str(len(model.classifier._modules) - 1)  # Last head layer of prev model
        current_head = model.classifier._modules[head_layer_idx]
        assert isinstance(current_head, torch.nn.Linear), "NO VALID HEAD IDX"

        # Get head of a prev model corresponding to task
        target_heads = utilities.utils.get_prev_heads(args.head_paths, head_layer_idx)
        target_head_idx = 0  # first in list
        print("EVAL on prev heads: ", args.head_paths)
        assert len(target_heads) == 1

        accuracy = test_network.test_model(manager.method, model, args.dset_path, target_head_idx, subset=args.test_set,
                                           target_head=target_heads, batch_size=args.batch_size,
                                           task_idx=args.eval_dset_idx)
        return accuracy


class FinetuneRehearsalPartialMem(Method):
    name = "finetuning_rehearsal_partial_mem"
    eval_name = name
    category = Category.BASELINE
    extra_hyperparams_count = 0
    arg_string = 'baseline_rehearsal_partial_mem'
    hyperparams = {}
    grid_chkpt = True
    start_scratch = True
    no_framework = True

    def get_output(self, images, args):
        offset1, offset2 = args.model.compute_offsets(args.current_head_idx,
                                                      args.model.cum_nc_per_task)  # No shared head
        outputs = args.model(Variable(images), args.current_head_idx)[:, offset1: offset2]
        return outputs

    @staticmethod
    def grid_train(args, manager, lr):
        return FinetuneRehearsalFullMem.grid_train(args, manager, lr)

    @staticmethod
    def grid_poststep(args, manager):
        Finetune.grid_poststep(args, manager)

    @staticmethod
    def inference_eval(args, manager):
        return FinetuneRehearsalFullMem.inference_eval(args, manager)


class FinetuneRehearsalFullMem(Method):
    name = "finetuning_rehearsal_full_mem"
    eval_name = name
    category = Category.BASELINE
    extra_hyperparams_count = 0
    arg_string = 'baseline_rehearsal_full_mem'
    hyperparams = {}
    grid_chkpt = True
    start_scratch = True
    no_framework = True

    def get_output(self, images, args):
        offset1, offset2 = args.model.compute_offsets(args.current_head_idx,
                                                      args.model.cum_nc_per_task)  # No shared head
        outputs = args.model(Variable(images), args.current_head_idx)[:, offset1: offset2]
        return outputs

    @staticmethod
    def grid_train(args, manager, lr):
        print("RAW REHEARSAL BASELINE")

        # Need 1 head, because also loss on exemplars of prev tasks is performed
        nc_per_task = manager.datasets.get_nc_per_task(manager.dataset)
        total_outputs = sum(nc_per_task)
        print("nc_per_task = {}, TOTAL OUTPUTS = {}".format(nc_per_task, total_outputs))

        print("RUNNING {} mode".format(manager.method.arg_string))
        overwrite_args = {
            'weight_decay': args.weight_decay,
            'task_name': args.task_name,
            'task_count': args.task_counter,
            'prev_model_path': manager.previous_task_model_path,
            'save_path': manager.gridsearch_exp_dir,
            'n_outputs': total_outputs,
            'method': manager.method.arg_string,
            'n_memories': args.mem_per_task,
            'n_epochs': args.num_epochs,
            'cuda': True,
            'dataset_path': manager.current_task_dataset_path,
            'n_tasks': manager.dataset.task_count,
            'batch_size': args.batch_size,
            'lr': lr,
            'finetune': True,  # Crucial
            'is_scratch_model': args.task_counter == 1
        }
        return trainRehearsal.main(overwrite_args, nc_per_task)

    @staticmethod
    def grid_poststep(args, manager):
        Finetune.grid_poststep(args, manager)

    @staticmethod
    def inference_eval(args, manager):
        """ Inference for testing."""
        model = torch.load(args.eval_model_path)
        target_head_idx = args.eval_dset_idx
        target_heads = None

        print("EVAL on prev head idx: ", target_head_idx)
        accuracy = test_network.test_model(manager.method, model, args.dset_path, target_head_idx, subset=args.test_set,
                                           target_head=target_heads, batch_size=args.batch_size,
                                           task_idx=args.eval_dset_idx)
        return accuracy


class Joint(Method):
    name = "joint"
    eval_name = name
    category = Category.BASELINE
    extra_hyperparams_count = 0
    hyperparams = {}
    grid_chkpt = True
    start_scratch = True
    no_framework = True

    def get_output(self, images, args):
        raise NotImplementedError("JOINT has custom testing method for shared head.")

    @staticmethod
    def grid_train(args, manager, lr):
        return Finetune.grid_train(args, manager, lr)

    @staticmethod
    def grid_datafetch(args, dataset):
        current_task_dataset_path = dataset.get_task_dataset_path(task_name=None, rnd_transform=True)

        if current_task_dataset_path is not None:  # Available preprocessed JOINT dataset
            print("Running JOINT for all tasks as 1 batch, dataset = ", current_task_dataset_path)
            return current_task_dataset_path

        # Merge current task dataset with all prev task ones
        max_task = dataset.task_count  # Include all datasets in the list
        current_task_dataset_path = [dataset.get_task_dataset_path(
            task_name=dataset.get_taskname(ds_task_counter), rnd_transform=False)
            for ds_task_counter in range(1, max_task + 1)]
        print("Running JOINT for task ", args.task_name, " on datasets: ", current_task_dataset_path)
        return current_task_dataset_path

    @staticmethod
    def grid_poststep(args, manager):
        Finetune.grid_poststep(args, manager)

    @staticmethod
    def compose_dataset(dataset_path, batch_size):
        return Finetune.compose_dataset(dataset_path, batch_size)

    @staticmethod
    def train_args_overwrite(args):
        args.starting_task_count = 1
        args.max_task_count = args.starting_task_count

    @staticmethod
    def inference_eval(args, manager):
        return test_network.test_task_joint_model(args.model_path, args.dataset_path, args.dataset_index,
                                                  args.task_lengths, batch_size=args.batch_size, subset='test',
                                                  print_per_class_acc=False, debug=False, tasks_idxes=args.tasks_idxes)


# In[21]:


import os
import configparser
import sys
import random
import numpy
import warnings
import shutil
import datetime
import copy

import torch.nn as nn
import torch


def init():
    set_random()


########################################
# CONFIG PARSING
########################################
def get_root_src_path():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))


def get_parsed_config():
    """ Using this file as reference to get src root: src/<dir>/utils.py"""
    config = configparser.ConfigParser()
    src_path = get_root_src_path()
    config.read(os.path.join(src_path, 'config.init'))

    # Replace "./" with abs src root path
    for key, path in config['DEFAULT'].items():
        if '.' == read_from_config(config, key).split(os.path.sep)[0]:
            config['DEFAULT'][key] = os.path.join(src_path, read_from_config(config, key)[2:])
            create_dir(config['DEFAULT'][key], key)

    return config


def read_from_config(config, key_value):
    return os.path.normpath(config['DEFAULT'][key_value]).replace("'", "").replace('"', "")


def parse_str_to_floatlist(str_in):
    return list(map(float, str_in.replace(' ', '').split(',')))


########################################
# DETERMINISTIC
########################################
def set_random(seed=7):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_now():
    return str(datetime.datetime.now().date()) + "_" + ':'.join(str(datetime.datetime.now().time()).split(':')[:-1])


########################################
# PYTORCH UTILS
########################################
def replace_last_classifier_layer(model, out_dim):
    last_layer_index = str(len(model.classifier._modules) - 1)
    num_ftrs = model.classifier._modules[last_layer_index].in_features
    model.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, out_dim).cuda()
    return model


def get_first_FC_layer(seq_module):
    """
    :param seq_module: e.g. classifier or feature Sequential module of a model.
    """
    for module in seq_module.modules():
        if isinstance(module, nn.Linear):
            return module
    raise Exception("No LINEAR module in sequential found...")


def save_cuda_mem_req(out_dir, out_filename='cuda_mem_req.pth.tar'):
    """
    :param out_dir: /path/to/best_model.pth.tar
    """
    out_dir = os.path.dirname(out_dir)
    out_path = os.path.join(out_dir, out_filename)

    mem_req = {}
    mem_req['cuda_memory_allocated'] = torch.cuda.memory_allocated(device=None)
    mem_req['cuda_memory_cached'] = torch.cuda.memory_cached(device=None)

    torch.save(mem_req, out_path)
    print("SAVED CUDA MEM REQ {} to path: {}".format(mem_req, out_path))


def save_preprocessing_time(out_dir, time, out_filename='preprocess_time.pth.tar'):
    if os.path.isfile(out_dir):
        out_dir = os.path.dirname(out_dir)
    out_path = os.path.join(out_dir, out_filename)
    torch.save(time, out_path)
    print_timing(time, "PREPROCESSING")


def print_timing(timing, title=''):
    title = title.strip() + ' ' if title != '' else title
    print("{}TIMING >>> {} <<<".format(title, str(timing)))


def reset_stats():
    try:
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        print("RESETTED STATS")
    except:
        print("PYTORCH VERSION NOT ENABLING RESET STATS")


def print_stats():
    print("CUDA MAX MEM ALLOC >>> {} <<<".format(torch.cuda.max_memory_allocated()))
    print("CUDA MAX MEM CACHE >>> {} <<<".format(torch.cuda.max_memory_cached()))


########################################
# EXP PATHS
########################################
def get_exp_name(args, method):
    exp_name = ["dm={}".format(args.drop_margin),
                "df={}".format(args.decaying_factor),  # framework_hyperparams
                "e={}".format(args.num_epochs),
                "bs={}".format(args.batch_size)]
    if args.weight_decay != 0:
        exp_name.append("L2={}".format(args.weight_decay))
    for h_key, h_val in method.hyperparams.items():
        exp_name.append("{}={}".format(h_key, h_val))
    if hasattr(method, 'static_hyperparams'):
        for h_key, h_val in method.static_hyperparams.items():
            exp_name.append("{}={}".format(h_key, h_val))
    exp_name = '_'.join(exp_name)
    return exp_name


def get_starting_model_path(root_path, dataset_obj, model_name, exp_name, method_name, append_filename=True):
    """
    All methods have the same model for the same task, as no forgetting mechanism is applied.
    Nevertheless, SI estimates during training time, therefore we train SI first model and use it as starting model for
    all other methods sharing the same model and data set.

    Target, e.g.:
        /survey/exp_results/tiny_imgnet/SI/small_VGG9_cl_128_128/gridsearch/
        first_task_basemodel/<exp_name>/best_model.pth.tar
    :param exp_name: vanilla, L2=0.01,...
    """

    path = os.path.join(root_path, dataset_obj.train_exp_results_dir, method_name, model_name)
    path = os.path.join(path, 'gridsearch', 'first_task_basemodel', exp_name, 'task_1', 'TASK_TRAINING')

    if append_filename:
        path = os.path.join(path, 'best_model.pth.tar')
    return path


def get_test_results_path(root_path, dataset_obj, method_name, model_name,
                          gridsearch_name=None, exp_name=None, subset='test', create=False):
    """
    Util method to get the path of the testing results. (e.g. testing performances,...)
    :param dataset_obj:     CustomDataset object
    :param method_name:      str method name or Method object
    """
    path = os.path.join(root_path, 'results', dataset_obj.test_results_dir, method_name, model_name)

    if gridsearch_name is not None:
        path = os.path.join(path, gridsearch_name)
    if exp_name is not None:
        if subset != 'test':
            exp_name = '{}_{}'.format(exp_name, subset)
        path = os.path.join(path, exp_name)
    if create:
        create_dir(path)
    return path


def get_test_results_filename(method_name, task_number):
    return "test_method_performances" + method_name + str(int(task_number) - 1) + ".pth"


def get_train_results_path(tr_results_root_path, dataset_obj, method_name=None, model_name=None, gridsearch=True,
                           gridsearch_name=None, exp_name=None, filename=None, create=False):
    """
    Util method to get the path of the training results. (e.g. models, hyperparams during training,...)

    :param dataset_obj:     CustomDataset object
    :param method_name:      str method name or Method object
    """

    if create and filename is not None:
        print("WARNING: filename is not being created, but superdirs are if not existing.")

    path = os.path.join(tr_results_root_path, dataset_obj.train_exp_results_dir)
    if method_name is not None:
        path = os.path.join(path, method_name)
    if model_name is not None:
        path = os.path.join(path, model_name)
    if gridsearch:
        path = os.path.join(path, 'gridsearch')
    if gridsearch_name is not None:
        path = os.path.join(path, gridsearch_name)
    if exp_name is not None:
        path = os.path.join(path, exp_name)
    if create:
        create_dir(path)
    if filename is not None:
        path = os.path.join(path, filename)
    return path


def get_perf_output_filename(method_name, dataset_index, joint_full_batch=False):
    """
    Performances filename saved during training. (e.g. accuracy,...)
    :param dataset_index:  task_idx - 1
    """
    if joint_full_batch:
        return 'test_method_performancesJOINT_FULL_BATCH.pth'
    else:
        return 'test_method_performances' + method_name + str(dataset_index) + ".pth"


def get_hyperparams_output_filename():
    return 'hyperparams.pth.tar'


def get_prev_heads(prev_head_model_paths, head_layer_idx):
    """
    Last of the head_model_paths is the target head. Evaluating e.g. on Task 3, means heads of Task 1,2,3 available in
    head_model_paths. A model trained up to a certain task, has all heads available up to this task.
    Can be used to analyse per-head performance.

    :param prev_head_model_paths:   Previous Models to extract head from (the model's last task)
    :param head_layer_idx:      Model idx you want to know accuracy from
    """
    if not isinstance(prev_head_model_paths, list):
        prev_head_model_paths = [prev_head_model_paths]

    if len(prev_head_model_paths) == 0:
        return []

    heads = []
    # Add prev model heads
    for head_model_path in prev_head_model_paths:
        previous_model_ft = torch.load(head_model_path)
        if isinstance(previous_model_ft, dict):
            previous_model_ft = previous_model_ft['model']

        head = previous_model_ft.classifier._modules[head_layer_idx]
        assert isinstance(head, torch.nn.Linear), type(head)
        heads.append(copy.deepcopy(head.cuda()))
        del previous_model_ft

    return heads


########################################
# PATH UTILS
########################################
def get_immediate_subdirectories(parent_dir_path, path_mode=False, sort=False):
    """

    :param parent_dir_path: dir to take subdirs from
    :param path_mode: if true, returns subdir paths instead of names
    :return: List with names (not paths) of immediate subdirs
    """
    if not path_mode:
        dirs = [name for name in os.listdir(parent_dir_path)
                if os.path.isdir(os.path.join(parent_dir_path, name))]
    else:
        dirs = [os.path.join(parent_dir_path, name) for name in os.listdir(parent_dir_path)
                if os.path.isdir(os.path.join(parent_dir_path, name))]
    if sort:
        dirs.sort()
    return dirs


def get_relative_path(absolute_path, segments=1):
    """ Returns relative path with last #segments of the absolute path. """
    return os.path.sep.join(list(filter(None, absolute_path.split(os.path.sep)))[-segments:])


def attempt_move(src_path, dest_path):
    try:
        shutil.move(src_path, dest_path)
    except Exception:
        if not os.path.exists(dest_path):  # Don't print if already transfered
            print("Dest path not existing: ", dest_path)
        if not os.path.exists(src_path):
            print("SRC path not existing: ", src_path)


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def append_to_file(filepath, msg):
    """ Append a new line to a file, and create if file doesn't exist yet. """
    write_mode = 'w' if not os.path.exists(filepath) else 'a'
    with open(filepath, write_mode) as f:
        f.write(msg + "\n")


def rm_dir(path_to_dir, delete_subdirs=True, content_only=True):
    if path_to_dir is not None and os.path.exists(path_to_dir):
        for the_file in os.listdir(path_to_dir):
            file_path = os.path.join(path_to_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path) and delete_subdirs:
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)
        print("REMOVED CONTENTS FROM DIR: ", path_to_dir)

        if not content_only:
            try:
                shutil.rmtree(path_to_dir)
            except Exception as e:
                print(e)
            print("REMOVED DIR AND ALL ITS CONTENTS: ", path_to_dir)


def create_dir(dirpath, print_description=""):
    try:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, mode=0o750)
    except Exception as e:
        print(e)
        print("ERROR IN CREATING ", print_description, " PATH:", dirpath)


def create_symlink(src, ln):
    if not os.path.exists(ln):
        create_dir(os.path.dirname(ln))
        os.symlink(src, ln)


########################################
# MISCELLANEOUS UTILS
########################################
def float_to_scientific_str(value, sig_count=1):
    """
    {0:.6g}.format(value) also works

    :param value:
    :param sig_count:
    :return:
    """
    from decimal import Decimal
    format_str = '%.' + str(sig_count) + 'E'
    return format_str % Decimal(value)


def debug_add_sys_args(string_cmd, set_debug_option=True):
    """
    Add debug arguments as params, this is for IDE debugging usage.
    :param string_cmd:
    :return:
    """
    warnings.warn("=" * 20 + "SEVERE WARNING: DEBUG CMD ARG USED, TURN OF FOR REAL RUNS" + "=" * 20)
    args = string_cmd.split(' ')
    if set_debug_option:
        args.insert(0, "--debug")
    for arg in args:
        sys.argv.append(str(arg))


# In[22]:


import numpy as np
import pylab
from matplotlib import rcParams

rcParams['font.family'] = 'DejaVu Serif'
rcParams['font.sans-serif'] = ['DejaVuSerif']
import matplotlib.pyplot as plt


def plot_line_horizontal_sequence(plots_data, colors, linestyles, labels, markers, markersizes, save_img_path=None,
                                  ylim=None,
                                  legend="out",
                                  ylabel="Accuracy % after learning all tasks",
                                  y_label_fontsize=19,
                                  xlabel="Training Sequence Per Task",
                                  x_label_fontsize=19,
                                  start_y_zero=False,
                                  labelmode='minor',
                                  single_dot_idxes=None,
                                  taskcount=10):
    """
    Checkout for markers: https://matplotlib.org/api/markers_api.html

    :param curves_data: Ordered array of arrays [ [<data_seq_curve1>], [<data_seq_curve2>], ... ]
    :param labels: Ordered array of labels
    :param legend: best or "upper/lower/center right/left/center"
    """
    legend_col = 4  # 5
    height_inch = 8
    width_inch = 20
    x_tick_fontsize = 16
    y_tick_fontsize = 18
    legendsize = 16
    bg_alpha = 1
    bg_color = 'whitesmoke'
    plt.ion()

    task_idxs = [0, 4, 9, 14, 19] if taskcount > 10 else [i for i in range(0, taskcount)]
    print("task_idxs={}".format(task_idxs))

    panel_length = len(plots_data[0][0])  # Length of 1 plot panel
    curves_per_plot = len(plots_data[0])  # Curves in 1 plot
    plot_count = len(task_idxs)  # Amount of stacked plots next to eachother

    print("panel_length={}".format(panel_length))
    print("curves_per_plot={}".format(curves_per_plot))
    print("plot_count={}".format(plot_count))

    if single_dot_idxes is None:
        single_dot_idxes = []

    fig, ax = plt.subplots(figsize=(width_inch, height_inch))
    print('Adding plot data')
    for i, plot_idx in enumerate(task_idxs):  # horizontal subplots
        curves_data = plots_data[plot_idx]
        print("Plot idx = {}".format(plot_idx))
        for curve_idx, curve_data in enumerate(curves_data):  # curves in 1 subplot
            # Shift to graph + offset of not testing on prev task
            plot_Xshift = i * panel_length + 1 * plot_idx

            X = np.arange(len(curve_data)) + plot_Xshift
            label = labels[curve_idx] if i == 0 else None
            marker = markers[curve_idx]
            markersize = markersizes[curve_idx]
            print("Plot X = {}".format(X))
            print("Xshift={}".format(plot_Xshift))

            if curve_idx in single_dot_idxes:  # Plot e.g. JOINT as single point at the end
                X = X[-1]
                curve_data = curve_data[-1]
                markersize = 12

            ax.plot(X, curve_data, color=colors[curve_idx], label=label, linewidth=1.5, marker=marker,
                    linestyle=linestyles[curve_idx], markersize=markersize)
    # No Top/right border
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # Put X-axis ticks
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    subplot_offset = 0.1  # TODO

    if ylim is not None:
        ax.set_ylim(top=ylim)

    # Background
    print('Adding plot span')
    for idx, task_idx in enumerate(task_idxs):
        ax.axvspan(idx * panel_length + subplot_offset, (1 + idx) * panel_length - subplot_offset,
                   facecolor=bg_color, alpha=bg_alpha)

    ##############################
    # X-axis gridline positions
    # Major
    # XgridlinesPosMajor = np.linspace(0, (panel_length) * plot_count, num=plot_count)

    # Minor
    # offset_idx = 0
    upper_ticksoffset = -4
    XgridlinesPosMinor, XgridlinesPosMajor = [], []
    for idx, task_idx in enumerate(task_idxs):
        XgridlinesPosMinor.append(idx * panel_length + task_idx)
        XgridlinesPosMajor.append(int(idx * panel_length + panel_length / 2 + upper_ticksoffset))
        # offset_idx += 1
    XgridlinesPosMajor = np.asarray(XgridlinesPosMajor)
    print("XgridlinesPosMinor={}".format(XgridlinesPosMinor))
    print("XgridlinesPosMajor={}".format(XgridlinesPosMajor))

    ###############################
    # Labels
    print("Setting labels")
    Xtick_minorlabels = ['T{}'.format(idx + 1) for idx in task_idxs]
    Xtick_majorlabels = np.repeat('T1', len(XgridlinesPosMajor))

    if labelmode == 'major':
        # Labels Major labeling only
        # Xticks = np.linspace(0, 10, ticks_per_plot)
        Xticks = XgridlinesPosMajor

        # Set
        ax.set_xticks(Xticks, minor=False)
        ax.set_xticklabels(Xtick_majorlabels, minor=False)
    elif labelmode == 'both':
        # Labels both major minor gridlines
        offset_idx = 0
        Xticks_gridlines = []
        Xtick_labels = []
        for idx, task_idx in enumerate(task_idxs):
            Xticks_gridlines.append(idx * panel_length)
            Xtick_labels.append('T{}'.format((task_idx + 1) % panel_length))
            if offset_idx > 0:
                Xticks_gridlines.append(idx * panel_length + offset_idx)
                Xtick_labels.append('T{}'.format((task_idx + 1) % panel_length))
            offset_idx += 1
        Xticks = Xticks_gridlines
        # Set
        plt.xticks(Xticks, Xtick_labels, fontsize=10, color='black')

    elif labelmode == 'minor':
        # Labels only on minor gridlines
        Xticks = XgridlinesPosMinor

        # Set
        ax.set_xticks(Xticks, minor=True)
        ax.set_xticklabels(Xtick_minorlabels, minor=True)
        ax.set_xticklabels([], minor=False)

    print("Setting ticks")
    # Actual Ticks with Labels
    ax.tick_params(axis='y', which='major', labelsize=y_tick_fontsize)
    ax.tick_params(axis='x', which='minor', labelsize=x_tick_fontsize)
    ax.tick_params(axis='x', which='major', labelsize=x_tick_fontsize, length=0)

    # Axis titles
    ax.set_xlabel(xlabel, fontsize=x_label_fontsize, labelpad=5)
    ax.set_ylabel(ylabel, fontsize=y_label_fontsize, labelpad=5)
    ax.set_xlim(-1, len(task_idxs) * taskcount + 1)

    # Grid lines
    ax.set_xticks(XgridlinesPosMinor, minor=True)
    ax.set_xticks(XgridlinesPosMajor, minor=False)

    ax.xaxis.grid(True, linestyle='--', alpha=0.4, which='minor')
    ax.xaxis.grid(True, linestyle='-', alpha=0.8, which='major', color='white')

    # y-axis
    if start_y_zero:
        ax.set_ylim(bottom=0)

    # Legend
    print("Setting legend")
    if legend == "top":
        leg = ax.legend(bbox_to_anchor=(0., 1.20, 1., 0.1), loc='upper center', ncol=legend_col,
                        prop={'size': legendsize},
                        mode="expand", fancybox=True, fontsize=24)  # best or "upper/lower/center right/left/center"
    else:
        leg = ax.legend(bbox_to_anchor=(0., -0.36, 1., -.136), loc='upper center', ncol=legend_col,
                        prop={'size': legendsize},
                        mode="expand", fancybox=True, fontsize=24)  # best or "upper/lower/center right/left/center"

    # Update legend linewidths
    for idx, legobj in enumerate(leg.legendHandles):
        if idx not in single_dot_idxes:
            legobj.set_linewidth(2.0)
        else:
            legobj.set_linewidth(0)
        legobj._legmarker.set_markersize(8.0)

    # TOP axis
    print("Setting axes")
    ax_top = ax.twiny()
    print("chkpt{}".format(1))
    ax_top.set_xlim(-1, taskcount * len(task_idxs) + 1)  # MUST BE SAME AS ax
    top_ticks = XgridlinesPosMajor + 5
    print("chkpt{}".format(2))

    ax_top.set_xticks(top_ticks, minor=False)
    ax_top.set_xticklabels(Xtick_minorlabels, minor=False)
    print("chkpt{}".format(3))

    ax_top.tick_params(axis=u'both', which=u'both', length=0)
    ax_top.tick_params(axis='x', which='major', labelsize=x_tick_fontsize)
    print("chkpt{}".format(4))

    ax_top.set_xlabel('Evaluation on Task', fontsize=x_label_fontsize, labelpad=10)
    plt.setp(ax_top.get_xaxis().get_offset_text(), visible=False)
    print("chkpt{}".format(5))

    # Format Plot
    # plt.tight_layout()

    print("Saving to {}".format(save_img_path))
    if save_img_path is not None:
        plt.axis('on')
        pylab.savefig(save_img_path, bbox_inches='tight')
        pylab.clf()
    else:
        pylab.show()  # Show always also when saving


def imshow_tensor(inp, title=None, denormalize=True,
                  mean=np.array([0.485, 0.456, 0.406]),
                  std=np.array([0.229, 0.224, 0.225])):
    """
    Imshow for Tensor.

    :param inp: input Tensor of img
    :param title:
    :param denormalize: denormalize input or not
    :param mean: imgnet mean by default
    :param std: imgnet std by default
    :return:
    """

    inp = inp.cpu().numpy().transpose((1, 2, 0))
    if denormalize:
        inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

    if title is not None:
        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated


# In[23]:


import os

# Define variables
MY_PYTHON = "python3"  # Use python3 if python is not recognized
EXEC = "./framework/main.py"

# Get the root path and set the PYTHONPATH
this_script_path = os.path.dirname(os.path.abspath("__file__"))
root_path = os.path.join(this_script_path, "../src/")
os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{root_path}"
print(f"Project src root '{root_path}' added to Python path")


# In[24]:


test_results_root_path='./results/test'
tr_results_root_path='./results/train'
models_root_path='./data/models'
ds_root_path='./data/datasets'


# In[25]:


# RUN THIS AS INIT
from data.dataset import *

from models.net import *
from utilities.main_postprocessing import *

# CONFIG
config = utils.get_parsed_config()
test_results_root_path = utils.read_from_config(config, 'test_results_root_path')
tr_results_root_path = utils.read_from_config(config, 'tr_results_root_path')
models_root_path = utils.read_from_config(config, 'models_root_path')

dataset = TinyImgnetDataset()
model = SmallVGG9(models_root_path, dataset.input_size)

# Turn on/off
plot_SI = True

# PARAMS
img_extention = 'png'  # 'eps' for latex
save_img = True

plot_seq_acc = True
plot_seq_forgetting = False
hyperparams_selection = []

label_segment_idxs = [0]
exp_name_contains = None

# INIT
method_names = []
method_data_entries = []

#############################################
# MAS METHOD
if plot_SI:
    method = SI()
    method_names.append(method.name)
    label = None

    tuning_selection = []
    gridsearch_name = "reproduce"
    method_data_entries.extend(
        collect_gridsearch_exp_entries(test_results_root_path, tr_results_root_path, dataset, method, gridsearch_name,
                                       model, tuning_selection, label_segment_idxs=label_segment_idxs,
                                       exp_name_contains=exp_name_contains))

#############################################
# ANALYZE
#############################################
print(method_data_entries)
out_name = None
if save_img:
    out_name = '_'.join(['DEMO', dataset.name, "(" + '_'.join(method_names) + ")", model.name])

analyze_experiments(method_data_entries, hyperparams_selection=hyperparams_selection, plot_seq_acc=plot_seq_acc,
                    plot_seq_forgetting=plot_seq_forgetting, save_img_parent_dir=out_name, img_extention=img_extention)


# In[26]:


get_ipython().system('ls /tf/CLsurvey/CLsurvey/src/data/datasets/tiny-imagenet/tiny-imagenet-200/val/')


# In[32]:


get_ipython().system('curl -O http://cs231n.stanford.edu/tiny-imagenet-200.zip')
get_ipython().system('unzip tiny-imagenet-200.zip')


# In[36]:


# List the contents of the tiny-imagenet-200 directory
get_ipython().system('ls /tf/CLsurvey/CLsurvey/src/data/datasets/tiny-imagenet/tiny-imagenet-200/')
# List the contents of the val directory
get_ipython().system('ls /tf/CLsurvey/CLsurvey/src/data/datasets/tiny-imagenet/tiny-imagenet-200/val/')


# In[37]:


get_ipython().system('ls /tf/CLsurvey/CLsurvey/src/data/datasets/tiny-imagenet/')


# In[39]:


get_ipython().system('ls /tf/CLsurvey/CLsurvey/src/data/datasets/tiny-imagenet/tiny-imagenet-200/')


# In[40]:


get_ipython().system('mkdir -p /tf/CLsurvey/CLsurvey/src/data/datasets/tiny-imagenet/')
get_ipython().system('curl -O http://cs231n.stanford.edu/tiny-imagenet-200.zip')
get_ipython().system('unzip tiny-imagenet-200.zip -d /tf/CLsurvey/CLsurvey/src/data/datasets/tiny-imagenet/')


# In[41]:


get_ipython().system('curl -O http://cs231n.stanford.edu/tiny-imagenet-200.zip')


# In[42]:


get_ipython().system('ls -lh tiny-imagenet-200.zip')


# In[43]:


get_ipython().system('ls /tf/CLsurvey/CLsurvey/src/data/datasets/tiny-imagenet/tiny-imagenet-200/')


# In[ ]:




