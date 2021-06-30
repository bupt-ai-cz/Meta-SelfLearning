import os
import sys
import re
import six
import math
import lmdb
import torch

from natsort import natsorted
import itertools
from PIL import Image
from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms


class Batch_Balanced_Dataset(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.

        opt.batch_ratio: 一个batch中包含的不同数据集的比例
        opt.total_data_usage_ratio: 对于每一个数据及，使用这个数据集的百分之多少，默认是1（100%）
        """
        self.opt = opt
        log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        # 为每个dataloader应用collate函数，直接输出一整个batch，
        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset) # 当前数据集包含的图片数量
            log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio)) # 使用的比例
            if opt.fix_dataset_num != -1: number_dataset = opt.fix_dataset_num
            dataset_split = [number_dataset, total_number_dataset - number_dataset] # List[int] e.g. [50, 50]
            indices = range(total_number_dataset)
            
            # accumulate函数： _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
            # Subset就是根据indices取一个数据集的子集，indice根据opt.total_data_usage_ratio来取值
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=False, drop_last=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self, meta_target_index=-1, no_pseudo=False): # 如果指定了meta_target_index，则忽略第meta_target_index个数据集
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            if i == meta_target_index: continue
            # 如果要求不采样伪标签数据集，且目前包含伪标签数据集则跳过
            if i == len(self.dataloader_iter_list) - 1 and no_pseudo and self.has_pseudo_label_dataset(): continue 
            try:
                image, text = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration: # 如果一个数据集图片数量不够了，则重新构建迭代器进行训练
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts
    
    def get_meta_test_batch(self, meta_target_index=-1): # 如果指定了meta_target_index，则忽略第meta_target_index个数据集
        
        if meta_target_index == self.opt.source_num:
            assert len(self.data_loader_list) == self.opt.source_num + 1, 'There is no target dataset'
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            if i == meta_target_index:
                try:
                    image, text = data_loader_iter.next()
                    balanced_batch_images.append(image)
                    balanced_batch_texts += text
                except StopIteration: # 如果一个数据集图片数量不够了，则重新构建迭代器进行训练
                    self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                    image, text = self.dataloader_iter_list[i].next()
                    balanced_batch_images.append(image)
                    balanced_batch_texts += text
                except ValueError:
                    pass
        # print(balanced_batch_images[0].shape)
        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts

    def add_target_domain_dataset(self, dataset, opt):
        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        avg_batch_size = opt.batch_size // opt.source_num
        batch_size = len(dataset) if len(dataset) <= avg_batch_size else avg_batch_size
        self_training_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size,
                        shuffle=True,  # 'True' to check training progress with validation function.
                        num_workers=int(opt.workers), pin_memory=False, collate_fn=_AlignCollate, drop_last=True)
        if self.has_pseudo_label_dataset():
            self.data_loader_list[opt.source_num] = self_training_loader
            self.dataloader_iter_list[opt.source_num] = (iter(self_training_loader))
        else:
            self.data_loader_list.append(self_training_loader)
            self.dataloader_iter_list.append(iter(self_training_loader))

    def add_pseudo_label_dataset(self, dataset, opt):
        avg_batch_size = opt.batch_size // opt.source_num
        batch_size = len(dataset) if len(dataset) <= avg_batch_size else avg_batch_size
        self_training_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size,
                        shuffle=True,  # 'True' to check training progress with validation function.
                        num_workers=int(opt.workers), pin_memory=False, collate_fn=self_training_collate)
        if self.has_pseudo_label_dataset():
            self.data_loader_list[opt.source_num] = self_training_loader
            self.dataloader_iter_list[opt.source_num] = (iter(self_training_loader))
        else:
            self.data_loader_list.append(self_training_loader)
            self.dataloader_iter_list.append(iter(self_training_loader))

    def add_residual_pseudo_label_dataset(self, dataset, opt):
        avg_batch_size = opt.batch_size // opt.source_num
        batch_size = len(dataset) if len(dataset) <= avg_batch_size else avg_batch_size
        self_training_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size,
                        shuffle=True,  # 'True' to check training progress with validation function.
                        num_workers=int(opt.workers), pin_memory=False, collate_fn=self_training_collate)
        if self.has_residual_pseudo_label_dataset():
            self.data_loader_list[opt.source_num + 1] = self_training_loader
            self.dataloader_iter_list[opt.source_num + 1] = (iter(self_training_loader))
        else:
            self.data_loader_list.append(self_training_loader)
            self.dataloader_iter_list.append(iter(self_training_loader))

    def has_pseudo_label_dataset(self):
        return True if len(self.data_loader_list) > self.opt.source_num else False

    def has_residual_pseudo_label_dataset(self):
        return True if len(self.data_loader_list) > self.opt.source_num + 1 else False
        
class Batch_Balanced_Sampler(object):
    def __init__(self, dataset_len, batch_size):
        dataset_len.insert(0,0)
        self.dataset_len = dataset_len
        self.start_index = list(itertools.accumulate(self.dataset_len))[:-1]
        self.batch_size = batch_size # 每个子数据集的batchsize
        self.counter = 0

    def __len__(self):
        return self.dataset_len

    def __iter__(self):
        data_index = []
        while True:
            for i in range(len(self.start_index)):
                data_index.extend([self.start_index[i] + (self.counter * self.batch_size + j) % self.dataset_len[i + 1] for j in range(self.batch_size)])
            yield data_index
            data_index = []
            self.counter += 1
        

class Batch_Balanced_Dataset0(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.

        opt.batch_ratio: 一个batch中包含的不同数据集的比例
        opt.total_data_usage_ratio: 对于每一个数据及，使用这个数据集的百分之多少，默认是1（100%）
        """
        self.opt = opt
        log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        # 为每个dataloader应用collate函数，直接输出一整个batch，
        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        self.batch_size_list = []
        Total_batch_size = 0

        self.dataset_list = []
        self.dataset_len_list = []

        self.pseudo_dataloader = None
        self.pseudo_batch_size = -1

        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset) # 当前数据集包含的图片数量
            
            
            log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio)) # 使用的比例
            if opt.fix_dataset_num != -1: number_dataset = opt.fix_dataset_num
            dataset_split = [number_dataset, total_number_dataset - number_dataset] # List[int] e.g. [50, 50]
            indices = range(total_number_dataset)
            
            # accumulate函数： _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
            # Subset就是根据indices取一个数据集的子集，indice根据opt.total_data_usage_ratio来取值
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            self.batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            self.dataset_list.append(_dataset)
            self.dataset_len_list.append(number_dataset)



        concatenated_dataset = ConcatDataset(self.dataset_list)
        assert len(concatenated_dataset) == sum(self.dataset_len_list)

        batch_sampler = Batch_Balanced_Sampler(self.dataset_len_list, _batch_size)
        self.data_loader = iter(torch.utils.data.DataLoader(
            concatenated_dataset,
            batch_sampler=batch_sampler,
            num_workers=int(opt.workers),
            collate_fn=_AlignCollate, pin_memory=False))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(self.batch_size_list)
        self.batch_size_list = list(map(int, self.batch_size_list))
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self, meta_target_index=-1, no_pseudo=False): # 如果指定了meta_target_index，则忽略第meta_target_index个数据集
        
        imgs, texts = next(self.data_loader)
        # 如果未指定或指定为伪标签数据集，则直接返回所有
        if meta_target_index == -1 or meta_target_index >= len(self.batch_size_list): return imgs, texts
        start_index_list = list(itertools.accumulate(self.batch_size_list))
        start_index_list.insert(0, 0)

        ret_imgs, ret_texts = [], []
        for i in range(len(self.batch_size_list)): 
            if i == meta_target_index: continue
            ret_imgs.extend(imgs[start_index_list[i] : start_index_list[i] + self.batch_size_list[i]])
            ret_texts.extend(texts[start_index_list[i] : start_index_list[i] + self.batch_size_list[i]])
        ret_imgs = torch.stack(ret_imgs, 0)
        
        # assert self.has_pseudo_label_dataset() == True, 'Pseudo label dataset can\'t be empty'
        if self.has_pseudo_label_dataset():
            try:
                psuedo_imgs, pseudo_texts = next(self.pseudo_dataloader_iter)
            except StopIteration:
                self.pseudo_dataloader_iter = iter(self.pseudo_dataloader)
                psuedo_imgs, pseudo_texts = next(self.pseudo_dataloader_iter)
            ret_imgs = torch.cat([ret_imgs, psuedo_imgs], 0)
            ret_texts += pseudo_texts

        return ret_imgs, ret_texts

    def get_meta_test_batch(self, meta_target_index=-1): # 如果指定了meta_target_index，则忽略第meta_target_index个数据集
        
        assert meta_target_index != -1, 'Meta target index should be specified'
        if meta_target_index >= len(self.batch_size_list) and self.has_pseudo_label_dataset(): 
            try:
                img, text = next(self.pseudo_dataloader_iter)
            except StopIteration:
                self.pseudo_dataloader_iter = iter(self.pseudo_dataloader)
                img, text = next(self.pseudo_dataloader_iter)

            return img, text
        
        imgs, texts = next(self.data_loader)
        start_index_list = list(itertools.accumulate(self.batch_size_list))
        start_index_list.insert(0, 0)
        ret_img, ret_text = None, None
        for i in range(len(self.batch_size_list)): 
            if i == meta_target_index:
                ret_img = imgs[start_index_list[i]:start_index_list[i] + self.batch_size_list[i]]
                ret_text = texts[start_index_list[i]:start_index_list[i] + self.batch_size_list[i]]

        return ret_img, ret_text

    def add_pseudo_label_dataset(self, dataset, opt):
        avg_batch_size = opt.batch_size // opt.source_num
        batch_size = len(dataset) if len(dataset) <= avg_batch_size else avg_batch_size
        self.pseudo_batch_size = batch_size
        self.pseudo_dataloader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size,
                        shuffle=True,  # 'True' to check training progress with validation function.
                        num_workers=int(opt.workers), pin_memory=False, collate_fn=self_training_collate)
        self.pseudo_dataloader_iter = iter(self.pseudo_dataloader)


    def has_pseudo_label_dataset(self):
        return True if self.pseudo_dataloader else False


def hierarchical_dataset(root, opt, select_data='/', pseudo=False):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root+'/', followlinks=True):
        print(dirpath, dirnames, filenames)
        if not dirnames: # 当dirnames为空，即当前dirpath下只包含（lmdb)文件时，进行操作
            select_flag = False
            for selected_d in select_data: # select_data为字符串，e.g. 'MJ','ST'
                if selected_d in dirpath: # 如果dirpath中包含了select_data 说明当前的目录是目标目录，select_flag置True
                    select_flag = True
                    break

            if select_flag:
                dataset = LmdbDataset(dirpath, opt, pseudo=pseudo)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    # 把所有数据集拼接在一起，以MJ为例，dataset_list中包括了MJ_train, MJ_valid和MJ_test
    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):

    def __init__(self, root, opt, pseudo=False):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            if self.opt.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    if self.opt.pseudo_dataset_num != -1 and pseudo and index > self.opt.pseudo_dataset_num:
                        break
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')
                    # print(label)

                    if len(label) > self.opt.batch_max_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.opt.character}]'
                    # if re.search(out_of_char, label.lower()): # 根据车牌场景进行了修改，因为车牌里只有大写字母，如果调用了lower，因为opt.char里面不包含小写字母，则所有车牌均被过滤
                    if re.search(out_of_char, label):
                        continue

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            # if not self.opt.sensitive:
            #     label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.opt.character}]'
            label = re.sub(out_of_char, '', label)

        return (img, label)


class RawDataset(Dataset):

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


def self_training_collate(batch):
    imgs, labels = [], []
    for img, label in batch:
        imgs.append(img)
        labels.append(label)
    
    return torch.stack(imgs), labels

class SelfTrainingDataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels
    
    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]

    def __len__(self):
        assert len(self.imgs) == len(self.labels)
        return len(self.imgs)



def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
