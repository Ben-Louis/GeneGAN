import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import random
import itertools


class CelebAData(Dataset):

    """
    The file structure must follows:
    data_root
        |- Img
        |    |- img_align_celeba
        |    |    |- 000001.jpg
        |    |    +- ...
        |    +- ...
        |- Anno
        |    |- list_attr_celeba.txt
        |    |- identity_CelebA.txt
        |    +- ...
        +- Eval
            +- list_eval_partition.txt

    """

    def __init__(self, data_root, image_size, crop_size, selected_attrs=['Bangs'], mode='train'):

        # path
        self.image_dir = os.path.join(data_root, 'Img', 'img_align_celeba')
        self.attr_path = os.path.join(data_root, 'Anno', 'list_attr_celeba.txt')

        self.selected_attrs = selected_attrs
        self.attr2idx = {}
        self.idx2attr = {}
        self._preprocess(selected_attrs)

        self.mode = mode
        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

        trans = []
        if mode == 'train':
            trans.append(transforms.RandomHorizontalFlip())
        trans.append(transforms.CenterCrop(crop_size))
        trans.append(transforms.Resize(image_size))
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(trans)

    def _preprocess(self, selected_attrs):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(666)
        random.shuffle(lines)
        self.train_dataset = []
        self.test_dataset = []
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])    

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label) # label: ont-hot

    def get_loader(self, **kwargs):
        return DataLoader(self, **kwargs)



"""
5_o_Clock_Shadow 
Arched_Eyebrows 
Attractive 
Bags_Under_Eyes 
Bald Bangs 
Big_Lips 
Big_Nose 
Black_Hair 
Blond_Hair 
Blurry 
Brown_Hair 
Bushy_Eyebrows 
Chubby 
Double_Chin 
Eyeglasses 
Goatee 
Gray_Hair 
Heavy_Makeup 
High_Cheekbones 
Male 
Mouth_Slightly_Open 
Mustache 
Narrow_Eyes 
No_Beard 
Oval_Face 
Pale_Skin 
Pointy_Nose 
Receding_Hairline 
Rosy_Cheeks 
Sideburns 
Smiling 
Straight_Hair 
Wavy_Hair 
Wearing_Earrings 
Wearing_Hat
Wearing_Lipstick 
Wearing_Necklace 
Wearing_Necktie 
Young 
"""


class CelebaData_GeneGAN(CelebAData):


    def _preprocess(self, selected_attrs):
        """Preprocess the CelebA attribute file."""
        assert len(selected_attrs) == 1, 'len(selected_attrs) must be 1'
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(666)
        random.shuffle(lines)
        pos, neg = [], []

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                if values[idx] == '1':
                    pos.append(filename)
                else:
                    neg.append(filename)

        #print('pos:{}, neg:{}'.format(len(pos), len(neg)))
        self.train_dataset = (pos[100:], neg[100:])
        self.test_dataset = (pos[:100], neg[:100])


    def __len__(self):
        if self.mode == 'train':
            return len(self.train_dataset[0])
        else:
            return len(self.test_dataset[0])


    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        pos_file = dataset[0][index]
        neg_file = random.choice(dataset[1])

        image_pos = Image.open(os.path.join(self.image_dir, pos_file))
        image_neg = Image.open(os.path.join(self.image_dir, neg_file))
        return self.transform(image_pos), self.transform(image_neg)

    def get_test(self, index, n=10):
        neg_files = self.test_dataset[1][index:index+n]
        pos_files = random.sample(self.test_dataset[0], n)

        neg_imgs, pos_imgs = [], []
        for f in neg_files:
            neg_imgs.append(self.transform(Image.open(os.path.join(self.image_dir, f))))
        for f in pos_files:
            pos_imgs.append(self.transform(Image.open(os.path.join(self.image_dir, f))))

        return torch.stack(pos_imgs), torch.stack(neg_imgs)



