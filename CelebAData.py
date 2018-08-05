import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import random


class CelebAData(Dataset):

	"""
	The file structure must follows:
	data_root
		|- Img
		|	|- img_align_celeba
		|	|	|- 000001.jpg
		|	|	+- ...
		|	+- ...
		|- Anno
		|	|- list_attr_celeba.txt
		|	|- identity_CelebA.txt
		|	+- ...
		+- Eval
			+- list_eval_partition.txt

	"""

	def __init__(self, data_root, image_size, crop_size, selected_attrs=['Bangs'], mode='train'):

		# path
		self.data_root = data_root
		self.attr_path = os.path.join(self.data_root, 'Anno', 'list_attr_celeba.txt')
		self._preprocess(selected_attrs)

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
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_datasetransforms.append([filename, label])
            else:
                self.train_datasetransforms.append([filename, label])	

    def setmode(self, mode):
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