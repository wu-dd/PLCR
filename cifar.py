from PIL import Image
import os
import os.path
import numpy as np
import pickle
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision import datasets, transforms
from augment.cutout import Cutout
from augment.autoaugment_extra import CIFAR10Policy, ImageNetPolicy
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip, RandomErasing, \
    ToPILImage
import copy
import torchvision.datasets as dsets
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from scipy.special import comb


class MY_CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, rate_partial=0.3):

        super(MY_CIFAR10, self).__init__(root, transform=transform,
                                         target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        # print(len(self.targets))
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.transform = Compose([
            RandomHorizontalFlip(),
            RandomCrop(32, 4, padding_mode='reflect'),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform1 = Compose([
            RandomHorizontalFlip(),
            RandomCrop(32, 4, padding_mode='reflect'),
            ToTensor(),
            Cutout(n_holes=1, length=16),
            ToPILImage(),
            CIFAR10Policy(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.rate_partial = rate_partial
        self.partial_labels = self.generate_partial_labels()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, partial_label = self.data[index], self.targets[index], self.partial_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img_ori = self.transform(img)
            img1 = self.transform1(img)
            img2 = self.transform1(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_ori, img1, img2, target, partial_label, index

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

    def generate_partial_labels(self):
        if (self.rate_partial != -1):
            def binarize_class(y):
                y = torch.tensor(y)
                label = y.reshape(len(y), -1)
                enc = OneHotEncoder(categories='auto')
                enc.fit(label)
                label = enc.transform(label).toarray().astype(np.float32)
                label = torch.from_numpy(label)
                return label

            new_y = binarize_class(self.targets)
            n = len(self.targets)
            c = max(self.targets) + 1
            avgC = 0
            partial_rate = self.rate_partial
            print(partial_rate)
            for i in range(n):
                row = new_y[i, :]
                row[np.where(np.random.binomial(1, partial_rate, c) == 1)] = 1
                while torch.sum(row) == 1:
                    row[np.random.randint(0, c)] = 1
                avgC += torch.sum(row)
                new_y[i] = row

            avgC = avgC / n
            print("Finish Generating Candidate Label Sets:{}!\n".format(avgC))
            new_y = new_y.cpu().numpy()
            return new_y
        else:
            def binarize_class(y):
                label = y.reshape(len(y), -1)
                enc = OneHotEncoder(categories='auto')
                enc.fit(label)
                label = enc.transform(label).toarray().astype(np.float32)
                label = torch.from_numpy(label)
                return label

            def create_model(ds, feature, c):
                from partial_models.resnet import resnet
                from partial_models.mlp import mlp_phi
                if ds in ['kmnist', 'fmnist']:
                    net = mlp_phi(feature, c)
                elif ds in ['cifar10']:
                    net = resnet(depth=32, n_outputs=c)
                else:
                    pass
                return net

            with torch.no_grad():
                c = max(self.targets) + 1
                data = torch.from_numpy(self.data)
                y = binarize_class(torch.tensor(self.targets, dtype=torch.long))

                f = np.prod(list(data.shape)[1:])
                batch_size = 2000
                rate = 0.4
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                weight_path = ('./weights/' + 'cifar10' + '/400.pt')
                model = create_model('cifar10', f, c).to(device)
                model.load_state_dict(torch.load(weight_path, map_location=device))
                train_X, train_Y = data.to(device), y.to(device)

                train_X = train_X.permute(0, 3, 1, 2).to(torch.float32)
                train_p_Y_list = []
                step = train_X.size(0) // batch_size
                for i in range(0, step):
                    _, outputs = model(train_X[i * batch_size:(i + 1) * batch_size])
                    train_p_Y = train_Y[i * batch_size:(i + 1) * batch_size].clone().detach()
                    partial_rate_array = F.softmax(outputs, dim=1).clone().detach()
                    partial_rate_array[torch.where(train_Y[i * batch_size:(i + 1) * batch_size] == 1)] = 0
                    partial_rate_array = partial_rate_array / torch.max(partial_rate_array, dim=1, keepdim=True)[0]
                    partial_rate_array = partial_rate_array / partial_rate_array.mean(dim=1, keepdim=True) * rate
                    partial_rate_array[partial_rate_array > 1.0] = 1.0
                    m = torch.distributions.binomial.Binomial(total_count=1, probs=partial_rate_array)
                    z = m.sample()
                    train_p_Y[torch.where(z == 1)] = 1.0
                    train_p_Y_list.append(train_p_Y)
                train_p_Y = torch.cat(train_p_Y_list, dim=0)
                assert train_p_Y.shape[0] == train_X.shape[0]
            final_y = train_p_Y.cpu().clone()
            pn = final_y.sum() / torch.ones_like(final_y).sum()
            print("Partial type: instance dependent, Average Label: " + str(pn * 10))
            return final_y.cpu().numpy()


class MY_CIFAR100(MY_CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]