from PIL import Image
import os
import os.path
import errno
import copy
import codecs
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from copy import deepcopy
from torchvision.datasets.vision import VisionDataset
import torchvision.datasets as dsets
from sklearn.preprocessing import OneHotEncoder
from augment.cutout import Cutout
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip, RandomErasing, ToPILImage

class MY_KMNIST(VisionDataset):
    """`Kuzushiji-MNIST <https://github.com/rois-codh/kmnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    urls = [
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, rate_partial=0.3):

        super(MY_KMNIST, self).__init__(root, transform=transform,
                                        target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.training_file
        else:
            downloaded_list = self.test_file

        self.data, self.targets = torch.load(
            os.path.join(self.root, self.processed_folder, downloaded_list))



        self.transform = Compose([
            RandomHorizontalFlip(),
            RandomCrop(28, 4, padding_mode='reflect'),
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5]),
        ])

        self.transform1 = Compose([
            RandomHorizontalFlip(),
            RandomCrop(28, 4, padding_mode='reflect'),
            ToTensor(),
            Cutout(n_holes=1, length=16),
            ToPILImage(),
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5]),
        ])

        self.rate_partial = rate_partial
        self.partial_labels = self.generate_partial_labels()


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target, partial_label = self.data[index], self.targets[index], self.partial_labels[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img_ori = self.transform(img)
            img1 = self.transform1(img)
            img2 = self.transform1(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_ori, img1, img2, target, partial_label, index

    def __len__(self):
        return len(self.data)

    def generate_partial_labels(self):
        if(self.rate_partial!=-1):
            def binarize_class(y):
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
                c=max(self.targets)+1
                data=self.data
                y=binarize_class(torch.tensor(self.targets,dtype=torch.long))

                f = np.prod(list(data.shape)[1:])
                batch_size = 2000
                rate = 0.4
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                weight_path = ('./weights/' + 'kmnist' + '/400.pt')
                model = create_model('kmnist', f, c).to(device)
                model.load_state_dict(torch.load(weight_path, map_location=device))
                train_X, train_Y = data.to(device), y.to(device)
                train_X = train_X.view(train_X.shape[0], -1).float()
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

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the Kuzushiji-MNIST data if it doesn't exist in processed_folder already."""
        import urllib.request
        import requests
        import gzip

        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)
