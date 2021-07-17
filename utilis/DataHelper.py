from torch.utils.data import Dataset
import PIL.Image as Image
import os


def train_dataset(img_root, label_root):
    imgs = []
    n = len(os.listdir(img_root))
    for i in range(n-1):
        img = os.path.join(img_root, "img%d.png" % (i+1))  # img = //1.png
        label = os.path.join(label_root, "label%d.png" % (i+1)) # label = //1_mask.png
        imgs.append((img, label))
    return imgs

def eval_dataset(img_root, label_root):
    imgs = []
    n = len(os.listdir(img_root))
    for i in range(n):
        img = os.path.join(img_root, "img%d.png" % (i+1))  # img = //1.png
        label = os.path.join(label_root, "label%d.png" % (i+1)) # label = //1_mask.png
        imgs.append((img, label))
    return imgs

def test_dataset(img_root):
    imgs = []
    n = len(os.listdir(img_root))
    for i in range(n):
        img = os.path.join(img_root, "img%d.png" % (i+1))
        imgs.append(img)
    return imgs


class TrainDataset(Dataset):
    def __init__(self, img_root, label_root, transform=None, target_transform=None):
        imgs = train_dataset(img_root, label_root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

class EvalDataset(Dataset):
    def __init__(self, img_root, label_root, transform=None, target_transform=None):
        imgs = eval_dataset(img_root, label_root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)


class TestDataset(Dataset):
    def __init__(self, img_root, transform=None, target_transform=None):
        imgs = test_dataset(img_root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path = self.imgs[index]
        img_x = Image.open(x_path).resize((512, 512), Image.BICUBIC)
        if self.transform is not None:
            img_x = self.transform(img_x)
        return img_x

    def __len__(self):
        return len(self.imgs)









