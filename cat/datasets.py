import os
import re
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Dataset Processor
class DatasetProcessor(object):
  def __init__(self, file_path='data', image_folder='data', train_set='train.csv', test_set='test.csv'):
    self.image_folder = image_folder
    # Load the train imageid, labels, and captions
    with open(os.path.join(file_path, train_set)) as file:
      lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    self.df_train = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
    # Load the test imageid, and captions
    with open(os.path.join(file_path, test_set)) as file:
      lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    self.df_test = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
    # Retrieve the maximum length of the captions
    self.max_length = self.df_train['Caption'].str.len().max()
    # Retrieve the minimum length of the captions
    self.min_length = self.df_train['Caption'].str.len().min()
    label_series = self.df_train['Labels'].apply(lambda x: list(map(int, x.split(" "))))
    self.binarizer = MultiLabelBinarizer()
    self.label_binary = self.binarizer.fit_transform(label_series).tolist()
    # Compute the number of classes
    self.num_classes = len(self.label_binary[0])
  
  def get_test(self):
    # Extract the imageid, and captions
    imgid_test = self.df_test['ImageID'].to_numpy()
    caption_test = self.df_test['Caption'].to_numpy()
    return imgid_test, caption_test
  
  def get_train_validate(self):
    # Extract the imageid, labels, and captions
    imgid_raw = self.df_train['ImageID'].to_numpy()
    caption_raw = self.df_train['Caption'].to_numpy()
    # Split the train set into train and validation sets
    random_state = random.randint(0, 1e5)
    imgid_train, imgid_valid, label_train, label_valid = train_test_split(imgid_raw, self.label_binary, 
                                                                          test_size=0.1, random_state=random_state)
    caption_train, caption_valid, label_train, label_valid = train_test_split(caption_raw, self.label_binary, 
                                                                            test_size=0.1, random_state=random_state)
    label_binary_tensor = torch.tensor(self.label_binary)
    label_train_tensor = torch.tensor(label_train)
    label_valid_tensor = torch.tensor(label_valid)
    return (imgid_raw, caption_raw, label_binary_tensor), (imgid_train, caption_train, label_train_tensor), (imgid_valid, caption_valid, label_valid_tensor)

  def decode(self, y_pred_encode):
    if isinstance(y_pred_encode, torch.Tensor):
      y_pred_numpy = y_pred_encode.numpy()
    elif isinstance(y_pred_encode, np.ndarray):
      y_pred_numpy = y_pred_encode
    
    if len(y_pred_numpy.shape) == 1:
      y_pred_numpy = y_pred_numpy.reshape(1, -1)
    y_pred_tuplst = self.binarizer.inverse_transform(y_pred_numpy)
    y_pred = []
    for tup in y_pred_tuplst:
      lst = list(tup)
      str_lst = list(map(str, lst))
      y_pred.append(" ".join(str_lst))
    y_pred = np.array(y_pred)
    if len(y_pred_encode.shape) == 1:
      return y_pred[0]
    else:
      return y_pred
  
  def display_image(self, data_loader):
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    img = images[0]
    label = self.decode(labels[0])
    img_numpy = img.numpy()
    plt.imshow(np.transpose(img_numpy, (1, 2, 0)))
    plt.title(f"Label: {label}.")
    plt.show()
  
  def display_text(self, data_loader, tokenizer, n_samples=10):
    dataiter = iter(data_loader)
    data = next(dataiter)
    txt, mask, labs = data['input_ids'], data['attention_mask'], data['label']
    txt_ = [tokenizer.decode(t.squeeze(), skip_special_tokens=True) for t in txt[:n_samples]]
    mask_ = [int(m.sum()) for m in mask[:n_samples]]
    labs_ = [self.decode(l) for l in labs[:n_samples]]
    txt_samples = pd.DataFrame({
        'Text': txt_,
        'Actual input length': mask_,
        'Label': labs_
    })
    return txt_samples

# Customise ImageDataset
class ImageDataset(Dataset):
  def __init__(self, imgid_label, img_folder, transform='normalize'):
    self.imgid_label = imgid_label
    self.img_folder = img_folder
    if transform == 'augment':
      self.transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.Resize(232),
        transforms.RandomResizedCrop((224, 224), scale=(0.1, 1), ratio=(0.7, 1.0/0.7)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    elif transform == 'normalize':
      self.transform = transforms.Compose([
        ResizeLongEdgeAndPad(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

  def __getitem__(self, idx):
    imgid_ = self.imgid_label[idx][0]
    label_ = self.imgid_label[idx][1]
    img_path = os.path.join(self.img_folder, imgid_)
    img = Image.open(img_path).convert("RGB")
    if self.transform:
      img = self.transform(img)
    return img, label_

  def __len__(self):
    return len(self.imgid_label)

# Define a normalisation transformation, including Resize and Padding
class ResizeLongEdgeAndPad(object):
  def __init__(self, size, padding_mode='constant', fill=0):
    self.size = size
    self.padding_mode = padding_mode
    self.fill = fill

  def __call__(self, img):
    w, h = img.size
    if h > w:
      new_h = self.size
      new_w = int(self.size * w / h)
    else:
      new_w = self.size
      new_h = int(self.size * h / w)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    pad_w = self.size - new_w
    pad_h = self.size - new_h
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    return ImageOps.expand(img, padding, fill=self.fill)

# Define a text data augmentation - random swap
def random_swap(words, n):
  words = words.copy()
  for _ in range(n):
    if len(words) < 2:
      break
    idx1, idx2 = random.sample(range(len(words)), 2)
    words[idx1], words[idx2] = words[idx2], words[idx1]
  return words

# Customise TextDataset
class TextDataset(Dataset):
  def __init__(self, txt_label, tokenizer, max_length, min_length=23, random_swap_=False):
    self.txt_label = txt_label
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.min_length = min_length
    self.random_swap_ = random_swap_

  def __len__(self):
    return len(self.txt_label)

  def __getitem__(self, idx):
    text = self.txt_label[idx][0]
    label = self.txt_label[idx][1]
    if self.random_swap_:
      words = text.split()
      words = random_swap(words, n=self.min_length)
      text = ' '.join(words)

    encoding = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=self.max_length,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'label': torch.tensor(label, dtype=torch.long)
    }

class MultimodalDataset(Dataset):
  def __init__(self, imgid_txt_label, img_folder, tokenizer, max_length, transform=None, random_swap_=False):
    self.imgid_txt_label = imgid_txt_label
    self.img_folder = img_folder
    self.tokenizer = tokenizer
    self.max_length = max_length
    if transform == 'augment':
      self.transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.Resize(232),
        transforms.RandomResizedCrop((224, 224), scale=(0.1, 1), ratio=(0.7, 1.0/0.7)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    elif transform == 'normalize' or transform == 'normalise' or transform is None:
      self.transform = transforms.Compose([
        ResizeLongEdgeAndPad(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    self.random_swap_ = random_swap_

  def __len__(self):
    return len(self.imgid_txt_label)

  def __getitem__(self, idx):
    imgid_ = self.imgid_txt_label[idx][0]
    text = self.imgid_txt_label[idx][1]
    label = self.imgid_txt_label[idx][2]
    # Image operations
    img_path = os.path.join(self.img_folder, imgid_)
    img = Image.open(img_path).convert("RGB")
    if self.transform:
      img = self.transform(img)
    # Text operations
    if self.random_swap_:
      words = text.split()
      words = random_swap(words, n=5)
      text = ' '.join(words)

    encoding = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=self.max_length,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    txt_dict = {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'label': torch.tensor(label, dtype=torch.long)
    }
    return img, txt_dict