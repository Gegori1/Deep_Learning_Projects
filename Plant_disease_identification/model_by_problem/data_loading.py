# %% libraries
import os
import sys
import shutil
import numpy as np

from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import transforms
from torch.utils.data import Subset
from torch.utils.data import Dataset


# %% train and eval data loader

def load_train_eval_data(path_to_data: str,  dataset_loader: Dataset, batch_size: int = 32, eval_size: float = 0.2, resize: int = 224, random_state: int = 42, workers: int = 4, pin_memory_device: object = ...):
    """
    Load train and evaluation data from a directory containing subdirectories with images.

    Args:
    path_to_data: str: Path to the directory containing subdirectories with images.
    eval_size: float: Size of the eval set.
    random_state: int: Random state for the train_eval_split function.
    
    Returns:
    train_data: DataLoader: Train data.
    eval_data: DataLoader: eval data.
    """
    # Create a transformation
    trnsf = transforms.Compose([
        transforms.RandomCrop(size=resize),
        transforms.Resize([resize, resize]),
        transforms.RandomAdjustSharpness(2),
        transforms.RandomAutocontrast(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        # transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    # Create a dataset
    data = dataset_loader(root=path_to_data, transform=trnsf)
    
    targets = data.targets

    # Perform a stratified split
    train_index, eval_index, targets_train, targets_eval = train_test_split(
        range(len(targets)), 
        targets, 
        test_size=eval_size, 
        stratify=targets, 
        random_state=random_state
    )
  	
    subset_train, subset_eval = Subset(data, train_index), Subset(data, eval_index)
    
    # set a weighted sampler for the DataLoader
    count_class_train, count_class_eval = dict(Counter(targets_eval)), dict(Counter(targets_eval))
    
    weights_train = [1 / count_class_train[i] for i in targets_train]
    weights_eval = [1 / count_class_eval[i] for i in targets_eval]

    sampler_train = WeightedRandomSampler(weights_train, num_samples=len(weights_train), replacement=True)
    sampler_eval = WeightedRandomSampler(weights_eval, num_samples=len(weights_eval), replacement=True)

    if "cuda" in pin_memory_device.type:
      train_data = DataLoader(dataset=subset_train, batch_size=batch_size, sampler=sampler_train, num_workers=workers, pin_memory=True, pin_memory_device=pin_memory_device.type)
      eval_data = DataLoader(dataset=subset_eval, batch_size=batch_size, sampler=sampler_eval, num_workers=workers, pin_memory=True, pin_memory_device=pin_memory_device.type)
    else:
      train_data = DataLoader(dataset=subset_train, batch_size=batch_size, sampler=sampler_train, num_workers=workers)
      eval_data = DataLoader(dataset=subset_eval, batch_size=batch_size, sampler=sampler_eval, num_workers=workers)
    
    return train_data, eval_data


# %% test data loader

def load_train_data(path_to_data: str,  dataset_loader: Dataset, batch_size: int = 32, resize: int = 224, workers: int = 4, pin_memory_device: object = ...):
  """
    Load train data from a directory containing subdirectories with images.

    Args:
    path_to_data: str: Path to the directory containing subdirectories with images.
    
    Returns:
    train_data: DataLoader: train data.
  """

  trnsf = transforms.Compose([
          transforms.Resize([resize, resize]),
          # transforms.ToTensor(),
          transforms.Normalize(mean=0.5, std=0.5)
    ])

    # Create a dataset
  data = dataset_loader(root=path_to_data, transform=trnsf)

  targets = data.targets

  # set a weighted sampler for the DataLoader
  count_class = dict(Counter(targets))
  weights = [1 / count_class[i] for i in targets]
  sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    # Create a DataLoader
  if "cuda" in pin_memory_device.type:
    train_data = DataLoader(dataset=data, batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=True, pin_memory_device=pin_memory_device.type)
  else:
    train_data = DataLoader(dataset=data, batch_size=batch_size, sampler=sampler, num_workers=workers)
    
  return train_data


def load_test_data(path_to_data: str,  dataset_loader: Dataset, batch_size: int = 32, resize: int = 224, workers: int = 4, pin_memory_device: object = ...):
  """
    Load test data from a directory containing subdirectories with images.

    Args:
    path_to_data: str: Path to the directory containing subdirectories with images.
    
    Returns:
    test_data: DataLoader: Test data.
  """

  trnsf = transforms.Compose([
          transforms.Resize([resize, resize]),
          # transforms.ToTensor(),
          transforms.Normalize(mean=0.5, std=0.5)
    ])

    # Create a dataset
  data = dataset_loader(root=path_to_data, transform=trnsf)
    
    # Create a DataLoader
  if "cuda" in pin_memory_device.type:
    test_data = DataLoader(dataset=data, batch_size=batch_size, num_workers=workers, pin_memory=True, pin_memory_device=pin_memory_device.type)
  else:
    test_data = DataLoader(dataset=data, batch_size=batch_size, num_workers=workers)
    
  return test_data