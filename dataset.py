import numpy as np
import pandas as pd
from PIL import Image
from typing import Literal

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class TrainSet(Dataset):
    def __init__(self):
        