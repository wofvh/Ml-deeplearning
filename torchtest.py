#BERT 

import os 
import re
import json
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from tqdm import tqdm
import tensorflow as tf
from tensorflow import *
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import early_stopping, ModelCheckpoint