import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import KFold
import matplotlib
import matplotlib.pyplot as plt
from dateutil.parser import parse
from datetime import datetime, timedelta
from collections import deque



#sdf
print("Pandas version:      ", pd.__version__)
print("NumPy version:       ", np.__version__)
print("SciKit Learn version:", sklearn.__version__)
print("TensorFlow version:  ", tf.__version__)
print("MatPlotLib version:  ", matplotlib.__version__)

seed = 8
tf.set_random_seed(seed)
np.random.seed(seed)
dataframe = pd.read_csv('prices.csv')
dataframe.describe()
