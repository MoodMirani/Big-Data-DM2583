import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from main import Create_Training_Set

trainingData = Create_Training_Set()

print(trainingData.tail)
