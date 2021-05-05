# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

df = pd.read_csv('./data/challengeA_train.csv')
train_count = df.groupby('emotion').count()['image_id']
train_count.plot(kind='bar')