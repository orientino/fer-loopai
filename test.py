# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

df = pd.read_csv('./data/challengeA_train.csv')
train_count = df.groupby('emotion').count()['image_id']
train_count.plot(kind='bar')

# %%
# compute class weights 
# w_j = n_samples / (n_classes * n_samples_j)
n_classes = len(df['emotion'].unique())
class_weight = [len(df)/(n_classes*len(df[df['emotion']==i])) 
                for i in range(n_classes)]
class_weight