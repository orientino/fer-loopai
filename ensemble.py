"""
Main module to test the ensemble results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import *
from utils.data import *
from utils.model import *
from utils.train import *
from utils.infer import *
from utils.plot import *
plt.style.use('ggplot')

# load dataset
transform_valid = transforms.Compose([ transforms.ToTensor() ])

batch_size = 8
path = './data/'
df_train = pd.read_csv(os.path.join(path, 'challengeA_train_clean_stratify.csv'), index_col=0)
df_valid = pd.read_csv(os.path.join(path, 'challengeA_valid_clean_stratify.csv'), index_col=0)[:300] # testing only with 300 images
df_test = pd.read_csv(os.path.join(path, 'challengeA_test.csv'), index_col=0)
class_weights = torch.tensor(get_class_weights(df_train['emotion']))
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

train_loader, valid_loader, test_loader = get_dataloaders_loopai(df_train, df_valid, df_test, path, transform_valid, transform_valid, batch_size)

import torch.nn as nn
from models.vit import *
from models.densenet import *
from models.vggnet import *
from models.resnet_narrow_nocp import *
from efficientnet_pytorch import EfficientNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ensemble_models = []
path = "./pretrained"

# mobilenet = MobileNetV3L().to(device)
# load(os.path.join(path, "MobileNetV3_1.0_13_0.618.tar"), mobilenet, device=device)
# ensemble_models.append(mobilenet)

vit = ViT().to(device)
load(os.path.join(path, "ViT1.0_19_0.638.tar"), vit, device=device)
ensemble_models.append(vit)

vgg = VGGNet().to(device)
load(os.path.join(path, "VGG4.0_6_0.691.tar"), vgg, device=device)
ensemble_models.append(vgg)

resnet = resnet18().to(device)
load(os.path.join(path, "RESNET18_2.2_2_0.677.tar"), resnet, device=device)
ensemble_models.append(resnet)

densenet = densenet121().to(device)
load(os.path.join(path, "DENSENET121_2.1_4_0.672.tar"), densenet, device=device)
ensemble_models.append(densenet)

# efficientnet = EfficientNet.from_name(model_name="efficientnet-b3", in_channels=1, image_size=48, num_classes=7).to(device)
# load(os.path.join(path, "EFFICIENTNET_B3_1.0_6_0.666.tar"), efficientnet, device=device)
# ensemble_models.append(efficientnet)

preds = ensemble(0, ensemble_models, valid_loader, device=device)
print(f"Accuracy ensemble: {accuracy_score(df_valid['emotion'], preds)}")
