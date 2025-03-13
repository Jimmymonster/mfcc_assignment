from model_old import LSTMEmotionModel
from torchinfo import summary
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMEmotionModel(num_classes=5).to(device)
summary(model, input_size=(1, 216, 128))
# print(model)A