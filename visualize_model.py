from model import D, Net
import torch
import torchvision
from torchview import draw_graph
from torchviz import make_dot
import config

config.domain_backprop = True
config.tags.append("ensemble_training")

# input_size=(1,4,64,64,64)
# x = torch.randn(input_size)


input_size = ((1, 1, 16, 16, 16), (1, 1, 16, 16, 16))
x = (torch.randn(input_size[0]), torch.randn(input_size[1]))

model = Net()
model.train()

model_graph = draw_graph(model, input_data=x, expand_nested=True, save_graph=True, filename="DomainBackprop")
