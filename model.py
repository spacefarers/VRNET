import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch
import config
import numpy as np


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.kaiming_uniform_(m.weight.data)
    elif classname.find("Linear") != -1:
        init.kaiming_uniform_(m.weight.data)
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def BuildResidualBlock(channels, dropout, kernel, depth, bias):
    layers = []
    for i in range(int(depth)):
        layers += [nn.Conv3d(channels, channels, kernel_size=kernel, stride=1, padding=kernel // 2, bias=bias),
                   # nn.BatchNorm3d(channels),
                   nn.ReLU(True)]
        if dropout:
            layers += [nn.Dropout(0.5)]
    layers += [nn.Conv3d(channels, channels, kernel_size=kernel, stride=1, padding=kernel // 2, bias=bias),
               # nn.BatchNorm3d(channels),
               ]
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout, kernel, depth, bias):
        super(ResidualBlock, self).__init__()
        self.block = BuildResidualBlock(channels, dropout, kernel, depth, bias)

    def forward(self, x):
        out = x + self.block(x)
        return out


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel // 2
        self.Gates = nn.Conv3d(input_size + hidden_size, 4 * hidden_size, kernel, padding=pad)

    def forward(self, input_, prev_hidden=None, prev_cell=None):
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        if prev_hidden is None and prev_cell is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_hidden = torch.zeros(state_size)
            prev_cell = torch.zeros(state_size)
        prev_hidden = prev_hidden.to(config.device)
        prev_cell = prev_cell.to(config.device)
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)
        return hidden, cell


class Encoder(nn.Module):
    def __init__(self, inc, init_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(inc, init_channels, 4, 2, 1)
        self.rb1 = ResidualBlock(init_channels, dropout=False, kernel=3, depth=2, bias=False)
        self.conv2 = nn.Conv3d(init_channels, 2 * init_channels, 4, 2, 1)
        self.rb2 = ResidualBlock(2 * init_channels, dropout=False, kernel=3, depth=2, bias=False)
        self.conv3 = nn.Conv3d(2 * init_channels, 4 * init_channels, 4, 2, 1)
        self.rb3 = ResidualBlock(4 * init_channels, dropout=False, kernel=3, depth=2, bias=False)
        self.conv4 = nn.Conv3d(4 * init_channels, 8 * init_channels, 4, 2, 1)
        self.rb4 = ResidualBlock(8 * init_channels, dropout=False, kernel=3, depth=2, bias=False)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x1 = self.rb1(x1)
        x2 = F.relu(self.conv2(x1))
        x2 = self.rb2(x2)
        x3 = F.relu(self.conv3(x2))
        x3 = self.rb3(x3)
        x4 = F.relu(self.conv4(x3))
        x4 = self.rb4(x4)
        return [x1, x2, x3, x4]


def voxel_shuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width, in_depth = input.size()
    channels //= upscale_factor ** 3

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor
    out_depth = in_depth * upscale_factor

    input_view = input.reshape(batch_size, channels, upscale_factor, upscale_factor, upscale_factor, in_height,
                               in_width, in_depth)

    return input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(batch_size, channels, out_height, out_width, out_depth)


class VoxelShuffle(nn.Module):
    def __init__(self, inchannels, outchannels, upscale_factor):
        super(VoxelShuffle, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv3d(inchannels, outchannels * (upscale_factor ** 3), 3, 1, 1)

    def forward(self, x):
        x = voxel_shuffle(self.conv(x), self.upscale_factor)
        return x


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, 4, 2, 1)
        self.conv2 = nn.Conv3d(64, 128, 4, 2, 1)
        self.lstm = LSTMCell(128, 128, 3)
        self.conv3 = nn.Conv3d(128, 1, 4, 2, 1)

    def forward(self, x):
        num = x.size()[1]
        h = None
        c = None
        comps = []
        for i in range(num):
            f = F.relu(self.conv1(x[:, i:i + 1, :, :, :, ]))
            f = F.relu(self.conv2(f))
            # h, c = self.lstm(f, h, c)
            # f = self.conv3(h)
            f = self.conv3(f)
            f = F.avg_pool3d(f, f.size()[2:]).view(-1)
            comps.append(f)
        comps = torch.stack(comps)
        comps = torch.squeeze(comps)
        return comps


class Decoder(nn.Module):
    def __init__(self, outc, init_channels):
        super(Decoder, self).__init__()
        self.deconv41 = nn.ConvTranspose3d(init_channels, init_channels // 2, 4, 2, 1)
        self.conv_u41 = nn.Conv3d(init_channels, init_channels // 2, 3, 1, 1)
        self.deconv31 = nn.ConvTranspose3d(init_channels // 2, init_channels // 4, 4, 2, 1)
        self.conv_u31 = nn.Conv3d(init_channels // 2, init_channels // 4, 3, 1, 1)
        self.deconv21 = nn.ConvTranspose3d(init_channels // 4, init_channels // 8, 4, 2, 1)
        self.conv_u21 = nn.Conv3d(init_channels // 4, init_channels // 8, 3, 1, 1)
        self.deconv11 = nn.ConvTranspose3d(init_channels // 8, init_channels // 16, 4, 2, 1)
        self.conv_u11 = nn.Conv3d(init_channels // 16, outc, 3, 1, 1)

    def forward(self, features):
        u11 = F.relu(self.deconv41(features[-1]))
        u11 = F.relu(self.conv_u41(torch.cat((features[-2], u11), dim=1)))
        u21 = F.relu(self.deconv31(u11))
        u21 = F.relu(self.conv_u31(torch.cat((features[-3], u21), dim=1)))
        u31 = F.relu(self.deconv21(u21))
        u31 = F.relu(self.conv_u21(torch.cat((features[-4], u31), dim=1)))
        u41 = F.relu(self.deconv11(u31))
        out = self.conv_u11(u41)
        out = torch.tanh(out)
        return out


class UNet(nn.Module):
    def __init__(self, inc, outc, init_channels):
        super(UNet, self).__init__()
        self.encoder = Encoder(inc, init_channels)
        self.decoder = Decoder(outc, init_channels * 8)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def Encode(self, x):
        return self.encoder(x)

    def Decode(self, x):
        return self.decoder(x)


def conv3d(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    return F.conv3d(input, weight.to(config.device), bias.to(config.device), stride, padding, dilation, groups)


def relu(input):
    return F.threshold(input, 0, 0, inplace=True)


def vs(input, weight, bias=None, factor=1, stride=1, padding=1, dilation=1, groups=1):
    x = F.conv3d(input, weight.to(config.device), bias.to(config.device), stride, padding, dilation, groups)
    return voxel_shuffle(x, factor)


class Siren(nn.Module):
    def __init__(self, omega=30):
        super(Siren, self).__init__()
        self.omega = omega

    def forward(self, x):
        return torch.sin(self.omega * x)


class Switch(nn.Module):
    def __init__(self, beta=1.0):
        super(Switch, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class RDB(nn.Module):  # Residual Dense Block
    def __init__(self, init_channels, outchannels, active='relu'):
        super(RDB, self).__init__()
        self.conv1 = nn.Conv3d(init_channels, 2 * init_channels, 3, 1, 1)
        self.conv2 = nn.Conv3d(3 * init_channels, 4 * init_channels, 3, 1, 1)
        self.conv3 = nn.Conv3d(4 * init_channels + 2 * init_channels + init_channels, outchannels, 3, 1, 1)
        if active == 'relu':
            self.ac = nn.ReLU(inplace=True)
        elif active == 'tanh':
            self.ac = nn.Tanh()
        elif active == 'siren':
            self.ac = Siren()
        elif active == 'switch':
            self.ac = Switch()

    def forward(self, x):
        x1 = self.ac(self.conv1(x))
        x2 = self.ac(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.ac(self.conv3(torch.cat((x, x1, x2), dim=1)))
        return x3


class Upscale(nn.Module):
    def __init__(self, inc, outc):
        super(Upscale, self).__init__()
        self.deconv = VoxelShuffle(inc, outc, 2)
        self.up = VoxelShuffle(inc, outc, 2)
        self.conv = nn.Conv3d(outc, outc, 3, 1, 1)
        self.conv1 = nn.Conv3d(outc, outc, 3, 1, 1)
        self.conv2 = nn.Conv3d(2 * outc, outc, 3, 1, 1)

    def forward(self, x):
        x1 = F.relu(self.deconv(x))
        x1 = torch.sigmoid(self.conv(self.up(x))) * x1
        x2 = F.relu(self.conv1(x1))
        x3 = self.conv2(torch.cat((x1, x2), dim=1))
        return x3


class FeatureExtractor(nn.Module):
    def __init__(self, inc):
        super(FeatureExtractor, self).__init__()

        self.s = nn.Sequential(*[RDB(inc, 16),
                                 RDB(16, 32),
                                 RDB(32, 64),
                                 RDB(64, 64)])

    def forward(self, x):
        return self.s(x)


from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class AdvancedDomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.conv1 = nn.Conv3d(64, 128, 4, 2, 1)
        self.bn1 = nn.BatchNorm3d(128)
        self.conv2 = nn.Conv3d(128, 256, 4, 2, 1)
        self.bn2 = nn.BatchNorm3d(256)
        self.conv3 = nn.Conv3d(256, 512, 4, 2, 1)
        self.bn3 = nn.BatchNorm3d(512)
        self.lstm = LSTMCell(512, 512, 3)
        self.fc1 = nn.Linear(int(np.prod(config.crop_size)), 1024)
        self.fc2 = nn.Linear(1024, 1)  # choose between source and target

    def forward(self, x):  # x.shape: [batch_size, frames: 4, 64, crop_size[0], crop_size[1], crop_size[2]]
        h = None
        c = None
        for t in range(x.shape[1]):
            f = F.relu(self.bn1(self.conv1(x[:, t, :, :, :, :])))
            f = F.relu(self.bn2(self.conv2(f)))
            f = F.relu(self.bn3(self.conv3(f)))
            h, c = self.lstm(f, h, c)
        x = h.view(h.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1)
        return x


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('fc1', nn.Linear((config.interval+2)*int(np.prod(config.crop_size)) * 64, 100))
        self.domain_classifier.add_module('relu1', nn.ReLU(True))
        self.domain_classifier.add_module('dpt1', nn.Dropout())
        self.domain_classifier.add_module('fc2', nn.Linear(100, 1))

    def forward(self, x):  # x.shape: [batch_size, frames: 4, 64, crop_size[0], crop_size[1], crop_size[2]]
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return self.domain_classifier(x).squeeze(1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.interval = config.interval
        self.scale = config.scale
        self.s = FeatureExtractor(1)

        self.t = nn.Sequential()
        for k in range(0, self.interval):
            self.t.add_module('temporal' + str(k + 1), FeatureExtractor(2))

        if self.scale == 4:
            self.upscaler = nn.Sequential(*[Upscale(64, 64),
                                            nn.ReLU(True),
                                            Upscale(64, 32),
                                            nn.ReLU(True),
                                            nn.Conv3d(32, 1, 3, 1, 1)
                                            ])
        elif self.scale == 8:
            self.upscaler = nn.Sequential(*[Upscale(64 // 2, 64 // 2),
                                            nn.ReLU(True),
                                            Upscale(64 // 2, 32 // 2),
                                            nn.ReLU(True),
                                            Upscale(32 // 2, 16 // 2),
                                            nn.ReLU(True),
                                            nn.Conv3d(16 // 2, 1, 3, 1, 1)
                                            ])

        self.domain_classifier = None

    def load_model(self, model):
        try:
            self.load_state_dict(model)
        except RuntimeError:
            self.domain_classifier = DomainClassifier()
            self.load_state_dict(model)

    def forward(self, s, e, alpha=None):
        features = [self.s(s)]
        for k in range(0, self.interval):
            features.append(self.t._modules['temporal' + str(k + 1)](torch.cat((s, e), dim=1)))
        features.append(self.s(e))
        domain_output = None
        if self.training and config.domain_backprop:
            if self.domain_classifier is None:
                self.domain_classifier = DomainClassifier()
            reverse_features = [ReverseLayerF.apply(i, alpha).unsqueeze(1) for i in features]
            domain_output = self.domain_classifier(torch.cat(reverse_features, dim=1))
        output = torch.cat([self.upscaler(i) for i in features], dim=1)
        return output, domain_output


def prep_model(model):
    model = model.to(config.device)
    model = nn.DataParallel(model)
    model.apply(weights_init_kaiming)
    return model


class MetaClassifier(nn.Module):
    def __init__(self, models: list[Net]):
        super(MetaClassifier, self).__init__()
        self.models = models
        for model in models:
            for p in model.parameters():
                p.requires_grad = False
        self.fc1 = nn.Linear(len(config.pretrain_vars), 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, s, e):
        results = []
        for ind, model in enumerate(self.models):
            results.append(model(s, e).unsqueeze(0))
        x = torch.cat(results, dim=0)
