'''
Created on Jun 8, 2017

@author: jendrik
'''

import numpy as np
import functools

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch import optim
import torch.nn.init as init
#from tf.contrib.keras import MaxPooling2D


class SegNet(nn.Module):
    
    def __init__(self, number_of_classes, dropProb=.4, ckpt_file=None):
        super(SegNet, self).__init__()
        self.number_of_classes = number_of_classes
        self.convs = {}
        self.norms = {}
        self.activations = {}
        self.drops = {}
        self.pools = {}
        self.deconvs = {}
        self.dec_norms = {}
        self.dec_activations = {}
        self.dec_drops = {}
        self.upscales = {}

        # Build Yolov2 as feature extractor
        in_channels = 3
        layer_sizes_yolo = {i: size for i, size in enumerate(
            [32, 64, 128, 64, 128, 256, 128, 256, 512, 256, 512, 256, 512, 1024, 512, 1024])}
        kernel_sizes_yolo = {i: size for i, size in enumerate([3, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 1, 3, 3, 1, 3])}
        max_pools_yolo = [0, 1, 4, 7, 12]
        self.feed_outs_yolo = [1, 4, 7, 12]
        feed_sizes = {}
        drops_yolo = [4, 7, 12]
        for i, size in layer_sizes_yolo.items():
            if i in self.feed_outs_yolo:
                feed_sizes[i] = size
                print(i, size)
            conv = nn.Conv2d(in_channels=in_channels, out_channels=size,
                             kernel_size=kernel_sizes_yolo[i],
                             stride=1, bias=False)
            setattr(self, "conv" + str(i), conv)

            self.convs.update({i: conv})

            bn = nn.BatchNorm2d(size)
            self.norms.update({i: bn})
            setattr(self, "bn"+str(i), bn)

            act = nn.LeakyReLU(negative_slope=0.1)
            self.activations.update({i: act})
            if i in max_pools_yolo:
                pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                self.pools.update({i: pool})
            if i in drops_yolo:
                drop = nn.Dropout2d(dropProb)
                setattr(self, "drop" + str(i), drop)
                self.drops.update({i: drop})
            in_channels = size

        # Build deconvolution
        layer_sizes_dec = {i: size for i, size in enumerate(
            [1024, 512, 512, 512, 256, 512, 256, 256, 512, 256, 128, 256, 128, 128, 64, self.number_of_classes]
        )}
        kernel_sizes_dec = {i: size for i, size in enumerate([3, 1, 3, 3, 1, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 3])}
        upscale = [2, 7, 10, 13, 14]
        self.feed_in = [2, 7, 10, 13]
        feed_in_sizes = {}
        feed_size_values = feed_sizes.values()
        feed_size_values.sort()
        print(feed_sizes.values())
        for i, size in zip(upscale, feed_size_values[::-1]):
            feed_in_sizes[i] = size
        drops_dec = [0, 2, 7, 10, 13]
        for i, size in layer_sizes_dec.items():
            if (i-1) in self.feed_in:
                in_channels += feed_in_sizes[i-1]
                print(feed_in_sizes[i-1])
            deconv = nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=size,
                kernel_size=kernel_sizes_dec[i], stride=1, bias=False)
            setattr(self, "deconv" + str(i), deconv)

            self.deconvs.update({i: deconv})

            bn = nn.BatchNorm2d(size)
            self.dec_norms.update({i: bn})
            setattr(self, "deconv_bn"+str(i), bn)

            act = nn.ELU()
            self.dec_activations.update({i: act})
            if i in upscale:
                upsample = nn.Upsample(scale_factor=2)
                self.upscales.update({i: upsample})
            if i in drops_dec:
                drop = nn.Dropout2d(dropProb)
                setattr(self, "deconv_drop" + str(i), drop)
                self.dec_drops.update({i: drop})
            in_channels = size

        if ckpt_file:
            self.yolo_trainable = False
            self.trainPhase = False
            self.load_state_dict(torch.load(ckpt_file), strict=False)
            for param in self.parameters():
                param.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        in_shape = x.size()
        feed_storer = []
        try:
            for i in self.convs.keys():
                conv = self.convs[i]
                x = conv(x) 
                x = self.norms[i](x)
                if i in self.activations:
                    x = self.activations[i](x)
                if i in self.drops:
                    # otherwise memory issues
                    x = self.drops[i](x)
                if i in self.feed_outs_yolo:
                    feed_storer.append(x)
                if i in self.pools:
                    x = self.pools[i](x)
            for i in self.deconvs.keys():
                if i-1 in self.feed_in:
                    feed = feed_storer.pop()
                    m = nn.ZeroPad2d((feed.size()[3] - x.size()[3], 0,
                                      feed.size()[2] - x.size()[2], 0))
                    x = m(x)
                    x = torch.cat((x, feed), 1)
                if i == len(self.deconvs.keys())-1:
                    index = len(self.deconvs.keys())-1
                    m = nn.ZeroPad2d((in_shape[3] - (self.deconvs[index].kernel_size[0] - 1) - x.size()[3], 0,
                                      in_shape[2] - (self.deconvs[index].kernel_size[1] - 1) - x.size()[2], 0))
                    x = m(x)
                if i in self.dec_drops:
                    # otherwise memory issues
                    x = self.dec_drops[i](x)
                deconv = self.deconvs[i]
                x = deconv(x)
                x = self.dec_norms[i](x)
                if i in self.dec_activations:
                    x = self.dec_activations[i](x)
                if i in self.upscales:
                    x = self.upscales[i](x)
            return x
        except Exception as e:
            print(i, x.size())
            raise e

    @staticmethod
    def num_vectors(x):
        size = x.size()[2:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class SegNetDepth(nn.Module):

    def __init__(self, number_of_classes, dropProb=.4, ckpt_file=None):
        super().__init__()
        self.number_of_classes = number_of_classes
        self.convs = {}
        self.norms = {}
        self.activations = {}
        self.drops = {}
        self.pools = {}
        self.deconvs = {}
        self.dec_norms = {}
        self.dec_activations = {}
        self.dec_drops = {}
        self.upscales = {}

        # Build Yolov2 as feature extractor
        in_channels = 3
        layer_sizes_yolo = {i: size for i, size in enumerate(
            [32, 64, 128, 64, 128, 256, 128, 256, 512, 256, 512, 256, 512, 1024, 512, 1024])}
        kernel_sizes_yolo = {i: size for i, size in enumerate([3, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 1, 3, 3, 1, 3])}
        max_pools_yolo = [0, 1, 4, 7, 12]
        self.feed_outs_yolo = [1, 4, 7, 12]
        feed_sizes = {}
        drops_yolo = [4, 7, 12]
        for i, size in layer_sizes_yolo.items():
            conv = nn.Conv2d(in_channels=in_channels, out_channels=size,
                             kernel_size=kernel_sizes_yolo[i],
                             stride=1, bias=False)
            setattr(self, "conv" + str(i), conv)

            self.convs.update({i: conv})

            bn = nn.BatchNorm2d(size)
            self.norms.update({i: bn})
            setattr(self, "bn" + str(i), bn)

            act = nn.LeakyReLU(negative_slope=0.1)
            self.activations.update({i: act})
            if i in max_pools_yolo:
                pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                self.pools.update({i: pool})
            if i in self.feed_outs_yolo:
                feed_sizes[i] = size
            if i in drops_yolo:
                drop = nn.Dropout2d(dropProb)
                setattr(self, "drop" + str(i), drop)
                self.drops.update({i: drop})
            in_channels = size

        # Build deconvolution
        layer_sizes_dec = {i: size for i, size in enumerate(
            [1024, 512, 512, 512, 256, 512, 256, 256, 512, 256, 128, 256, 128, 128, 96, self.number_of_classes+1])}
        kernel_sizes_dec = {i: size for i, size in enumerate([3, 1, 3, 3, 1, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 3])}
        upscale = [2, 7, 10, 13, 14]
        self.feed_in = [2, 7, 10, 13]
        feed_in_sizes = {}
        for i, size in zip(upscale, list(feed_sizes.values())[::-1]):
            feed_in_sizes[i] = size
        drops_dec = [0, 2, 7, 10, 13]
        for i, size in layer_sizes_dec.items():
            if (i - 1) in self.feed_in:
                in_channels += feed_in_sizes[i - 1]
            deconv = nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=size,
                kernel_size=kernel_sizes_dec[i], stride=1, bias=False)
            setattr(self, "deconv" + str(i), deconv)

            self.deconvs.update({i: deconv})

            bn = nn.BatchNorm2d(size)
            self.dec_norms.update({i: bn})
            setattr(self, "deconv_bn" + str(i), bn)

            act = nn.ELU()
            self.dec_activations.update({i: act})
            if i in upscale:
                upsample = nn.Upsample(scale_factor=2)
                self.upscales.update({i: upsample})
            if i in drops_dec:
                drop = nn.Dropout2d(dropProb)
                setattr(self, "deconv_drop" + str(i), drop)
                self.dec_drops.update({i: drop})
            in_channels = size

        if ckpt_file:
            self.yolo_trainable = False
            self.trainPhase = False
            self.model.load_state_dict(torch.load(ckpt_file), strict=False)
            for param in self.model.parameters():
                param.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        in_shape = x.size()
        feed_storer = []
        try:
            for i in self.convs.keys():
                conv = self.convs[i]
                x = conv(x)
                x = self.norms[i](x)
                if i in self.activations:
                    x = self.activations[i](x)
                if i in self.drops:
                    # otherwise memory issues
                    x = self.drops[i](x)
                if i in self.feed_outs_yolo:
                    feed_storer.append(x)
                if i in self.pools:
                    x = self.pools[i](x)
            for i in self.deconvs.keys():
                if i - 1 in self.feed_in:
                    feed = feed_storer.pop()
                    m = nn.ZeroPad2d((feed.size()[3] - x.size()[3], 0,
                                      feed.size()[2] - x.size()[2], 0))
                    x = m(x)
                    x = torch.cat((x, feed), 1)
                    # print(x.shape)
                if i == len(self.deconvs.keys()) - 1:
                    index = len(self.deconvs.keys()) - 1
                    m = nn.ZeroPad2d((in_shape[3] - (self.deconvs[index].kernel_size[0] - 1) - x.size()[3], 0,
                                      in_shape[2] - (self.deconvs[index].kernel_size[1] - 1) - x.size()[2], 0))
                    x = m(x)
                if i in self.dec_drops:
                    # otherwise memory issues
                    x = self.dec_drops[i](x)
                deconv = self.deconvs[i]
                x = deconv(x)
                x = self.dec_norms[i](x)
                if i in self.dec_activations:
                    x = self.dec_activations[i](x)
                if i in self.upscales:
                    x = self.upscales[i](x)
            classes = x[:, :self.number_of_classes]
            depth = x[:, self.number_of_classes]
            return classes, 350*nn.Sigmoid()(depth)
        except Exception as e:
            print(i, x.size())
            raise e
    
    
    
    
    
    
