'''
Created on Feb 11, 2019

@author: jendrik
'''

from SegNet.Network.Network import SegNet, SegNetDepth
from matplotlib import image as mpimg
import torch.nn.functional as F
from numpy import random
from datetime import datetime
import os
import torch
from torch import nn
from torch import optim
import numpy as np
from torch.autograd import Variable
import functools
from tensorboardX import SummaryWriter
import cv2
import traceback
import logging
logger = logging.getLogger()

def get_trainer(model_class, **kwargs):
    if model_class == SegNet:
        return SegNet(**kwargs)
    else:
        raise NotImplementedError()


class Trainer(object):
    '''
    classdocs
    '''

    def __init__(self, model, log_path, log_prefix):
        '''
        Constructor
        '''
        self.model = model
        self.log_path = log_path
        self.log_prefix = log_prefix

    def train_net(self, x, y):
        raise NotImplementedError()

    def evaluate(self, x, y):
        raise NotImplementedError()

    def auto_train(self, epochs, datasets, data_probs, batches_per_epoch,
                   batches_per_val, patience, dist=None):
        pass


class SegNetTrainer(Trainer):
    def __init__(self, model, optim_class, lr, weight_decay,
                 scheduler_class, log_path, log_prefix,
                 weights=None, **kwargs):
        super().__init__(model, log_path, log_prefix)

        if isinstance(model, SegNet) or isinstance(model, SegNetDepth):
            self.base = model
        else:
            self.base = model.module
        self.model = model
        ctime_str = datetime.now().isoformat()
        self.log_dir = (log_path + ctime_str + f'SegNet'
                        + log_prefix + '/')
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        with open(self.log_dir + 'params.csv', 'w') as f:
            for k in locals():
                f.write(str(k) + '\n')

        self.loss_criterion = nn.BCELoss(weights, reduction='none')
        # self.loss_criterion = nn.CrossEntropyLoss(weights)
        self.optimizer = optim_class(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)
        self.scheduler = scheduler_class(self.optimizer, **kwargs)
        self.yolo_trainable = True
        self.trainPhase = False

        self.logits_initialised = False
        self.running_logits = None

    def one_hot_encode(self, target):
        batch_size, h, w = target.size()
        labels_one_hot = torch.FloatTensor(batch_size, h, w, self.base.number_of_classes).zero_().to(self.base.device)
        labels_one_hot.scatter_(3, target.unsqueeze(-1), 1)
        return labels_one_hot

    def train_net(self, x, target):
        self.model.train(True)
        self.optimizer.zero_grad()  # zero the gradient buffers
        output = self.model(x)
        output = output.permute(0, 2, 3, 1)
        output = F.softmax(output, dim=3)
        labels_one_hot = self.one_hot_encode(target)
        loss = (self.base.number_of_classes *
                (self.loss_criterion(output, labels_one_hot)
                 * (target.unsqueeze(0).unsqueeze(-1) != 0).float()
                 ).mean())
        try:
            acc = self.calculate_acc(output, labels_one_hot, target)
        except:
            traceback.print_exc()
        loss.backward()
        self.optimizer.step()
        return loss.data.cpu().numpy(), acc

    def auto_train(self, epochs, datasets, data_probs, batches_per_epoch,
                   batches_per_val, patience, dist=None):
        switch = False
        count = 0
        cross_entropy_comp = 10000.
        logger.log(100, 'Starting training')
        writer = SummaryWriter(self.log_dir)
        for i in np.arange(epochs):
            with open(self.log_dir+'datasets.csv', 'w') as f:
                for dataset in datasets:
                    f.write(dataset.__class__.__name__ + '\n')
            if i == 0:
                dataset = datasets[0]
            else:
                dataset = random.choice(datasets)
            test_image = dataset.__next__(train=True)
            self.scheduler.step()
            if i == 0:
                self.set_weights('./darknet19_448.weights')
                print('Yolo untrainable')
                self.set_yolo_trainable(False)
                image = Variable(torch.from_numpy(test_image[0]['input_img'])).float().permute(0, 3, 1, 2)
                target = Variable(torch.from_numpy(test_image[1]['output_img'])).long()
                self.train_net(image.to(self.base.device), target.to(self.base.device))
                res = dataset.convert_target_to_image(self.eval(image.to(self.base.device)).data.cpu().numpy()[0])
                mpimg.imsave('./images/initialImage.png', res)  # cv2.addWeighted(res, 0.5, imageTemp, 0.5,0))

            batch_ce = 0
            train_acc = 0
            for j in range(batches_per_epoch):
                number = np.random.randint(0, max(data_probs))
                for k in range(len(data_probs)):
                    if data_probs[k] > number:
                        dataset = datasets[k]
                        break
                data = dataset.__next__(train=True)
                tmp_ce, tmp_acc = self.train_net(
                    Variable(torch.from_numpy(data[0]['input_img'])).float().permute(0, 3, 1, 2).to(self.base.device),
                    Variable(torch.from_numpy(data[1]['output_img'])).long().to(self.base.device))
                print("Step: %d of %d, Train Loss: %g Train Acc %g %s" % (j, batches_per_epoch, tmp_ce, tmp_acc, data[2]))
                # print(data[0]['input_img'].shape, data[1]['output_img'].shape, np.unique(data[0]['input_img']))
                if np.isnan(tmp_ce):
                    raise ValueError(f'cross_entropy is {tmp_ce}, accurcay is {tmp_acc}')
                batch_ce += tmp_ce
                train_acc += tmp_acc
            batch_ce /= batches_per_epoch
            train_acc /= batches_per_epoch
            writer.add_scalar('Train CrossEntropy', batch_ce, i)
            writer.add_scalar('Train Accuracy', train_acc, i)
            test_ce = 0
            test_acc = 0
            for j in range(batches_per_val):
                number = np.random.randint(0, max(data_probs))
                for k in range(len(data_probs)):
                    if data_probs[k] > number:
                        dataset = datasets[k]
                        break
                data = dataset.__next__(train=False)
                print(data[2])
                tmp_ce, tmp_acc = self.val(
                    Variable(torch.from_numpy(data[0]['input_img']),
                             requires_grad=False).float().permute(0, 3, 1, 2).to(
                        self.base.device),
                    Variable(torch.from_numpy(data[1]['output_img']),
                             requires_grad=False).long().to(self.base.device))
                test_ce += tmp_ce
                test_acc += tmp_acc
            test_ce /= batches_per_val
            test_acc /= batches_per_val
            writer.add_scalar('Val CrossEntropy', test_ce, i)
            writer.add_scalar('Val Accuracy', test_acc, i)
            # Add it to the Tensorboard summary writer
            # Make sure to specify a step parameter to get nice graphs over time
            res = dataset.convert_target_to_image(self.eval(image.to(self.base.device)).data.cpu().numpy()[0])
            image_temp = (255.*image[0].permute(1, 2, 0).data.numpy()).astype('uint8')
            if image_temp.shape[0] > res.shape[0]:
                mpimg.imsave('./images/Epoch%04d.png' % i, cv2.addWeighted(res, 0.5, image_temp[::2, ::2], 0.5, 0))
            else:
                mpimg.imsave('./images/Epoch%04d.png' % i, cv2.addWeighted(res, 0.5, image_temp, 0.5, 0))

            count += 1

            if test_ce < (cross_entropy_comp - .001):
                torch.save(self.base.state_dict(), self.log_dir + 'bestModel.ckpt')
                count = 0
                cross_entropy_comp = test_ce
            if count >= patience:
                if not switch:
                    switch = True
                    count = 0
                    patience = 8
                    print('Yolo trainable')
                    self.set_yolo_trainable(True)
                else:
                    break
        writer.close()

    @staticmethod
    def calculate_acc(output, labels_one_hot, target):
        try:
            acc = (torch.argmax(output, dim=3) == torch.argmax(labels_one_hot, dim=3)).float()
            mask = (target != 0).float()
            # print(mask)
            # print(mask.sum())
            acc = acc * mask
            mask_sum = mask.sum().data.cpu()
            # print(mask_sum)
            if mask_sum == 0:
                return 0
            else:
                return (torch.sum(acc) / torch.sum(mask)).data.cpu().numpy()
        except RuntimeError:
            return 0

    def val(self, x, target):
        self.model.train(False)
        self.trainPhase = False
        with torch.no_grad():
            output = self.model(x)
            output = output.permute(0, 2, 3, 1)
            output = F.softmax(output, dim=3)
            labels_one_hot = self.one_hot_encode(target)
            loss = (self.base.number_of_classes *
                    (self.loss_criterion(output, labels_one_hot)
                     * (target.unsqueeze(0).unsqueeze(-1) != 0).float()
                     ).mean())

            try:
                acc = self.calculate_acc(output, labels_one_hot, target)
            except:
                traceback.print_exc()
        return loss.data.cpu().numpy(), acc

    def set_yolo_trainable(self, trainable):
        self.yolo_trainable = trainable
        for i in range(len(self.base.convs)):
            for param in self.base.convs.get(i).parameters():
                param.requires_grad = trainable
            for param in self.base.norms.get(i).parameters():
                param.requires_grad = trainable

    def eval(self, x):
        self.model.train(False)
        with torch.no_grad():
            _, res = nn.Softmax2d()(self.model(x)).max(1)
        return res

    def eval_with_avg(self, x):
        self.model.train(False)
        with torch.no_grad():
            softmax = nn.Softmax2d()(self.model(x))
            if not self.logits_initialised:
                self.running_logits = softmax
                self.logits_initialised = True
            else:
                self.running_logits = .5 * self.running_logits + .5 * softmax
            _, res = self.running_logits.max(1)
        return res

    def set_weights(self, weight_file):
        weight_file = open(weight_file, 'rb')  # ./darknet19_448.weights
        weights_header = np.ndarray(
            shape=(4,), dtype='int32', buffer=weight_file.read(16))
        print('Weights Header: ', weights_header)
        weight_loader = functools.partial(self.load_weights,
                                          weight_file=weight_file)
        print(self.base.state_dict().keys())
        # for param in self.parameters():
        #    print(param)
        state_dict = {}
        for i in range(len(self.base.convs)):
            conv = self.base.convs[i]
            bn = self.base.norms[i]
            weights = weight_loader(int(conv.out_channels), int(conv.kernel_size[0]), True, int(conv.in_channels))
            state_dict['conv%d.weight' % i] = torch.from_numpy(weights['kernel']).to(self.model.device)
            state_dict['bn%d.bias' % i] = torch.from_numpy(weights['bias']).to(self.model.device)
            state_dict['bn%d.weight' % i] = torch.from_numpy(weights['gamma']).to(self.model.device)
            state_dict['bn%d.running_mean' % i] = torch.from_numpy(weights['movingMean']).to(self.model.device)
            state_dict['bn%d.running_var' % i] = torch.from_numpy(weights['movingVariance']).to(self.model.device)
        self.base.load_state_dict(state_dict, strict=False)
        print('Unused Weights: ', len(weight_file.read()) / 4)

    @staticmethod
    def load_weights(filters, size, batch_normalisation, prev_layer_filter, weight_file):
        weights = {}
        weights_shape = (size, size, prev_layer_filter, filters)
        # Caffe weights have a different order:
        darknet_w_shape = (filters, weights_shape[2], size, size)
        weights_size = np.product(weights_shape)
        print(weights_shape)
        print(weights_size)

        conv_bias = np.ndarray(
            shape=(filters,),
            dtype='float32',
            buffer=weight_file.read(filters * 4))
        weights.update({'bias': conv_bias})

        if batch_normalisation:
            bn_weights = np.ndarray(
                shape=(3, filters),
                dtype='float32',
                buffer=weight_file.read(filters * 12))

            weights.update({'gamma': bn_weights[0]})
            weights.update({'movingMean': bn_weights[1]})
            weights.update({'movingVariance': bn_weights[2]})

        conv_weights = np.ndarray(
            shape=darknet_w_shape,
            dtype='float32',
            buffer=weight_file.read(weights_size * 4))

        # DarkNet conv_weights are serialized Caffe-style:
        # (out_dim, in_dim, height, width)
        # We would like to set these to Tensorflow order:
        # (height, width, in_dim, out_dim)
        # TODO: Add check for Theano dim ordering.
        # conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
        weights.update({'kernel': conv_weights})
        return weights

class SegNetDepthTrainer(SegNetTrainer):
    def __init__(self, model, optim_class, lr, weight_decay,
                 scheduler_class, log_path, log_prefix,
                 weights=None, **kwargs):
        super().__init__(model, optim_class, lr, weight_decay,
                         scheduler_class, log_path, log_prefix,
                         weights, ckpt_file, **kwargs)
        ctime_str = datetime.now().isoformat()
        self.log_dir = (log_path + ctime_str + f'SegNetDepth'
                        + log_prefix + '/')
        self.depth_loss_criterion = nn.MSELoss()

    def train_net(self, x, target, depth=None):
        self.model.train(True)
        self.optimizer.zero_grad()  # zero the gradient buffers
        classes, pred_depth = self.model(x)
        classes = classes.permute(0, 2, 3, 1)
        classes = F.softmax(classes, dim=3)
        labels_one_hot = self.one_hot_encode(target)
        mask = (target.unsqueeze(0).unsqueeze(-1) != 0).float()
        loss = (self.base.number_of_classes *
                (self.loss_criterion(classes, labels_one_hot)
                 * mask).mean())
        if depth is not None:
            # depth_mask = (depth < 327).float()
            loss += self.depth_loss_criterion(pred_depth / 350, depth / 350) * 4  # * depth_mask
        try:
            acc = self.calculate_acc(classes, labels_one_hot, target)
        except:
            traceback.print_exc()
        loss.backward()
        self.optimizer.step()
        return loss.data.cpu().numpy(), acc

    def auto_train(self, epochs, datasets, data_probs, batches_per_epoch,
                   batches_per_val, patience, dist=None):
        switch = False
        count = 0
        cross_entropy_comp = 10000.
        logger.log(100, 'Starting training')
        writer = SummaryWriter(self.log_dir)
        for i in np.arange(epochs):
            with open(self.log_dir + 'datasets.csv', 'w') as f:
                for dataset in datasets:
                    f.write(dataset.__class__.__name__ + '\n')
            if i == 0:
                dataset = datasets[0]
            else:
                dataset = random.choice(datasets)
            test_image = dataset.__next__(train=True)
            self.scheduler.step()
            if i == 0:
                self.set_weights('./darknet19_448.weights')
                print('Yolo untrainable')
                self.set_yolo_trainable(False)
                comp_image = Variable(torch.from_numpy(test_image[0]['input_img'])).float().permute(0, 3, 1, 2)
                target = Variable(torch.from_numpy(test_image[1]['output_img'])).long()
                if 'depth_img' in test_image[1]:
                    depth = Variable(torch.from_numpy(test_image[1]['depth_img'])).float().to(self.base.device)
                else:
                    depth = None
                self.train_net(comp_image.to(self.base.device), target.to(self.base.device), depth)
                res, depth = self.eval(comp_image.to(self.base.device))
                res = dataset.convert_target_to_image(res.data.cpu().numpy()[0])
                mpimg.imsave('./images/initialImage.png', res)  # cv2.addWeighted(res, 0.5, imageTemp, 0.5,0))
                mpimg.imsave('./images/initialDepth.png' % i, depth[0].data.cpu().numpy())

            batch_ce = 0
            train_acc = 0
            for j in range(batches_per_epoch):
                number = np.random.randint(0, max(data_probs))
                for k in range(len(data_probs)):
                    if data_probs[k] > number:
                        dataset = datasets[k]
                        break
                data = dataset.__next__(train=True)
                image = Variable(torch.from_numpy(data[0]['input_img'])).float().permute(0, 3, 1, 2)
                target = Variable(torch.from_numpy(data[1]['output_img'])).long()
                if 'depth_img' in data[1]:
                    depth = Variable(torch.from_numpy(data[1]['depth_img'])).float().to(self.base.device)
                else:
                    depth = None
                tmp_ce, tmp_acc = self.train_net(image.to(self.base.device), target.to(self.base.device), depth)
                print("Step: %d of %d, Train Loss: %g Train Acc %g %s" % (
                    j, batches_per_epoch, tmp_ce, tmp_acc, data[2]))
                # print(data[0]['input_img'].shape, data[1]['output_img'].shape, np.unique(data[0]['input_img']))
                if np.isnan(tmp_ce):
                    raise ValueError(f'cross_entropy is {tmp_ce}, accurcay is {tmp_acc}')
                batch_ce += tmp_ce
                train_acc += tmp_acc
            batch_ce /= batches_per_epoch
            train_acc /= batches_per_epoch
            writer.add_scalar('Train CrossEntropy', batch_ce, i)
            writer.add_scalar('Train Accuracy', train_acc, i)
            test_ce = 0
            test_acc = 0
            for j in range(batches_per_val):
                number = np.random.randint(0, max(data_probs))
                for k in range(len(data_probs)):
                    if data_probs[k] > number:
                        dataset = datasets[k]
                        break
                data = dataset.__next__(train=False)
                print(data[2])
                image = Variable(torch.from_numpy(data[0]['input_img'])).float().permute(0, 3, 1, 2)
                target = Variable(torch.from_numpy(data[1]['output_img'])).long()
                if 'depth_img' in data[1]:
                    depth = Variable(torch.from_numpy(data[1]['depth_img'])).float().to(self.base.device)
                else:
                    depth = None
                tmp_ce, tmp_acc = self.val(image.to(self.base.device), target.to(self.base.device), depth)
                test_ce += tmp_ce
                test_acc += tmp_acc
            test_ce /= batches_per_val
            test_acc /= batches_per_val
            writer.add_scalar('Val CrossEntropy', test_ce, i)
            writer.add_scalar('Val Accuracy', test_acc, i)
            # Add it to the Tensorboard summary writer
            # Make sure to specify a step parameter to get nice graphs over time
            res, depth = self.eval(comp_image.to(self.base.device))
            res = dataset.convert_target_to_image(res.data.cpu().numpy()[0])
            image_temp = (255. * comp_image[0].permute(1, 2, 0).data.numpy()).astype('uint8')
            if image_temp.shape[0] > res.shape[0]:
                mpimg.imsave('./images/Epoch%04d.png' % i, cv2.addWeighted(res, 0.5, image_temp[::2, ::2], 0.5, 0))
            else:
                mpimg.imsave('./images/Epoch%04d.png' % i, cv2.addWeighted(res, 0.5, image_temp, 0.5, 0))
            mpimg.imsave('./images/depth_Epoch%04d.png' % i, depth[0].data.cpu().numpy())

            count += 1

            if test_ce < (cross_entropy_comp - .001):
                torch.save(self.base.state_dict(), self.log_dir + 'bestModel.ckpt')
                count = 0
                cross_entropy_comp = test_ce
            if count >= patience:
                if not switch:
                    switch = True
                    count = 0
                    patience = 8
                    print('Yolo trainable')
                    self.set_yolo_trainable(True)
                else:
                    break
        writer.close()

    def val(self, x, target, depth=None):
        self.model.train(False)
        with torch.no_grad():
            classes, pred_depth = self.model(x)
            classes = classes.permute(0, 2, 3, 1)
            classes = F.softmax(classes, dim=3)
            labels_one_hot = self.one_hot_encode(target)
            loss = (self.base.number_of_classes *
                    (self.loss_criterion(classes, labels_one_hot)
                     * (target.unsqueeze(0).unsqueeze(-1) != 0).float()
                     ).mean())
            if depth is not None:
                loss += self.depth_loss_criterion(pred_depth / 350, depth / 350)

            try:
                acc = self.calculate_acc(classes, labels_one_hot, target)
            except:
                traceback.print_exc()
        return loss.data.cpu().numpy(), acc

    def set_yolo_trainable(self, trainable):
        self.yolo_trainable = trainable
        for i in range(len(self.base.convs)):
            for param in self.base.convs.get(i).parameters():
                param.requires_grad = trainable
            for param in self.base.norms.get(i).parameters():
                param.requires_grad = trainable

    def eval(self, x):
        self.model.train(False)
        with torch.no_grad():
            classes, depth = self.model(x)
            _, res = nn.Softmax2d()(classes).max(1)
        return res, depth

    def eval_with_avg(self, x):
        self.model.train(False)
        with torch.no_grad():
            classes, depth = self.model(x)
            softmax = nn.Softmax2d()(classes)
            if not self.logits_initialised:
                self.running_logits = softmax
                self.logits_initialised = True
            else:
                self.running_logits = .5 * self.running_logits + .5 * softmax
            _, res = self.running_logits.max(1)
        return res, depth

    def set_weights(self, weight_file):
        weight_file = open(weight_file, 'rb')  # ./darknet19_448.weights
        weights_header = np.ndarray(
            shape=(4,), dtype='int32', buffer=weight_file.read(16))
        print('Weights Header: ', weights_header)
        weight_loader = functools.partial(self.load_weights,
                                          weight_file=weight_file)
        print(self.base.state_dict().keys())
        # for param in self.parameters():
        #    print(param)
        state_dict = {}
        for i in range(len(self.base.convs)):
            conv = self.base.convs[i]
            bn = self.base.norms[i]
            weights = weight_loader(int(conv.out_channels), int(conv.kernel_size[0]), True, int(conv.in_channels))
            state_dict['conv%d.weight' % i] = torch.from_numpy(weights['kernel']).to(self.model.device)
            state_dict['bn%d.bias' % i] = torch.from_numpy(weights['bias']).to(self.model.device)
            state_dict['bn%d.weight' % i] = torch.from_numpy(weights['gamma']).to(self.model.device)
            state_dict['bn%d.running_mean' % i] = torch.from_numpy(weights['movingMean']).to(self.model.device)
            state_dict['bn%d.running_var' % i] = torch.from_numpy(weights['movingVariance']).to(self.model.device)
        self.base.load_state_dict(state_dict, strict=False)
        print('Unused Weights: ', len(weight_file.read()) / 4)

    @staticmethod
    def load_weights(filters, size, batch_normalisation, prev_layer_filter, weight_file):
        weights = {}
        weights_shape = (size, size, prev_layer_filter, filters)
        # Caffe weights have a different order:
        darknet_w_shape = (filters, weights_shape[2], size, size)
        weights_size = np.product(weights_shape)
        print(weights_shape)
        print(weights_size)

        conv_bias = np.ndarray(
            shape=(filters,),
            dtype='float32',
            buffer=weight_file.read(filters * 4))
        weights.update({'bias': conv_bias})

        if batch_normalisation:
            bn_weights = np.ndarray(
                shape=(3, filters),
                dtype='float32',
                buffer=weight_file.read(filters * 12))

            weights.update({'gamma': bn_weights[0]})
            weights.update({'movingMean': bn_weights[1]})
            weights.update({'movingVariance': bn_weights[2]})

        conv_weights = np.ndarray(
            shape=darknet_w_shape,
            dtype='float32',
            buffer=weight_file.read(weights_size * 4))

        # DarkNet conv_weights are serialized Caffe-style:
        # (out_dim, in_dim, height, width)
        # We would like to set these to Tensorflow order:
        # (height, width, in_dim, out_dim)
        # TODO: Add check for Theano dim ordering.
        # conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
        weights.update({'kernel': conv_weights})
        return weights


