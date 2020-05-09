"""
UNet for artifact-removal PACT
Author: Tri Vu - Photacoustic Imaging Lab - Duke University
For more information, please contact tqv@duke.edu
"""

from __future__ import print_function, division

import scipy

from keras.datasets import mnist

# from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add, MaxPool2D, Conv2DTranspose
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.layers import Lambda

from keras.applications import VGG19

from keras.models import Sequential, Model

from keras.optimizers import Adam

from keras.backend import expand_dims

import datetime

import matplotlib.pyplot as plt

import sys

# from data_loader import DataLoader

import numpy as np

import scipy.io as sio
import tensorflow as tf

import os

import datetime
import matplotlib.pyplot as plt
import getBatch as gb
import time
import readMatFile as rmf
from keras.models import load_model
from matplotlib.colors import Normalize
import cv2
import tensorflow as tf
# %matplotlib inline

# prefix = "homo_2D_high_random_disc_"
# path = "C:/Users/Tri Vu/OneDrive - Duke University/PI Lab/Limited-View DL/Data Generation/2D homo high disc line/"
# suffix = ".mat"
# full_path = path + prefix + str(200) + suffix
#
# path_to_save = "C:/Users/Tri Vu/OneDrive - Duke University/PI Lab/Limited-View DL/Main Project/images/"

import keras.backend as K


class UNet():

    def __init__(self, channels=3, img_size=128, eta=0.0001, beta1=0.99, save_log=None,
                 save_img=None, prefix=None,
                 suffix=None, path=None, n_residual_blocks=8, n=100, epochs=500, save_model=None, test_index=None,
                 img_hr=False, cont_train=None, batch_size=10):

        # Input shape
        self.channels = channels
        self.img_size = img_size  # Artifact height
        self.eta = eta
        self.beta1 = beta1
        self.prefix = prefix
        self.suffix = suffix
        self.path = path
        self.save_log = save_log
        self.save_img = save_img
        self.img_shape = (self.img_size, self.img_size, self.channels)
        self.img_hr = img_hr
        if img_hr:
            self.hr_size = self.img_size * 4
            self.hr_shape = (self.hr_size, self.hr_size, self.channels)
        self.n_residual_blocks = n_residual_blocks
        self.gf = 64  # # of maps in G first layer
        self.df = 64  # # of maps in D first layer
        self.n = n  # # of instances
        self.epochs = epochs
        self.n_features = self.img_size * self.img_size
        self.save_model = save_model
        self.test_index = test_index
        self.full_path = path + prefix + str(self.test_index) + suffix
        self.cont_train = cont_train
        self.batch_size = batch_size

        optimizer = Adam(self.eta, self.beta1)

        # if self.img_hr:
        #     patch = int(self.hr_size / 2 ** 4)
        # else:
        #     patch = int(self.img_size / 2 ** 4)
        # self.disc_patch = (patch, patch, 1)

        # Build the generator

        self.model = self.build_unet()
        self.model.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        # self.model.summary()

    def build_unet(self):
        def conv_layer(x, filters, dropout=0, concat_layer=None):

            if concat_layer is not None:
                x = Conv2DTranspose(filters, kernel_size=2, strides=1, padding='same')(x)
                x = UpSampling2D(size=2)(x)
                x = Concatenate()([x, concat_layer])

            ## This is the construction of conventional Unet
            c = Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)
            # Use dropout as in this report http://cs230.stanford.edu/files_winter_2018/projects/6937642.pdf and pix2pix paper
            if dropout > 0 and concat_layer is not None:
                c = Dropout(dropout)(c)
            c = Activation('relu')(c)
            # c = LeakyReLU()(c)
            c = BatchNormalization(momentum=0.8)(c)
            c = Conv2D(filters, kernel_size=3, strides=1, padding='same')(c)
            # Use dropout as in this report http://cs230.stanford.edu/files_winter_2018/projects/6937642.pdf and pix2pix paper
            if dropout > 0 and concat_layer is not None:
                c = Dropout(dropout)(c)
            # c = LeakyReLU()(c)
            c = Activation('relu')(c)
            # c = Activation('relu')(c)
            c = BatchNormalization(momentum=0.8)(c)
            # if dropout > 0:
            #     c = Dropout(dropout)(c)

            return c

        # Upsampling layers which are not needed in this project
        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            # u = LeakyReLU()(u)
            return u

        # Free-artifact image input

        arti = Input(shape=self.img_shape)

        down1 = conv_layer(arti, self.gf, 0)
        down1_mp = MaxPool2D(pool_size=(2, 2))(down1)
        down2 = conv_layer(down1_mp, 2 * self.gf)
        down2_mp = MaxPool2D(pool_size=(2, 2))(down2)
        down3 = conv_layer(down2_mp, 4 * self.gf)
        down3_mp = MaxPool2D(pool_size=(2, 2))(down3)
        down4 = conv_layer(down3_mp, 8 * self.gf)
        down4_mp = MaxPool2D(pool_size=(2, 2))(down4)
        down5 = conv_layer(down4_mp, 16 * self.gf)

        # print(down5.shape)
        up4 = conv_layer(down5, 8 * self.gf, concat_layer=down4)
        up3 = conv_layer(up4, 4 * self.gf, concat_layer=down3)
        up2 = conv_layer(up3, 2 * self.gf, concat_layer=down2)
        up1 = conv_layer(up2, self.gf, concat_layer=down1)

        # Upsampling
        if self.img_hr:
            u1 = deconv2d(up1)
            u2 = deconv2d(u1)

            # Generate high resolution output
            gen_af = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)
        else:
            gen_af = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(up1)

        return Model(arti, gen_af)

    def train(self, sample_interval=10, verbose=1):
        batch_size = self.batch_size
        if self.save_model and self.cont_train is None:
            """ If the model is new, create new dir to save G and D """
            current_directory = os.getcwd()
            trained_path = os.path.join(current_directory, "model", self.save_model, "trained model")
            os.makedirs(trained_path)

        if self.cont_train is not None:
            """ If continue to train the mode, load all the models in this GAN """
            current_directory = os.getcwd()
            trained_path = os.path.join(current_directory, "model", self.cont_train, "trained model")
            self.model.load_weights(trained_path+"/model.h5")
            # print(self.generator)

        if self.test_index != 'phantom' and self.save_img:
            a_o = rmf.readReconsMat(self.full_path)
            a_f = rmf.readTrueMat(self.full_path)
            # if self.channels == 1:
            # a_o = np.dstack((a_o, a_o, a_o))
            a_o = a_o.reshape([1, self.img_size,self.img_size, 1])
            current_directory = os.getcwd()
            img_path = os.path.join(current_directory, "model", self.save_model, "image")
            os.makedirs(img_path)
            to_save_img = True
        else: to_save_img = False

        loss_report = []
        n_batches = int(np.ceil(self.n / batch_size))

        if self.save_log:
            current_directory = os.getcwd()
            log_path = os.path.join(current_directory, "model", self.save_model, "log")
            os.makedirs(log_path)
            write = tf.summary.FileWriter(log_path)

        loss = None
        loss_summary = tf.Summary()
        loss_summary.value.add(tag='Training Loss', simple_value=loss)

        iter_count = 0
        initialTime = time.time()
        for epoch in range(self.epochs):
            preTrainTime = time.time()
            idx = gb.ShuffleIdx(self.n)
            # ----------------------
            #  Train Discriminator
            # ----------------------

            for i in range(n_batches):
                iter_count += 1

                if verbose == 1:
                    print('Epoch: ' + str(epoch) + ', iteration: ' + str(iter_count))
                ii = gb.getBatchIndices(i, idx, batch_size)
                arti, arti_free = gb.getBatchTrain(ii, batch_size, self.n_features, self.path, self.prefix, self.suffix,
                                                   channels=self.channels, hr=self.img_hr)

                # arti, arti_free = np.array(arti), np.array(arti_free)
                # From low res. image generate high res. version

                loss = self.model.train_on_batch(arti, arti_free)

                if self.save_log:
                    loss_summary.value[0].simple_value = loss[0]
                    write.add_summary(loss_summary, iter_count-1)

            if (epoch+1) % sample_interval == 0 or epoch == 0:
                remainingTime = (time.time() - preTrainTime) * (self.epochs - epoch) / 60

                if self.save_model is not None:
                    save = os.path.join(trained_path, "epoch%i" % epoch)
                    os.makedirs(save)
                    # if epoch == 0:
                    # current_directory = os.getcwd()
                    # save_trained_path = os.path.join(current_directory, "trained model", self.save_model)
                    # os.makedirs(save_trained_path)
                    # if self.cont_train is None:
                    self.model.save(save + "/model.h5")

                if verbose == 1:
                    print("Remaining training time: " + str(remainingTime))
                    print(epoch)
                    print("Loss: " + str(loss))

                # np.savetxt(r"lossArray_" + datetime.datetime.now().strftime(
                #     "%Y%m%d") + "_2.txt", loss_report)

                if to_save_img and self.test_index != 'phantom':
                    fake_hr = self.model.predict(a_o, steps=1)

                    # Display output image when channels = 3
                    if self.channels == 3:
                        # self.sample_images(epoch)
                        fig = plt.figure(figsize=(20, 10))
                        ax = fig.add_subplot(2, 3, 1)
                        ax.set_title("Ground Truth")
                        h = plt.imshow(a_f)
                        ax = fig.add_subplot(2, 3, 2)
                        ax.set_title("Original Image with Artifacts")
                        h = plt.imshow(a_o[0, :, :, 0])
                        ax = fig.add_subplot(2, 3, 3)
                        ax.set_title("After GAN All Channels - epoch %i" % epoch)
                        h = plt.imshow(fake_hr[0, :, :, :])
                        ax = fig.add_subplot(2, 3, 4)
                        ax.set_title("Channel 1")
                        h = plt.imshow(fake_hr[0, :, :, 0])
                        ax = fig.add_subplot(2, 3, 5)
                        ax.set_title("Channel 2")
                        h = plt.imshow(fake_hr[0, :, :, 1])
                        ax = fig.add_subplot(2, 3, 6)
                        ax.set_title("Channel 3")
                        h = plt.imshow(fake_hr[0, :, :, 2])

                    # Display output image when channels = 1
                    if self.channels == 1:
                        fig = plt.figure(figsize=(20, 10))
                        ax = fig.add_subplot(1, 3, 1)
                        ax.set_title("Ground Truth")
                        h = plt.imshow(a_f)
                        ax = fig.add_subplot(1, 3, 2)
                        ax.set_title("Original Image with Artifacts")
                        h = plt.imshow(a_o.reshape([self.img_size, self.img_size]))
                        ax = fig.add_subplot(1, 3, 3)
                        ax.set_title("After GAN All Channels - epoch %i" % epoch)
                        if self.img_hr:
                            h = plt.imshow(fake_hr.reshape([self.hr_size, self.hr_size]))
                        else:
                            h = plt.imshow(fake_hr.reshape([self.img_size, self.img_size]))

                    # Save these images to file if save_img is given
                    if self.save_img:
                        plt.savefig(
                            img_path + "/output_UNet_BLdisc_" + datetime.datetime.now().strftime(
                                "%Y%m%d-%H%M%S") + "_epoch%i" % epoch + ".png")

                if self.test_index == 'phantom':
                    PREDICT_IMG_PATH = "D:/OneDrive - Duke University/PI Lab/Limited-View DL/Needed results/phantom.mat"
                    data = sio.loadmat(PREDICT_IMG_PATH)
                    # ground_truth = data['p0_true'] # For disc data
                    # arti = data['p0_recons']
                    ground_truth = data['p0_TV_DL']  # For phantom data
                    arti = data['p0_DAS']
                    arti = arti.reshape([1, self.img_size, self.img_size, 1])
                    arti = (arti - arti.min()) / arti.max()
                    pred = self.model.predict(arti, steps=1)
                    pred = pred.reshape([self.hr_size, self.hr_size])
                    fig = plt.figure(figsize=(20, 10))
                    ax = fig.add_subplot(1, 3, 1)
                    ax.set_title("Ground Truth")
                    h = plt.imshow(ground_truth)
                    ax = fig.add_subplot(1, 3, 2)
                    ax.set_title("Original Image with Artifacts")
                    h = plt.imshow(arti.reshape([self.img_size, self.img_size]))
                    ax = fig.add_subplot(1, 3, 3)
                    ax.set_title("After Super-Resolution GAN")
                    h = plt.imshow(pred)
                    # plt.show()
                    plt.savefig(
                        self.save_img + "output_testSRGAN_SRUnet_BLdisc_phantom_" + datetime.datetime.now().strftime(
                            "%Y%m%d-%H%M%S") + "_epoch%i" % epoch + ".png")

            if epoch == self.epochs - 1:
                if self.save_log:
                    fig = plt.figure()
                    plt.plot(loss_report)
                    plt.ylabel('Generator Loss')
                    plt.xlabel('Epoch')
                    # plt.savefig(self.save_log + "loss_report_testSRGAN_SR_BLdisc_" + datetime.datetime.now().strftime(
                    #         "%Y%m%d-%H%M%S") + ".png")

        totalTrainingTime = (time.time() - initialTime) / 60

        return totalTrainingTime, self.model.count_params(), iter_count, self.model
