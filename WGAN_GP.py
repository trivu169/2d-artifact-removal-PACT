"""
Wasserstein GAN with Gradient Clipping for artifact-removal PACT
This code is modified based on the original version from
https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
Please refer to this repo and the original paper https://arxiv.org/abs/1704.00028 for
more details
Author: Tri Vu - Photacoustic Imaging Lab - Duke University
For more information, please contact tqv@duke.edu
"""



from __future__ import print_function, division
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add, MaxPool2D, Conv2DTranspose
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
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
import keras.backend as K
from keras.layers.merge import _Merge
from functools import partial



class WGAN():

    def __init__(self, channels=3, img_size=128, eta=0.0001, beta1=0.5, beta2=0.9, save_log=False, save_img=False, prefix=None,
                 suffix=None, path=None, n_residual_blocks=8, n=100, epochs=500, save_model=None, test_index=None,
                 img_hr = False, cont_train=None, batch_size=10, n_critic=5, l2=20, num_trained_layer=1, cont_epoch=50):

        """
        :param channels:
        :param img_size:
        :param eta:
        :param beta1:
        :param beta2:
        :param save_log:
        :param save_img:
        :param prefix:
        :param suffix:
        :param path:
        :param n_residual_blocks:
        :param n:
        :param epochs:
        :param save_model:
        :param test_index:
        :param img_hr:
        :param cont_train:
        :param batch_size:
        """
        # Input shape
        self.channels = channels
        self.img_size = img_size                 # Artifact height
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.prefix=prefix
        self.suffix = suffix
        self.path = path
        self.save_log = save_log
        self.save_img = save_img
        self.img_shape = (self.img_size, self.img_size, self.channels)
        self.img_hr = img_hr
        if img_hr:
            self.hr_size = self.img_size*4
            self.hr_shape = (self.hr_size, self.hr_size, self.channels)
        self.n_residual_blocks = n_residual_blocks
        self.gf = 64 # # of maps in G first layer
        self.df = 64 # # of maps in D first layer
        self.n = n # # of instances
        self.epochs = epochs
        self.n_features = self.img_size*self.img_size
        self.save_model = save_model
        self.test_index = test_index
        self.full_path = path + prefix + str(self.test_index) + suffix
        self.cont_train = cont_train
        self.cont_epoch = cont_epoch
        self.batch_size = batch_size
        self.n_critic = n_critic
        self.l2 = l2
        self.num_trained_layer = num_trained_layer
        BATCH_SIZE = batch_size

        optimizer = Adam(self.eta, self.beta1, self.beta2)

        def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                                  gradient_penalty_weight):

            """
            Gradient penalty loss function obtained from
            https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py.
            Please refer to this repo for more detailed explanation
            :param y_pred: output from Generator
            :param y_true: artifact-free ground truth
            :param averaged_samples:
            :param gradient_penalty_weight:
            :return:
            """

            # first get the gradients:
            #   assuming: - that y_pred has dimensions (batch_size, 1)
            #             - averaged_samples has dimensions (batch_size, nbr_features)
            # gradients afterwards has dimension (batch_size, nbr_features), basically
            # a list of nbr_features-dimensional gradient vectors

            gradients = K.gradients(y_pred, averaged_samples)[0]

            # compute the euclidean norm by squaring ...
            gradients_sqr = K.square(gradients)

            #   ... summing over the rows ...
            gradients_sqr_sum = K.sum(gradients_sqr,
                                      axis=np.arange(1, len(gradients_sqr.shape)))

            #   ... and sqrt
            gradient_l2_norm = K.sqrt(gradients_sqr_sum)

            # compute lambda * (1 - ||grad||)^2 still for each single sample
            gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)

            # return the mean as loss over all the batch samples
            return K.mean(gradient_penalty)

        class RandomWeightedAverage(_Merge):

            """
            Obtain Random Weight Average between two tensors
            Also adapted from
            https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py.
            Minor change is that batch_size is a part of the input, not a DEFINITION
            Please refer to this repo for more detailed explanation
            """

            def _merge_function(self, inputs):
                weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
                return (weights * inputs[0]) + ((1 - weights) * inputs[1])

        def w_loss(x, y):

            """
            Generate Wasserstein loss between output from Critic (Discriminator) and the label
            """

            return K.mean(x * y)

        # os.makedirs("/logs")
        # We use a pre-trained VGG19 model to extract image features from the high resolution

        # and the generated high resolution images and minimize the mse between them

        # self.vgg = self.build_vgg()
        # self.vgg.trainable = False
        # self.vgg.compile(loss='mse',
        #     optimizer=optimizer,
        #     metrics=['accuracy'])

        # Calculate output shape of D (PatchGAN)

        if self.img_hr:
            patch = int(self.hr_size/2**4)
        else: patch = int(self.img_size / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Build the discriminator (critic in this WGAN-GP case)

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        # self.generator.compile(optimizer=optimizer, loss=['mse'])

        self.generator.trainable = False

        arti_free = Input(shape=self.img_shape)
        g_in_D = Input(shape=self.img_shape)  # Image with artifact input for G to get output
        # for D
        g_out_D = self.generator(g_in_D)  # Corrected image from G
        d_out_fake = self.discriminator(g_out_D)  # Critic of the G output
        d_out_true = self.discriminator(arti_free)

        # The next step is to set up the gradient clipping for the loss of D using
        # randomly averaged sample
        averaged_samples = RandomWeightedAverage()([arti_free, g_out_D])

        averaged_samples_out = self.discriminator(averaged_samples)

        partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=averaged_samples,
                                  gradient_penalty_weight=10)
        partial_gp_loss.__name__ = 'gradient_penalty'

        # After this setup, we have had the averaged_sample for the gradient clipping, together with
        # d_out_fake and d-Out_true as the output of D. Now we are ready to setup and compile D

        self.d_model = Model(inputs=[arti_free, g_in_D],
                             outputs=[d_out_true, d_out_fake, averaged_samples_out])

        # self.d_model = multi_gpu_model(self.d_model, gpus=2)
        self.d_model.compile(optimizer=optimizer, loss=[w_loss, w_loss, partial_gp_loss])
        # Build the generator


        # for layer in self.discriminator.layers:
        #     layer.trainable = False
        # self.discriminator.trainable = False

        arti = Input(shape=self.img_shape)

        # Generate high res. version from low res.

        self.generator.trainable = True

        fake_arti = self.generator(arti)

        self.discriminator.trainable = False
        # Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_arti)

        # print(validity.shape, fake_features.shape)
        self.gen_model = Model(arti, [validity, fake_arti])
        # self.gen_model = multi_gpu_model(self.gen_model, gpus=2)
        self.gen_model.compile(loss=[w_loss, 'mse'],
                               loss_weights=[1, self.l2],
                              optimizer=optimizer) # Combine mse and binary crossentropy for perceptual loss
        # self.gen_model.summary()


        # self.d_model.summary()

    def build_generator(self):
        def conv_layer(x, filters, dropout=0, concat_layer=None):
            # print(x.shape)
            if concat_layer is not None:
                x = Conv2DTranspose(filters, kernel_size=2, strides=1, padding='same')(x)
                # x = LeakyReLU()(x)
                # x = BatchNormalization()(x)
                x = UpSampling2D(size=2)(x)
                x = Concatenate()([x, concat_layer])
                # print(x.shape)
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
            c = BatchNormalization(momentum=0.8)(c)
            # if dropout > 0:
            #     c = Dropout(dropout)(c)

            return c

        # def residual_block(layer_input, filters):
        #
        #     """Residual block described in paper"""
        #
        #     d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
        #     d = Activation('relu')(d)
        #     d = BatchNormalization(momentum=0.8)(d)
        #     d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
        #     d = BatchNormalization(momentum=0.8)(d)
        #     d = Add()([d, layer_input])
        #
        #     return d

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

        # # Pre-residual block
        #
        # c1 = Conv2D(self.gf, kernel_size=9, strides=1, padding='same')(arti)
        # c1 = Activation('relu')(c1)
        #
        # # Propagate through residual blocks
        #
        # r = residual_block(c1, self.gf)
        # for _ in range(self.n_residual_blocks - 1):
        #     r = residual_block(r, self.gf)
        #
        # # Post-residual block
        #
        # c2 = Conv2D(self.gf, kernel_size=3, strides=1, padding='same')(r)
        # c2 = BatchNormalization(momentum=0.8)(c2)
        # c2 = Add()([c2, c1])

        down1 = conv_layer(arti, self.gf,0)
        down1_mp = MaxPool2D(pool_size=(2, 2))(down1)
        down2 = conv_layer(down1_mp, 2*self.gf)
        down2_mp = MaxPool2D(pool_size=(2, 2))(down2)
        down3 = conv_layer(down2_mp, 4*self.gf)
        down3_mp = MaxPool2D(pool_size=(2, 2))(down3)
        down4 = conv_layer(down3_mp, 8*self.gf)
        down4_mp = MaxPool2D(pool_size=(2, 2))(down4)
        down5 = conv_layer(down4_mp, 16*self.gf)

        # print(down5.shape)
        up4 = conv_layer(down5, 8*self.gf, concat_layer=down4)
        up3 = conv_layer(up4, 4*self.gf, concat_layer=down3)
        up2 = conv_layer(up3, 2*self.gf, concat_layer=down2)
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

    def build_discriminator(self):
        def d_block(layer_input, filters, strides=1, bn=False):

            """Discriminator layer"""

            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same',
                       kernel_initializer='he_normal')(layer_input)
            d = LeakyReLU()(d)

            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input img

        if self.img_hr:
            d0 = Input(shape=self.hr_shape)
        else: d0 = Input(shape=self.img_shape)
        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df*2)
        d4 = d_block(d3, self.df*2, strides=2)
        d5 = d_block(d4, self.df*4)
        d6 = d_block(d5, self.df*4, strides=2)
        d7 = d_block(d6, self.df*8)
        d8 = d_block(d7, self.df*8, strides=2)
        # d8 = Flatten()(d8)
        d9 = Dense(self.df*16, kernel_initializer='he_normal')(d8)
        d10 = LeakyReLU()(d9)

        validity = Dense(1, kernel_initializer='he_normal')(d10)

        return Model(d0, validity)



    def train(self, sample_interval=10, verbose=1):
        """
        Main training for WGAN-GP of PACT
        :param sample_interval: the time step to record the log and images of the model
        :return: param counts and total training time of this model
        """

        batch_size = self.batch_size

        """
        Pre-process log and image folders
        """
        if self.save_model and self.cont_train is None:
            """ If the model is new, create new dir to save G and D """
            current_directory = os.getcwd()
            trained_path = os.path.join(current_directory, "model", self.save_model, "trained model")
            os.makedirs(trained_path)

        if self.cont_train is not None:
            """ If continue to train the mode, load all the models in this GAN """
            current_directory = os.getcwd()
            trained_path = os.path.join(current_directory, "model", self.cont_train, "trained model",
                                        "epoch" + str(self.cont_epoch - 1))
            self.generator.load_weights(trained_path + "/generator.h5")
            self.d_model.load_weights(trained_path + "/discriminator.h5")

            for layer in self.generator.layers[:-self.num_trained_layer]:
                layer.trainable = False

            for layer in self.generator.layers:
                print(layer, layer.trainable)

            print(self.generator.layers[round(len(self.generator.layers)/2)-4])

            raise AttributeError('Stop.')
        if self.save_img:
            if self.test_index != 'phantom':
                a_o = rmf.readHilbertMat(self.full_path)
                a_f = rmf.readTrueMat(self.full_path)
                # if self.channels == 1:
                # a_o = np.dstack((a_o, a_o, a_o))
                a_o = a_o.reshape([1, self.img_size, self.img_size, 1])

            current_directory = os.getcwd()
            if self.cont_train is None:
                img_path = os.path.join(current_directory, "model", self.save_model, "image")
                os.makedirs(img_path)
            else:
                img_path = os.path.join(current_directory, "model", self.cont_train, "image")
            to_save_img = True
        else:
            to_save_img = False

        loss_report = []
        n_batches = int(np.ceil(self.n / batch_size))

        if self.save_log:
            current_directory = os.getcwd()
            log_path = os.path.join(current_directory, "model", self.save_model, "log")
            os.makedirs(log_path)
            write = tf.summary.FileWriter(log_path)

        gloss = None
        gloss_summary = tf.Summary()
        gloss_summary.value.add(tag='Total Generator Loss', simple_value=gloss)

        gloss_wl = None
        gloss_wl_summary = tf.Summary()
        gloss_wl_summary.value.add(tag='Generator W Loss', simple_value=gloss_wl)

        gloss_mse = None
        gloss_mse_summary = tf.Summary()
        gloss_mse_summary.value.add(tag='Generator MSE Loss', simple_value=gloss_mse)

        dloss = None
        dloss_summary = tf.Summary()
        dloss_summary.value.add(tag='Total Discriminator Loss', simple_value=dloss)

        dloss_real = None
        dloss_real_summary = tf.Summary()
        dloss_real_summary.value.add(tag='Discriminator Loss (real)', simple_value=dloss_real)

        dloss_fake = None
        dloss_fake_summary = tf.Summary()
        dloss_fake_summary.value.add(tag='Discriminator Loss (fake)', simple_value=dloss_fake)

        gploss = None
        gploss_summary = tf.Summary()
        gploss_summary.value.add(tag='Gradient Penalty Loss', simple_value=gploss)

        """
        Training start
        """
        positive_y = np.ones((self.batch_size, ) + self.disc_patch, dtype=np.float32)
        negative_y = -positive_y
        dummy_y = np.zeros((self.batch_size, ) + self.disc_patch, dtype=np.float32)

        iter_count = 0
        start_epoch = 0
        if self.cont_train is not None:
            iter_count = self.cont_epoch * n_batches
            start_epoch = self.cont_epoch
        initialTime = time.time()
        for epoch in range(self.epochs):
            preTrainTime = time.time()
            idx = gb.ShuffleIdx(self.n)
            idx_d = np.random.randint(1, self.n, batch_size*self.n_critic)
            # ----------------------
            #  Train Discriminator
            # ----------------------

            for i in range(n_batches):
                iter_count += 1

                if verbose == 1:
                    print('Epoch: ' + str(epoch) + ', iteration: ' + str(iter_count))
                # print(ii)

                # arti, arti_free = np.array(arti), np.array(arti_free)
            # From low res. image generate high res. version

                for i_critic in range(self.n_critic):
                    ii = gb.getBatchIndices(i_critic, idx_d, batch_size)
                    arti, arti_free = gb.getBatchTrain(ii, batch_size, self.n_features, self.path,
                                                       self.prefix, self.suffix,
                                                       channels=self.channels, hr=self.img_hr)
                    d_loss = self.d_model.train_on_batch([arti_free, arti],
                                                [positive_y, negative_y, dummy_y])

                ii = gb.getBatchIndices(i, idx, batch_size)
                arti, arti_free = gb.getBatchTrain(ii, batch_size, self.n_features, self.path, self.prefix, self.suffix,
                                               channels=self.channels, hr=self.img_hr)

                g_loss = self.gen_model.train_on_batch(arti, [positive_y, arti_free])

                if self.save_log:
                    dloss_summary.value[0].simple_value = d_loss[0]
                    write.add_summary(dloss_summary, iter_count - 1)

                    dloss_real_summary.value[0].simple_value = d_loss[1]
                    write.add_summary(dloss_real_summary, iter_count - 1)

                    dloss_fake_summary.value[0].simple_value = d_loss[2]
                    write.add_summary(dloss_fake_summary, iter_count - 1)

                    gloss_summary.value[0].simple_value = g_loss[0]
                    write.add_summary(gloss_summary, iter_count - 1)

                    gloss_wl_summary.value[0].simple_value = g_loss[1]
                    write.add_summary(gloss_wl_summary, iter_count - 1)

                    gloss_mse_summary.value[0].simple_value = g_loss[2]
                    write.add_summary(gloss_mse_summary, iter_count - 1)

                    gploss_summary.value[0].simple_value = d_loss[3]
                    write.add_summary(gploss_summary, iter_count - 1)

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
                    self.generator.save(save + "/generator.h5")
                    self.d_model.save(save + "/discriminator.h5")
                    # self.combined.save(save + "/combined.h5")

                if verbose == 1:
                    print("Remaining training time: " + str(remainingTime))
                    print(epoch)
                    print("G loss: " + str(g_loss))
                    print("D loss: " + str(d_loss))

                # np.savetxt(r"lossArray_" + datetime.datetime.now().strftime(
                #     "%Y%m%d") + "_2.txt", loss_report)

                if to_save_img and self.test_index != 'phantom':
                    fake_hr = self.generator.predict(a_o, steps=1)
                    print(fake_hr.shape)

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
                        h = plt.imshow(fake_hr[0,:,:,:])
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
                        h = plt.imshow(a_o.reshape([self.img_size,self.img_size]))
                        ax = fig.add_subplot(1, 3, 3)
                        ax.set_title("After GAN All Channels - epoch %i" % epoch)
                        if self.img_hr:
                            h = plt.imshow(fake_hr.reshape([self.hr_size,self.hr_size]))
                        else: h = plt.imshow(fake_hr.reshape([self.img_size,self.img_size]))

                    # Save these images to file if save_img is given
                    if self.save_img:
                        plt.savefig(img_path + '/' + self.save_model + "_epoch%i" % epoch + ".png")

                if self.test_index == 'phantom':
                    PREDICT_IMG_PATH = "D:/OneDrive - Duke University/PI Lab/Limited-View DL/Needed results/phantom.mat"
                    data = sio.loadmat(PREDICT_IMG_PATH)
                    # ground_truth = data['p0_true'] # For disc data
                    # arti = data['p0_recons']
                    ground_truth = data['p0_TV_DL']  # For phantom data
                    arti = data['p0_DAS']
                    arti = arti.reshape([1, self.img_size, self.img_size, 1])
                    arti = (arti - arti.min()) / arti.max()
                    pred = self.gen_model.predict(arti, steps=1)
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
                    plt.savefig(img_path + '/' + self.save_model + "_epoch%i" % epoch + ".png")

            if epoch == self.epochs-1:
                if self.save_log:
                    fig = plt.figure()
                    plt.plot(loss_report)
                    plt.ylabel('Generator Loss')
                    plt.xlabel('Epoch')
                    # plt.savefig(self.save_log + "loss_report_testSRGAN_SR_BLdisc_" + datetime.datetime.now().strftime(
                    #         "%Y%m%d-%H%M%S") + ".png")

        totalTrainingTime = (time.time() - initialTime) / 60

        return totalTrainingTime, self.gen_model.count_params()+self.d_model.count_params(), iter_count, self.generator