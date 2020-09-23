import U_Net
import scipy.io as sio
import matplotlib.pyplot as plt
import datetime
from keras.models import load_model
import WGAN_GP
import numpy as np
import scipy.misc as sm
import os
import time
import keras.backend as K

from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_mse as mse
from scipy.misc import imresize

""" Choose which gpu to run the training """
gpu = 1  # 0 for first gpu, 1 for 2nd gpu, 2 for both
if gpu == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif gpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

"""Initialize params"""

MODEL = 'wgan-gp'
# PREFIX = "homo_2D_nne_"
PREFIX = "homo_2D_disc_"
# PATH = "../Data Generation/2D homo nne dir filtered rotated/"
PATH = "../Data Generation/2D homo disc BL lr/"
SUFFIX = ".mat"
EPOCH = 100
SAVE_LOG = True
SAVE_IMG = True
N = 4000
IMG_SIZE = 256
SAVE_MODEL_PATH = "./model/"
SAVE_MODEL_NAME = MODEL + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # + ".h5"
SAVE_MODEL_PATH = SAVE_MODEL_PATH+SAVE_MODEL_NAME
# LOAD_MODEL_PATH = "./trained model/SRGAN_PACT_SRUnet_nne_20181102-163119/epoch24/generator.h5"
MODEL_NAME = 'wgan-gp20200511-135418'
LOAD_MODEL_PATH = './model/' + MODEL_NAME + '/trained model/epoch59/generator.h5'
CONT_MODEL_NAME = None
CONT_EPOCH = 50
TEST_INDEX = 10
PREDICT_IMG_PATH = PATH+PREFIX+str(TEST_INDEX)+SUFFIX
# PREDICT_IMG_PATH = "D:/OneDrive - Duke University/PI Lab/Limited-View DL/Needed results/phantom.mat"
# TEST_INDEX = "phantom" # Use number for simulated data
HR = False

""" Below are the execution options available for this code 
    - TRAIN_FLAG: train the model if True. Specify SAMPLE_INTERVAL for sample result and BATCH_SIZE for each iteration
    - TEST_FLAG: test the model on a single data if True
    - EVAL_FLAG: generate SSIM result of the model given the test range. Specify EVAL_IMG_PATH to the data folder for 
    evalution and the TEST_RANGE indices
    - INVIVO_FLAG: test the model on the invivo data. Specify INVIVO_PATH to the invivo data.
    - LEARNING_CURVE: plot the SSIM wrt to increased training size
"""

TRAIN_FLAG = False

SAMPLE_INTERVAL = 10
BATCH_SIZE = 5

TEST_FLAG = False
EVAL_FLAG = False
INVIVO_FLAG = True
PLOT_FLAG = False
LEARNING_CURVE = False

INVIVO_PATH = './'
INVIVO_NAME = 'pa_3D_2'  # Name of the variable corresponding to the image matrix
# CONT_TRAIN_FLAG = False

# EVAL_IMG_PATH = "../Data Generation/2D homo nne dir filtered rotated/homo_2D_nne_"
EVAL_IMG_PATH = "./2D homo disc BL lr/homo_2D_disc_"
# VALIDATION_RANGE = [7201, 9000]
VALIDATION_RANGE = [4501, 5000]

PLOT_PATH = "./"

""" Execution for each options """
if TRAIN_FLAG:
    """Train model"""

    # For training test (no model saving and result recording)

    # For unet training
    if MODEL == 'unet':
        model = U_Net.UNet(prefix=PREFIX, path=PATH, suffix=SUFFIX, epochs=EPOCH, save_img=SAVE_IMG,
                           save_model=SAVE_MODEL_NAME, save_log=SAVE_LOG, n=N, test_index=TEST_INDEX,
                             channels=1, img_hr=HR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

    # For wgan-gp training
    if MODEL == 'wgan-gp':
        LAMBDA = 20
        model = WGAN_GP.WGAN(prefix=PREFIX, path=PATH, suffix=SUFFIX, epochs=EPOCH, save_img=SAVE_IMG,
                             save_model=SAVE_MODEL_NAME, save_log=SAVE_LOG, n=N, test_index=TEST_INDEX,
                             channels=1, img_hr=HR, img_size=IMG_SIZE, batch_size=BATCH_SIZE, n_critic=5, l2=LAMBDA,
                             cont_train=CONT_MODEL_NAME, cont_epoch=CONT_EPOCH)

    trainingTime, totalParams, iter_count, _ = model.train(sample_interval=SAMPLE_INTERVAL)

    # Create README for the model
    if CONT_MODEL_NAME is not None:
        f = open(SAVE_MODEL_PATH + '\README_cont.md', "w+")
        f.write('Continue model from: ' + CONT_MODEL_NAME + '\n')
    else:
        f = open(SAVE_MODEL_PATH + '\README.md', 'w+')
    f.write('MODEL INFORMATION:\n')
    f.write('Model name:    ' + SAVE_MODEL_NAME + '\n')
    f.write('Epochs:    ' + str(EPOCH) + '\n')
    f.write('Training size: ' + str(N) + '\n')
    f.write('Training data name: ' + PATH + '\n')
    f.write('Total parameters:  ' + str(totalParams) + '\n')
    f.write('Total training time:   ' + str(trainingTime) + '\n')
    f.write('Number of iterations:  ' + str(iter_count) + '\n')
    f.write('Additional notes: \n')
    if MODEL == 'wgan-gp':
        f.write('Lambda_2 = ' + str(LAMBDA) + '\n')
    # f.write('Revised nne data with directivity\nwith 50 first frame and 30 last frames removed in each nne stack\n')  # Put additional notes here
    # f.write('Inputs were rotated 90 degrees to make them longitudinal vessels\nAfter-Hilbert-transform input\n')
    f.close()

if TEST_FLAG:
    def w_loss(x, y):
        """
        Generate Wasserstein loss between output from Critic (Discriminator) and the label
        """

        return K.mean(x * y)

    # Load model and data images
    if MODEL == 'wgan-gp':
        model = load_model(LOAD_MODEL_PATH, custom_objects={'w_loss': w_loss})
    else:
        model = load_model(LOAD_MODEL_PATH)
    data = sio.loadmat(PREDICT_IMG_PATH)
    ground_truth = data['p0_true'] # For disc data
    arti = data['p0_hil']
    # print(data['DATA_SNR'])
    # ground_truth = data['p0_TV_DL']  # For phantom data
    # arti = data['p0_DAS']
    arti = arti.reshape([1, IMG_SIZE, IMG_SIZE, 1])
    testTime = time.time()
    pred = model.predict(arti, steps=1)
    pred = pred.reshape([IMG_SIZE,IMG_SIZE])
    runTime = (time.time() - testTime)
    print('Run Time: ', str(runTime))

    # Display imgs
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 3, 1)
    ax.set_title("Ground Truth")
    h = plt.imshow(ground_truth)
    plt.colorbar(shrink=0.5)
    ax = fig.add_subplot(1, 3, 2)
    ax.set_title("Original Image with Artifacts")
    h = plt.imshow(arti.reshape([IMG_SIZE, IMG_SIZE]))
    plt.colorbar(shrink=0.5)
    ax = fig.add_subplot(1, 3, 3)
    ax.set_title(MODEL)
    h = plt.imshow(pred)
    plt.colorbar(shrink=0.5)
    plt.show()

    # Setup up path to save mat
    get_path = os.path.split(LOAD_MODEL_PATH)
    get_path = get_path[0]
    sio.savemat(get_path+'/'+PREFIX+str(TEST_INDEX), dict([('p0_out', pred), ('p0_recon', arti.reshape([256, 256])),
                                                       ('p0_true', ground_truth), ('model', LOAD_MODEL_PATH)]))

def getResults(data, model):
    if HR == False:
        ground_truth = data['p0_true']
        das = data['p0_recons']
    else:
        ground_truth = data['p0_hr']
        das = imresize(data['p0_recons'], [512, 512])
    arti = data['p0_hil']
    arti = arti.reshape([1, 256, 256, 1])
    pred = model.predict(arti, steps=1)
    if HR == True:
        pred = pred.reshape([512, 512])
    else:
        pred = pred.reshape([256, 256])

    temp = ssim(ground_truth, pred, data_range=pred.max()-pred.min())
    temp2 = psnr(ground_truth, pred, data_range=pred.max()-pred.min())
    temp3 = mse(ground_truth, pred)

    return temp, temp2, temp3

""" For running the SSIM """
if EVAL_FLAG:
    if TEST_FLAG == False:
        model = load_model(LOAD_MODEL_PATH)
    ss_out, ss_das, ss_mse = [], [], []

    for i in range(VALIDATION_RANGE[0], VALIDATION_RANGE[1]+1):
        print(str(i))
        try:
            data = sio.loadmat(EVAL_IMG_PATH + str(i) + SUFFIX)
            temp, temp2, temp3 = getResults(data, model)
        except:
            print('zlib.error -3 occurred at file: ' + str(i))
            data = sio.loadmat(EVAL_IMG_PATH + str(1) + SUFFIX)
            temp, temp2, temp3 = getResults(data, model)

        ss_out.append(temp)
        ss_das.append(temp2)
        ss_mse.append(temp3)

            # snr.append(data['DATA_SNR'])

    sio.savemat(r"./eval/" + MODEL + ".mat", dict([('ssim', np.asmatrix(ss_out)), ('psnr', np.asmatrix(ss_das)),
                                                   ('mse', np.asmatrix(ss_mse))]))
    # sio.savemat('RA4paaReconPINet_20181102-163119-24_test.mat', dict([('p0_SRUNet', recon)]))

    print("Average SSIM for PI: ", np.mean(ss_out))
    print("Std SSIM for PI: ", np.std(ss_out))
    print("Average SSIM for DAS: ", np.mean(ss_das))
    print("Std SSIM for DAS: ", np.std(ss_das))

if INVIVO_FLAG:
    model = load_model(LOAD_MODEL_PATH)
    data = sio.loadmat(INVIVO_PATH)
    arti = data[INVIVO_NAME]
    # arti = arti-arti.min()
    # arti = arti/arti.max()
    # arti = (arti-arti.min())/(arti.max()-arti.min())
    arti = arti.reshape([arti.shape[0], arti.shape[1], -1])

    # This line is for a single frame extraction
    # arti = arti[:, :, 265]
    # arti = arti.reshape([arti.shape[0], arti.shape[1], 1])
    # arti[:, :] = arti.min()
    # print(arti.shape)

    # Because the shape of the in vivo image is not always a multiple of 128 in each dimension, we have to create a zero
    # -padded matrix to hold it
    if arti.shape[0] % IMG_SIZE == 1:
        num_x = (arti.shape[0]//IMG_SIZE+1)
        dim_x = num_x*IMG_SIZE
    else:
        num_x = arti.shape[0]//IMG_SIZE
        dim_x = arti.shape[0]
    if arti.shape[1] % IMG_SIZE == 1:
        num_y = (arti.shape[1]//IMG_SIZE+1)
        dim_y = num_y*IMG_SIZE
    else:
        num_y = arti.shape[1]//IMG_SIZE
        dim_y = arti.shape[1]
    arti_padded = np.zeros([dim_x, dim_y, arti.shape[2]])
    arti_padded = arti
    # print(num_y, num_x)
    recon = np.ones([dim_x, dim_y, arti.shape[2]])

    # Now split the padded in vivo image into multiple of 128x128 image and parse thru model
    for i in range(0, recon.shape[2]):
        print("Frame: " + str(i))
        for j in range(0, num_x):
            for k in range(0, num_y):
                invivo = arti_padded[j*IMG_SIZE:(j+1)*IMG_SIZE, k*IMG_SIZE:(k+1)*IMG_SIZE, i]
                invivo = invivo.reshape([1, IMG_SIZE, IMG_SIZE,1])
                pred = model.predict(invivo, steps=1)
                recon[j*IMG_SIZE:(j+1)*IMG_SIZE, k*IMG_SIZE:(k+1)*IMG_SIZE, i] = pred.reshape([IMG_SIZE,
                                                                                               IMG_SIZE])

    get_path = os.path.split(INVIVO_PATH)
    invivo_name = get_path[1]
    invivo_name = invivo_name.split('.')
    invivo_name = invivo_name[0]
    get_path = get_path[0]

    sio.savemat(get_path + '/' + invivo_name + '_' + MODEL_NAME,
                dict([('p0_TR', arti),
                      ('p0_recon', recon),
                      # ('x_img0', data['x_img0']),
                      # ('z_img0', data['z_img0']),
                      ('model', LOAD_MODEL_PATH)]))
    # sio.savemat('RA4paaReconPINet_20181102-163119-24_test.mat', dict([('p0_SRUNet', recon)]))
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(np.squeeze(recon[:, :, 10]))
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(np.squeeze(arti[:, :, 10]))
    plt.show()

if PLOT_FLAG:
    plot_list = os.listdir(PLOT_PATH)
    gen_loss = []
    dis_loss = []

    " Make D and G loss arrays from csv files in the PLOT_PATH folder "
    for i in range(len(plot_list)):
        list_i = plot_list[i].split(".")
        list_i = list_i[0].split("_")
        if list_i[0] == 'discriminator':
            dis = np.loadtxt(PLOT_PATH + '/' + plot_list[i], skiprows=1, delimiter=',')
            dis_loss = np.append(dis_loss, dis[:, 2])
        if list_i[0] == 'generator':
            gen = np.loadtxt(PLOT_PATH + '/' + plot_list[i], skiprows=1, delimiter=',')
            gen_loss = np.append(gen_loss, gen[:, 2])

    x = np.arange(len(gen_loss))*SAMPLE_INTERVAL

    " Plot G and D Loss "
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Generator Loss")
    plt.plot(x, gen_loss)
    plt.ylabel("Loss")
    plt.xlabel("Batch Step")
    plt.xlim([min(x), max(x)])
    plt.title("Generator Loss")
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Discriminator Loss")
    plt.plot(x, dis_loss)
    plt.ylabel("Loss")
    plt.xlabel("Batch Step")
    plt.xlim([min(x), max(x)])
    plt.title("Discriminator Loss")
    plt.show()

# TRAINING_SIZE_RANGE = np.arange(3600, 4601, 200)
TRAINING_SIZE_RANGE = np.arange(1000, 3501, 500)
# TRAINING_SIZE_RANGE = [4600]
# VALIDATION_RANGE = [4501, 5000]
VALIDATION_RANGE = [9001, 9500]

if LEARNING_CURVE:
    """ Generate learning curve to determine a reasonable training size """
    SAVE_LOG = False
    SAVE_IMG = False
    SAVE_MODEL_NAME = None
    lc_ssim, lc_range = [], []
    EPOCH = 1
    BATCH_SIZE = 5

    for i in TRAINING_SIZE_RANGE:

        print(i)
        N = i

        # For unet training
        if MODEL == 'unet':
            model = U_Net.UNet(prefix=PREFIX, path=PATH, suffix=SUFFIX, epochs=EPOCH, save_img=SAVE_IMG,
                               save_model=SAVE_MODEL_NAME, save_log=SAVE_LOG, n=N, test_index=TEST_INDEX,
                               channels=1, img_hr=HR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

        # For wgan-gp training
        if MODEL == 'wgan-gp':
            LAMBDA = 20
            model = WGAN_GP.WGAN(prefix=PREFIX, path=PATH, suffix=SUFFIX, epochs=EPOCH, save_img=SAVE_IMG,
                                 save_model=SAVE_MODEL_NAME, save_log=SAVE_LOG, n=N, test_index=TEST_INDEX,
                                 channels=1, img_hr=HR, img_size=IMG_SIZE, batch_size=BATCH_SIZE, n_critic=5,
                                 l2=LAMBDA)

        _, _, _, trained_model = model.train(sample_interval=SAMPLE_INTERVAL, verbose=0)

        ss_ssim = []

        for j in range(VALIDATION_RANGE[0], VALIDATION_RANGE[1] + 1):
            try:
                data = sio.loadmat(EVAL_IMG_PATH + str(j) + SUFFIX)
                temp, temp2, temp3 = getResults(data, trained_model)
            except:
                print('zlib.error -3 occurred at file: ' + str(j))
                data = sio.loadmat(EVAL_IMG_PATH + str(1) + SUFFIX)
                temp, temp2, temp3 = getResults(data, trained_model)

            ss_ssim.append(temp)

        lc_ssim.append(np.mean(ss_ssim))
        del model, trained_model

        sio.savemat("./eval/learning_curve_" + MODEL + "_" + PREFIX + str(TRAINING_SIZE_RANGE[0]) + "_" +
                   str(i) + '_1',
                   dict([('lc', np.asmatrix([lc_ssim]))]))

    plt.plot(TRAINING_SIZE_RANGE, lc_ssim)
    plt.show()

VIS_MODEL = True
if VIS_MODEL:
    model = load_model(LOAD_MODEL_PATH)
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')