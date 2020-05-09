import readMatFile as rMF
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import rescale, resize, downscale_local_mean
import cv2

def ShuffleIdx(i):
    idx = np.random.permutation(i)
    return idx

# Get the shuffled indices of the batch
def getBatchIndices(batch_idx, idx, batch_size):
    idx = idx[batch_idx * batch_size:(batch_idx + 1) * batch_size]
    return idx

# Extract the batch with the shuffled indices
def getBatchTrain(idx, batch_size, n_features, path, prefix, suffix, channels=1,
                  hr=False, rescale_factor=1):

    if hr == False and rescale_factor < 1:
        raise AttributeError(
            'Cannot downscale the image to less than the size of DAS image. Suggested fix: set hr = True.')
    if hr == True and rescale_factor < 0.25:
        raise AttributeError(
            'Cannot downscale the image to less than the size of DAS image. Suggested fix: set rescale_factor >= 0.25.')
    if rescale_factor > 1:
        raise Warning(
            'You are trying to up-scale the image using conventional interpolation. Consider to reset rescale_factor to 1.')
    if channels == 3 and hr == True:
        raise AttributeError(
            'Current version does not support both HR and 3-channel image due to limitation of memory.')

    if channels == 3:
        X = np.ones([batch_size, int(np.sqrt(n_features)), int(np.sqrt(n_features)), 3])
        y = np.ones([batch_size, int(np.sqrt(n_features)), int(np.sqrt(n_features)), 3])
    else:
        X = np.ones([batch_size, int(np.sqrt(n_features)), int(np.sqrt(n_features)), 1])
        if hr:
            y = np.ones([batch_size, int(np.sqrt(n_features)*4*rescale_factor),
                         int(np.sqrt(n_features)*4*rescale_factor), 1])
        else: y = np.ones([batch_size, int(np.sqrt(n_features)), int(np.sqrt(n_features)), 1])

    if batch_size > len(idx):
        batch_size = len(idx)

    for i in range(0,batch_size-1):
        if (idx[i+1]) == 0:
            (idx[i + 1]) = 1 # Prevent the case when i = 0

        full_path = path + prefix + str(idx[i+1]) + suffix

        def get_xy(full_path, rec):
            if hr:
                true = rMF.readHrMat(full_path)
                true = rescale(true, rescale_factor, anti_aliasing=True)
            else: true = rMF.readTrueMat(full_path)

            # true = np.rot90(true)

            # Initiate scaler and fit the data into batch
            # scaler = MinMaxScaler()

            # Fit real part into batch
            # scaler.fit((rec))
            if channels == 3:
                X[i] = np.dstack((rec,rec,rec))
                # X[i] = cv2.cvtColor(rec, cv2.COLOR_GRAY2RGB)
            # scaler.fit((true))
                y[i] = np.dstack((true,true,true))
                # y[i] = cv2.cvtColor(true, cv2.COLOR_GRAY2RGB)
            else:
                X[i] = np.reshape(rec, (int(np.sqrt(n_features)), int(np.sqrt(n_features)), 1))
                if hr:
                    y[i] = np.reshape(true, (int(np.sqrt(n_features)*4*rescale_factor),
                                            int(np.sqrt(n_features)*4*rescale_factor), 1))
                else: y[i] = np.reshape(true, (int(np.sqrt(n_features)),int(np.sqrt(n_features)), 1))
            # Fit imag part into batch
            # scaler.fit(rMF.getFFTImag(rec))
            # X[i+1] = np.reshape(scaler.transform(rMF.getFFTImag(rec)), (n_features))
            # scaler.fit(rMF.getFFTImag(true))
            # y[i+1] = np.reshape(scaler.transform(rMF.getFFTImag(true)), (n_features))

            return X, y

        # print(idx[i+1])
        # Get matrix of both the ground truth and the recon
        try:
            # rec = rMF.readReconsMat(full_path)  # For before-Hilbert input
            rec = rMF.readHilbertMat(full_path)  # For after-Hilbert input
            # rec = np.rot90(rec)
            X, y = get_xy(full_path, rec)
        except:
            print('zlib.error -3 occurred at file: ' + str(idx[i+1]))
            full_path = path + prefix + str(np.random.randint(1, 10)) + suffix
            # rec = rMF.readReconsMat(full_path)  # For before-Hilbert input
            rec = rMF.readHilbertMat(full_path)  # For after-Hilbert input
            # rec = np.rot90(rec)  # Since the nne data has mostly unwanted diving vessels for training, we gonna rotate
            #                      # the input 90 degrees to make them longitudinal vessels
            X, y = get_xy(full_path, rec)

    return X, y

def getBatchTest(batch_size, n_features, path, prefix, suffix,channels=1):
    arr = np.arange(4000,5000)
    idx = np.random.permutation((arr))
    idx = idx[:batch_size]
    if channels == 3:
        X = np.ones([int(np.sqrt(n_features)), int(np.sqrt(n_features)), 3])
        y = np.ones([int(np.sqrt(n_features)), int(np.sqrt(n_features)), 3])
    else:
        X = np.ones([int(np.sqrt(n_features)), int(np.sqrt(n_features)), 1])
        y = np.ones([int(np.sqrt(n_features)), int(np.sqrt(n_features)), 1])

    for i in range(0,batch_size-1):
        if (idx[i+1]) == 0:
            (idx[i + 1]) = 1 # Prevent the case when i = 0

        # Get the path
        full_path = path + prefix + str(idx[i+1]) + suffix

        # Get matrix of both the ground truth and the recon
        rec = rMF.readReconsMat(full_path)
        true = rMF.readTrueMat(full_path)

        # Initiate scaler and fit the data into batch
        if channels == 3:
            X[i] = np.dstack((rec,rec,rec))
        # scaler.fit((true))
        #     X[i] = cv2.cvtColor(rec, cv2.COLOR_GRAY2RGB)
            y[i] = np.dstack((true,true,true))
            # y[i] = cv2.cvtColor(rec, cv2.COLOR_GRAY2RGB)
        else:
            X[i] = rec
            y[i] = true

        # Fit imag part into batch
        # scaler.fit(rMF.getFFTImag(rec))
        # X[i+1] = np.reshape(scaler.transform(rMF.getFFTImag(rec)), (n_features))
        # scaler.fit(rMF.getFFTImag(true))
        # y[i+1] = np.reshape(scaler.transform(rMF.getFFTImag(true)), (n_features))

    return X, y

def getBatchNoI(ii,n_features,batch_size, path, prefix, suffix):
    arr = np.arange(1,ii)
    idx = np.random.permutation((arr))
    idx = idx[:batch_size]
    print(idx)
    X = np.ones([batch_size,n_features])
    y = np.ones([batch_size,n_features])

    for i in range(0,batch_size-1):
        if (idx[i+1]) == 0:
            (idx[i + 1]) = 1 # Prevent the case when i = 0

        # Get the path
        full_path = path + prefix + str(idx[i+1]) + suffix

        # Get matrix of both the ground truth and the recon
        rec = rMF.readReconsMat(full_path)
        true = rMF.readTrueMat(full_path)

        # Initiate scaler and fit the data into batch
        scaler = MinMaxScaler()

        # Fit real part into batch
        scaler.fit((rec))
        X[i] = np.reshape(scaler.transform((rec)),(n_features))
        scaler.fit((true))
        y[i] = np.reshape(scaler.transform((true)),(n_features))

    return y, X

def getBatch(n_features,batch_size, path, prefix, suffix):
    X = np.ones([batch_size,n_features])
    y = np.ones([batch_size,n_features])

    for i in range(0,batch_size-1):

        # Get the path
        full_path = path + prefix + str(i+1) + suffix

        # Get matrix of both the ground truth and the recon
        rec = rMF.readReconsMat(full_path)
        true = rMF.readTrueMat(full_path)

        # Initiate scaler and fit the data into batch
        scaler = MinMaxScaler()

        # Fit real part into batch
        scaler.fit((rec))
        X[i] = np.reshape(scaler.transform((rec)),(n_features))
        scaler.fit((true))
        y[i] = np.reshape(scaler.transform((true)),(n_features))

    return y, X