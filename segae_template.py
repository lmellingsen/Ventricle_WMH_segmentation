import os
import nibabel as nib
import subprocess
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, losses, utils, Input, Model, optimizers
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda, GaussianNoise, Conv3D, BatchNormalization
from  tensorflow.keras.layers import UpSampling3D, concatenate, Multiply
from tensorflow.keras.constraints import NonNeg
from pdb import *


def normalize(data):
        temp = data[np.nonzero(data)]
        q = np.percentile(temp, 99.0)
        temp = temp[temp <= q]
        return data / np.max(temp)

def get_volumes(subject):
    """Here you can write code to load your T1, T2, and FLAIR images, as well as bias corrected T1, T2, and FLAIR images"""
    img = nib.load(filepath)
    images[image_type] = normalize(np.asarray(img.get_fdata()))
    
    data = np.stack((images['t1'], images['t2'], images['flair'],
                    images['t1_n4'], images['t2_n4'], images['flair_n4']), axis=-1)
    return data

def patchify(volumes, patch_shape):
    X, Y, Z, channels = volumes.shape
    x, y, z = patch_shape
    shape = (X - x + 1, Y - y + 1, Z - z + 1, x, y, z, channels)
    X_str, Y_str, Z_str, channels_str = volumes.strides
    strides = (X_str, Y_str, Z_str, X_str, Y_str, Z_str, channels_str)
    return np.lib.stride_tricks.as_strided(volumes, shape=shape, strides=strides)

def sliding_patches(volumes, BSZ):
    volumes_ext = np.stack([np.pad(volumes[..., i], ((BSZ[0] // 2, BSZ[0] //2),
                                                     (BSZ[1] // 2, BSZ[1] //2),
                                                     (BSZ[2] // 2, BSZ[2] //2)),'constant')
                            for i in range(volumes.shape[-1])], -1)
    return patchify(volumes_ext, BSZ)

def get_patch_set(subject, BSZ, stride, training=True):
    Y_list = []
    coordinates = {}

    volumes = get_volumes(subject)
    patches_coord = sliding_patches(volumes, BSZ)
    nonzero_img = np.nonzero(volumes[..., 0])
    coord = []
    coord.append([i for i in range(max(np.min(nonzero_img[0]), int(BSZ[0]/2)),
          min(np.max(nonzero_img[0] + 1), volumes[..., 0].shape[0] - BSZ[0]//2), stride[0])])
    coord.append([i for i in range(max(np.min(nonzero_img[1]), int(BSZ[1]/2)),
                              min(np.max(nonzero_img[1]) + 1, volumes[..., 0].shape[1] - BSZ[1]//2), stride[1])])
    coord.append([i for i in range(max(np.min(nonzero_img[2]), int(BSZ[2]/2)),
                              min(np.max(nonzero_img[2]) + 1, volumes[..., 0].shape[2] - BSZ[2]//2), stride[2])])
    coordinates = np.array(coord)
    seg_patches = np.empty((int(len(coord[0])*len(coord[1])*len(coord[2])), BSZ[0], BSZ[1], BSZ[2], volumes.shape[-1]))
    cnt = 0
    for i in coord[0]:
        for j in coord[1]:
            for k in coord[2]:
                seg_patches[cnt, ...] = patches_coord[i, j, k, ...]
                cnt = cnt+1

    data = seg_patches
    return data, coordinates


    def assemble_patches(prediction, coordinates, volume_shape):
        num_abundances = prediction.shape[-1]
        dim1 = coordinates[0]
        dim2 = coordinates[1]
        dim3 = coordinates[2]
        cnt=0
        abundances = np.zeros(volume_shape + (num_abundances,))
        weights = np.zeros(volume_shape)
        hBSZ = tuple(i//2 for i in prediction.shape[1:-1])
        for i in dim1:
            for j in dim2:
                for k in dim3:
                    abundances[i-hBSZ[0]:i+hBSZ[0], j-hBSZ[1]:j+hBSZ[1], k-hBSZ[2]:k+hBSZ[2], :] = \
                        abundances[i-hBSZ[0]:i+hBSZ[0], j-hBSZ[1]:j+hBSZ[1], k-hBSZ[2]:k+hBSZ[2], :] \
                        + prediction[cnt, ...]
                    weights[i-hBSZ[0]:i+hBSZ[0], j-hBSZ[1]:j+hBSZ[1], k-hBSZ[2]:k+hBSZ[2]] = \
                                    weights[i-hBSZ[0]:i+hBSZ[0], j-hBSZ[1]:j+hBSZ[1], k-hBSZ[2]:k+hBSZ[2]] \
                                                                            + np.ones(prediction.shape[1:-1])
                    cnt = cnt+1
        weights = np.expand_dims(weights, axis=-1)
        abundances = np.divide(abundances, weights, where=(weights>0))
        return abundances

#This is the regularizer:
def abundance_corr(x):
    x = K.l2_normalize(x, axis=[1,2,3])
    for i in range(x.shape[-1]):
        xi_corr = x * K.expand_dims(x[..., i], axis=-1)
        xi_corrsum = K.expand_dims(K.sum(xi_corr, axis=-1), axis=-1)
        if i == 0:
            x_corrsum = xi_corrsum
        else:
            x_corrsum = K.concatenate([x_corrsum, xi_corrsum])
    return  0.02*K.mean(x_corrsum) #Here you have to find a good value to replace 0.02 to determine suitable amount of "blending" between materials in the segmentation layer

#This is the loss function:
def variation_corr(y_true, y_pred):
    laplace = tf.constant([0., 0., 0., 0., 1., 0., 0., 0., 0.,
                          0., 1., 0., 1., -6., 1., 0., 1., 0.,
                          0., 0., 0., 0., 1., 0., 0., 0., 0.], shape=[3, 3, 3])
    laplace = K.expand_dims(laplace, axis=-1)
    laplace = K.concatenate([laplace, laplace, laplace], axis=-1)
    laplace = K.expand_dims(laplace, axis=-1)

    y_true_diff = tf.nn.convolution(y_true, laplace, padding="VALID")
    y_pred_diff = tf.nn.convolution(y_pred, laplace, padding="VALID")
    y_true_diff = K.spatial_3d_padding(y_true_diff, padding=((1, 1), (1, 1), (1, 1)), data_format=None)
    y_pred_diff = K.spatial_3d_padding(y_pred_diff, padding=((1, 1), (1, 1), (1, 1)), data_format=None)

    y_true =K.l2_normalize(y_true, axis=[1,2,3])
    y_pred =K.l2_normalize(y_pred, axis=[1,2,3])

    y_true_diff =K.l2_normalize(y_true_diff, axis=[1,2,3])
    y_pred_diff =K.l2_normalize(y_pred_diff, axis=[1,2,3])

    return K.mean(-(y_true * y_pred) - (y_true_diff * y_pred_diff))

LR =layers.LeakyReLU(0.1)
LR.__name__='LRELU'


def segae_model(output='reconstruction', abundances=4, input_shape=(80,80,80,3)):
    input_img = Input(shape=input_shape)
    #Here I create a brain mask tensor:
    brain = Lambda(lambda x: K.expand_dims(K.cast(K.not_equal(K.sum(x, axis=-1), 0), 'float32'), -1))(input_img)
    
    if output=='reconstruction':
        x = Lambda(lambda x: x * K.abs(K.random_normal(shape=(1, 1, 1, 3), mean=1, stddev=0.5)))(input_img)
    elif output=='segmentation':
            x = GaussianNoise(0.001)(input_img)
    
    x = GaussianNoise(0.03)(x)
    x = Conv3D(32, (3, 3, 3), activation=LR, padding='same', name='conv1')(x)
    x1 = BatchNormalization(name='bn1')(x)

    x = Conv3D(64, (3, 3, 3), strides=2, activation=LR, use_bias=False, padding='same', name='conv2')(x1)
    x = BatchNormalization(name='bn2')(x)
    x = Conv3D(64, (3, 3, 3), activation=LR, use_bias=False, padding='same', name='conv3')(x)
    x2 = BatchNormalization(name='bn3')(x)
    
    x = Conv3D(128, (3, 3, 3), strides=2, activation=LR, use_bias=False, padding='same', name='conv4')(x2)
    x = BatchNormalization(name='bn4')(x)
    x = Conv3D(128, (3, 3, 3), activation=LR, use_bias=False, padding='same', name='conv5')(x)
    x = BatchNormalization(name='bn5')(x)

    x = UpSampling3D((2, 2, 2))(x)
    x = concatenate([x, x2])
    x = Conv3D(64, (3, 3, 3), activation=LR, use_bias=False, padding='same', name='conv6')(x)
    x = BatchNormalization(name='bn6')(x)
    x = Conv3D(64, (3, 3, 3), activation=LR, use_bias=False, padding='same', name='conv7')(x)
    x = BatchNormalization(name='bn7')(x)
    
    x = UpSampling3D((2, 2, 2))(x)
    x = concatenate([x, x1])
    x = Conv3D(32, (3, 3, 3), activation=LR, use_bias=False, padding='same', name='conv8')(x)
    x = BatchNormalization(name='bn8')(x)
    x = Conv3D(32, (3, 3, 3), activation=LR, use_bias=False, padding='same', name='conv9')(x)
    x = BatchNormalization(name='bn9')(x)
    x = Conv3D(16, (1, 1, 1), activation=LR, use_bias=False, padding='same', name='conv10')(x)
    x = BatchNormalization(name='bn10')(x)

    x = Multiply()([x, brain])
    x = Conv3D(abundances, (1, 1, 1), padding='same', use_bias=False,
               activity_regularizer=abundance_corr, activation='softmax', name='segmentation')(x)
    output_seg = Multiply()([x, brain])
    output_img = Conv3D(3, (1, 1, 1), padding='same',
                        kernel_constraint=keras.constraints.non_neg(), use_bias=False, name='LastLayer')(output_seg)
    if output == 'reconstruction':
        return Model(input_img, output_img)
    elif output == 'segmentation':
        return Model(input_img, output_seg)




model_path = #Your path to the saved model

# If True, this script will train a model, if False, it will use a previously trained model to segment images
if True:
    subjects = ['subjectID_1', 'subjectID_2', 'subjectID_3']
    for subject in subjects:
        # Input a subject id or path to the images and a patch size (BSZ) and stride to create a set of patches for each subject:
        subject_patches, subject_coordinates = get_patch_set(subject, BSZ=(80, 80, 80), stride=(40, 40, 40)) 
        if subject == subjects[0]:
            training_set = subject_patches
        else:
            training_set = np.concatenate((training_set, subject_patches), axis=0)

    # Here I remove 50% of the patches that contain the most background voxels to reduce the training data size:
    brainmask_patches = (np.not_equal(np.sum(training_set, axis=-1), 0)).astype(int)
    importance = np.sum(brainmask_patches, axis=(1,2,3))
    print("Importance: {}".format(importance))
    mask = importance > np.percentile(importance, 50)
    training_set = training_set[mask, ...]

    model = segae_model(abundances=5, output='reconstruction', input_shape=(80, 80, 80, 3))
    #If you want to retrain another model you can put this here:
    #model.load_weights(model_path + """YOUR_MODEL NAME""",
    #                   by_name=True)
    model.compile(
        optimizer=optimizers.Nadam(lr=0.001),
        loss=variation_corr)

    history = model.fit(training_set[..., :3], training_set[...,3:],
                        epochs=80,
                        batch_size=1,
                        shuffle=True)

    model.save_weights(model_path + """YOUR MODEL NAME""")

else:

    trained_model_name="""YOUR MODEL NAME"""

    abundance_model = segae_model(output='segmentation',,
                                  abundances=5, input_shape=(80, 80, 80, 3))
    abundance_model.load_weights(model_path + trained_model_name, by_name=True)

    num_abundances=5
    print_abundances = [0, 1, 2, 3, 4]

    for subject in ['subjectID_1', 'subjectID_2', 'subjectID_3']: 
        try:
            data, coordinates = get_patch_set(subject, BSZ=(80, 80, 80), stride=(40, 40, 40), training=False)
        except FileNotFoundError:
            print('Subject {} not found'.format(subject))
            continue

        X_test = data[..., :3]
        #Put a path to one of your images here to get the nibabel img header:
        t1_path =  t1_path =  glob.glob('/images/registeredToMNI/' + subject + '_*_MPRAGEPre_reg.nii.gz')[0] 
        img = nib.load(t1_path)

        prediction = abundance_model.predict(X_test, 1)
        abundances = assemble_patches(prediction, coordinates, img.shape)

        dest_path = """Your destination path for predicted segmentations (abundances)""" \
                    + trained_model_name  + '/' + subject

        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        for i in print_abundances:
            print('Print NIFTI for subject ' + subject)
            new_mask = nib.Nifti1Image(abundances[..., i], img.affine, img.header)
            nib.save(new_mask, 
                     os.path.join(dest_path,
                                  subject + '_abundance_' + str(i) + '.nii.gz'))

        abundances = np.concatenate((np.expand_dims(abundances[..., -1], axis=-1),
                                     abundances[..., :-1]), -1)
        new_mask = nib.Nifti1Image(np.argmax(abundances, axis=-1), img.affine, img.header)
        nib.save(new_mask, 
                 os.path.join(dest_path,
                              subject + '_segmentation.nii.gz'))
