import os
import nibabel as nib
from sklearn import utils, metrics
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
from functools import partial
import random
from scipy.ndimage import morphology
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
model_path = '/mnt/archive/bmap/res17/hans/cnn_ventricle_segmentation/keras_models/'

subjects_g1 = #list of subject IDs for group 1
subjects_g2 = #list of subject IDs for group 2
subjects_g3 = #list of subject IDs for group 3

def load_brainmask(subject, dtype=np.float32):
    FOLDER = '/mnt/archive/bmap/data/NPH/nifti_n120/'
    filename = subject + "/" + subject + \
                    ".t1.rai.resampled.N4.mni_MultiModalStripMask.nii"
    filepath = os.path.join(FOLDER, filename)
    img = nib.load(filepath)
    return np.asarray(img.get_fdata(), dtype=dtype)

def load_segae_seg(subject): 
    folder= '/mnt/archive/bmap/res17/hans/data_archive/' + subject + '/TM101_CNN_LMM_autoencoder/'
    seg_code = {'gm':2, 'wm':3, 'wmh':3, 'csf':1}
    #seg = []
    for i, material in enumerate(['csf', 'gm', 'wm', 'wmh']): #, 'meninges']
        file = subject + '_abundance_' + str(i) + '.nii.gz'
        filepath = os.path.join(folder, file)
        img = nib.load(filepath)
        
        if i == 0:
            seg = seg_code[material] * (np.asarray(img.get_fdata(), dtype=np.float) > 0.5).astype(int)
        else:
            seg = seg + seg_code[material] * (np.asarray(img.get_fdata(), dtype=np.float)  > 0.5).astype(int)      
    
    return np.where(np.isnan(seg), 0, seg)

def load_segae_seg_modified(subject):
    folder= '/mnt/archive/bmap/res17/hans/data_archive/' + subject + '/TM101_CNN_LMM_autoencoder/'
    segae = {}
    for i, material in enumerate(['csf', 'gm', 'wm', 'wmh', 'meninges']):
        filename = subject + '_abundance_' + str(i) + '.nii.gz'
        filepath = os.path.join(folder, filename)
        img = nib.load(filepath)
        seg = np.asarray(img.get_fdata(), dtype=np.float)
        segae[material] = np.where(np.isnan(seg), 0, seg)

    artifacts_wmhs = (segae['csf'] * segae['wmh'])
    artifacts_gm = (segae['csf'] * segae['gm'])
    segae['csf'] = segae['csf'] + artifacts_wmhs
    segae['wmh'] = segae['wmh'] - artifacts_wmhs
    segae['csf'] = segae['csf'] + artifacts_gm
    segae['gm'] = segae['gm'] - artifacts_gm
    segae['artifacts'] = artifacts_wmhs

    return segae


def load_rudolph2(subject, dtype=np.float32):
    folder = """RUDOLPH FOLDER """
    rudolph_file = subject + "***FILENAME***_rudolph_noblur_fixedTopology.nii"
    filepath = os.path.join(folder, rudolph_file)
    img = nib.load(filepath)
    return np.asarray(img.get_fdata(), dtype=dtype)

def get_rudolph2_ventricle_segmentation(subject, dtype=np.float32):
    seg = load_rudolph2(subject, dtype)    
    LLV = np.zeros(seg.shape)
    RLV = np.zeros(seg.shape)
    third = np.zeros(seg.shape)
    fourth = np.zeros(seg.shape)
    for label in [50, 52]:
        LLV = LLV + (seg == label).astype(int)
    for label in [49, 51]:
        RLV = RLV + (seg == label).astype(int)
    third = (seg == 4).astype(int)
    fourth = (seg == 11).astype(int)
    Y = np.zeros(seg.shape)
    Y = LLV + 2*RLV + 3*third + 4*fourth
    return Y.astype(dtype)


def get_volumes_modified(subject, dtype=np.float32):
    rudolph_labels = get_rudolph2_ventricle_segmentation(subject) 
    brainmask = load_brainmask(subject)

    segae = load_segae_seg_modified(subject)
    segmentations = []
    for i, material in enumerate(['csf', 'gm', 'wm', 'wmh', 'meninges']):
        if material == 'meninges':
            segmentations.insert(0, segae[material])
        else:
            segmentations.append(segae[material]) 
    segmentation = np.stack(segmentations, axis=-1)
    segae_seg = np.argmax(np.where(np.isnan(segmentation), 0, segmentation), axis=-1)
    segae_csf = (segae_seg == 1).astype(int)
    labels = np.unique(rudolph_labels)
    Y_train = np.equal.outer(rudolph_labels,
                            labels).astype(np.float32)      
    Y_train[..., 1:] = np.where( np.expand_dims(brainmask > 0.5,  axis = -1),
                                (Y_train[..., 1:] * np.expand_dims((segae_csf >= 0.50).astype(int), axis = -1)), 0)
    Y_train[..., 1] = morphology.binary_closing((Y_train[..., 1] >= 0.5).astype(int), np.ones((3,3,3)), iterations = 2)
    Y_train[..., 2] = morphology.binary_closing((Y_train[..., 2] >= 0.5).astype(int), np.ones((3,3,3)), iterations = 2)
    Y_train[..., 3] = morphology.binary_closing((Y_train[..., 3] >= 0.5).astype(int), np.ones((3,3,3)), iterations = 2)
    Y_train[..., 0] = np.where( brainmask > 0.5,
                                (1 - np.sum(Y_train[..., 1:], axis=-1)), 1)
    seg_code = {'gm':2, 'wm':3, 'wmh':3, 'csf':1, 'meninges':5}
    for i, material in enumerate(['csf', 'gm', 'wm', 'wmh', 'meninges']):
        if i == 0:
            seg = seg_code[material] * segae[material]
        else:
            seg = seg + seg_code[material] * segae[material]

    data = [seg, Y_train[..., 0],  Y_train[..., 1], Y_train[..., 2],
            Y_train[..., 3], Y_train[..., 4]]
    return np.stack(data , -1)


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

def get_patch_set(subjects, BSZ, stride):
    coordinates = {}
    for subject in subjects:
        volumes = get_volumes_modified(subject)
        print(volumes.shape)
        patches_coord = sliding_patches(volumes, BSZ)
        nonzero_img = np.nonzero(volumes[..., 0])
        coord = []
        coord.append([i for i in range(max(np.min(nonzero_img[0]), int(BSZ[0]/2)),
              min(np.max(nonzero_img[0] + 1), volumes[..., 0].shape[0] - BSZ[0]//2), stride[0])])
        coord.append([i for i in range(max(np.min(nonzero_img[1]), int(BSZ[1]/2)),
                                  min(np.max(nonzero_img[1]) + 1, volumes[..., 0].shape[1] - BSZ[1]//2), stride[1])])
        coord.append([i for i in range(max(np.min(nonzero_img[2]), int(BSZ[2]/2)),
                                  min(np.max(nonzero_img[2]) + 1, volumes[..., 0].shape[2] - BSZ[2]//2), stride[2])])
        coordinates[subject] = np.array(coord)
        seg_patches = np.empty((int(len(coord[0])*len(coord[1])*len(coord[2])), BSZ[0], BSZ[1], BSZ[2], volumes.shape[-1]))
        cnt = 0
        for i in coord[0]:
            for j in coord[1]:
                for k in coord[2]:
                    seg_patches[cnt, ...] = patches_coord[i, j, k, ...]
                    cnt = cnt+1
        print("Extracting patches from subject: {}".format(subject))
        if subject == subjects[0]:
            data = seg_patches
        else:
            data = np.concatenate((data, seg_patches), axis=0)
    return data, coordinates

"""
brainmask_patches = (np.not_equal(np.sum(X_train, axis=-1), 0)).astype(int)
importance = np.sum(brainmask_patches, axis=(1,2,3))
print("Importance: {}".format(importance))
mask = importance > np.percentile(importance, 50)

X_train = X_train[mask, ...]
output_patches = data[mask, ..., -1] #mask, ..., 3]

test = np.expand_dims(X_train[-1, ...], 0)
"""


def get_random_subject_patches(subjects_train, stride=(40, 40, 40)):    
    #data, coordinates, class_weights = get_patch_set(subjects_train, BSZ=(128,128,128), stride=stride)
    data, coordinates = get_patch_set(subjects_train, BSZ=(128,128,128), stride=stride)
    print('data shape = {}'.format(data.shape[-1]))
    if data.shape[-1] == 3:
        output_patches = data[..., -1]
        labels = np.unique(output_patches)
        num_classes = labels.shape[0]
        Y_train = np.equal.outer(output_patches,
                                labels).astype(np.float32)
        Y_train[..., 1:] = np.where( np.expand_dims(data[..., 1] > 0.5,  axis = -1),
                                    (Y_train[..., 1:] * np.expand_dims(data[..., 0], axis = -1)), 0)
        Y_train[..., 0] = np.where( data[..., 1] > 0.5,
                                    (1 - data[..., 0]), 1)
        X_train = data[..., 1]

        print("Y_train shape: {}, data0 vals: {}".format(Y_train.shape, data[..., 0].shape ))
    else:
        X_train = data[..., 0]
        Y_train = data[..., 1:]
    
    X_train = np.expand_dims(X_train, axis=-1)

    """
    class_weights = utils.compute_class_weight('balanced',
                                           labels,
                                           output_patches.flatten()).tolist()
    """
    # print("Class_weights: {}".format(class_weights))
    
    return X_train, Y_train#, class_weights

def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
  denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
  return 1 - (numerator + 1) / (denominator + 1)


LR = layers.LeakyReLU(0.1)
LR.__name__='LRELU'

def weighted_binary_crossentropy(y_true, y_pred, weights):
    labels = y_true
    logits = y_pred
    class_weights = tf.constant([[[weights]]])
    
    weights = tf.reduce_sum(class_weights * labels, axis=-1)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    print(unweighted_losses.shape, weights.shape)
    weighted_losses = unweighted_losses * weights #K.expand_dims(weights, axis=-1)
    loss = tf.reduce_mean(weighted_losses)
    return loss
#ncce = partial(weighted_binary_crossentropy, weights=class_weights)
#ncce.__name__='weighted_binary_crossentropy'

ff=1
def recon_model(output='reconstruction', abundances=4, input_shape=(80,80,80,3)):
    input_img = Input(shape=input_shape)
    brain = Lambda(lambda x: K.expand_dims(K.cast(K.not_equal(K.sum(x, axis=-1), 0), 'float32'), -1))(input_img)
    if output=='segmentation':
        x = GaussianNoise(0.005)(input_img)
    else:
        x = Lambda(lambda x: x * K.abs(K.random_normal(shape=(1, 1, 1, input_shape[-1]), mean=1, stddev=0.5)))(input_img)
    x = GaussianNoise(0.05)(x)
    x = Conv3D(32*ff, (3, 3, 3), activation=LR, padding='same', name='conv1')(x)#input_img)
    x1 = BatchNormalization(name='bn1')(x)

    x = Conv3D(64*ff, (3, 3, 3), strides=2, activation=LR, use_bias=False, padding='same', name='conv2')(x1)
    x = BatchNormalization(name='bn2')(x)
    x = Conv3D(64*ff, (3, 3, 3), activation=LR, use_bias=False, padding='same', name='conv3')(x)
    x2 = BatchNormalization(name='bn3')(x)

    
    x = Conv3D(128*ff, (3, 3, 3), strides=2, activation=LR, use_bias=False, padding='same', name='conv4')(x2)
    x = BatchNormalization(name='bn4')(x)
    x = Conv3D(128*ff, (3, 3, 3), activation=LR, use_bias=False, padding='same', name='conv5')(x)
    x3 = BatchNormalization(name='bn5')(x)

    x = Conv3D(256*ff, (3, 3, 3), strides=2, activation=LR, use_bias=False, padding='same', name='conv6')(x3)
    x = BatchNormalization(name='bn6')(x)
    x = Conv3D(256*ff, (3, 3, 3), activation=LR, use_bias=False, padding='same', name='conv7')(x)
    x4 = BatchNormalization(name='bn7')(x)

    #adding this resolution scale for ventricle segmentation, remove for brainsegmentation
    x = Conv3D(512*ff, (3, 3, 3), strides=2, activation=LR, use_bias=False, padding='same', name='convVS1')(x4)
    x = BatchNormalization(name='bnVS1')(x)
    x = Conv3D(512*ff, (3, 3, 3), activation=LR, use_bias=False, padding='same', name='convVS2')(x)
    x = BatchNormalization(name='bnVS2')(x)

    x = UpSampling3D((2, 2, 2))(x)
    x = concatenate([x, x4])
    x = Conv3D(256*ff, (3, 3, 3), activation=LR, use_bias=False, padding='same', name='convVS3')(x)
    x = BatchNormalization(name='bnVS3')(x)
    x = Conv3D(256*ff, (3, 3, 3), activation=LR, use_bias=False, padding='same', name='convVS4')(x)
    x = BatchNormalization(name='bnVS4')(x)
    #x = SpatialDropout3D(0.5)(x)
    x = UpSampling3D((2, 2, 2))(x)
    x = concatenate([x, x3])
    x = Conv3D(128*ff, (3, 3, 3), activation=LR, use_bias=False, padding='same', name='conv8')(x)
    x = BatchNormalization(name='bn8')(x)
    x = Conv3D(128*ff, (3, 3, 3), activation=LR, use_bias=False, padding='same', name='conv9')(x)
    x = BatchNormalization(name='bn9')(x)

    x = UpSampling3D((2, 2, 2))(x)
    x = concatenate([x, x2])
    x = Conv3D(64*ff, (3, 3, 3), activation=LR, use_bias=False, padding='same', name='conv10')(x)
    x = BatchNormalization(name='bn10')(x)
    x = Conv3D(64*ff, (3, 3, 3), activation=LR, use_bias=False, padding='same', name='conv11')(x)
    x = BatchNormalization(name='bn11')(x)
    
    x = UpSampling3D((2, 2, 2))(x)
    x = concatenate([x, x1])
    x = Conv3D(32*ff, (3, 3, 3), activation=LR, use_bias=False, padding='same', name='conv12')(x)
    x = BatchNormalization(name='bn12')(x)
    x = Conv3D(32*ff, (3, 3, 3), activation=LR, use_bias=False, padding='same', name='conv13')(x)
    x = BatchNormalization(name='bn13')(x)
    x = Conv3D(16*ff, (1, 1, 1), activation=LR, use_bias=False, padding='same', name='conv14')(x)
    x_x = BatchNormalization(name='bn14')(x)
    
    x = Multiply()([x_x, brain])
    x = Conv3D(abundances, (1, 1, 1), padding='same', use_bias=False,
               activity_regularizer=abundance_corr, activation='softmax', name='segmentation')(x)
    output_seg = Multiply()([x, brain])
    output_img = Conv3D(3, (1, 1, 1), padding='same',
                        kernel_constraint=keras.constraints.non_neg(), use_bias=False, name='LastLayer')(output_seg)

    if output == 'reconstruction':
        return Model(input_img, output_img)
    elif output == 'brainmask':
        output_brainmask = Conv3D(2, (1, 1, 1), activation='softmax', padding='same', name='conv15')(x_x)
        return Model(input_img, output_brainmask)
    elif output == 'csf_segmentation':
        output_csf = Conv3D(5, (1, 1, 1), activation='softmax', padding='same')(x_x)
        return Model(input_img, output_csf)
    elif output == 'segmentation':
        return Model(input_img, output_seg)


brainmask=False
if False:  #Training if True, otherwise prediction using a previously trained model
    if brainmask:
        model = recon_model(output='brainmask')
        model.compile(
        optimizer=optimizers.Nadam(lr=0.00001),
        loss=ncce,
        metrics=[corr, 'accuracy'])
        history = model.fit(X_train, Y_train,
            epochs=150,
            batch_size=5,
            shuffle=True)
    else:
        for i, learning_rate in enumerate([0.000003, 0.000003, 0.000003, 0.000003]):
            subjects_train = random.sample(subjects_g1, 5) + random.sample(subjects_g2, 5) + random.sample(subjects_g3, 5)
            X_train, Y_train = get_random_subject_patches(subjects_train)
            print('Starting round {}'.format(str(i+1)))
            model = recon_model(output='csf_segmentation', input_shape=(128,128,128,1))
            if i > 0:
                model.load_weights(model_path + 'TM17c_csf_segmentation_unet')
            model.compile(
            optimizer=optimizers.Nadam(lr=learning_rate),
            loss=dice_loss,
            metrics=[corr, 'accuracy'])
            model.fit(X_train, Y_train,
                epochs=50,
                batch_size=1,
                shuffle=True)
            print('saving csf_segmentation_unet')
            model.save_weights(model_path + 'TM19_csf_segmentation_unet')
            K.clear_session()
    
else:
    if brainmask:
        model = recon_model(output='brainmask')
        model.load_weights(model_path + 'TM4_brainmask_unet', by_name=True)
        model.compile(
            optimizer=optimizers.Nadam(lr=0.0000001),
            loss=ncce,
            metrics=[corr, 'accuracy'])
        history = model.fit(X_train, Y_train,
            epochs=50,
            batch_size=5,
            shuffle=True)
    else:
        model = recon_model(output='csf_segmentation', input_shape=(128,128,128,1))
        model.load_weights(model_path + 'TM20_csf_segmentation_unet')

        for i, learning_rate in enumerate([0.000000003]):
            subjects_train = random.sample(subjects_g1, 5) + random.sample(subjects_g2, 5) \
                             + random.sample(subjects_g3, 5)
            
            X_train, Y_train = get_random_subject_patches(subjects_train)
            print('Starting round {}'.format(str(i+1)))
            model.compile(
            optimizer=optimizers.Nadam(lr=learning_rate),
            loss=dice_loss,
            metrics=[corr, 'accuracy'])
            model.fit(X_train, Y_train,
                epochs=15,
                batch_size=1,
                shuffle=True)
            print('saving csf_segmentation_unet')
            model.save_weights(model_path + 'TM20_csf_segmentation_unet')
            #K.clear_session()
