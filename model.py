from keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Concatenate,
    UpSampling2D, BatchNormalization, Activation
)
from keras.metrics import binary_accuracy
from keras.models import Model, Input
from keras.optimizers import Adam

from metrics_utils import custom_sparse_categorical_accuracy


def unet(input_channel_num=1, output_class_nbr=2, pretrained_weights=None):
    """ Returns a Keras U-Net model.

    ["U-Net: Convolutional Networks for Biomedical Image Segmentation",
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox, Computer Science
        Department and BIOSS Centre for Biological Signalling Studies,
        University of Freiburg, Germany, May 2015. arXiv:1505.04597]

    The output is a grayscale image if it's a binary segmentation, otherwise
    it generates as many channels as output class number.

    Parameters
    ----------
        input_channel_num : int
            number of channels of the input images
        output_class_nbr : int
            number of the output class handle by the U-Net. If it's
            above 2 class, it's a multi-class, otherwise
            it's a binary segmentation
        pretrained_weights : str
            path to pretrained weights (hdf5 or h5 file) to be loaded.
            The architecture is expected to be unchanged.

    Returns
    -------
        a U-Net model
    """
# 128x128x3
    inputs = Input(shape=(None, None, input_channel_num))
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = Concatenate(axis=3)([drop4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)

    if output_class_nbr == 2:
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)

        model.compile(optimizer=Adam(lr=1e-4),
                      loss='binary_crossentropy', metrics=[binary_accuracy])

    elif output_class_nbr > 2:
        conv10 = Conv2D(filters=output_class_nbr, kernel_size=(1, 1))(conv9)
        conv10 = BatchNormalization()(conv10)
        conv10 = Activation('softmax')(conv10)

        model = Model(inputs=inputs, outputs=conv10)

        model.compile(optimizer=Adam(lr=1e-4),
                      loss='sparse_categorical_crossentropy',
                      metrics=[custom_sparse_categorical_accuracy])
    else:
        raise ValueError("output class number should be higher than 1")

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
