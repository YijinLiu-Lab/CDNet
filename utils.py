import numpy as np
import math

def split_datasets(x_array, y_array, split_ratio=0.75):
    """Split a dataset into two sets of array. The split is identical for x_array and y_array
    The input array shape must be
    - (number, width, height)
    - (number, width, height, 1)

    Parameters
    ----------
        x_array : array
            input x
        y_array : array
            input y
        split_ratio : float
            split ratio between the two data sets

    Returns
    -------
        two datasets of x and y
    """

    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError("input x_array and y_array first dimensions must be equal")

    limit = int(math.floor(x_array.shape[0] * split_ratio))

    return (
        x_array[:limit, :, :],
        y_array[:limit, :, :],
        x_array[limit:, :, :],
        y_array[limit:, :, :]
        )

def preprocess_input(x):
    if len(x.shape) != 4:
        raise ValueError("incorrect 'x' shape, must be a 4d array \
            with shape=(number, width, height, 1)")
    n, _, _, _ = x.shape
    if isinstance(x, np.ndarray):
        x_tmp = x.astype(dtype=np.float32)
        x_tmp = np.reshape(x_tmp, [x_tmp.shape[0], np.product(x_tmp.shape[1:])])
        x_mean = np.mean(x_tmp, axis=1).reshape((n, 1, 1, 1))
        x_std = np.std(x_tmp, axis=1).reshape((n, 1, 1, 1))

        return (x - x_mean) / np.maximum(x_std, 1e-7)
    else:
        raise TypeError("'x' must be a numpy array")

def random_crop(img_x, img_y, random_crop_width, random_crop_height):
    """2D random crop in XY plan

    Parameters
    ----------
        img_x : array
            input x
        img_y : array
            input y
        random_crop_width : int
            crop width
        random_crop_height : int
            crop height

    Returns
    -------
        two cropped datasets of img_x and img_y
    """

    # Dimensions of img_x and img_y are similar
    width, height = img_x.shape[0], img_x.shape[1]
    dx = random_crop_width
    dy = random_crop_height
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img_x[x:(x+dx), y:(y+dy), :], img_y[x:(x+dx), y:(y+dy), :]

def crop_generator(batches, crop_width, crop_height, nb_channelsx=1, nb_channelsy=1):
    """Keras generator enable input crop
    Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.

    Parameters
    ----------
        batches : Keras ImageGen
            Keras image generator
        crop_width : int
            crop width
        crop_height : int
            crop height
        nb_channels : int
            image channels number

    Returns
    -------
        random cropped datasets from image's batches
    """

    while True:
        batch_x, batch_y = next(batches)
        img_nbr = batch_x.shape[0]
        batch_crops_x = np.zeros((img_nbr, crop_width, crop_height, nb_channelsx), np.float32)
        batch_crops_y = np.zeros((img_nbr, crop_width, crop_height, nb_channelsy), np.float32)
        for i in range(img_nbr):
            # Random crop
            batch_crops_x[i], batch_crops_y[i] = random_crop(
                batch_x[i], batch_y[i], crop_width, crop_height)

        yield (batch_crops_x, batch_crops_y)

def clear_old_model(folder_path, model_name):
    """Clear old model if exist
    Model extensions tested are hdf5 and h5

    Parameters
    ----------
        folder_path : str
            folder path
        model_name : str
            model name without extension

    Returns
    -------
        error message if an element was locked
    """

    old_model_list = glob.glob(os.path.join(folder_path, model_name) + '.h*')
    for model_file in old_model_list:
        try:
            os.unlink(model_file)
        except Exception as e:
            print(e)

def expand_array_size_with_padding(x, channel_number=1, multiple_factor=1):
    """Expand the array size to have proper channel number, width and
    height multiple of multiple_factor.
    The input shape is
    - (number, width, height, nb_channels)
    The output shape is
    - (number, width_resized, height_resized, nb_channels_resized)

    Parameters
    ----------
        x : numpy array
            array to resize
        channel_number : int
            number of output's channel
        multiple_factor : int
            output width and height divisible factor

    Returns
    -------
        array with proper dimensions
    """

    if len(x.shape) != 4:
        raise ValueError("'x' shape not managed")
    _, w, h, d = x.shape

    pad_w = int(np.ceil(float(w)/multiple_factor)*multiple_factor-w)
    pad_h = int(np.ceil(float(h)/multiple_factor)*multiple_factor-h)
    pad_d = channel_number-d

    # Image border padding using symmetric
    x_pad = np.pad(x, [(0, 0), (0, pad_w), (0, pad_h), (0, 0)], 'symmetric')
    # Channels duplication
    x_pad = np.pad(x_pad, [(0, 0), (0, 0), (0, 0), (0, pad_d)], 'edge')

    return x_pad

def crop_array_size(x, target_w, target_h):
    """Reduce the array size
    It's the expand_array_size_with_padding() inverse transform
    The input shape is
    - (number, width, height, nb_channels)
    The output shape is
    - (number, target_w, target_h, 1)

    Parameters
    ----------
        x : numpy array
            array to resize
        target_w : int
            output width
        target_h : int
            output height

    Returns
    -------
        array with proper dimensions
    """

    n, w, h, d = x.shape

    if target_w > w or target_h > h:
        return x[:, :, :, 0:1]

    # If the model output channels number is higher than one, reduce the
    # predicted image depth to only one channel
    x_crop = x[:, :, :, 0:1]

    # Crop
    x_crop = x_crop[:, :target_w, :target_h, :]

    return x_crop


# if __name__ == '__main__':
#     a = np.ones((100,100,3))
#     b,c = random_crop(a,a,512.,512.)
#     print()