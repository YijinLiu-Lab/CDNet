import numpy as np
import dxchange
from utils import *
import os
import tensorflow as tf
from keras import backend as Keras
from keras.models import load_model
from keras.models import model_from_json
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".tiff",".tif"])
def red_stack_tiff(path):
    files = os.listdir(path)
    prj = []
    for n,file in enumerate(files):
        if is_image_file(file):
            p = dxchange.read_tiff(path + file)
            prj.append(p)
    pr = np.array(prj)
    return pr
def unet_evaluate():
    opt = get_args()
    path_input = opt.evaluation_dataroot
    path_model = opt.model_dataroot+'/u-net.json'
    input_x = red_stack_tiff(path_input)
    where_are_inf = np.isnan(input_x)
    input_x[where_are_inf] = 0.0
    xc,xr, xl = input_x.shape
    input_x = input_x.reshape(xc, xr, xl, 1)
    x_norm = preprocess_input(input_x)
    x_norm = expand_array_size_with_padding(
        x_norm, 1, 32)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    Keras.set_session(tf.Session(config=config))
    weights_path = opt.model_dataroot+'/u_net1.hdf5'

    json_file = open(path_model, 'r')
    model = model_from_json(json_file.read())
    model.load_weights(weights_path)
    y_pred = model.predict(x_norm, batch_size=32, verbose=0)
    y_pred = np.squeeze(y_pred)
    y_pred = y_pred[:,:xr,:xl]
    fil = 'out_'
    ph_re = opt.out_dataroot
    for i_na, re in enumerate(y_pred):
        if i_na < 10:
            dxchange.writer.write_tiff(re, ph_re + '%s000%s.tiff' % (fil, i_na))
        elif 9 < i_na < 100:

            dxchange.writer.write_tiff(re, ph_re + '%s00%s.tiff' % (fil, i_na))
        elif 99 < i_na < 1000:
            dxchange.writer.write_tiff(re, ph_re + '%s0%s.tiff' % (fil, i_na))
        else:
            dxchange.writer.write_tiff(re, ph_re + '%s%s.tiff' % (fil, i_na))

def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--evaluation_dataroot', type=str, default='./')
    parser.add_argument('--model_dataroot', type=str, default='./')
    parser.add_argument('--out_dataroot', type=str, default='./')

if __name__ == '__main__':
    unet_evaluate()