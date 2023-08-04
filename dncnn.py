import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import window
import warnings
from PIL import Image
import random
from skimage.util import crop
from scipy.ndimage import uniform_filter

'''
# --------------------------------------------
# Codes Below Taken from: 
# https://github.com/cszn/KAIR/blob/master/utils/utils_image.py
# --------------------------------------------
'''

'''
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''
NON_ZERO_FLOAT = 1e-8 

def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)

'''
# --------------------------------------------
# image processing process on numpy image
# channel_convert(in_c, tar_type, img_list):
# rgb2ycbcr(img, only_y=True):
# bgr2ycbcr(img, only_y=True):
# ycbcr2rgb(img):
# --------------------------------------------
'''

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    rlt = np.clip(rlt, 0, 255)
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

'''
# --------------------------------------------
# read image from path
# opencv is fast, but read BGR numpy image
# --------------------------------------------
'''

# --------------------------------------------
# get uint8 image of size HxWxn_channles (RGB)
# --------------------------------------------
def imread_uint(path, n_channels=3):
    #  input: path
    #  output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img

# --------------------------------------------
# numpy(single) [0, 1] <--->  numpy(uint)
# --------------------------------------------
def uint2single(img):
    return np.float32(img/255.)

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())

# convert single (HxWxC) to 4-dimensional torch tensor
def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)

# convert 2/3/4-dimensional torch tensor to uint
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

# convert uint to 4-dimensional torch tensor
def uint2tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)

# convert torch tensor to single
def tensor2single(img):
    img = img.data.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    return img

def imshow(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


class DnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(DnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = conv(nc, out_nc, mode='C', bias=bias)

        self.model = sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        n = self.model(x)
        return x-n

# --------------------------------------------
# FDnCNN
# --------------------------------------------
# Compared with DnCNN, FDnCNN has three modifications:
# 1) add noise level map as input
# 2) remove residual learning and BN
# 3) train with L1 loss
# may need more training time, but will not reduce the final PSNR too much.
# --------------------------------------------
class FDnCNN(nn.Module):
    def __init__(self, in_nc=2, out_nc=1, nc=64, nb=20, act_mode='R'):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        """
        super(FDnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = conv(nc, out_nc, mode='C', bias=bias)

        self.model = sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x

'''
########################################## End ##########################################
'''

'''
# --------------------------------------------
# Codes below Modified from:
# https://github.com/awsaf49/artifact/blob/main/data/transform.py
# --------------------------------------------
'''

'''
# --------------------------------------------
# Random Crop + JPEG Compression (Modified)
# --------------------------------------------
'''

# Configuirations
OUTPUT_SIZE  = 200
CROPSIZE_MIN = 160 # minimum allowed crop size
CROPSIZE_MAX = 2048 # maximum allowed crop size
CROPSIZE_RATIO = (5,8)
QF_RANGE = (65, 100)
    
def random_crop_resize(img):
    """
    This function takes an input image, randomly crops a square region from it, 
    resizes the cropped region to a fixed size, and returns the resulting image.

    img - Image represented in numpy array
    """
    height, width = img.shape[0], img.shape[1]
    # select the size of crop
    cropmax = min(min(width, height), CROPSIZE_MAX)
    if cropmax < CROPSIZE_MIN:
        warnings.warn("({},{}) is too small".format(height, width))
        return None

    # try to ensure the crop ratio is at least 5/8
    cropmin = max(cropmax * CROPSIZE_RATIO[0] // CROPSIZE_RATIO[1], CROPSIZE_MIN)
    cropsize = random.randint(cropmin, cropmax)

    # select the type of interpolation
    # determines the type of interpolation to use during the resizing step. 
    # It uses cv2.INTER_AREA if the crop size is larger than a constant OUTPUT_SIZE, 
    # otherwise, it uses cv2.INTER_CUBIC.
    interp = cv2.INTER_AREA if cropsize > OUTPUT_SIZE else cv2.INTER_CUBIC

    # select the position of the crop
    x1 = random.randint(0, width - cropsize)
    y1 = random.randint(0, height - cropsize)

    # perform the random crop
    cropped_img = img[y1:y1+cropsize, x1:x1+cropsize]

    # perform resizing
    resized_img = cv2.resize(cropped_img, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=interp)

    # return the cropped image array
    return resized_img

'''
########################################## End ##########################################
'''

'''
# --------------------------------------------
# Codes below Referred & Modified from:
# https://github.com/scikit-image/scikit-image/blob/main/skimage/metrics/_structural_similarity.py 
# --------------------------------------------
'''

'''
# --------------------------------------------
# Local Correlation Computation
# --------------------------------------------
'''
def average_local_correlation(im1, im2, win_size=None):
    """
    This functions compute the local correlation between two images patches from two different images. 

    im1         - Image 1
    im2         - Image 2
    win_size    - Window Size
    """
    if win_size == None:
        win_size = 7
    K2 = 0.03
    ndim = im1.ndim
    NP = win_size ** ndim
    cov_norm = NP / (NP - 1)  # sample covariance
    L = 255 # max pixel values
    C2 = (K2 * L) ** 2

    # Compute (weighted) means
    ux = uniform_filter(im1, size=win_size)
    uy = uniform_filter(im2, size=win_size)

    # Compute (weighted) variances and covariances
    uxx = uniform_filter(im1 * im1, size=win_size)
    uyy = uniform_filter(im2 * im2, size=win_size)
    uxy = uniform_filter(im1 * im2, size=win_size)
    vx = cov_norm * (uxx - ux * ux) # variance of x
    vy = cov_norm * (uyy - uy * uy) # variance of y
    vxy = cov_norm * (uxy - ux * uy) # covariance of x and y

    num = vxy + 0.5*C2
    denom = np.sqrt(vx+NON_ZERO_FLOAT) + np.sqrt(vy+NON_ZERO_FLOAT) + 0.5*C2

    S = num / denom
    # to avoid edge effects will ignore filter radius strip around edges
    # pad = 3
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim. Use float64 for accuracy.
    mcorr = crop(S, pad).mean(dtype=np.float64)

    return mcorr

'''
########################################## End ##########################################
'''

'''
# --------------------------------------------
# Codes below Taken from: https://www.astrobetter.com/wiki/tiki-index.php?page=python_radial_profiles
# Use Cases Referred from: https://www.astrobetter.com/wiki/tiki-index.php?page=python_image_fft 
# --------------------------------------------
'''

'''
# --------------------------------------------
# Compute 2D Power Spectrum Density(PSD) and azimuthally averaged 1D PSD
# --------------------------------------------
'''

def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

'''
########################################## End ##########################################
'''

'''
# --------------------------------------------
# Self-defined Utility Functions
# --------------------------------------------
'''
# normalize image pixel values back to range [0, RANGE_MAX] using linear scaling
def normalize_image(img):
    RANGE_MAX=255
    # None              - return the output directly
    # (0, RANGE_MAX)    - the range of pixel values
    # cv2.NORM_MINMIAX  - normalization is done using minimum-maximum scaling
    # cv2.CV_8U         - output will be an 8-bit unsigned integer image, with pixel values ranging from 0 to 255.
    return cv2.normalize(img, None, 0, RANGE_MAX, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# random crop an input image to certain size
def random_crop(img, height=200, width=200):
    assert img.shape[0] >= height, f"crop height must be smaller than image height"
    assert img.shape[1] >= width, f"crop width must be smaller than image width"
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img

# resize image to specific height and weight
# interpolation method depends on the input dimension and the output dimension
def resize(img, height=256, width=256):
    img_h, img_w = img.shape[0], img.shape[1]
    # select the type of interpolation
    interp = cv2.INTER_AREA if (img_h > height or img_w > width) else cv2.INTER_CUBIC
    # perform resizing
    resized_img = cv2.resize(img, (height, width), interpolation=interp)
    # return the cropped image array
    return resized_img

# HxW to HxWx1
def expand_dim(img):
    if img.ndim == 2:
        return np.expand_dims(img, axis=2)  # HxWx1
    return img

# check if image is in grayscale
def is_gray(img):
    """
    Returns if the img is in grayscale
    """
    if len(img.shape) < 3:
        return True
    if img.shape[2]  == 1:
        return True
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    if (b==g).all() and (b==r).all():
        return True
    return False

# check if an image is represented an 8-bit unsigned integer array or still a filepath 
def is_path(image):
    return isinstance(image, str)

def fft_preprocess(image):
    """
    FFT preprocess the image
    """
    if is_gray(image):
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
    else:
        fshift = []
        for idx in range(3):
            # Splitting the color image into separate color channels
            channel = image[..., idx]

            # Applying Fourier transform on each color channel
            fft = np.fft.fft2(channel)
            # Applying fftshift to each color channel
            fft_shifted = np.fft.fftshift(fft)
            fshift.append(fft_shifted)

        fshift = np.stack(fshift, axis=-1)
    magnitude_spectrum = np.abs(fshift)
    return magnitude_spectrum.astype(np.float64)

def autocorrelation_2d(img):
    # Shift the image to the origin
    # shifted_image = image - np.mean(image)
    if is_gray(img):
        # Compute the 2D Fourier transform
        fft_image = np.fft.fft2(img)
        # Compute the autocorrelation using the power spectrum
        psd = np.abs(fft_image) ** 2
        autocorr = np.fft.ifft2(np.abs(psd) ** 2)
        # Normalize the autocorrelation
        autocorr /= (autocorr[0, 0] * (img.shape[0] * img.shape[1]))
        # Shift the zero-frequency component to the center
        autocorr = np.fft.fftshift(autocorr)
        return np.real(autocorr)
    else:
        autocorr = []
        for idx in range(3):
            autocorr.append(autocorrelation_2d(img[...,idx]))
        fshift = np.stack(autocorr, axis=-1)
        return fshift



def high_pass_fft_filtered(image, cutoff=25):
    """Performing high-pass fft filtering on the image"""
    if not is_gray(image):
        # Splitting the color image into separate color channels
        temp = []
        for idx in range(3):
            channel = image[...,idx]
            filtered = high_pass_fft_filtered(channel, cutoff)
            temp.append(filtered)
        filtered_image = np.stack(temp, axis=-1)
        return filtered_image
    else:
        f = np.fft.fft2(image)
        f_shifted = np.fft.fftshift(f)
        h, w = image.shape
        half_h, half_w = h//2, w//2
        # remove frequencies below cutoff
        f_shifted[half_h-cutoff:half_h+cutoff+1, half_w-cutoff:half_w+cutoff+1] = 0
        f_ishift = np.fft.ifftshift(f_shifted)
        filtered_image = np.fft.ifft2(f_ishift)
        filtered_image = np.abs(filtered_image)
        return filtered_image

def high_pass_dct_filtered(image, factor=2):
    """Performing high-pass filtering on the image"""
    image = image.astype(np.float64)
    if not is_gray(image):
        temp = []
        # Splitting the color image into separate color channels 
        for idx in range(3):
            channel = image[...,idx]
            # Applying Fourier transform on each color channel
            filtered = high_pass_dct_filtered(channel, factor)
            temp.append(filtered)
        filtered_image = np.stack(temp, axis=-1)
        return filtered_image
    else:
        # Apply DCT to the image
        # Convert the image to floating-point data type
        dct_image = cv2.dct(image)
        # Set high-frequency coefficients to zero
        h, w = dct_image.shape
        dct_image[:h//factor, :w//factor] = 0
        # Apply inverse DCT to obtain the modified image
        filtered_image = cv2.idct(dct_image)
        return filtered_image

def low_pass_dct_filtered(image, factor=2):
    """Performing low-pass filtering on the image"""
    image = image.astype(np.float64)
    if not is_gray(image):
        temp = []
        # Splitting the color image into separate color channels 
        for idx in range(3):
            channel = image[...,idx]
            # Applying Fourier transform on each color channel
            filtered = low_pass_dct_filtered(channel, factor)
            temp.append(filtered)
        filtered_image = np.stack(temp, axis=-1)
        return filtered_image
    else:
        # Apply DCT to the image
        # Convert the image to floating-point data type
        dct_image = cv2.dct(image)
        # Set high-frequency coefficients to zero
        h, w = dct_image.shape
        half_h, half_w = h//2, w//2
        dct_image[half_h:, :] = 0
        dct_image[:, half_w:] = 0
        # Apply inverse DCT to obtain the modified image
        filtered_image = cv2.idct(dct_image)
        return filtered_image

# apply contrast to images
def adjust_brightness_contrast(img, perc=50, contrast=1.5, hue=False):
    # Convert the image to a float representation
    img = img.astype(np.float64)

    img = img - np.percentile(img, perc)

    # Clip all the negative pixel values to 0
    img = np.maximum(0, img)

    # Apply contrast adjustment
    img = img * contrast

    # Return normalized image
    return normalize_image(img, hue)

# apply hanning window
def image_windowing(img):
    return img * window('hann', img.shape)

def is_std_zero(img):
    # grayscale
    if is_gray(img):
        return np.std(img)==0
    # rgb
    else:
        return (np.std(img[:, :, 0])==0 or np.std(img[:, :, 1])==0 or np.std(img[:, :, 2])==0)

'''
#------------------------------
# Zhang et. al. denoiser
# wrapper functions to estimate image residuals
#------------------------------ 
'''

n_channels = 3 # 1 for grayscale, 3 for color
nb = 20 # 17 for fixed noise level, 20 for blind
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise_level_img = 15                 # noise level for noisy image
noise_level_model = noise_level_img  # noise level for model

# model = DnCNN(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')
# model.load_state_dict(torch.load('./dncnn_color_blind.pth'), strict=True)
# model = model.to(device)
# model.eval()

def dn_cnn_filtered_residual(model, image, n_channels=n_channels):
    if is_path(image):
        image = imread_uint(image, n_channels=n_channels)
    img_L = uint2single(image)
    img_L = single2tensor4(img_L)
    img_L = img_L.to(device)
    img_E = model(img_L)
    img_N = img_L - img_E
    img_N = tensor2uint(img_N)
    return img_N

n_channels = 1 # 1 for grayscale, 3 for color
nb = 17 # 17 for fixed noise level, 20 for blind

# model = DnCNN(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')
# model.load_state_dict(torch.load('./dncnn_15.pth'), strict=True)
# model = model.to(device)
# model.eval()

def dn_cnn_filtered_residual(model, image, n_channels=n_channels):
    if is_path(image):
        image = imread_uint(image, n_channels=n_channels)
    img_L = uint2single(expand_dim(image))
    img_L = single2tensor4(img_L)
    img_L = img_L.to(device)
    img_E = model(img_L)
    img_N = img_L - img_E
    img_N = tensor2uint(img_N)
    return img_N