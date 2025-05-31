import numpy as np
from imageio import imread
from scipy.io import loadmat
import matplotlib as mpl
import matplotlib.cm as cm
import torchvision.transforms as transforms
import cv2
import torch

def enhance_image(image):
    #after hist_99, do clache + bilateral filtering
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe = clahe.apply(image)
    bilateral = cv2.bilateralFilter(clahe, 5, 20, 15)
    return bilateral

def hist_99(image):
    im_srt = np.sort(image.reshape(-1))
    upper_bound = im_srt[round(len(im_srt) * 0.99) - 1]
    lower_bound = im_srt[round(len(im_srt) * 0.01)]

    img = image
    img[img < lower_bound] = lower_bound
    img[img > upper_bound] = upper_bound
    image_out = ((img - lower_bound) / (upper_bound - lower_bound)) * 255.0
    image_out = image_out.astype(np.uint8)
    return image_out

def align_contrast(imgL, imgR):
    imgR_mean = torch.mean(imgR)
    imgL_mean = torch.mean(imgL)
    if imgR_mean < imgL_mean:
        imgR = imgR * imgL_mean / imgR_mean
        imgR = torch.clamp(imgR, 0, 1)
    else:
        imgL = imgL * imgR_mean / imgL_mean
        imgL = torch.clamp(imgL, 0, 1)
    return imgL, imgR

def process_image(imageL, imageR, type):
    if type == "minmax":
        imageL_out = (imageL - np.min(imageL)) / (np.max(imageL) - np.min(imageL)) * 255
        imageR_out = (imageR - np.min(imageR)) / (np.max(imageR) - np.min(imageR)) * 255
    elif type == "hist_99":
        imageL_out, imageR_out = hist_99(imageL), hist_99(imageR)
    elif type == "intensity_bind":
        if np.max(imageL) < 35000:
            imageL_out = (imageL - np.min(imageL)) / (np.max(imageL) - np.min(imageL)) * 255
            imageR_out = (imageR - np.min(imageR)) / (np.max(imageR) - np.min(imageR)) * 255
        else:
            intensity, bin_edges = np.histogram(imageL, bins=4)
            upper_bound = bin_edges[1]
            lower_bound = bin_edges[0]

            imageL_out = np.zeros_like(imageL, dtype=np.uint8)
            mask = (imageL >= lower_bound) & (imageL <= upper_bound)
            imageL_out[mask] = ((imageL[mask] - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
            imageL_out[imageL > upper_bound] = np.mean(imageL_out[imageL <= upper_bound])
            imageL_out = (imageL_out - np.min(imageL_out)) / (np.max(imageL_out) - np.min(imageL_out)) * 255

            intensity, bin_edges = np.histogram(imageR, bins=4)
            upper_bound = bin_edges[1]
            lower_bound = bin_edges[0]
            imageR_out = np.zeros_like(imageR, dtype=np.uint8)
            mask = (imageR >= lower_bound) & (imageR <= upper_bound)
            imageR_out[mask] = ((imageR[mask] - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
            imageR_out[imageR > upper_bound] = np.mean(imageR_out[imageR <= upper_bound])
            imageR_out = (imageR_out - np.min(imageR_out)) / (np.max(imageR_out) - np.min(imageR_out)) * 255
    else:
        imageL_out, imageR_out = imageL / 255, imageR / 255
    
    imageL_final, imageR_final = enhance_image(imageL_out.astype(np.uint8)), enhance_image(imageR_out.astype(np.uint8))
    print(np.max(imageL_final), np.max(imageR_final))
    # cv2.imwrite("imageL.png", imageL_final)
    # cv2.imwrite("imageR.png", imageR_final)
    return imageL_final, imageR_final

def process_one_image(imageL, type):
    type=='hist_99'
    if type == "minmax":
        imageL_out = (imageL - np.min(imageL)) / (np.max(imageL) - np.min(imageL)) * 255
    elif type == "hist_99":
        imageL_out = hist_99(imageL)
    elif type == "intensity_bind":
        if np.max(imageL) < 35000:
            imageL_out = (imageL - np.min(imageL)) / (np.max(imageL) - np.min(imageL)) * 255
        else:
            intensity, bin_edges = np.histogram(imageL, bins=4)
            upper_bound = bin_edges[1]
            lower_bound = bin_edges[0]

            imageL_out = np.zeros_like(imageL, dtype=np.uint8)
            mask = (imageL >= lower_bound) & (imageL <= upper_bound)
            imageL_out[mask] = ((imageL[mask] - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
            imageL_out[imageL > upper_bound] = np.mean(imageL_out[imageL <= upper_bound])
            imageL_out = (imageL_out - np.min(imageL_out)) / (np.max(imageL_out) - np.min(imageL_out)) * 255
    else:
        imageL_out= imageL / 255
    

    # print(imageL_out.shape)
    # imageL_final = imageL_out#.astype(np.uint8)
    imageL_final = enhance_image(imageL_out.astype(np.uint8))
    # print(np.max(imageL_final), np.max(imageR_final))
    # cv2.imwrite("imageL.png", imageL_final)
    # cv2.imwrite("imageR.png", imageR_final)
    return imageL_final


class invNormalize(object):
    def __init__(self):
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        self.invTrans = transforms.Compose(
            [
                transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / std[0], 1 / std[1], 1 / std[2]]),
                transforms.Normalize(mean=[-mean[0], -mean[1], -mean[2]], std=[1.0, 1.0, 1.0]),
            ]
        )

    def __call__(self, x):
        return self.invTrans(x)


def get_transform():
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

def get_normalize():
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]

    return transforms.Compose(
        [
            transforms.Normalize(mean=mean, std=std)
        ]
    )


def load_fireview_img(path):
    agc = True
    img = imread(path).astype(np.float32)
    if len(img.shape) == 2:  # for non-agc images
        img = np.expand_dims(img, axis=2)
        agc = False
    return img, agc


def load_as_float_img(path):
    img = imread(path).astype(np.float32)
    if len(img.shape) == 2:  # for NIR and thermal images
        img = np.expand_dims(img, axis=2)
    return img


def load_as_float_depth(path):
    if "png" in path:
        depth = np.array(imread(path).astype(np.float32))
    elif "npy" in path:
        depth = np.load(path).astype(np.float32)
    elif "mat" in path:
        depth = loadmat(path).astype(np.float32)
    return depth


# Tempearture to raw thermal radiation value
# For the parameters (R,B,F,O), we use default values of FLIR A65C camera
def Celsius2Raw(celcius_degree):
    R = 380747
    B = 1428
    F = 1
    O = -88.539
    raw_value = R / (np.exp(B / (celcius_degree + 273.15)) - F) + O
    return raw_value


# Raw thermal radiation value to tempearture
def Raw2Celsius(Raw):
    R = 380747
    B = 1428
    F = 1
    O = -88.539
    Celsius = B / np.log(R / (Raw - O) + F) - 273.15
    return Celsius


def visualize_disp_as_numpy(disp, cmap="jet"):
    """
    Args:
        data (HxW): disp data
        cmap: color map (inferno, plasma, jet, turbo, magma, rainbow)
    Returns:
        vis_data (HxWx3): disp visualization (RGB)
    """

    disp = disp.cpu().numpy()
    disp = np.nan_to_num(disp)  # change nan to 0

    vmin = np.percentile(disp[disp != 0], 0)
    vmax = np.percentile(disp[disp != 0], 95)

    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    vis_data = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(np.uint8)
    return vis_data


def visualize_depth_as_numpy(depth, cmap="jet", is_sparse=True):
    """
    Args:
        data (HxW): depth data
        cmap: color map (inferno, plasma, jet, turbo, magma, rainbow)
    Returns:
        vis_data (HxWx3): depth visualization (RGB)
    """

    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0

    inv_depth = 1 / (x + 1e-6)

    if is_sparse:
        vmax = 1 / np.percentile(x[x != 0], 5)
    else:
        vmax = np.percentile(inv_depth, 95)

    normalizer = mpl.colors.Normalize(vmin=inv_depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    vis_data = (mapper.to_rgba(inv_depth)[:, :, :3] * 255).astype(np.uint8)
    if is_sparse:
        vis_data[inv_depth > vmax] = 0
    return vis_data
