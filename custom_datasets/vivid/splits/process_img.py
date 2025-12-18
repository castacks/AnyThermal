#! /usr/bin/env python

import rosbag
import rospy
import ros_numpy
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import numpy.matlib
import math
import copy
import cv2
from PIL import Image
from pyproj import Proj, transform
import os
import cv2
# Add fieldscale path
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/fieldscale/python")
from fieldscale import Fieldscale

# Fieldscale parameter
params_200_no_clahe = {
    'max_diff': 200, 'min_diff': 400, 'iteration': 7, 'gamma': 1.5,
    'clahe': False, 'video': False
}  # Base config, CLAHE toggled dynamically


## global variables
#f = open(folder+'imglist.txt','w')
imgno = 0
imglist = []
cvbdg = CvBridge()
proj_UTMK = Proj(init='epsg:5178')
proj_WGS84 = Proj(init='epsg:4326')
initpt = np.array([988118.672713562, 1818806.4875518726]) ## origin of campus, just for convenience.

def apply_fieldscale(img, use_clahe=False, folder_name=None, seqname=None):
    params = params_200_no_clahe.copy()
    params['clahe'] = use_clahe
    fs = Fieldscale(**params)
    tmp_path = f"{folder_name}_{seqname}_tmp_fieldscale_input.png"
    cv2.imwrite(tmp_path, img)
    rescaled = fs(tmp_path)
    os.remove(tmp_path)  # Clean up temporary file
    return rescaled
    
def write_gps(msg, f_gps):
    global proj_UTMK, proj_WGS84
    tnow = msg.header.stamp.to_sec()
    lat = msg.latitude
    lon = msg.longitude
    x, y = transform(proj_WGS84,proj_UTMK,lon,lat)
    f_gps.write('%.6f, %.6f, %.6f\n'%(x-initpt[0],y-initpt[1],tnow))


def apply_clahe(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def normalize_16bit_to_8bit(image: np.ndarray, apply_clahe=False) -> np.ndarray:
    p1, p99 = np.percentile(image, (1, 99))
    stretched = np.clip((image - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
    return stretched


def write_images(msg, folder,modality,scenario,seqname=None):
    global f, imgno, imglist, cvbdg
    imgno += 1
    if (imgno % 10) != 0:
        return
    tnow = msg.header.stamp.to_sec()
    cv_img = cvbdg.imgmsg_to_cv2(msg,desired_encoding="passthrough")
    if cv_img is None:
        print("Image is None, skipping...")
        return
    # import pdb; pdb.set_trace()
    if "driving" in scenario:
        if modality == "thermal":
            DIM = (640, 512)
            K_1 = np.array([[445.34173838383924, 0., 310.74708274781557], [0., 446.40695195454197, 249.54892754326676], [0., 0., 1.]])
            D_1 = np.array([[0.248994091117758, -2.909390816436668, 8.052903070857168, -7.363581411738435]])
        elif modality == "color":
            DIM = (1280, 1024)
            K_1 = np.array([[702.6030497884977, 0., 644.9296487349911], [0., 703.4541726858521, 526.4572414665469], [0., 0., 1.]])
            D_1 = np.array([[0.5424763304847061, -1.7515099175022195, 6.050489512760127,-5.786170959900578]])
    elif "handheld" in scenario:
        if modality == "thermal":

            DIM = (640, 512)
            K_1 = np.array([[437.38861083256637, 0., 323.5284494924228], [0., 437.29475745770907, 256.36315482047905], [0., 0., 1.]])
            D_1 = np.array([[-0.431640703150235, 0.17811649959498482, -0.0014905118708651568,-0.0011389876158246133]])
        elif modality == "color":
            DIM = (640, 480)
            K_1 = np.array([[531.0787710720505, 0., 322.0434384516226], [0., 530.9280349280389, 239.58555406488455], [0., 0., 1.]])
            D_1 = np.array([[0.029762942995503135, -0.0835531221056956, 0.000913361749036359,0.0029817041017012043]])
    else:
        print("Unknown scenario: " + scenario)
        return
    map1_1, map2_1 = cv2.fisheye.initUndistortRectifyMap(K_1, D_1, np.eye(3), K_1, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(cv_img, map1_1, map2_1, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    if modality == "thermal":
        folder_name = "thermal_raw"
    else:
        folder_name = modality
    cv2.imwrite(folder+f"/{folder_name}/"+'%.6f.png'%(tnow), undistorted_img)
    if modality == "thermal":
        cv_img_8bit = normalize_16bit_to_8bit(cv_img)
        cv_img_8bit_clahe = apply_clahe(cv_img_8bit)
        try:
            cv_img_fieldscale = apply_fieldscale(cv_img, False,folder_name,seqname)
            cv_img_fieldscale_clahe = apply_fieldscale(cv_img, True,folder_name,seqname)
            cv2.imwrite(folder+f"/{modality}_fieldscale/"+'%.6f.png'%(tnow), cv_img_fieldscale)
            cv2.imwrite(folder+f"/{modality}_fieldscale_clahe/"+'%.6f.png'%(tnow), cv_img_fieldscale_clahe)
        except:
            print(f"Fieldscale processing failed. Skipping fieldscale images for {folder} {seqname} {modality} at {tnow}")
        cv2.imwrite(folder+f"/{modality}_8/"+'%.6f.png'%(tnow), cv_img_8bit)
        cv2.imwrite(folder+f"/{modality}_clahe/"+'%.6f.png'%(tnow), cv_img_8bit_clahe)

#    imglist.append([str(imgno) + "%06i.png"%(imgno)])
#    f.write("%12.12f, %06i\n"%(tnow,imgno))

def raw_to_kelvin(val):
    return ( 1428.0 / log( 408825.0 / ( val + 58.417 ) + 1.0 ) )

def main(folder_name, seqname):
    path_prefix = os.path.join("extracted_data", folder_name, seqname + '/')
    os.makedirs(path_prefix, exist_ok=True)
    print("Extracting bag file: " + os.path.join(folder_name,seqname)+'.bag')
    bag = rosbag.Bag(os.path.join(folder_name,seqname)+'.bag')
    print("Loaded bag file: " + os.path.join(folder_name,seqname)+'.bag')
    folder = path_prefix+'img'
    os.makedirs(folder, exist_ok=True)
    os.makedirs(folder+"/thermal_raw", exist_ok=True)
    os.makedirs(folder+"/thermal_clahe", exist_ok=True)
    os.makedirs(folder+"/thermal_8", exist_ok=True)
    os.makedirs(folder+"/thermal_fieldscale", exist_ok=True)
    os.makedirs(folder+"/thermal_fieldscale_clahe", exist_ok=True)
    os.makedirs(folder+"/thermal", exist_ok=True)

    os.makedirs(folder+"/color", exist_ok=True)

    f_gps = open(path_prefix+'gpslist.txt','w')

    # write images with timestamp assignment
    # for topic, msg, t in bag.read_messages(topics=['/gps']):
    #     write_gps(msg, f_gps)

#    for topic, msg, t in bag.read_messages(topics=['/camera/image_color']):
#        if 'camera
    print("PRocessing for scenario: " + folder_name)
    # for topic, msg, t in bag.read_messages(topics=['/thermal/image_raw','/camera/image_color','/rgb/image']):
    for topic, msg, t in bag.read_messages(topics=['/thermal/image_raw']):
        if 'thermal' in topic:
            write_images(msg, folder,"thermal",folder_name,seqname)
        elif 'color' in topic or "rgb" in topic:
            write_images(msg, folder,"color",folder_name)


    print('saved '+str(imgno)+' images timstamps to detections.')

    bag.close()

if __name__ == '__main__':
    folder_name = sys.argv[1]
    seq_name = sys.argv[2][:-4]  # Remove the '.bag' extension
    main(folder_name,seq_name)
