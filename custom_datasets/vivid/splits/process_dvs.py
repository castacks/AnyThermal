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

from pytictoc import TicToc
from PIL import Image
from pyproj import Proj, transform
from pathlib import Path
import os
## global variables
t = TicToc()
width = 0
height = 0
# path_prefix = 'vivid/driving/'
imglist = []
cvbdg = CvBridge()
class allevents():
    def __init__(self):
        self.x = np.zeros((0,),dtype=np.uint16)
        self.y = np.zeros((0,),dtype=np.uint16)
        self.p = np.zeros((0,),dtype=np.bool)
        self.t = np.zeros((0,),dtype=np.float32)
        self.nevents = 0

def write_imu(msg, f_imu):
    tnow = msg.header.stamp.to_sec()
    rot_vel = msg.angular_velocity
    f_imu.write('%.6f, %.6f, %.6f, %.6f\n'%(rot_vel.x, rot_vel.y, rot_vel.z,tnow))

def update_event_q(msg, evtlist):
    global width, height
    width = msg.width
    height = msg.height
    nevents = np.shape(msg.events)[0]
    x_arr = np.zeros((nevents,),dtype=np.uint16)
    y_arr = np.zeros((nevents,),dtype=np.uint16)
    p_arr = np.zeros((nevents,),dtype=np.bool)
    t_arr = np.zeros((nevents,),dtype=np.float64)
    for i in range(nevents):
        x_arr[i] = msg.events[i].x
        y_arr[i] = msg.events[i].y
        p_arr[i] = msg.events[i].polarity
        t_arr[i] = msg.events[i].ts.to_sec()
    order = np.argsort(t_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    p_arr = p_arr[order]
    t_arr = t_arr[order]
    evtlist.x = np.concatenate((evtlist.x, x_arr));
    evtlist.y = np.concatenate((evtlist.y, y_arr));
    evtlist.p = np.concatenate((evtlist.p, p_arr));
    evtlist.t = np.concatenate((evtlist.t, t_arr));
    evtlist.nevents += nevents
    return evtlist

def generate_event_img(evpath, evtlist, im_stamp):
    global width, height
    dt = 0.005
    tstamp = im_stamp
    dvs_img0 = np.zeros((height,width, 3), dtype=np.uint8)

    e_idx = abs(evtlist.t - tstamp)<dt
    if np.count_nonzero(e_idx) < (0.01 * height * width):
        return
    itvl = np.round(np.count_nonzero(e_idx), 0)
    ev_x = evtlist.x[e_idx]
    ev_y = evtlist.y[e_idx]
    ev_p = evtlist.p[e_idx]
    ev_t = evtlist.t[e_idx]

    dvs_img0[ev_y[0*itvl:1*itvl], ev_x[0*itvl:1*itvl], ev_p[0*itvl:1*itvl]*2] = 255

    im0 = Image.fromarray(dvs_img0)
    im0.save(evpath + '/%6.6f.png' % tstamp)

def ther_minmax(msg, f_ther):
    cv_img = cvbdg.imgmsg_to_cv2(msg,desired_encoding="passthrough")
    f_ther.write('%.6f, %.6f\n'%(np.min(cv_img), np.max(cv_img)))

def main(folder_name, seqname):
    path_prefix = 'extracted_data/' + folder_name + '/'+seq_name + '/'
    os.makedirs(path_prefix,exist_ok=True)
    print("Extracting bag file: " + os.path.join(folder_name,seqname)+'.bag')

    bag = rosbag.Bag(os.path.join(folder_name,seqname)+'.bag')
    print("Loaded bag file: " + os.path.join(folder_name,seqname)+'.bag')
#    f_imu = open('imulist_'+seqname+'.txt','w')
#    f_ther = open('therminmax_'+seqname+'.txt','w')
    # gpsfile = path_prefix + 'gpslist.txt'
    # gpslist = open(gpsfile,"r")
    event_dir = path_prefix + 'evt/'
    os.makedirs(event_dir, exist_ok=True) 
    print("")

    image_dir = Path(path_prefix+'img/thermal/')
    im_list = list(image_dir.glob('*.png'))
    im_stamps = sorted([float(str(x)[-21:-4]) for x in im_list])
    im_stamps = [x for i, x in enumerate(im_stamps) if np.remainder(i,5)==0]
    im_stamps = np.array(im_stamps)

#    for topic, msg, _ in bag.read_messages(topics=['/dvs/imu','/thermal/image_raw']):
#        if topic == '/dvs/imu':
#            write_imu(msg, f_imu)
#        elif topic == '/thermal/image_raw':
#            ther_minmax(msg,f_ther)
    ## write images with timestamp assignment
    for i in range(len(im_stamps)):
        startT = rospy.Time.from_sec(im_stamps[i]-0.2)
        endT = rospy.Time.from_sec(im_stamps[i]+0.2)
        evtlist = allevents()
        for _, msg, _ in bag.read_messages(topics=['/dvs/events'], start_time=startT, end_time=endT):
            evtlist = update_event_q(msg, evtlist)
        generate_event_img(event_dir, evtlist, im_stamps[i])

    print('processed '+seqname+','+' total event images are : '+str(len(im_stamps)))

    bag.close()

if __name__ == '__main__':
    folder_name = sys.argv[1]
    seq_name = sys.argv[2][:-4]  # remove .bag extension
    main(folder_name,seq_name)
