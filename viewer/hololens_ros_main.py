#!/usr/bin/env python3

import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm

import rospy
from hololens_ros_utils import broadcast_tf, publish_camera_data
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import TransformBroadcaster

# Settings --------------------------------------------------------------------

# HoloLens address
host = "192.169.1.45"

# Operating mode
# 0: video
# 1: video + camera pose
# 2: query calibration (single transfer)
mode = hl2ss.StreamMode.MODE_1

enable_mrc = False # Enable Mixed Reality Capture (Holograms)
shared = False # Enable Shared Capture

# Camera parameters
width     = 1920
height    = 1080
framerate = 30

divisor = 1 # Effective FPS is framerate / divisor
profile = hl2ss.VideoProfile.H265_MAIN # Video encoding profile
decoded_format = 'bgr24' # Decoded format

def main():
    # Initialilze ROS node
    rospy.init_node('hololens2_data_publisher')
    # rate = rospy.Rate(10) # set loop rate

    # Declare pub and br
    image_pub = rospy.Publisher("/camera/image_rect", Image, queue_size=10)
    info_pub = rospy.Publisher("/camera/camera_info", CameraInfo, queue_size=10)
    br = TransformBroadcaster()   

    # Start the client object to get HoloLens2 data
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=enable_mrc, shared=shared)
    client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height, framerate=framerate, divisor=divisor, profile=profile, decoded_format=decoded_format)
    client.open()

    # loop inside this
    while not rospy.is_shutdown():
        # Get the next data packet from HoloLens2
        data = client.get_next_packet()

        # Publish image data + broadcast tf
        broadcast_tf(data.pose, br=br) 
        publish_camera_data(data.payload.image, image_pub=image_pub, info_pub=info_pub) 

    # after the loop, exit
    client.close()
    hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
