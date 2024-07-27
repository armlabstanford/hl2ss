#------------------------------------------------------------------------------
# This script receives video from the HoloLens front RGB camera and plays it.
# The camera supports various resolutions and framerates. See
# https://github.com/jdibenes/hl2ss/blob/main/etc/pv_configurations.txt
# for a list of supported formats. The default configuration is 1080p 30 FPS. 
# The stream supports three operating modes: 0) video, 1) video + camera pose, 
# 2) query calibration (single transfer).
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm

# RAP-SAM (TODO: change the path) --------------------------------------------------------------------
from pycocotools import mask as coco_mask # to decode segmented mask (coco rle format)
import demo.segmentation as rapsam
config_path = '/home/yu/RAP-SAM/configs/rap_sam/eval_rap_sam_coco.py'
checkpoint_path = '/home/yu/RAP-SAM/rapsam_r50_12e.pth'
# image_path = '/home/yu/RAP-SAM/demo/demo2_s.jpg'
# output_path = '/home/yu/RAP-SAM/output/vis'

# Initialize the model
model = rapsam.init_inferencer(config_path, checkpoint_path)

# Settings --------------------------------------------------------------------

# HoloLens address
host = "192.169.1.35"

# Operating mode
# 0: video
# 1: video + camera pose
# 2: query calibration (single transfer)
mode = hl2ss.StreamMode.MODE_1

# Enable Mixed Reality Capture (Holograms)
enable_mrc = False

# Enable Shared Capture
# If another program is already using the PV camera, you can still stream it by
# enabling shared mode, however you cannot change the resolution and framerate
shared = False

# Camera parameters
# Ignored in shared mode
width     = 640
height    = 360
framerate = 30

# Framerate denominator (must be > 0)
# Effective FPS is framerate / divisor
divisor = 10

# Video encoding profile
profile = hl2ss.VideoProfile.H265_MAIN

# Decoded format
# Options include:
# 'bgr24'
# 'rgb24'
# 'bgra'
# 'rgba'
# 'gray8'
decoded_format = 'bgr24'

#------------------------------------------------------------------------------

hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=enable_mrc, shared=shared)

if (mode == hl2ss.StreamMode.MODE_2):
    data = hl2ss_lnm.download_calibration_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width, height, framerate)
    print('Calibration')
    print(f'Focal length: {data.focal_length}')
    print(f'Principal point: {data.principal_point}')
    print(f'Radial distortion: {data.radial_distortion}')
    print(f'Tangential distortion: {data.tangential_distortion}')
    print('Projection')
    print(data.projection)
    print('Intrinsics')
    print(data.intrinsics)
    print('RigNode Extrinsics')
    print(data.extrinsics)
else:
    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.esc
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height, framerate=framerate, divisor=divisor, profile=profile, decoded_format=decoded_format)
    client.open()

    while (enable):
        data = client.get_next_packet()

        # print(f'Pose at time {data.timestamp}')
        # print(data.pose)
        # print(f'Focal length: {data.payload.focal_length}')
        # print(f'Principal point: {data.payload.principal_point}')
        img_rgb = data.payload.image
        cv2.imshow('Video', img_rgb)

        # Real-time segmentation on RGB input (visualization)
        inference = rapsam.infer(model, img_rgb) # inference result:dict
        img_seg = inference['visualization'][0] # visualization of inference
        cv2.imshow('Segmentation', img_seg)

        # TODO: select based on label index == object
        mask_binary = rapsam.get_combined_binary_mask(inference)
        if mask_binary is not None:
            # Apply the mask to your images if it exists
            img_rgb_masked = img_rgb * mask_binary[:, :, None]  # Apply mask to RGB image
            img_seg_masked = img_seg * mask_binary[:, :, None]  # Apply mask to segmentation image
        else:
            # If no mask, use the original images
            img_rgb_masked = img_rgb * 0
            img_seg_masked = img_seg * 0

        cv2.imshow('Mask for Object 0', img_rgb_masked)
        cv2.waitKey(1)

    client.close()
    listener.join()

hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
