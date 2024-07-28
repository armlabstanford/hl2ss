#------------------------------------------------------------------------------
# This script demonstrates how to create aligned RGBD images, which can be used
# with Open3D, from the depth and front RGB cameras of the HoloLens.
# Press space to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import multiprocessing as mp
import numpy as np
import open3d as o3d
import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv


import torch
import matplotlib.pyplot as plt
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

import numpy as np

# Depth Anything Setup --------------------------------------------------------------------
# load the depth anything model
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitb' # or 'vits', 'vitb'
dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20 # 20 for indoor model, 80 for outdoor model

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
model.load_state_dict(torch.load(f'/home/yu/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))

model = model.to(DEVICE).eval()

# RAP-SAM (TODO: change the path) --------------------------------------------------------------------
from pycocotools import mask as coco_mask # to decode segmented mask (coco rle format)
import demo.segmentation as rapsam
config_path = '/home/yu/RAP-SAM/configs/rap_sam/eval_rap_sam_coco.py'
checkpoint_path = '/home/yu/RAP-SAM/rapsam_r50_12e.pth'
# image_path = '/home/yu/RAP-SAM/demo/demo2_s.jpg'
# output_path = '/home/yu/RAP-SAM/output/vis'

# Initialize the model
model_seg = rapsam.init_inferencer(config_path, checkpoint_path)

# Settings --------------------------------------------------------------------

# HoloLens address
host = '192.169.1.35'

# Calibration path (must exist but can be empty)
calibration_path = '../calibration'

# Front RGB camera parameters
pv_width = 640
pv_height = 360
pv_framerate = 30

# Buffer length in seconds
buffer_length = 10

# Maximum depth in meters
max_depth = 3.0

#------------------------------------------------------------------------------
# D and D alignment through least square fitting (frame by frame)
def learn_scale_and_offset_raw(dense_depth, sparse_depth):
    dense_depth_flat = dense_depth.flatten()
    sparse_depth_flat = sparse_depth.flatten()

    valid_mask = sparse_depth_flat > 0
    dense_depth_valid = dense_depth_flat[valid_mask]
    sparse_depth_valid = sparse_depth_flat[valid_mask]

    # Prepare the design matrix (A) and target vector (b)
    A = np.vstack([dense_depth_valid, np.ones_like(dense_depth_valid)]).T
    b = sparse_depth_valid

    # Solve the normal equation A.T * A * x = A.T * b
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    scale, offset = x
    return scale, offset
#------------------------------------------------------------------------------

if __name__ == '__main__':
    # Keyboard events ---------------------------------------------------------
    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.space
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Start PV Subsystem ------------------------------------------------------
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Get RM Depth Long Throw calibration -------------------------------------
    # Calibration data will be downloaded if it's not in the calibration folder
    calibration_lt = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, calibration_path)

    uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)

    # Create Open3D visualizer ------------------------------------------------
    o3d_lt_intrinsics = o3d.camera.PinholeCameraIntrinsic(hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT, calibration_lt.intrinsics[0, 0], calibration_lt.intrinsics[1, 1], calibration_lt.intrinsics[2, 0], calibration_lt.intrinsics[2, 1])

    
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # pcd = o3d.geometry.PointCloud()
    # first_pcd = True

    # Start PV and RM Depth Long Throw streams --------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height, framerate=pv_framerate, decoded_format='rgb24'))
    producer.configure(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
    producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, pv_framerate * buffer_length)
    producer.initialize(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS * buffer_length)
    producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.start(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
    sink_depth = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, manager, ...)

    sink_pv.get_attach_response()
    sink_depth.get_attach_response()

    # Initialize PV intrinsics and extrinsics ---------------------------------
    pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
    pv_extrinsics = np.eye(4, 4, dtype=np.float32)
 
    # Main Loop ---------------------------------------------------------------
    while (enable):
        # Wait for RM Depth Long Throw frame ----------------------------------
        sink_depth.acquire()

        # Get RM Depth Long Throw frame and nearest (in time) PV frame --------
        _, data_lt = sink_depth.get_most_recent_frame()
        if ((data_lt is None) or (not hl2ss.is_valid_pose(data_lt.pose))):
            continue

        _, data_pv = sink_pv.get_nearest(data_lt.timestamp)
        if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
            continue

        # Preprocess frames ---------------------------------------------------
        raw_depth_lt = hl2ss_3dcv.rm_depth_undistort(data_lt.payload.depth, calibration_lt.undistort_map)
        normalized_depth_lt = hl2ss_3dcv.rm_depth_normalize(raw_depth_lt, scale)
        img_rgb = data_pv.payload.image # RGB, HxW
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB) # opencv bgr to rgb

        # RGB to Depth (from Depth-Anything) ---------------------------------------------------
        raw_depth_DA = model.infer_image(img_rgb) # HxW raw depth map in numpy
        normalized_depth_DA = cv2.normalize(raw_depth_DA, None, 0, 255, cv2.NORM_MINMAX) # Normalize the depth map for display
        normalized_depth_DA = normalized_depth_DA.astype('uint8')
        colored_depth_DA = cv2.applyColorMap(normalized_depth_DA, cv2.COLORMAP_JET) # DEPTH, HxW

        # # print the metric depth of the center pixel
        # center_pixel = (int(normalized_depth.shape[1] / 2), int(normalized_depth.shape[0] / 2))
        # print(f"Depth at center pixel: {depth_output[center_pixel[1], center_pixel[0]]}")g

        # Update PV intrinsics ------------------------------------------------
        # PV intrinsics may change between frames due to autofocus
        pv_intrinsics = hl2ss.update_pv_intrinsics(pv_intrinsics, data_pv.payload.focal_length, data_pv.payload.principal_point)
        color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)
        
        # Generate aligned RGBD image -----------------------------------------
        lt_points         = hl2ss_3dcv.rm_depth_to_points(xy1, normalized_depth_lt)
        lt_to_world       = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(data_lt.pose)
        world_to_lt       = hl2ss_3dcv.world_to_reference(data_lt.pose) @ hl2ss_3dcv.rignode_to_camera(calibration_lt.extrinsics)
        world_to_pv_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(color_extrinsics) @ hl2ss_3dcv.camera_to_image(color_intrinsics)
        world_points      = hl2ss_3dcv.transform(lt_points, lt_to_world)
        pv_uv             = hl2ss_3dcv.project(world_points, world_to_pv_image)
        color_remapped    = cv2.remap(img_rgb, pv_uv[:, :, 0], pv_uv[:, :, 1], cv2.INTER_LINEAR)

        raw_depth_remapped= cv2.remap(raw_depth_DA, pv_uv[:, :, 0], pv_uv[:, :, 1], cv2.INTER_LINEAR)
        colored_depth_remapped    = cv2.remap(colored_depth_DA, pv_uv[:, :, 0], pv_uv[:, :, 1], cv2.INTER_LINEAR)

        mask_uv = hl2ss_3dcv.slice_to_block((pv_uv[:, :, 0] < 0) | (pv_uv[:, :, 0] >= pv_width) | (pv_uv[:, :, 1] < 0) | (pv_uv[:, :, 1] >= pv_height))
        normalized_depth_lt[mask_uv] = 0


        # extra for DD alignment
        raw_depth_remapped = np.expand_dims(raw_depth_remapped, axis=-1)
        raw_depth_lt = np.expand_dims(raw_depth_lt, axis=-1)
        raw_depth_remapped[mask_uv]=0 # [m]
        raw_depth_lt[mask_uv]=0 # [mm]
        # ----------------------- check the mean value of the depth 
        mean_array1 = np.mean(raw_depth_lt)
        mean_array2 = np.mean(raw_depth_remapped *1000)
        # print(f"Average of raw depth lt: {mean_array1}, raw depth DA: {mean_array2}")

        # ----------------------- calculate the alignment parameters
        # raw_depth_remapped = raw_depth_remapped * 1000 # [m] to [mm]
        raw_depth_lt = raw_depth_lt / 1000 # [mm] to [m]
        scale_ , offset_= learn_scale_and_offset_raw(raw_depth_remapped, raw_depth_lt)
        print(f'Alignment parameters: scale = {scale_} and offset = {offset_} [m]') 

        # print(f'normalized depth:{normalized_depth_lt.shape}, raw depth:{raw_depth_lt.shape} ')
        # print(f'normalized depth:{normalized_depth_lt[10]}, raw depth:{raw_depth_lt[10]} ')

        # Display RGBD --------------------------------------------------------
        # image = np.hstack((hl2ss_3dcv.rm_depth_to_rgb(normalized_depth_lt) / 8, color / 255)) # Depth scaled for visibility
        # cv2.imshow('RGBD', image)

        # image = np.hstack((hl2ss_3dcv.rm_depth_to_rgb(normalized_depth_lt) / 8, colored_depth_remapped / 255)) # Depth scaled for visibility
        # cv2.imshow('DD', image)

        #-----------------------COMPUTE ALIGNED DEPTH -----------------------
        aligned_depth = raw_depth_DA * scale_ + offset_ # [m]
        normalized_aligned_depth_DA = cv2.normalize(aligned_depth, None, 0, 255, cv2.NORM_MINMAX) # Normalize the depth map for display
        normalized_aligned_depth_DA = normalized_aligned_depth_DA.astype('uint8')
        colored_aligned_depth_DA = cv2.applyColorMap(normalized_aligned_depth_DA, cv2.COLORMAP_JET) # DEPTH, HxW

        # -----------------------SEGMENTATION -----------------------
        inference = rapsam.infer(model_seg, img_rgb) # inference result:dict
        img_seg = inference['visualization'][0] # visualization of inference
        cv2.imshow('Segmentation', img_seg)

        # TODO: select based on label index == object
        mask_binary = rapsam.get_combined_binary_mask(inference)
        if mask_binary is not None:
            # Apply the mask to your images if it exists
            img_rgb_masked = img_rgb * mask_binary[:, :, None]  # Apply mask to RGB image
            img_depth_masked = colored_aligned_depth_DA * mask_binary[:, :, None]  # Apply mask to segmentation image
        else:
            # If no mask, use the original images
            img_rgb_masked = img_rgb * 0
            img_depth_masked = colored_aligned_depth_DA * 0

        

        # ------------------- VISUALIZATION ------------------ 
        cv2.imshow('RGB (from PV)', img_rgb)
        cv2.imshow('Depth (aligned)', colored_aligned_depth_DA)
        # cv2.imshow('RGB (segmented)', img_rgb_masked)
        cv2.imshow('Depth (segmented)', img_depth_masked)
        cv2.waitKey(1)

        

        # # depth sanity check
        # center_pixel = (int(normalized_aligned_depth_DA.shape[1] / 2), int(normalized_aligned_depth_DA.shape[0] / 2))
        # print(f"Depth at center pixel: {aligned_depth[center_pixel[1], center_pixel[0]]}")

        # Convert to Open3D RGBD image and create pointcloud ------------------
        # color_image_lt = o3d.geometry.Image(color_remapped)
        # depth_image_lt = o3d.geometry.Image(normalized_depth_lt)
        color_image = o3d.geometry.Image(img_rgb)
        depth_image = o3d.geometry.Image(aligned_depth)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image, depth_scale=1, depth_trunc=max_depth, convert_rgb_to_intensity=False)
        # rgbd_lt = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image_lt, depth_image_lt, depth_scale=1, depth_trunc=max_depth, convert_rgb_to_intensity=False)
        o3d_pv_intrinsics = o3d.camera.PinholeCameraIntrinsic(pv_width, pv_height, color_intrinsics[0, 0], color_intrinsics[1, 1], color_intrinsics[2, 0], color_intrinsics[2, 1])
        tmp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_pv_intrinsics)
        # tmp_pcd_lt = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_lt, o3d_lt_intrinsics)

        # Display pointcloud --------------------------------------------------
        # pcd.points = tmp_pcd.points
        # pcd.colors = tmp_pcd.colors

        # # print(f'point cloud length: {len(pcd.points)} ')
        # print(f'point cloud first point: {pcd.points[0]} ')
        # print(f'point cloud last point: {pcd.points[-1]} ')
        # # print(f'point cloud middle point?: {pcd.points[len(pcd.points)/2]} ')

        # if (first_pcd):
        #     vis.add_geometry(pcd)
        #     first_pcd = False
        # else:
        #     vis.update_geometry(pcd)

        # vis.poll_events()
        # vis.update_renderer()

    # Stop PV and RM Depth Long Throw streams ---------------------------------
    sink_pv.detach()
    sink_depth.detach()
    producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.stop(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

    # Stop PV subsystem -------------------------------------------------------
    hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Stop keyboard events ----------------------------------------------------
    listener.join()