# #!/usr/bin/env python

import rospy
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf_conversions import transformations 

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

# INITIALIZE NODE
# rospy.init_node('hololens2_data_publisher', anonymous=True)

# DECLARE 
# NOTE: Also declare publisher and broadcaster outside the loop
# image_pub = rospy.Publisher("/camera/image_rect", Image, queue_size=10)
# info_pub = rospy.Publisher("/camera/camera_info", CameraInfo, queue_size=10)
# br = TransformBroadcaster()

# DECLARE CONSTANTS
HEIGHT = 1920 # image height
WIDTH = 1080 # image width

# camera info: from camera_calibration on not broken hololens (NO.3)
D = [0.01578507813331108, -0.0425577690521911, 0.001535108606462915, -0.0005645046966343704, 0.0] # distortion
K = [1481.8690580024659, 0.0, 936.7853966108089, 0.0, 1485.0033867902052, 511.25986612371463, 0.0, 0.0, 1.0] # pv camera instrinsics (3x3)
R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] # rectification (only for stereo)
P = [1480.5687990452186, 0.0, 935.294874799051, 0.0, 0.0, 1487.8845772206278, 512.3236664290798, 0.0, 0.0, 0.0, 1.0, 0.0] # projection matrix (3x4)

# ------------------------- publisher for /camera/camera_info and /camera/image_rect (for apriltag_ros) -----------------------------
def publish_camera_data(image, image_pub, info_pub):
    bridge = CvBridge()

    # Set a fixed camera info (adjust parameters as needed)
    camera_info = CameraInfo()
    camera_info.header.frame_id = "camera_cv"
    camera_info.height = HEIGHT
    camera_info.width = WIDTH
    
    camera_info.D = D
    camera_info.K = K
    camera_info.R = R
    camera_info.P = P

    camera_info.header.stamp = rospy.Time.now()
    image_msg = bridge.cv2_to_imgmsg(image, "bgr8")
    image_msg.header = camera_info.header
    image_pub.publish(image_msg)
    info_pub.publish(camera_info)

    # rate = rospy.Rate(120)  # 10hz
    # rate.sleep()

# ------------------------- tf broadcaster -----------------------------
def broadcast_tf(camera_pose, br):
    # Create a TransformStamped message for world to camera
    world_to_camera = TransformStamped()
    world_to_camera.header.frame_id = "hl_world"
    world_to_camera.child_frame_id = "camera_cv" # broadcast converted camera frame in OpenCV from OpenGL

    # get camera pose cv
    camera_pose_gl = camera_pose.T # transpose because the camera pose from hl2ss is a transposed transformation 
    camera_pose_cv = opengl_to_opencv(camera_pose_gl) # gl to cv

    # Extract translation from the matrix
    translation = camera_pose_cv[0:3, 3]
    world_to_camera.transform.translation.x = translation[0]
    world_to_camera.transform.translation.y = translation[1]
    world_to_camera.transform.translation.z = translation[2]

    # Extract quaternion from 4x4 transformation matrix SE3 (tho only the rotation matrix is used (SO3))
    # https://github.com/ros/geometry/issues/64
    quaternion = transformations.quaternion_from_matrix(camera_pose_cv)
    world_to_camera.transform.rotation.x = quaternion[0]
    world_to_camera.transform.rotation.y = quaternion[1]
    world_to_camera.transform.rotation.z = quaternion[2]
    world_to_camera.transform.rotation.w = quaternion[3]

    world_to_camera.header.stamp = rospy.Time.now()
    br.sendTransform(world_to_camera)

def opengl_to_opencv(camera_pose_gl):
    # transformation from gl to cv (gl pose in cv frame)
    T_gl_to_cv = np.array([
                            [1, 0, 0, 0],
                            [0,-1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]
                        ])
    camera_pose_cv = T_gl_to_cv @ camera_pose_gl
    return camera_pose_cv

# TESTER
# if __name__ == '__main__':
#     try:
#         camera_pose = []
#         broadcast_transforms(camera_pose)
#     except rospy.ROSInterruptException:
#         pass
