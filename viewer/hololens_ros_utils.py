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

# ------------------------- publisher for /camera/camera_info and /camera/image_rect (for apriltag_ros) -----------------------------
def publish_camera_data(image, image_pub, info_pub):
    bridge = CvBridge()

    # Set a fixed camera info (adjust parameters as needed)
    camera_info = CameraInfo()
    camera_info.header.frame_id = "camera"
    camera_info.height = 1080
    camera_info.width = 1920
    camera_info.K = [-1474.3679, 0, 0.0, 0, 1474.7207, 0.0, 0, 0, 1] # Example intrinsic matrix
    camera_info.P = [-1474.3679, 0, 0.0, 0, 0, 1474.7207, 0.0, 0, 0, 0, 1, 0]

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
    world_to_camera.child_frame_id = "camera"

    # Extract translation from the matrix
    camera_pose = camera_pose.T # transpose because the output from hl2ss data.pose is transposed
    # print(f"Camera pose {camera_pose}")
    translation = camera_pose[0:3, 3]

    # print(f"translation {translation}")
    world_to_camera.transform.translation.x = translation[0]
    world_to_camera.transform.translation.y = translation[1]
    world_to_camera.transform.translation.z = translation[2]

    # Extract rotation matrix and convert to quaternion
    quaternion = transformations.quaternion_from_matrix(camera_pose)
    world_to_camera.transform.rotation.x = quaternion[0]
    world_to_camera.transform.rotation.y = quaternion[1]
    world_to_camera.transform.rotation.z = quaternion[2]
    world_to_camera.transform.rotation.w = quaternion[3]
    

    
    world_to_camera.header.stamp = rospy.Time.now()
    br.sendTransform(world_to_camera)

    # rate = rospy.Rate(10)  # 10 Hz
    # rate.sleep()

# TESTER
# if __name__ == '__main__':
#     try:
#         camera_pose = []
#         broadcast_transforms(camera_pose)
#     except rospy.ROSInterruptException:
#         pass
