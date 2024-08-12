import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import rospy
from hololens_ros_utils import broadcast_tf, publish_camera_data
from sensor_msgs.msg import Image, CameraInfo,PointCloud2 , PointField
from std_msgs.msg import Header , Bool
from tf2_ros import TransformBroadcaster
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2

# Settings --------------------------------------------------------------------

# HoloLens address
host = "192.169.1.41"

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
framerate = 15

divisor = 1 # Effective FPS is framerate / divisor
profile = hl2ss.VideoProfile.H265_MAIN # Video encoding profile
decoded_format = 'bgr24' # Decoded format

apriltag_detection_status = False

def april_tag_callback(status_msg):
    global apriltag_detection_status
    apriltag_detection_status = status_msg.data
    rospy.loginfo(f"detection {apriltag_detection_status}")

def generate_empty_cloud(num_points=0):
    points = np.random.rand(num_points,3)
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "tag_2"
    fields = [
        PointField(name ="x" , offset=0, datatype = PointField.FLOAT32, count=1),
        PointField(name ="y" , offset=4, datatype = PointField.FLOAT32, count=1),
        PointField(name ="z" , offset=8, datatype = PointField.FLOAT32, count=1)
    ]
    point_cloud = pc2.create_cloud(header, fields, points)
    return point_cloud


def generate_random_point_cloud(num_points=1000):
    points = np.random.rand(num_points,3)
    points = points * 0.1
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "tag_2"
    fields = [
        PointField(name ="x" , offset=0, datatype = PointField.FLOAT32, count=1),
        PointField(name ="y" , offset=4, datatype = PointField.FLOAT32, count=1),
        PointField(name ="z" , offset=8, datatype = PointField.FLOAT32, count=1)
    ]

    print(header.frame_id)

    point_cloud = pc2.create_cloud(header, fields, points)
    return point_cloud


def main():
    # Initialilze ROS node
    rospy.init_node('hololens2_data_publisher')
    # rate = rospy.Rate(10) # set loop rate

    # Declare pub and br
    image_pub = rospy.Publisher("/camera/image_raw", Image, queue_size=10)
    info_pub = rospy.Publisher("/camera/camera_info", CameraInfo, queue_size=10)
    pointcloud_pub = rospy.Publisher("random_point_cloud", PointCloud2, queue_size=10)
    rospy.Subscriber('apriltag_status', Bool, april_tag_callback)

    br = TransformBroadcaster()

    # # Set the rates for the image publisher and the TF broadcaster
    # image_rate = rospy.Rate(120)  
    # tf_rate = rospy.Rate(30)     

    # Start the client object to get HoloLens2 data
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=enable_mrc, shared=shared)
    client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height, framerate=framerate, divisor=divisor, profile=profile, decoded_format=decoded_format)
    client.open()
    rate = rospy.Rate(10) 

    # loop inside this
    while not rospy.is_shutdown():
        # Get the next data packet from HoloLens2
        data = client.get_next_packet()
        # cv2.imshow('Video', data.payload.image)
        # cv2.waitKey(1)
        # Publish image data + broadcast tf
        broadcast_tf(data.pose, br=br) # ~10 HZ
        publish_camera_data(data.payload.image, image_pub=image_pub, info_pub=info_pub) 

        # # Sleep to maintain the loop rate
        # tf_rate.sleep()
        # image_rate.sleep()
        if apriltag_detection_status:
            point_cloud_msg = generate_random_point_cloud()
            rospy.loginfo("Publishing random point cloud")
        else:
            point_cloud_msg = generate_empty_cloud()
            rospy.loginfo("Tag not detected")
        pointcloud_pub.publish(point_cloud_msg)


    # after the loop, exit
    client.close()
    hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass





