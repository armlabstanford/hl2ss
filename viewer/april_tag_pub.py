import rospy
import tf
from sensor_msgs.msg import PointCloud2 , Image
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from apriltag_ros.msg import AprilTagDetectionArray
import numpy as np

apriltag_detection_status = False
tf_listener = None
new_point_cloud_publisher = None
 
def point_cloud_callback(msg):
    global apriltag_detection_status
    
    
    if tf_listener is None:
        rospy.logwarn("Tf_listener not initialized")
        return 
    
    points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    points_lists = list(points)
    # rospy.loginfo(f"Received point cloud with {len(points_lists)} points")
    # for point in points_lists[:10]:
    #     rospy.loginfo(f"Point: x={point[0]}, y={point[1]}, z={point[2]}")

    tranformed_points = []
    for point in points_lists:
        point_transformed = transform_point(point)
        tranformed_points.append(point_transformed)
        # rospy.loginfo(f"Trasformed Point: x={point_transformed[0]}, y ={point_transformed[1]}, z={point_transformed[2]}")

    if not tranformed_points:
        rospy.logwarn("No points were transferred")
    # else:
    #     rospy.loginfo(f"Transformed point cloud with {len(tranformed_points)}points")
    
    try:
    # Republish the received PointCloud2 message
        msg = pc2.create_cloud_xyz32(msg.header, tranformed_points)
        rospy.loginfo("Created new pointcloud2 message")
        new_point_cloud_publisher.publish(msg)
    except Exception as e:
        rospy.logerr(f"Failed to create")


# empty_cloud = pc2.create_cloud_xyz32(msg.header, [])
#         new_point_cloud_publisher.publish(empty_cloud)
#         return
def apriltag_callback(apriltag_msg):
    global apriltag_detection_status
    apriltag_detection_status_publisher = rospy.Publisher("apriltag_status", Bool , queue_size=10)
    if not apriltag_msg.detections:
        msg = Bool()
        msg.data = apriltag_detection_status
        apriltag_detection_status_publisher.publish(apriltag_detection_status)
        rospy.loginfo("AprilTag not detected")
        apriltag_detection_status = False
    else:
        apriltag_detection_status = True
        msg = Bool()
        msg.data = apriltag_detection_status
        apriltag_detection_status_publisher.publish(apriltag_detection_status)
        for detection in apriltag_msg.detections:
            pose = detection.pose.pose.pose

            rospy.loginfo("AprilTag detected at: %s", pose)



def transform_point(point):
    global tf_listener , new_point_cloud_publisher
    if tf_listener is None:
        rospy.logwarn("tf-listener not initialized")
        return point 
    try:
    
        (trans,rot) = tf_listener.lookupTransform('/camera' , '/tag_2' , rospy.Time(0))
        transform_matrix = tf. transformations.compose_matrix(translate= trans, angles= tf.transformations.euler_from_quaternion(rot))
        point_h = np.array([point[0], point[1], point[2] , 1.0])
        point_transformed = np.dot(transform_matrix, point_h)
        return point_transformed[:3]
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        rospy.logwarn(f"TF lookup failed")
        return point 
        
def listener():
    global new_point_cloud_publisher
    global tf_listener
    global apriltag_detection_status
    rospy.init_node("point_cloud_subscriber", anonymous=True)
    
    # Initialize the publisher
    new_point_cloud_publisher = rospy.Publisher("new_point_cloud_topic", PointCloud2, queue_size=10)
    apriltag_detection_status_publisher = rospy.Publisher("apriltag_status", Bool , queue_size=10)

    tf_listener = tf.TransformListener()

    rospy.Subscriber("random_point_cloud", PointCloud2, point_cloud_callback)
    rospy.Subscriber("tag_detections", AprilTagDetectionArray, apriltag_callback)

    rospy.spin()

if __name__ == "__main__":
    bridge = CvBridge()
    listener()

