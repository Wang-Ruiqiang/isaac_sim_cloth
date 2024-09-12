import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()
        self.front_camera_subscription = self.create_subscription(
            Image,
            '/camera2/camera2/color/image_raw',
            self.front_camera_image_callback,
            10
        )

        self.side_camera_subscription = self.create_subscription(
            Image,
            '/camera1/camera1/color/image_raw',
            self.side_camera_image_callback,
            10
        )

        # 定义视频保存器
        self.out1 = cv2.VideoWriter('front_camera_output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480), True)
        self.out2 = cv2.VideoWriter('side_camera_output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480), True)
        # self.out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'avc1'), 30, (1024, 768))


    def front_camera_image_callback(self, msg):
        # 将ROS的图像消息转换为OpenCV格式
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 将图像写入视频文件
        self.out1.write(cv_image)

    def side_camera_image_callback(self, msg):
        # 将ROS的图像消息转换为OpenCV格式
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 将图像写入视频文件
        self.out2.write(cv_image)

    def destroy_node(self):
        # 关闭视频文件
        self.out.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    image_saver = ImageSaver()
    rclpy.spin(image_saver)
    image_saver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
