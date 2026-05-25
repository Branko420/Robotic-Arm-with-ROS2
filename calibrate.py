#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraCalibrationNode(Node):
    def __init__(self):
        super().__init__('camera_calibration_node')
        
        # Абонираме се за суровия видеопоток от камерата
        self.rgb_sub = self.create_subscription(
            Image, 
            '/aurora/rgb/image_raw', 
            self.image_callback, 
            10
        )
        
        # Издаваме видеото с мерника към нов топик
        self.publisher_ = self.create_publisher(Image, '/aurora/rgb/calibration_view', 10)
        self.br = CvBridge()
        
        self.get_logger().info('🟢 Калибровъчен режим стартиран!')
        self.get_logger().info('👉 Отвори RViz2 или rqt_image_view на топик: /aurora/rgb/calibration_view')
        self.get_logger().info('👉 Подравни червената точка точно с основата (base_link) на робота.')

    def image_callback(self, data):
        try:
            # Конвертираме ROS съобщението към OpenCV изображение
            frame = self.br.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Грешка при конвертиране на изображението: {e}")
            return

        # Взимаме размерите на екрана
        height, width, _ = frame.shape
        center_x = width // 2
        center_y = height // 2

        # --- ЧЕРТАЕНЕ НА МЕРНИКА ---
        # Вертикална линия (Синя)
        cv2.line(frame, (center_x, 0), (center_x, height), (255, 0, 0), 2)
        # Хоризонтална линия (Синя)
        cv2.line(frame, (0, center_y), (width, center_y), (255, 0, 0), 2)
        # Централна точка (Червена)
        cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), -1)
        
        # Добавяме информативен текст с черен фон за по-добра четимост
        text = "CALIBRATION MODE: Align red dot with Robot Base"
        cv2.rectangle(frame, (10, 10), (620, 45), (0, 0, 0), -1)
        cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        try:
            # Връщаме обработеното изображение обратно в ROS 2
            msg = self.br.cv2_to_imgmsg(frame, "bgr8")
            self.publisher_.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Грешка при публикуване: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🔴 Калибрацията е спряна от потребителя.')
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()