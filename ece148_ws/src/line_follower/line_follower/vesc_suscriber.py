"""

Subscribed to /controls topic
type is Twist, first number of the linear velocity is throttle and first number of angular is steering
"""

from VESC import VESC
import rclpy
from line_follower import Controller
from geometry_msgs.Twist import Twist

class vesc_subscriber(VESC):

    def __init__(self):
        super().__init__('vesc_subscriber')
        self.subscription = self.create_subscription(
            Twist,
            'control',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.vesc = VESC("/dev/ttyACM0")

    def listener_callback(self, msg):
        throttle = msg.linear.x
        steering = msg.angular.x
        self.vesc.run(anlge = steering, throttle = throttle)



def main(args=None):
    rclpy.init(args=args)

    vesc_suscriber = vesc_subscriber()

    rclpy.spin(vesc_suscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    vesc_suscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



