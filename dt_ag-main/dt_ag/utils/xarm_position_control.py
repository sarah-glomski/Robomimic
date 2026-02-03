#!/usr/bin/env python3
"""
XArm Position Controller Node

This node only handles commanding the XArm:
- Position commands
- Gripper commands
- Reset commands
- Pause/resume commands

It does NOT publish any state information.
"""

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Bool
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation as R


class XArmPositionController(Node):
    """
    ROS node that controls XArm position and gripper.
    """
    def __init__(self):
        super().__init__('xarm_position_controller')
        
        # Declare parameter for xarm IP address
        self.ip = self.declare_parameter('xarm_ip', '192.168.1.213').value
        self.get_logger().info(f'Connecting to xArm at IP: {self.ip}')

        # Initialize XArm for commanding only
        self.arm = XArmAPI(self.ip)
        self.setup_xarm()

        # Control state
        self.gripper = 0.0
        self.is_paused = False
        self.is_resetting = False

        # Subscribers for commands
        self.position_sub = self.create_subscription(
            PoseStamped, 'xarm_position', self.position_callback, 1)
        self.gripper_cmd_sub = self.create_subscription(
            Float32, 'gripper_position', self.gripper_callback, 1)
        self.reset_sub = self.create_subscription(
            Bool, '/reset_xarm', self.reset_callback, 10)
        self.pause_sub = self.create_subscription(
            Bool, '/pause_xarm', self.pause_callback, 10)

        # Initialize gripper and reset position
        self.initialize_robot()

        self.get_logger().info('XArm Position Controller initialized successfully')

    def setup_xarm(self):
        """
        Initialize the XArm for position control.
        """
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)  # Position control mode
        self.arm.set_state(state=0)  # Ready state
        time.sleep(1)
        self.get_logger().info('XArm initialized for position control')

    def initialize_robot(self):
        """
        Initialize gripper and reset robot to home position.
        """
        # Setup gripper
        self.arm.set_gripper_mode(0)      # location mode
        self.arm.set_gripper_enable(True) # power the driver
        self.arm.set_gripper_speed(5000)  # speed (1-5000)
        self.arm.clean_gripper_error()    # clear residual errors
        
        # Open gripper and reset position
        self.gripper_callback(Float32(data=0.0))
        self.reset_callback(Bool(data=True))

    def position_callback(self, pose_msg: PoseStamped):
        """
        Callback for receiving commanded positions.
        """
        if self.is_paused or self.is_resetting:
            return

        try:
            # Extract position (convert from m to mm)
            x_mm = pose_msg.pose.position.x * 1000.0
            y_mm = pose_msg.pose.position.y * 1000.0
            z_mm = pose_msg.pose.position.z * 1000.0

            # Extract quaternion orientation
            qw = pose_msg.pose.orientation.w
            qx = pose_msg.pose.orientation.x
            qy = pose_msg.pose.orientation.y
            qz = pose_msg.pose.orientation.z

            # Convert quaternion to Euler angles (roll, pitch, yaw) in radians
            ar = R.from_quat([qx, qy, qz, qw])
            roll_rad, pitch_rad, yaw_rad = ar.as_euler('xyz', degrees=False)

            # Set the position and orientation
            # speed = 30  # mm/s
            speed = 50
            code = self.arm.set_position(
                x=x_mm, y=y_mm, z=z_mm, 
                roll=roll_rad, pitch=pitch_rad, yaw=yaw_rad, 
                speed=speed, is_radian=True, wait=False
            )
            
            if code != 0:
                self.get_logger().warn(f'Position command failed with code: {code}')

        except Exception as e:
            self.get_logger().error(f'Error in position callback: {e}')

    def gripper_callback(self, gripper_msg: Float32):
        """
        Callback for receiving gripper position commands.
        """
        if self.is_paused or self.is_resetting:
            return
            
        try:
            # Set gripper position
            self.gripper = gripper_msg.data
            
            # Convert normalized value [0-1] to XArm gripper value [0-850]
            grasp = 850 - 850 * self.gripper
            code = self.arm.set_gripper_position(grasp, wait=False)
            
            if code != 0:
                self.get_logger().warn(f'Gripper command failed with code: {code}')
                
        except Exception as e:
            self.get_logger().error(f'Error in gripper callback: {e}')

    def reset_callback(self, msg: Bool):
        """
        Reset the XArm to a predefined position.
        """
        if msg.data:
            self.get_logger().info('Resetting XArm position...')
            self.is_resetting = True
            
            try:
                # Reset position (in mm and degrees)
                code = self.arm.set_position(
                    x=166.9, y=2.1, z=230.5, 
                    roll=179.2, pitch=0.1, yaw=1.3, 
                    speed=100, is_radian=False, wait=True
                )
                
                if code != 0:
                    self.get_logger().error(f'Reset position failed with code: {code}')
                else:
                    # Reset gripper
                    code = self.arm.set_gripper_position(850, wait=True)  # Fully open
                    if code != 0:
                        self.get_logger().error(f'Reset gripper failed with code: {code}')
                    else:
                        self.get_logger().info('XArm reset complete')
                
            except Exception as e:
                self.get_logger().error(f'Error during reset: {e}')
            finally:
                self.is_resetting = False

    def pause_callback(self, msg: Bool):
        """
        Pause or resume XArm motion.
        """
        self.is_paused = msg.data
        if self.is_paused:
            self.get_logger().info('XArm motion paused')
            # Optionally stop current motion
            try:
                self.arm.stop()
            except:
                pass
        else:
            self.get_logger().info('XArm motion resumed')

    def destroy_node(self):
        """
        Cleanly disconnect from XArm before shutting down.
        """
        self.get_logger().info('Disconnecting from XArm...')
        try:
            self.arm.disconnect()
        except:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    # Create position controller node
    node = XArmPositionController()
    
    try:
        node.get_logger().info('Running XArm Position Controller...')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()