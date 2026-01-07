#!/dl_env/bin/env python3

import rclpy
from rclpy.node import Node
import math
from geometry_msgs.msg import Quaternion
from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import SetEntityState

class Moving(Node):
    def __init__(self):
        super().__init__('moving_obstacle')
        
        self.set_model_client = self.create_client(SetEntityState,'/gazebo/set_entity_state')
        
        while not self.set_model_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
        
        self.model_states_sub = self.create_subscription(ModelStates,'/gazebo/model_states',
                                        self.model_states_callback, 10)

        self.angle = 0.0
    
    def model_states_callback(self, model):
        self.angle += 0.05 
        
        for i in range(len(model.name)):
            if model.name[i] == 'turtlebot3_dqn_obstacles':
                request = SetEntityState.Request()
                obstacle = EntityState()
                obstacle.name = 'turtlebot3_dqn_obstacles'
                
                obstacle.pose.position = model.pose[i].position
                obstacle.pose.orientation = Quaternion()
                obstacle.pose.orientation.z = math.sin(self.angle/2.0)
                obstacle.pose.orientation.w = math.cos(self.angle/2.0)
                obstacle.reference_frame = 'world'
                
                request.state = obstacle
                self.set_model_client.call_async(request)
                break


def main(args=None):
    rclpy.init(args=args)
    moving = Moving()
    rclpy.spin(moving)
    moving.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()