import collections
import json
import math
import os
import random
import sys
import time
import csv
from typing import List, Optional
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from turtlebot3_msgs.srv import Dqn
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


class DQNTest(Node):
    def __init__(self, stage: int, load_episode: int):
        super().__init__('dqn_test')
        
        # Initialize parameters
        self.stage = int(stage)
        self.state_size = 26
        self.action_size = 5
        self.episode_size = 3000
        self.load_episode = int(load_episode)
        self.max_goals = 20
        
        # DQN hyperparameters
        self.discount_factor = 0.99
        self.epsilon = 0.0  # Testing with no exploration
        self.learning_rate = 0.0007
        self.batch_size = 64
        
        # Metrics tracking
        self.total_goals = 0
        self.reached_goals = 0
        self.total_time = 0.0
        self.path_lengths: List[float] = []
        self.episode_times: List[float] = []
        self.start_time: Optional[float] = None
        self.last_pose: Optional[Pose] = None
        self.current_path_length = 0.0
        
        # Model setup
        self.model = self.build_model()
        self.model_dir_path = os.path.join('/home/grim/turtlebot3_ws/src/turtlebot3_dqn', 'model')
        self.load_model()
        
        # ROS2 setup
        qos = QoSProfile(depth=10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos)
        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        
        # Start testing
        self.process()

    def odom_callback(self, msg: Odometry):
        """Track robot movement for path length calculation"""
        current_pose = msg.pose.pose
        
        if self.last_pose is not None:
            dx = current_pose.position.x - self.last_pose.position.x
            dy = current_pose.position.y - self.last_pose.position.y
            self.current_path_length += math.sqrt(dx**2 + dy**2)
        
        self.last_pose = current_pose

    def load_model(self):
        """Load pre-trained model weights"""
        model_path = os.path.join(
            self.model_dir_path,
            f'stage{self.stage}_episode{self.load_episode}.h5'
        )
        json_path = os.path.join(
            self.model_dir_path,
            f'stage{self.stage}_episode{self.load_episode}.json'
        )
        
        if os.path.exists(model_path):
            self.model.load_weights(model_path)
            with open(json_path) as f:
                param = json.load(f)
                self.epsilon = param.get('epsilon', 0.0)
            self.get_logger().info(f"Loaded model from {model_path}")
        else:
            self.get_logger().error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")

    def build_model(self) -> Sequential:
        """Build the DQN model architecture"""
        if self.stage == 1:
            model = Sequential([
                Dense(512, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'),
                Dense(256, activation='relu', kernel_initializer='lecun_uniform'),
                Dense(128, activation='relu', kernel_initializer='lecun_uniform'),
                Dense(self.action_size, activation='linear', kernel_initializer='lecun_uniform')
            ])
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        if self.stage == 2:
            model = Sequential()
            model.add(Dense(64,input_shape=(self.state_size,),activation='relu',kernel_initializer='lecun_uniform'))
            model.add(Dense(self.action_size, activation='linear', kernel_initializer='lecun_uniform'))
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        model.summary()
        return model


    def get_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state.reshape(1, len(state)), verbose=0)
            return np.argmax(q_value[0])


    def process(self):
        """Run a fixed number of evaluation goals (20)"""
        metrics_file = open('turtlebot3_dqn/scripts/test_metrics_dqn_stage'+str(self.stage)+'.csv', 'w', newline='')
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow([
            'Goal Number', 'Goal Reached', 'Time Taken', 'Path Length',
            'Success Rate', 'Avg Time', 'Avg Path Length'
        ])

        try:
            for goal_number in range(1, self.max_goals + 1):
                self.start_episode()
                done = False
                state = []
                next_state = []
                score = 0
                local_step = 0

                while not done and rclpy.ok():
                    local_step += 1
                    action = self.select_action(local_step, state, next_state)
                    next_state, reward, done = self.execute_action(action, local_step == 1)
                    score += reward

                self.finish_episode(goal_number, done and reward > 0, metrics_writer)

        finally:
            self.print_final_metrics()
            metrics_file.close()
            self.get_logger().info("Finished all evaluation goals. Shutting down...")
            rclpy.shutdown()


    def start_episode(self):
        """Reset metrics for new episode"""
        self.current_path_length = 0.0
        self.last_pose = None
        self.start_time = time.time()
        time.sleep(1.0)  # Allow environment to stabilize


    def select_action(self, step: int, state: List[float], next_state: List[float]) -> int:
        """Select action based on current step and state"""
        return 2 if step == 1 else int(self.get_action(np.asarray(next_state)))


    def execute_action(self, action: int, is_init: bool) -> tuple:
        """Execute action and return next state, reward, done"""
        req = Dqn.Request()
        req.action = action
        req.init = is_init

        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

        future = self.rl_agent_interface_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            result = future.result()
            return result.state, result.reward, result.done
        else:
            self.get_logger().error(f'Service call failed: {future.exception()}')
            return [], 0, True


    def finish_episode(self, episode: int, success: bool, metrics_writer):
        """Handle episode completion and record metrics"""
        episode_time = time.time() - self.start_time
        self.total_time += episode_time
        self.total_goals += 1

        if success:
            self.reached_goals += 1
            self.path_lengths.append(self.current_path_length)
            self.episode_times.append(episode_time)

        # Calculate metrics
        success_rate = (self.reached_goals / self.total_goals) * 100
        avg_time = np.mean(self.episode_times) if self.episode_times else 0
        avg_path = np.mean(self.path_lengths) if self.path_lengths else 0

        # Log to CSV
        metrics_writer.writerow([
            episode, success, episode_time, self.current_path_length,
            f"{success_rate:.2f}%", f"{avg_time:.2f}", f"{avg_path:.2f}"
        ])

        # Print episode summary
        self.get_logger().info(
            f"Episode {episode}: {'Success' if success else 'Fail'} | "
            f"Time: {episode_time:.2f}s | Path: {self.current_path_length:.2f}m"
        )


    def print_final_metrics(self):
        """Print comprehensive testing metrics"""
        success_rate = (self.reached_goals / self.total_goals) * 100 if self.total_goals > 0 else 0
        avg_time = np.mean(self.episode_times) if self.episode_times else 0
        avg_path = np.mean(self.path_lengths) if self.path_lengths else 0

        print("\n=== Final Testing Metrics ===")
        print(f"Total Goals Given        : {self.total_goals}")
        print(f"Goals Reached           : {self.reached_goals}")
        print(f"Success Rate            : {success_rate:.2f}%")
        print(f"Total Time Taken        : {self.total_time:.2f} seconds")
        print(f"Average Time per Goal   : {avg_time:.2f} seconds")
        print(f"Average Path Length     : {avg_path:.2f} meters")

        # Also write to CSV
        with open('turtlebot3_dqn/scripts/test_metrics_dqn_stage'+str(self.stage)+'.csv', 
                  'a', newline='') as metrics_file:
            metrics_writer = csv.writer(metrics_file)
            metrics_writer.writerow([])  # empty row
            metrics_writer.writerow(['=== Final Summary ==='])
            metrics_writer.writerow(['Total Goals', self.total_goals])
            metrics_writer.writerow(['Goals Reached', self.reached_goals])
            metrics_writer.writerow(['Success Rate (%)', f"{success_rate:.2f}"])
            metrics_writer.writerow(['Total Time (s)', f"{self.total_time:.2f}"])
            metrics_writer.writerow(['Avg Time per Goal (s)', f"{avg_time:.2f}"])
            metrics_writer.writerow(['Avg Path Length (m)', f"{avg_path:.2f}"])



def main(args=None):
    rclpy.init(args=sys.argv if args is None else args)
    
    # Get stage and episode from command line
    args = sys.argv[1:] if args is None else args
    stage = args[0] if len(args) > 0 else '1'
    load_episode = args[1] if len(args) > 1 else '600'
    
    try:
        dqn_test = DQNTest(stage, load_episode)
        rclpy.spin(dqn_test)
    except KeyboardInterrupt:
        pass
    finally:
        dqn_test.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()