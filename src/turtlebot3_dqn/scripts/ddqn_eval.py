import collections
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from keras.layers import Dense
from keras.models import load_model, Sequential
from keras.optimizers import Adam
import numpy
import rclpy
from rclpy.node import Node
from turtlebot3_msgs.srv import Dqn


class DQNTest(Node):

    def __init__(self, stage, load_episode):
        super().__init__('dqn_test')

        self.stage = int(stage)
        self.state_size = 26
        self.action_size = 5
        # self.episode_size = 3000  # not used anymore for limiting episodes

        self.discount_factor = 0.99
        self.learning_rate = 0.0007
        self.epsilon = 0.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64

        self.memory = collections.deque(maxlen=50000)

        self.model = self.build_model()
        self.target_model = self.build_model()

        self.load_model = True
        self.load_episode = int(load_episode)
        self.model_dir_path = os.path.join('/home/grim/turtlebot3_ws/src/turtlebot3_dqn', 'model')

        self.model_path = os.path.join(
            self.model_dir_path,
            f'stage{self.stage}_episode{self.load_episode}_ddqn.h5'
        )

        if self.load_model:
            self.model.load_weights(self.model_path)
            with open(os.path.join(
                self.model_dir_path,
                f'stage{self.stage}_episode{self.load_episode}_ddqn.json')) as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')

        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')

        self.evaluate()

    def evaluate(self):
        num_eval_episodes = 20
        success_count = 0
        path_lengths = []
        durations = []
        log_rows = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join('/home/grim/turtlebot3_ws/src/turtlebot3_dqn/scripts', f'test_metrics_ddqn_stage{self.stage}.csv')
        for episode in range(1, num_eval_episodes + 1):
            local_step = 0
            state = []
            next_state = []
            done = False
            init = True
            score = 0
            episode_start_time = time.time()
            time.sleep(1.0)

            while not done:
                local_step += 1
                if local_step == 1:
                    action = 2
                else:
                    state = next_state
                    action = int(self.get_action(state))

                req = Dqn.Request()
                req.action = action
                req.init = init

                while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().info('rl_agent interface service not available, waiting...')

                future = self.rl_agent_interface_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)

                if future.result() is not None:
                    next_state = future.result().state
                    reward = future.result().reward
                    done = future.result().done
                    score += reward
                    init = False
                else:
                    self.get_logger().error(f'Error during service call: {future.exception()}')
                    break

                time.sleep(0.01)

            duration = time.time() - episode_start_time
            durations.append(duration)
            path_lengths.append(local_step)
            if score > 0:
                success_count += 1

            print(f"[Episode {episode}] Steps: {local_step}, Score: {score:.2f}, Time: {duration:.2f}s")

            log_rows.append([episode, score, local_step, duration])

        avg_path_length = sum(path_lengths) / len(path_lengths)
        max_path_length = max(path_lengths)
        avg_time = sum(durations) / len(durations)
        success_rate = (success_count / num_eval_episodes) * 100

        log_rows.append(['Summary', f'Success: {success_count}/{num_eval_episodes}',
                         f'Avg Len: {avg_path_length:.2f}', f'Avg Time: {avg_time:.2f}s'])

        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Score', 'Path Length', 'Duration (s)'])
            writer.writerows(log_rows)

        print("\n=== Evaluation Summary ===")
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Average Path Length: {avg_path_length:.2f}")
        print(f"Max Path Length: {max_path_length}")
        print(f"Average Time per Episode: {avg_time:.2f}s")
        print(f"Results saved to: {csv_file}")

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='lecun_uniform'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def get_action(self, state):
        if numpy.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = numpy.asarray(state)
            q_value = self.model.predict(state.reshape(1, len(state)), verbose=0)
            return numpy.argmax(q_value[0])


def main(args=None):
    if args is None:
        args = sys.argv
    stage = args[1] if len(args) > 1 else '2'
    load_episode = args[2] if len(args) > 2 else '600'
    rclpy.init(args=args)
    dqn_test = DQNTest(stage, load_episode)
    dqn_test.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
