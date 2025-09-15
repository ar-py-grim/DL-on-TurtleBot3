import collections
import datetime
import json
import math
import os
import random
import sys
import time
import numpy
import rclpy
from keras.layers import Dense
from keras.models import load_model, Sequential
from keras.optimizers import Adam
import tensorflow as tf
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
from turtlebot3_msgs.srv import Dqn

LOGGING = False
current_time = datetime.datetime.now().strftime('[%mm%dd-%H:%M]')

class DQNMetric(tf.keras.metrics.Metric):
    def __init__(self, name='dqn_metric'):
        super(DQNMetric, self).__init__(name=name)
        self.loss = self.add_weight(name='loss', initializer='zeros')
        self.episode_step = self.add_weight(name='step', initializer='zeros')

    def update_state(self, y_true, y_pred=0, sample_weight=None):
        self.loss.assign_add(y_true)
        self.episode_step.assign_add(1)

    def result(self):
        return self.loss / self.episode_step

    def reset_states(self):
        self.loss.assign(0)
        self.episode_step.assign(0)


class DQNAgent(Node):

    def __init__(self, stage_num, max_training_episodes):
        super().__init__('ddqn_agent')

        # Basic parameters
        self.stage = int(stage_num)
        self.train_mode = True
        self.state_size = 26
        self.action_size = 5
        self.max_training_episodes = int(max_training_episodes)

        # RL hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 0.0007
        self.epsilon = 1.0
        self.step_counter = 0
        self.epsilon_decay = 20000*self.stage
        self.epsilon_min = 0.05
        self.batch_size = 64

        # Experience replay
        # self.replay_memory = collections.deque(maxlen=100000)
        self.replay_memory = collections.deque(maxlen=50000)
        self.min_replay_memory_size = 5000

        # Build online & target networks
        self.model = self.create_qnetwork()
        self.target_model = self.create_qnetwork()
        self.update_target_model()
        self.update_target_after = 5000
        self.target_update_after_counter = 0

        # Model saving/loading
        self.load_model = True
        self.load_episode = 600
        self.model_dir_path = os.path.join(
            '/home/grim/turtlebot3_ws/src/turtlebot3_dqn', 'model'
        )
        self.model_path = os.path.join(
            self.model_dir_path,
            f'stage{self.stage}_episode{self.load_episode}_ddqn.h5'
        )
        if self.load_model:
            self.model.load_weights(self.model_path)
            with open(os.path.join(
                self.model_dir_path,
                f'stage{self.stage}_episode{self.load_episode}_ddqn.json'
            )) as f:
                p = json.load(f)
                self.epsilon = p.get('epsilon')
                self.step_counter = p.get('step_counter')

        # Logging metric
        if LOGGING:
            tb_name = f'{current_time}_stage{self.stage}_ep{self.load_episode}'
            log_dir = os.path.join('ddqn_logs', 'gradient_tape_ddqn', tb_name)
            self.dqn_reward_writer = tf.summary.create_file_writer(log_dir)
            self.dqn_reward_metric = DQNMetric()

        # ROS clients & publishers
        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        self.action_pub = self.create_publisher(Float32MultiArray, 'get_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, 'result', 10)

        # Start training loop
        self.process()
    
    def _save_checkpoint(self, episode: int):
        """Save model weights (.h5) and training metadata (.json)."""
        # build file names
        base = os.path.join(self.model_dir_path,
            f'stage{self.stage}_episode{episode}_ddqn'
        )
        h5_path   = base + '.h5'
        json_path = base + '.json'

        # save weights
        self.model.save(h5_path)
        # save metadata
        with open(json_path, 'w') as f:
            json.dump({
                'epsilon': self.epsilon,
                'step_counter': self.step_counter
            }, f)

        print(f'Model saved at: {h5_path} and {json_path}')


    def process(self):
        # 1) Make env
        self.env_make()
        time.sleep(1.0)

        # 2) Training episodes
        for episode in range(self.load_episode + 1, self.max_training_episodes + 1):
            state = self.reset_environment()
            total_reward = 0.0
            local_step = 0
            sum_max_q = 0.0

            time.sleep(1.0)
            # Episode loop
            while True:
                local_step += 1
                q_vals = self.model.predict(state, verbose=0)
                sum_max_q += float(numpy.max(q_vals))

                action = int(self.get_action(state))
                next_state, reward, done = self.step(action)
                total_reward += reward

                # Publish for visualization
                msg = Float32MultiArray()
                msg.data = [float(action), float(total_reward), float(reward)]
                self.action_pub.publish(msg)

                # Store & train
                if self.train_mode:
                    self.append_sample((state, action, reward, next_state, done))
                    self.train_model(done)

                state = next_state

                if done:
                    avg_max_q = sum_max_q / local_step if local_step > 0 else 0.0
                    # Publish result
                    msg = Float32MultiArray()
                    msg.data = [float(total_reward), float(avg_max_q)]
                    self.result_pub.publish(msg)

                    # TensorBoard log
                    if LOGGING:
                        self.dqn_reward_metric.update_state(total_reward)
                        with self.dqn_reward_writer.as_default():
                            tf.summary.scalar('dqn_reward',
                                              self.dqn_reward_metric.result(),
                                              step=episode)
                        self.dqn_reward_metric.reset_states()

                    print(f'Episode: {episode}, score: {total_reward}, '
                          f'memory: {len(self.replay_memory)}, '
                          f'epsilon: {self.epsilon:.3f}')
                    break

                time.sleep(0.01)

            if self.train_mode and episode % 20 == 0:
                self._save_checkpoint(episode)

    def env_make(self):
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('make_environment service not ready, retrying...')
        self.make_environment_client.call_async(Empty.Request())

    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('reset_environment service not ready, retrying...')
        future = self.reset_environment_client.call_async(Dqn.Request())
        rclpy.spin_until_future_complete(self, future)
        state = future.result().state
        return numpy.reshape(numpy.asarray(state), [1, self.state_size])

    def get_action(self, state):
        # Epsilon‑greedy
        self.step_counter += 1
        self.epsilon = max(self.epsilon_min,
            self.epsilon_min + (1.0 - self.epsilon_min) * math.exp(-self.step_counter / self.epsilon_decay)
        )

        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        q = self.model.predict(state, verbose=0)[0]
        return int(numpy.argmax(q))

    def step(self, action):
        # Call environment
        req = Dqn.Request()
        req.action = action
        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('rl_agent_interface not ready, retrying...')
        future = self.rl_agent_interface_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        ns, rw, dn = res.state, res.reward, res.done
        next_state = numpy.reshape(numpy.asarray(ns), [1, self.state_size])
        return next_state, rw, dn

    def append_sample(self, transition):
        self.replay_memory.append(transition)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_after_counter = 0
        print('* Target model updated *')

    def train_model(self, terminal):
        # Only once we have enough samples
        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        # Sample minibatch
        minibatch = random.sample(self.replay_memory, self.batch_size)
        states      = numpy.vstack([t[0] for t in minibatch]).reshape(self.batch_size, self.state_size)
        next_states = numpy.vstack([t[3] for t in minibatch]).reshape(self.batch_size, self.state_size)

        # Online and target Q-values
        q_online      = self.model.predict(states,      verbose=0)
        q_next_online = self.model.predict(next_states, verbose=0)
        q_next_target = self.target_model.predict(next_states, verbose=0)

        x_train, y_train = [], []

        for idx, (s, a, r, ns, done) in enumerate(minibatch):
            target_q = q_online[idx].copy()
            if done:
                target_q[a] = r
            else:
                # Double DQN update:
                best_next = int(numpy.argmax(q_next_online[idx]))
                target_q[a] = r + self.discount_factor * q_next_target[idx][best_next]
            x_train.append(s.reshape(self.state_size,))
            y_train.append(target_q)

        x_train = numpy.array(x_train)
        y_train = numpy.array(y_train)

        # Single batch update
        self.model.train_on_batch(x_train, y_train)

        # Periodic target network sync
        self.target_update_after_counter += 1
        if terminal and self.target_update_after_counter > self.update_target_after:
            self.update_target_model()

    def create_qnetwork(self):
        """2 layer MLP."""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.state_size,)),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        model.summary()
        return model


def main(args=None):
    rclpy.init(args=args)
    stage_num = sys.argv[1] if len(sys.argv) > 1 else '2'
    max_eps   = sys.argv[2] if len(sys.argv) > 2 else '10001'
    agent = DQNAgent(stage_num, max_eps)
    rclpy.spin(agent)
    agent.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
