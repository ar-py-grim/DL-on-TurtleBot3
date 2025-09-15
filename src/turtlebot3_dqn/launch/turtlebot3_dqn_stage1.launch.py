#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Ryan Shim

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    stage = LaunchConfiguration('stage', default='1')

    saved_model_file_name = 'stage1_'
    saved_model = os.path.join(
        get_package_share_directory('turtlebot3_dqn'),
        'model',
        saved_model_file_name)
    
    print(saved_model)

    gazebo_model = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'models/turtlebot3_square/goal_box/model.sdf')
    
    # executable name must match with 'console_scripts': [ 'name, = ] in setup.py file
    return LaunchDescription([
        # Node(
        #     package='turtlebot3_dqn',
        #     executable='turtlebot3_dqn',
        #     name='turtlebot3_dqn',
        #     output='screen',
        #     arguments=[saved_model]),

        Node(
            package='turtlebot3_dqn',
            executable='dqn_environment',
            name='dqn_environment',
            output='screen',
            arguments=[stage]),

        Node(
            package='turtlebot3_dqn',
            executable='action_graph',
            name='action_graph',
            output='screen'),

        Node(
            package='turtlebot3_dqn',
            executable='result_graph',
            name='result_graph',
            output='screen'),
    ])
