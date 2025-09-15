import rclpy
import random
import time
import math
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped
import tf_transformations

def create_pose_stamped(navigator, position_x, position_y, orientation_z):
    q_x, q_y, q_z, q_w = tf_transformations.quaternion_from_euler(0.0, 0.0, orientation_z)
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = position_x
    pose.pose.position.y = position_y
    pose.pose.position.z = 0.0
    pose.pose.orientation.x = q_x
    pose.pose.orientation.y = q_y
    pose.pose.orientation.z = q_z
    pose.pose.orientation.w = q_w
    return pose

def euclidean_dist(p1, p2):
    dx = p1.pose.position.x - p2.pose.position.x
    dy = p1.pose.position.y - p2.pose.position.y
    return math.sqrt(dx**2 + dy**2)

def main():
    rclpy.init()
    nav = BasicNavigator()

    # Set initial pose
    initial_pose = create_pose_stamped(nav, 0.0, 0.0, 0.0)
    nav.setInitialPose(initial_pose)

    # Wait for Nav2 to be active
    nav.waitUntilNav2Active()

    goal_poses = [
        create_pose_stamped(nav, 1.0, 0.0, 1.57),
        create_pose_stamped(nav, 2.0, -1.5, 1.57),
        create_pose_stamped(nav, 0.0, -2.0, 1.57),
        create_pose_stamped(nav, 2.0, 2.0, 1.57),
        create_pose_stamped(nav, 0.8, 2.0, 1.57),
        create_pose_stamped(nav, -1.9, 1.9, 1.57),
        create_pose_stamped(nav, -1.9, 0.2, 1.57),
        create_pose_stamped(nav, -1.9, -0.5, 1.57),
        create_pose_stamped(nav, -2.0, -2.0, 1.57),
        create_pose_stamped(nav, -0.5, -1.0, 1.57),
        create_pose_stamped(nav, -0.5, 2.0, 1.57),
        create_pose_stamped(nav, 2.0, -0.5, 1.57)
    ]

    # Randomly select 20 goals (with repeats)
    selected_goals = random.choices(goal_poses, k=20)

    nav.followWaypoints(selected_goals)

    total_goals = len(selected_goals)
    current_goal_index = 0
    reached_goals = 0
    time_per_goal = []
    goal_start_time = time.time()
    nav_start_time = goal_start_time
    last_waypoint_reported = -1

    while not nav.isTaskComplete():
        feedback = nav.getFeedback()
        if feedback and feedback.current_waypoint != last_waypoint_reported:
            last_waypoint_reported = feedback.current_waypoint
            pose = selected_goals[feedback.current_waypoint]
            x = pose.pose.position.x
            y = pose.pose.position.y
            yaw = tf_transformations.euler_from_quaternion([
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w
            ])[2]

            goal_time = time.time() - goal_start_time
            # Skip first "goal time" as it includes startup
            if current_goal_index != 0:  
                time_per_goal.append(goal_time)

            print(f"Reached Goal {current_goal_index + 1}/{total_goals}: "
                  f"x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} rad in {goal_time:.2f}s")

            goal_start_time = time.time()
            current_goal_index += 1
            reached_goals += 1

    total_time = time.time() - nav_start_time
    avg_time = sum(time_per_goal) / len(time_per_goal) if time_per_goal else 0.0

    total_path_length = sum(euclidean_dist(selected_goals[i], selected_goals[i-1])
                            for i in range(1, total_goals))
    avg_path_length = total_path_length / (total_goals - 1)

    print("\n--- Navigation Summary for stage 2 ---")
    print(f"Total Goals Given        : {total_goals}")
    print(f"Goals Reached            : {reached_goals}")
    print(f"Success Rate             : {(reached_goals / total_goals * 100):.2f}%")
    print(f"Total Time Taken         : {total_time:.2f} seconds")
    print(f"Average Time per Goal    : {avg_time:.2f} seconds")
    print(f"Average Path Length      : {avg_path_length:.2f} meters")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
