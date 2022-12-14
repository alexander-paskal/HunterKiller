from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='line_follower',
            node_executable='controller',
            node_name='ctrler',
            output='screen',
        ),
        Node(
            package='line_follower',
            node_executable='vesc_subscriber',
            node_name='vesc',
            output='screen'
        )
    ])
