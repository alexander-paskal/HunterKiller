from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='line_follower',
            node_executable='controller',
            node_name='controller',
            output='screen',
        ),
        Node(
            package='line_follower',
            node_executable='vesc_suscriber',
            node_name='vesc_subscriber',
            output='screen'
        )
    ])
