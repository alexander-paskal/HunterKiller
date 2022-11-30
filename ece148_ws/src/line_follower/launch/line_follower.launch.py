from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='line_follower',
            executable='controller',
            name='controller',
            output='screen',
        ),
        Node(
            package='line_follower',
            executable='vesc_suscriber',
            name='vesc_subscriber',
            output='screen'
        )
    ])
