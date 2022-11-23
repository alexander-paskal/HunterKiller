from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='oakd',
            executable='rgb_publisher',
            name='rgb',
            output='screen',
        ),
        Node(
            package='oakd',
            executable='depth_publisher',
            name='depth',
            output='screen'
        )
    ])
