from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='oakd',
            node_executable='rgb_publisher',
            node_name='rgb',
            output='screen',
        ),
        Node(
            package='oakd',
            node_executable='depth_publisher',
            node_name='depth',
            output='screen'
        )
    ])
