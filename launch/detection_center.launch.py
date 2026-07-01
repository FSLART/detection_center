import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('detection_center'),
        'config',
        'homography.yaml'
    )

    return LaunchDescription([
        Node(
            package='detection_center',
            executable='detection_center',
            name='zed_center',
            parameters=[config_file],
            output='screen',
        ),
    ])
