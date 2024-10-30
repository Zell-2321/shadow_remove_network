from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'shadow_remove_v1'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        (f'share/shadow_remove_v1/shadow_remove_v1/model', 
         glob('shadow_remove_v1/model/*.pth')),
        
        (f'share/shadow_remove_v1/shadow_remove_v1',
         glob('shadow_remove_v1/*.png')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zell',
    maintainer_email='773866146@qq.com',
    description='This is a package.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'shadow_remove_node = shadow_remove_v1.main_ros:main'
        ],
    },
)
