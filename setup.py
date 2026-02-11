from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'vacop_vision'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ayoub Hadi',
    maintainer_email='ayoubbiof2@gmail.com',
    description='Système de vision VACOP',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_node = vacop_vision.vision.vision_node:main',
        ],
    },
)
