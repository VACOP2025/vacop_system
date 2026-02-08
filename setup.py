from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'vacop_system'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # copier le contenu de 'models' dans install/share/vacop_system/models
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Vacop Team',
    maintainer_email='user@todo.todo',
    description='Système de vision VACOP',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_node = vacop_system.vision.vision_node:main',
        ],
    },
)