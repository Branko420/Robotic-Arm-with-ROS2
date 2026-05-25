from setuptools import find_packages, setup

package_name = 'servo_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi',
    maintainer_email='pi@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'servo_node = servo_pkg.servo_node:main',
            'laptop_brain = servo_pkg.laptop_brain:main',
            'laptop_test = servo_pkg.laptop_test:main',
            'virtual_servo_node = servo_pkg.virtual_servo_node:main',

        ],
    },
)
