from setuptools import find_packages, setup

package_name = 'drone_searching_behavior'

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
    maintainer='marsl',
    maintainer_email='marsl@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # package_name.node_name:function
            "search_node = drone_searching_behavior.search_node:main",
            'tag_overlay = drone_searching_behavior.tag_overlay:main',
        ],
    },
)
