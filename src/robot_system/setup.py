from setuptools import find_packages, setup

package_name = "robot_system"

setup(
    name=package_name,
    version="0.3.0",
    packages=find_packages(exclude=["tests"]),
    package_data={"robot_system": ["config/*.json"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md", "ARCHITECTURE.md"]),
    ],
    install_requires=[
        "setuptools",
        "openai-agents",
        "python-dotenv",
        "redis",
        "numpy",
        "sounddevice",
    ],
    zip_safe=False,
    maintainer="Branimir Kolev",
    maintainer_email="kolevb780@gmail.com",
    description="Maintainable realtime robotic-arm agent system",
    license="MIT",
    entry_points={
        "console_scripts": [
            "realtime_agent = robot_system.realtime.runtime:main",
            "workspace_scan_agent = robot_system.agents.workspace_scan_node:main",
            "planning_agent = robot_system.agents.planning_node:main",
            "colorized_pointcloud = robot_system.visualization.colorized_pointcloud:main",
        ]
    },
)
