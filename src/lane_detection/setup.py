from setuptools import find_packages, setup

package_name = "lane_detection"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="abhiyaan-nuc",
    maintainer_email="alanroyce2010@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["node = lane_detection.node:main", 
        "contours = lane_detection.lane_contours:main", 
        "process_ipm = lane_detection.lane_proc:main", 
        "image_drawer = lane_detection.image_drawer:main",
        "lane_logic = lane_detection.horiz_lane:main"
        "new_contours = lane_detection.lane_contours_new:main"]
    },
)
