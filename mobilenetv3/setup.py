from setuptools import setup

setup(
    name="mobilenetv3",
    version="1.0.0",
    package_dir={"mobilenetv3": "src"},
    install_requires=['opencv-python'],
    packages=["mobilenetv3"],
)