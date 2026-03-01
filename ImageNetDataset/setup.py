from setuptools import setup

setup(
    name="ImageNetDataset",
    version="1.0.0",
    package_dir={"ImageNetDataset": "src"},
    install_requires=['tqdm'],
    packages=["ImageNetDataset"],
)