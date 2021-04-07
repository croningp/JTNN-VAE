from distutils.core import setup
from setuptools import find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="jtnn",
    version="0.2.0",
    packages=find_packages(),
    package_data={},
    include_package_data=True,
    requires=requirements
)
