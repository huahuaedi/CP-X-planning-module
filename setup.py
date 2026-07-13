from os.path import dirname, realpath
from setuptools import setup, find_packages, Distribution
from opencda.version import __version__


def _read_requirements_file():
    """Return the elements in requirements.txt."""
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name='opencda-planning-module',
    version=__version__,
    packages=find_packages(),
    url='',
    license='MIT',
    author='',
    author_email='',
    description='OpenCDA-based CARLA planning module with behavior planning, global routing, MPC, and metrics.',
    long_description=open("README.md").read(),
    install_requires=_read_requirements_file(),
)
