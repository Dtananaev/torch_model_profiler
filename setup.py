from setuptools import find_packages, setup

from torchinspect import __version__

setup(
    name='torchinspect',
    version=__version__,
    packages=find_packages(exclude=['examples']),
    install_requires=[
        'numpy>=1.14',
        'torch>=1.4',
        'torchvision>=0.4',
    ],
    url='https://github.com/Dtananaev/torchinspect.git',
    license='MIT',
)