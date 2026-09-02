from logging import root
import os
import sys
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess

if __name__ == '__main__':
    setup(
        name='surrol',
        version='0.2.0',
        description='SurRoL: An Open-source Reinforcement Learning Centered and '
                    'dVRK Compatible Platform for Surgical Robot Learning',
        author='Med-AIR@CUHK',
        keywords='simulation, medical robotics, dVRK, reinforcement learning',
        packages=[
            'surrol',
        ],
        python_requires = '>=3.7',
        install_requires=[
            "gym>=0.15.6",
            "numpy>=1.21.1",
            "scipy",
            "pandas",
            "imageio",
            "imageio-ffmpeg",
            "opencv-python",
            "roboticstoolbox-python",
            "sympy",
            "panda3d==1.10.11",
            "trimesh",
            "kivymd"
        ],
        extras_require={
            # optional dependencies, required by evaluation, test, etc.
            "all": [
                "tensorflow-gpu==1.14",
                "baselines",
                "mpi4py",  # important for ddpg
                "ipython",
                "jupyter",
            ]
        }
    )
    

