from setuptools import find_packages, setup

setup(
    name='PrecoderNet',
    packages=find_packages(),
    version='1.1.15',
    description='PrecoderNet: Hybrid Beamforming for MM Wave Systems with DRL',
    author='mhyrzt',
    license='MIT',
    install_requires=[
        "tqdm",
        "numpy",
        "pandas",
        "matplotlib",
        "torch"
    ]
)