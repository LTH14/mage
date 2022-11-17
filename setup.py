from setuptools import setup, find_packages

setup(
    name='mage',
    version='0.0.1',
    description='Masked Generative Encoder',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
