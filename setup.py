from setuptools import setup, find_packages

setup(
    name="histopathology",
    version="0.1",
    packages=find_packages(include=['histopathology', 'histopathology.*']),
    install_requires=[
        'torch',
        'pytorch-lightning',
        'torchmetrics',
        'wandb',
        'torchvision',
        'lpips'
    ],
)
