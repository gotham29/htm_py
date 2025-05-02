from setuptools import setup, find_packages

setup(
    name='htm_py',
    version='1.0.0',
    description='Pure Python Hierarchical Temporal Memory (HTM) Implementation',
    author='Samuel Heiserman',
    author_email='samuel@example.com',
    url='https://github.com/your_username/htm_py',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.5',
        'matplotlib>=3.3.4'
    ],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
