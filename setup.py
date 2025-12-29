from setuptools import setup, find_packages

setup(
    name='dbnet-crnn-ctc',
    version='1.0.0',
    description='End-to-End OCR System with DBNet and CRNN',
    author='danteng1981',
    url='https://github.com/danteng1981/dbnet-crnn-ctc',
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'opencv-python>=4.5.0',
        'numpy>=1.19.0',
        'Pillow>=8.0.0',
        'shapely>=1.7.0',
        'pyclipper>=1.2.0',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language ::  Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
