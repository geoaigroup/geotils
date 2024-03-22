import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="geotils",
    version="0.0.1",
    author="Example Author",
    author_email="author@example.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'albumentations==1.3.1',
        'fiona==1.9.5',
        'matplotlib==3.8.0',
        'numpy==1.26.1',
        'opencv-python==4.8.1.78',
        'opencv-python-headless==4.8.1.78',
        'pandas==2.1.1',
        'Pillow==10.1.0',
        'scipy==1.11.3',
        'seaborn==0.13.1',
        'segment-anything @ git+ssh://git@github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588',
        'shapely==2.0.2',
        'tifffile==2023.9.26',
        'timm==0.9.2',
        'torchvision==0.16.0',
        'tqdm==4.66.1'
    ]
)