import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="word2vec_tf",
    version="0.0.1",
    author="Sergey Mkrtchyan",
    author_email="sergey.mkrtchyan@gmail.com",
    description="word2vec CBOW model implementation using TensorFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mkserge/word2vec_tf",
    packages=setuptools.find_packages(),
    install_requires=[
        'tensorflow',
        'numpy',
        'scipy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)