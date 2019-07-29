import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bigDATA",
    version="1.0.0",
    author="Justin Gonzalez",
    description="A python library for big data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jfgonzalez/bigDATA",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
