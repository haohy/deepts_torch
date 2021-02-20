import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="deepts",
  version="0.0.1",
  author="Hongyan Hao",
  author_email="haohy6@163.com",
  description="A package of deep learning based Time Series models with tensorflow 1.x and 2.x .",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/haohy/deepts",
  packages=setuptools.find_packages(),
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: Apache Software License",
  "Operating System :: OS Independent",
  ],
)