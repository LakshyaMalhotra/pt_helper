from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as desc_file:
    long_description = desc_file.read()


setup(
    name="pt_trainer",
    packages=find_packages(exclude=[]),
    version="0.1.0",
    license="MIT",
    description="pt_trainer, boilerplate code for training, logging and evaluation in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Lakshya Malhotra",
    author_email="lakshya9009@gmail.com",
    url="https://github.com/LakshyaMalhotra/pt_trainer",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "pytorch",
        "ml",
        "torchinfo",
        "model",
    ],
    install_requires=["torch>=1.8", "torchinfo>=1.6.5", "numpy>=1.22.3"],
    classifiers=[
        "Topic::Scientific/Engineering::Artificial Intelligence",
        "Programming Language::Python::3",
        "License::OSI Approved::MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
