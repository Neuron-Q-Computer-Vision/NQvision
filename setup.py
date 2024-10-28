from setuptools import setup, find_packages


print("Starting setup.py execution...")


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="NQvision",
    version="0.1.2",
    author="Neuron Q",
    author_email="debbichi.raki@neuronq.io",
    description="A library to simplify the development of AI-driven object detection and monitoring solutions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://neuronq.io",
    packages=find_packages(exclude=["tests", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "opencv-python",
        "Pillow",
        "deep_sort_realtime",
        "torch",
        "torchvision",
        "vidgear",
        "numba",
    ],
    license="MIT",
    keywords="computer vision, object detection, tracking, CUDA, enterprise",
)
