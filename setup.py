from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="reservoir-computing-framework",
    version="0.1.0",
    author="Paul Dinkler",
    author_email="paul-dinkler@t-online.de",
    description="A reservoir computing framework for exploring task complexity metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cubeskills/reservoir_computing_thesis.git",  
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    keywords=["reservoir computing", "neural networks", "machine learning", "complexity metrics"],
    project_urls={
        "Source": "https://github.com/cubeskills/reservoir_framework",
        "Documentation": "https://github.com/cubeskills/reservoir_framework#readme",
    },
    include_package_data=True,
    zip_safe=False,
)