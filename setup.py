from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# This call to setup() does all the work
setup(
    name="niteshade",
    version="0.0.1",
    description="Library for simulating data poisoning attacks against online learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oskarfernlund/niteshade",
    project_urls={
        "Documentation": "https://oskarfernlund.github.io/niteshade/",
        },
    author="Mart Bakler, Oskar Fernlund, Alex Ntemourtsidou, Jaime Sabal, Mustafa Saleem",
    author_email="mart.bakler21@imperial.ac.uk, oskar.fernlund21@imperial.ac.uk, alexandra.ntemourtsidou15@imperial.ac.uk, js921@ic.ac.uk, mustafa.saleem21@imperial.ac.uk",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=["matplotlib", "numpy", "scipy", "pandas", "scikit-learn",
        "torch", "torchvision", "tqdm"],
    python_requires=">=3.8",
)
