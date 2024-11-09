import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

reqs = [
    "tensorflow-macos==2.9.0;platform_system=='Darwin'",
    "tensorflow==2.9.0;platform_system!='Darwin'",
    "numpy==1.22.3",
    "matplotlib==3.5.2",
    "pandas==1.4.3",
    "scikit-learn==1.0.2",
    "xlrd==1.2.0",
    "yattag==1.14",
    "pillow==9.2.0",
    "pytest==7.1.2",
    "protobuf==3.20.0",
    "statsmodels==0.14.2",
    "quadprog==0.1.12",
    "cvxopt==1.3.2",
]

setuptools.setup(
    name="swot-ann-safeh2o",
    version="3.0.1",
    author="SafeH2O",
    author_email="support@safeh2o.app",
    description="SWOT ANN Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/safeh2o/swot-ann",
    project_urls={
        "Bug Tracker": "https://github.com/safeh2o/swot-ann/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["swotann"],
    python_requires=">=3.10",
    install_requires=reqs,
)
