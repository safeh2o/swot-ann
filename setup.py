import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

reqs = [
    "tensorflow-macos==2.16.2;platform_system=='Darwin'",
    "tensorflow==2.16.2;platform_system!='Darwin'",
    "numpy==1.26.4",
    "matplotlib==3.8.3",
    "pandas==2.2.1",
    "scikit-learn==1.7.1",
    "xlrd==2.0.1",
    "yattag==1.15.1",
    "pillow==10.2.0",
    "pytest==8.0.2",
    "protobuf==4.25.3",
    "statsmodels==0.14.2",
    "scipy==1.12.0",  # Added specific scipy version compatible with statsmodels
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
    python_requires=">=3.12",
    install_requires=reqs,
)