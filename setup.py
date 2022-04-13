import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

reqs = [
    "tensorflow<=2.8",
    "numpy<=1.20",
    "matplotlib<=3.4.1",
    "pandas<=1.2.4",
    "scikit-learn<=1.0.2",
    "xlrd<=1.2.0",
    "yattag<=1.14",
    "pillow<=8.2.0",
    "pytest<=6.2.3",
]

setuptools.setup(
    name="swot-ann-safeh2o",
    version="2.0.1",
    author="SafeH2O",
    author_email="support@safeh2o.app",
    description="SWOT ANN Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/safeh2o/swot-python-analysis",
    project_urls={
        "Bug Tracker": "https://github.com/safeh2o/swot-python-analysis/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["swotann"],
    python_requires=">=3.6",
    install_requires=reqs,
)
