from setuptools import find_packages, setup


setup(
    name="chronic-kidney-disease",
    author="Aurélien D.",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy~=1.23.3",
        "pandas~=1.5.3",
        "scikit-learn~=1.2.2",
    ],
    extras_require={
        "notebook": [
            "ipywidgets~=8.0.6",
            "liac-arff~=2.5.0",
            "openpyxl~=3.0.7",
            "pandas~=1.5",
            "scipy~=1.10.1",
            "ydata-profiling~=4.2.0",
        ]
    },
)
