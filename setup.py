from setuptools import setup, find_packages

setup(
    name="explainability",  # any name
    version="0.0.1",
    packages=find_packages(where="src/main/python"),
    package_dir={"": "src/main/python"},  # points Python to src/main/python
    install_requires=[  # if you have dependencies
        # e.g. 'torch>=2.0.0',
        # 'captum>=0.5.0',
    ],
)
