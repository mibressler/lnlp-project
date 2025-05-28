from setuptools import setup, find_packages

setup(
    name="lnlp_project",
    version="0.1",
    packages=find_packages(),  # Automatically includes dataset/, utils/ etc.
)

# run pip install -e . to install the package in editable mode - to use absolute imports