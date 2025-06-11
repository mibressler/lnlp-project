from setuptools import setup, find_packages

setup(
    name="lnlp_project",
    version="0.1",
    packages=find_packages(),  # Automatically include dataset/, utils/ etc.
)

# Run `pip install -e .` to install in editable mode for absolute imports
