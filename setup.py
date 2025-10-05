from setuptools import setup

setup(
    name="reassemblenet",
    py_modules=["reassemblenet"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
