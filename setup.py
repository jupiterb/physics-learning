from setuptools import setup, find_packages

requirements = ["numpy", "torch", "h5py", "pydicom", "scipy"]

setup(
    name="tumor_sim_pinns",
    version="0.0.0",
    description="",
    packages=find_packages(),
    install_requires=requirements,
)
