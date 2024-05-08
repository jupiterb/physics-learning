from setuptools import setup, find_namespace_packages

phynn_requirements = ["numpy", "torch", "h5py", "tqdm"]
phyprep_requirements = ["numpy", "pydicom", "nibabel"]


setup(
    name="physics-learning",
    version="0.0.0",
    description="",
    packages=find_namespace_packages(),
    extras_require={
        "phynn": phynn_requirements,
        "phyprep": phyprep_requirements,
    },
)
