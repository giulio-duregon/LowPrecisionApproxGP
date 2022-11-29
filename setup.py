from setuptools import setup, find_packages

readme = open("README.md").read()
setup(
    name="lp_approx_gp",
    version=0.1,
    description="An implementation of Gaussian Processes in Pytorch",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Giulio Duregon, Jonah Poctzobutt, Paul Koettering",
    url="https://github.com/giulio-duregon/LowPrecisionApproxGP",
    author_email="gjd9961@nyu.edu",
    packages=find_packages(exclude=["test", "test.*"]),
    python_requires=">=3.8",
    test_suite="test",
)
