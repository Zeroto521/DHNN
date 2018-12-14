import setuptools

from dhnn import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='dhnn',
    version=__version__,
    description='Discrete Hopfield Network (DHNN) implemented with Python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='yosukekatada,Zeroto521',
    author_email='Zeroto521@gmail.com',
    license="MIT",
    py_modules=['dhnn'],
    requires=['numpy', 'numba'],
    install_requires=['numpy', 'numba'],
    url='https://github.com/Zeroto521/DHNN',
    download_url='https://github.com/Zeroto521/DHNN/archive/master.zip',
    python_requires=">=3",
    package_data={"": ["*.jpg"]},
    keywords=[
        'machine learning',
        'neural networks',
        'hopfield', 'DHNN'
    ]
)
