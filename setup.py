import setuptools

from dhnn import __version__

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='dhnn',
    version=__version__,
    description='A Discrete Hopfield Neural Network Framework in python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='yosukekatada,Zeroto521',
    author_email='Zeroto521@gmail.com',
    maintainer='Zeroto521',
    maintainer_email='Zeroto521@gmail.com',
    license="MIT",
    py_modules=['dhnn'],
    requires=['numpy', 'numba'],
    install_requires=[],
    url='https://github.com/Zeroto521/DHNN',
    download_url='https://github.com/Zeroto521/DHNN/archive/master.zip',
    python_requires="2.7, 3.5, 3.6",
    package_data={"": ["assets/*.jpg"]},
    platforms=['linux', 'windows', 'macos'],
    keywords=[
        'machine learning',
        'neural networks',
        'hopfield', 'DHNN'
    ]
)
