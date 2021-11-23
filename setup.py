from setuptools import setup, find_packages

setup(
    name='imageMetrics',
    version='0.1',
    packages=['imageMetrics', 'imageMetrics.test'],
    description='Useful image metrics in python',
    long_description=open('README.md').read(),
    license='LICENSE.txt',
    install_requires=['numpy', 'pytest'],
    url='https://github.com/BillMills/python-package-example',
    author='Bendik Skarre Abrahamsen',
)