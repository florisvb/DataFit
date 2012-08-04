from distutils.core import setup

setup(
    name='FitData',
    version='0.0.1',
    author='Floris van Breugel',
    author_email='floris@caltech.edu',
    packages = ['ransac_fit', 'fit_data'],
    license='BSD',
    description='line and curve fitting',
    long_description=open('README.txt').read(),
)



