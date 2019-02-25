from distutils.core import setup

setup(
    name='PBTricks',
    version='0.1',
    packages=['pbtricks'],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'h5py', 'joblib', 'matplotlib'],
    url='https://github.com/pbashivan/pbtricks',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Pouya Bashivan',
    description="Pouya's bag of tricks >:)"
)
