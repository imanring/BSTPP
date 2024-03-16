from setuptools import setup

with open('README.md','r') as f:
    desc = f.read()


modules = ['dill>=0.3.5.1','geopandas>=0.14.0','importlib-resources','jax==0.4.23',
           'jaxlib==0.4.23','matplotlib>=3.5.2','multipledispatch>=0.6.0','numpy>=1.24.1',
           'numpyro>=0.10.0','opt-einsum>=3.3.0','packaging>=21.3','pandas>=1.4.3',
           'pyparsing>=3.0.9','scipy>=1.9.0','six>=1.16.0','tqdm>=4.64.0']

setup(
    name='BSTPP',
    version=0.1,

    url='https://github.com/imanring/BSTPP.git',
    author='Isaac Manring',
    author_email='isaacamanring@gmail.com',
    
    install_requires=modules,
    packages=['bstpp'],
    package_data={'bstpp': ['decoders/*','data/*']},
    
    license = 'MIT',
    py_modules=['bstpp'],
    description="Bayesian Spatiotemporal Point Process",
    long_description=desc,
)