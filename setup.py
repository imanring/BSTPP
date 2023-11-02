from setuptools import setup

with open('README.txt','r') as f:
    desc = f.read()

with open('requirements.txt','r') as f:
    modules = f.read().split('\n')

setup(
    name='BSTPP',
    version=0.1,

    url='https://github.com/imanring/Cox_Hawkes_Cov',
    author='Isaac Manring',
    author_email='isaacamanring@gmail.com',
    
    install_requires=modules,
    packages=['BSTPP'],
    
    licence = 'MIT',
    py_modules=['BSTPP'],
    description="Bayesian Spatiotemporal Point Process"
    long_description=desc,
)