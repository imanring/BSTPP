from setuptools import setup

with open('README.md','r') as f:
    desc = f.read()

with open('requirements.txt','r') as f:
    modules = f.read().split('\n')

setup(
    name='BSTPP',
    version=0.1,

    url='https://github.com/imanring/Cox_Hawkes_Cov.git',
    author='Isaac Manring',
    author_email='isaacamanring@gmail.com',
    
    install_requires=modules,
    packages=['bstpp'],
    package_data={'bstpp': ['decoders/*']},
    #data_files = ['bstpp/decoders/*'],
    
    license = 'MIT',
    py_modules=['bstpp'],
    description="Bayesian Spatiotemporal Point Process",
    long_description=desc,
)