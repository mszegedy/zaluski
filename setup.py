import subprocess
from setuptools import setup, find_packages

# Copied off of StackOverflow (https://stackoverflow.com/a/26367671/1249632):

__version__ = '0.2.0'
__build__ = subprocess.check_output('git describe --tags --always HEAD'
                                    .split()).decode().strip()

with open('zaluski/_version.py', 'w') as f:
    f.write('''\
# I will destroy any changes you make to this file.
# Sincerely,
# setup.py ;)

__version__ = '{}'
__build__ = '{}'
'''.format(__version__, __build__))

setup(name='zaluski',
      version=__version__,
      packages=find_packages(),
      install_requires=['sortedcontainers'])
