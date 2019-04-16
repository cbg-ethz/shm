from sys import version_info
from setuptools import setup, find_packages

if version_info[0] == 2:
    exit("Sorry, Python 2 is not supported. Move to Python 3 already.")


def readme():
    with open('README.rst') as fl:
        return fl.read()


test_deps = [
    'coverage',
    'findspark',
    'flake8',
    'pylint',
    'pytest>=3.6.2',
    'pytest-cov',
    'pytest-pep8',
    'yapf'
]

doc_deps = [
    'sphinx',
    'sphinx_fontawesome',
    'sphinxcontrib-fulltoc'
]

setup(
  name='shm',
  version='0.0.1',
  description='Structured hierarchical models using PyMC3',
  long_description=readme(),
  url='https://github.com/dirmeier/shm',
  author='Simon Dirmeier',
  author_email='simon.dirmeier@web.de',
  license='GPLv3',
  keywords='pymc bayes deep latent variable structure random field',
  packages=find_packages(),
  include_package_data=True,
  python_requires='>=3',
  install_requires=[
      'arviz',
      'matplotlib',
      'numpy',
      'pandas',
      'pymc3',
      'networkx',
      'scipy',
      'seaborn',
      'sklearn'
  ],
  test_requires=test_deps,
  extras_require={
      'test': test_deps,
      'doc': doc_deps,
      'dev': test_deps + doc_deps
  },
  classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: Python :: 3.7'
  ]
)
