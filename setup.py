"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib
import skbuild

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')




skbuild.setup(
    name='HydrOpTop', 
    version='1.0.1',  # Required
    description='Flexible and solver-independent topology optimization library in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MoiseRousseau/HydrOpTop',
    author='Moise Rousseau',  # Optional
    author_email='rousseau.moise@gmail.com', 
    classifiers=[ 
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Programming Language :: Python :: 3.10",
    ],
    keywords='topology optimization, mechanics, numerical simulation', 
    #package_dir={'': 'HydrOpTop'}, 
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.6, <4',
    install_requires=['numpy', 'scipy', 'nlopt', 'matplotlib', 'h5py'],
    extras_require={
        'ipopt': ['cyipopt'],
    },
    project_urls={ 
        'Bug Reports': 'https://github.com/MoiseRousseau/HydrOpTop/issues',
        'Source': 'https://github.com/MoiseRousseau/HydrOpTop/',
    },
)

