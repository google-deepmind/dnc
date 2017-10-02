from setuptools import setup

setup(
    name='dnc',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='1.0.0',

    description='Differentiable Neural Computer in Tensorflow',
    long_description='Differentiable Neural Computer in Tensorflow',

    # The project's main homepage.
    url='https://github.com/deepmind/dnc',

    # Author details
    author='Google Inc.',

    # Choose your license
    license='Apache License 2.0',

    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',

        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    # What does your project relate to?
    keywords='tensorflow differentiable neural computer dnc deepmind deep mind sonnet dm-sonnet machine learning',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['dnc'],
    install_requires=['dm-sonnet'],
)
