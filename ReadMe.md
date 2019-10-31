# STEEL (A STatistical sEmi-Empirical modeL)

This is the main directory, all scripts and notebooks in subdirectories should
add this folder from their path and work down. STEEL is run simply by running
python STEEL.py the input parameters are set at the bottom of the STEEL script.

Future work here inclues making a runfile in which running parameters are set. 

## Requirements
* Python 3
* GFortran compiler
* C++ compiler
* GNU Scientific Library (optional - see below)

Note, if you will be using conda to install all the dependencies, the GNU
Scientific Library will automatically be installed. You must ensure the other
requirements are met though. The installation methods for these vary depending
on the operating system, for example *brew* on Mac OS or *apt-get* on Debian-based
Linux distributions.

You can install all the python dependencies in the following ways. I recommend
the conda method:

### Using Conda
If you are using the Conda package manager as part of the
Anaconda (or Miniconda) Python distributions, you can install necessary packages
from the terminal. Navigate to the STEEL directory and run:

```
conda env create -f environment.yml
```
This will create a virtual environment called STEEL with all the necessary
dependencies. The environment can be activated with:
```
conda activate STEEL
```

### Using Pip
If you are using Python's native package manager, you can install the necessary
dependencies with the requirements.txt file. Navigate to the STEEL directory and
run:
```
pip install -r requirements.txt
```

## Setup
After you have the necessary dependencies, you must do 2 things before being
able to run STEEL.py

* Compile the cython file in STEEL/Functions/
* Compile the fortran file in STEEL/Functions/OtherModels/VDB13/

Navigate to STEEL/Functions/ and run:
```
python Setup.py build_ext --inplace
```

The navigate to STEEL/Functions/OtherModels/VDB13/ and run:
```
make -f Makefile.mke
```

Now you should be able to run STEEL.py!











