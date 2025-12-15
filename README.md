# asvtools - Arnoldi-SV Perturbations for Machine Learning Weather Prediction (MLWP)


This is the repository directly corresponding to the Arnoldi Singular Vector (A-SV) perturbation 
approach [Arnoldi Singular Vector perturbations for machine learning weather prediction](https://doi.org/10.48550/arXiv.2506.22450),
arXiv preprint: 2506.22450, 2025. 
It can compute a specific kind of Singular Vectors in complex meteorological systems in particular
to create Ensemble Systems based on initial condition perturbations.
Written by 
* Jens Winkler, Michael Denhard for Deutscher Wetterdienst (DWD).

The concept of Arnoldi Singular Vectors was theoretically developed by Jens Winkler, 
Michael Denhard, Bernhard A. Schmitt and is published as 
[Krylov methods for adjoint-free singular vector based perturbations in dynamical systems](https://doi.org/10.1002/qj.3668), 2020. 
See reference section, below.  

Please read this README completely first and start installation afterwards.


 
## What asvtools can do

The asvtools package, computes the Evolved Increment Matrix (EIM) based A-SV perturbations by using
the Arnoldi iteration method (for further information see our articles).
These inital condition perturbations can be used to create Ensemble Systems.

It uses DWD ICON analysis data (or other suitable user-provided analysis data) and (currently) 
the Pangu 6h or Pangu 24h MLWP forecast model from Huawei. 
Pangu-Repository: [Pangu-Weather](https://github.com/198808xc/Pangu-Weather), Pangu-Article:
[Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast](https://arxiv.org/abs/2211.02556)
, 2022.

You can use asvtools also if you are just interested in transforming atmospheric states on the
ICON grid to a latlon grid.

Note: asvtools in the given form can only used with the pangu model, but is is build in a form to
rather easy extend its usage to other arbitrary machine learing weather prediction models (MLWP). 
CURRENTLY this package is NOT under active development; it is provided as 'archiv'. In the first
place our intend is to provide the package in combination with our "Arnoldi Singular Vector 
perturbations for machine learning weather prediction" article for the public. 
Maybe this package will be further developed in the future, with some specific features / fixes be 
added.



## Requirements

### a) ecCodes 
 
ecCodes are a powerful bunch of tools from the ECMWF to handle meteorologial datasets in 
particular such data in the dataformat 'grib'.
[Installation information and downloads](https://confluence.ecmwf.int/display/ECC/ecCodes+installation)
It needs to be compliled by yourself.


### b) netCDF

The second standard data format in the meteorlogical community ist netCDF.
[Installation information and downloads](https://docs.unidata.ucar.edu/nug/current/getting_and_building_netcdf.html)
There are some precompiled versions. Probably you can use such a version, but it is saver 
to complile one by your own.


### c) CDO (climate data operations)

A useful set of commandline tools to handle and transform atmospheric data from the 
"Max-Planck-Institut für Meteorologie Hamburg".   
[Installation information and downloads](https://code.mpimet.mpg.de/projects/cdo)
See README of package there. Python bindings of cdo are not required.

IMPORTANT:  
    Make sure that the "configure" command contains a denotation of the ecCodes and netCDF paths. 
    This should be possible by the following command version:
        ```          
        ./configure --with-eccodes="PATH_TO_ECCODES_DIR"  --with-netcdf="PATH_TO_NETCDF_DIR"
        ```
    Please see also the latest CDO installation information to include ecCodes and netCDF.
    Of course, netcdf and eccodes must be installed before.


### d) Python + packages

Use python with a specific environment (see below).
Python 3 is required with following packages:
    netCDF4 , os, numpy, subprocess, xarray, glob, shutil, math, copy, sys, argparse, datetime,
    re, time, concurrent.futures, pycuda.driver, pycuda.autoinit
We do not provide a specific requirements file for asvtools. Probably you have an environment
by your own, which contains most of these packages.

Addionally you need python packages
        onnx 
    	    and either
        onnxruntime (for CPU) OR onnxruntime-gpu (for GPU) 
depending on your system. Do not install both at the same time. You may use two different 
environments. See following section.

Depending on your system, you might need to have special attention on your Nvidia specific software
in particular CUDA. We noticed some difficulties with the CUDAExecutionProvider. Probably helpful 
will be an installation of cuda and cudnn in python via onnxruntime-gpu itself. We explicitly 
recommand this by using the following command:  

        ```
        pip install onnxruntime-gpu[cuda,cudnn]
        ```

Evtually see the CUDAExecutionProvider relevant sections in the 'run_pangu_X.py' files and read the 
[onnxruntime docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).


Another thing which might help IN CASE OF TROUBLE, is to update the PATH and LD_LIBRARY_PATH 
variables with your cuda installation. Your CUDA installation might be found at '/usr/local/cuda' 
(or similar). Lets say a variable CUDA_PATH contains your path, then an update can 
be done by:

        ```       
        export PATH="${PATH}:${CUDA_PATH}/bin"
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_PATH}/lib64"
        ```


### e) .onnx files

Machine Learing Weather Prediction (MLWP) models exists with pretrained modelweights of usually  
several hours forecasting range for one model interference. 
If you require a 24h forecast you can use 24h trained network model once or e.g. four times the 6h 
model.
These pretrained neuronal network models are stored in "onnx" files. ONNX is the 
"Open Neuronal Network eXchange" Format. Onnx itself is not restricted to python but since python
is by far the most used environment for ML and at least asvtools itself is a python package, the 
onnx package for python is needed. As well as onnxruntime, which exist in two different versions 
(see above).

You need the 'pangu_weather_24.onnx' and 'pangu_weather_6.onnx' files. Use links provided at the 
[Pangu-Weather github repository](https://github.com/198808xc/Pangu-Weather) to download.


### f) Our environment

In order to make our setting transparent, we provide the software configuration we used

Python 3.10.11 with
numpy==1.26.1
onnx==1.20.0 (previously also 1.12.0)
onnxruntime-gpu==1.23.2 (previously also 1.19.2) 
xarray==2023.6.0
netCDF4==1.6.4
eccodes==1.6.1. We used 'onnxruntime-gpu[cuda,cudnn]' (see above). 


Addionally also
cdo 2.5.2.
eccodes 2.32.0
CUDA 12.4
using Nvidia A100 GPUs.




## Installation


### a) Install addional Software

Install ecCodes, netCDF, CDO and python itself (keep this order for the installation).
See to previous section for links to the projects and follow the installations instructions there.


### b) Download asvtools and pangu onnx files

Choose a suitable place for the project on your machine. Download asvtools and the mentioned 
pangu .onnx files.


### c) Virtual environment

asvtools and pangu itself need the mentioned extra python-packages. 
Therefor we recommand always to seperate the python environments. We also recommend to set a 
specific location/path for the environment.

Python venv is easy to use but you may use other solutions like e.g. conda.

Create a new python environment with name e.g. 'asvtools_venv_gpu' 
        ```
        python -m venv [defined_path_for_venv]/asvtools_venv_gpu
        ```
        (without the path it will be created in your home directory)


The environment can always be activated by using
        ```
        source [defined_path_of_venv]/asvtools_venv_gpu/bin/activate
        ```
Than the venv's name should occur in the beginnung of the shell prompt line.
After usage it can be deactivated by simply typing:
        ```
        deactivate
        ```


### d) python packages

Activate the venv (see above) and you can use python programs and install specific mentioned 
packages within the environment. Install in particular onnx AND either onnxruntime (for CPU 
machines) OR onnxruntime-gpu (for GPU machines).



    
## Using asvtools


We recommand to read our 
[Arnoldi Singular Vector perturbations for machine learning weather prediction](https://arxiv.org/abs/2506.22450) 
article in advance.

Note: We do not gurantee that all combinations of settings and functions are completely available.


### a) Set c.py

Set Parameters in 'asvtools/c.py'. See that file for more information.

The most important are:
LOOPS  		- number of loops for the arnoldi algorithm 
BLOCK_SIZE	- Block Size / number of parallel forecast ; size of 
			Krylov subspace = LOOPS * BLOCK_SIZE 
SV_CUTOF 	- maximal number of SV vectors to be computed; 
		    de facto  min( (LOOPS*BLOCK_SIZE) ; SV_CUTOF ) because
		    one can not compute more than the size of the underlying
		    Krylov Subspace. We reccommand not to use the tail of the
		    singular spectrum, rather the first half or two thirds of the sepectrum.

SV_VARS		- Which Variables should be used to be perturbed? We recommand T, U , V 
T_OPT		- Optimization time. Only possible to use multitudes of modul forecast time range.
		    We recommand to use something in the range of one to several days.
SV_NRS		- Define which SV are used for forecasts after generation. It is null-indexed.
AREA		- Which area to be used for singular vector computation? Choose 
			'north-hemi', 'south-hemi', 'earth' or 'europe'.
MODEL		- which forecasting model to be used? Currently only 'pangu_24' or 'pangu_6'
PROCESSUNIT	- is you machine a 'GPU' or 'CPU' machine?


### b) prepare and start a run.

The file
run_asvtools.py
is the one, which should be used to start the modules of the package. You will find some 
information there. Set the spefic variables for a asv run at the bottom of that file.

Run it with 'python run_asvtools.py' via command line.
Probably it might be benefical at your machine to start 'run_asvtools.py' via batch job. 
This should be defined by yourself, since you know your environment best.


### c) run_asvtools.py

The basic idea of 'run_asvtools.py' file, is to start the following parts:

#### i) Parameters at 'run_asvtools.py' file.

  At the bottom of the 'run_asvtools.py' file are addional parameters. Choose if the next run
  should include download of data or run of the arnoldi iteration itself, if forecasts of 
  specific SV_NRS should be done afterwards.


  Most important here: choose an amplitude (which is not trivial).


  The amplitudes within asvtools are rather big numerical values.
  Imagine you have two global ensemble-states. You use e.g. U + V, you compute the difference of 
  both and write it in one large vector. Because this vector is so large (approx. 72 Million 
  degrees of freedom for Pangu), the norm of it will be of relevant size.

  You can try amplitudes by your own, maybe also 100 but, do not choose 0.01 because that will be
  a butterfly^2.

  The larger the area, nr of chosen levels and chosen keys, the bigger your choice needs to be.

  A strategy for the choice might be: Use two nearby states (possibly of in terms of times 
  and / or space) compute the incement of both accourding the chosen keys and area and its norm. 
  Now take a relevant fraction of that norm e.g. 10% or 20% of that norm as your choice for the 
  amplitude. 

  If you used a subspace size of lets say >50 and you notice a shrinkage of the generated A-SV 
  perturbations, than this is a hint that you probably choose the amplitude too large.


  As an example
  
          ```
        amplitude=4000  # AREA='earth' , SV_VARS=[U,V]
        amplitude=500   # AREA='north_hemi' , SV_VARS=[T,U,V] 
          ```
  This should be the order of magnitude, but it is an partly open question.

  You may also use the strategy to generate perturbations with a specific amplitude and 
  afterwards use a posteriori scaling factor to shrink generated perturbations after generation 
  (currently not implemented). This would change the effectivly used perturbation-amplitude and 
  would not require any rerun of asvtools.


#### ii) prepare / clean folders

  Use 'asv.prep_arnoldi_dirs' in order to prepare and clean directories. E.g.:
  		```
      	asv.prep_arnoldi_dirs(clean_input_dir=True, keep_Q=False)
		```
  clean_input_dir= True  - vanishes all the input data (in particular the reference state)
  After changing a particular day / reference state you should clean the input dir 
  otherwise you can keep it in order to save time.
  
  keep_Q = True - Keeps the first Block of the Krylov subspace basis Q in order to save 
  time for the next run. The Q files are saved in 'iodir\run\iter' and each column is a full 
  state and is saved in one file. If you decide to use 'keep_Q=True' the first block, will not be
  removed, which allows you to exactly rerun a specific case (otherwise usually randomized q_i 
  for the start Block would be generated). 
  If you are not sure about the functionality use 'keep_Q = False'. This should be the default.
                


#### iii) download analysis state / provide input states.

  You need an analysis reference states. 
  
  The provided shell script 'get_icon_opendata.sh' 
  allows you to download a DWD ICON analysis state of the very day, when you run the script 
  and transforms it from ICON grid to a regular LAT-LON Grid suitable for asvtools.
  DWD does not provide a larger public archiv of past analysis data.

  You may have another source of suitable (past) netcdf analysis data (e.g. NOAA's GFS analysis
  or ECMWF's ERA5 Reanalysis Data) or you may set up a cron job to collect a bunch of DWD 
  analysis data e.g. for several months.

  See 'get_icon_opendata.sh -h' and file itself for instructions.	

  Inside 'run_asvtools.py' you may start 'get_icon_opendata.sh' using subprocess.
  e.g.:
  
          ```
          subprocess.check_call(['bash', os.path.join(asv.c.SV_WORK_DIR,'get_icon_opendata.sh'), 
                'opendata' , asv.c.SV_INPUT_DIR ] )
          ```
  or place one or more analysis netcdf-files within
  iodir/input.
  
  
  Note: There are almost none requirements for the analysis state filename, but one: The 
    current version requires a filename, which has to contain at least one '_' somewhere.


  In order to create an analysis file on your own use the following instructions.
  It needs to be NetCDF4, 0.25 degree lat-lon containing the variables and fields 
  of the ERA5 dataset but potentially using a slightly different naming. 
  Upper Air Variables 
    FI (Geopotential; often refered with variable 'Z', but needs to be denoted as 'FI')
    QV (Specific Humidity; also refered as 'Q', but needs to be denoted as 'QV')
      or RELHUM (Relative Humidity; will be transformed to QV in that case by asvtools)	
    T  (Temperature)
    U  (U Wind component)
    V  (V Wind component)
  on 13 pressure levels (as 'plev') (1000hPa, 925hPa, 850hPa, 700hPa, 600hPa, 500hPa, 400hPa,  
  300hPa, 250hPa, 200hPa, 150hPa, 100hPa and 50hPa). We provided the pressure levels in Pascal
  hence e.g. 500 hPa would be 50000.	
  
  
  Surface Variables
    PMSL (Mean Sea Level Pressure; also refered as 'MSLP', but needs to be denoted as 'PMSL')
    U_10M (U Wind component in 10m)
    V_10M (V Wind component in 10m)
    T_2M (Temperature in 10m)
  
  In both cases, the dimensions of 721 and 1440 represent the size along the latitude ('lat') and 
  longitude ('lon'), where the numerical range is (-90,90) deg and (0,359.75) deg, respectively. 
  The spacing is 0.25 degrees. Note: Pangu itself uses the opposite order accourding to lats. 
  It is not explicitly checked if lat / lon entries of input data are in the ascenending 
  or descending order. If order is wrong you will get outputs but they will be totally senseless.
  Please check by your own, if you build an adapter to your own input data. You will notice if one 
  of the orders is wrong e.g. by plotting.
  
  If you provide a proper netcdf reference in the input folder the program should handle it.
  
  If you have an ICON-grid grib file you can use 
      	```
    	./get_icon_opendata.sh    PATH_TO_DATA_ON_ICON_GRID_GRIBFILE   INPUT_DIR
    	```
  to transform it to a lat-lon netcdf file.
  
  
  We recommand to use analysis data, where 'lat' and 'lon' data is included as keys / variables.
  If you do not have such data, you may provide a specifc latlon file and save it perferibly as
  .npy (numpy) file within your workdir as 'll_025_latlon.npy' (or you adjust the line
  		
      	LATLON_FILE= os.path.join(SV_WORK_DIR, 'll_025_latlon.npy'
  )
  inside the file 'asvtools/c.py'. Maybe there is no full functionality for this variant, we 
  recommand analysis data, which contains lat-lon data.
  

#### iv) prepare analysis/reference file within asvtools for the run
   
   If you have a suitable analysis state, the package needs to be done some small preparations.
   
   ref_file_path   -  path to yout input analysis reference state file
        ```
        asv.prepare_ref(ref_file_path,'begin')
        ```
   
ii) - iv) need to be done only once for one reference state. If you do not want to change the 
analysis / reference file you do not need to do these parts again. Use suitable parameter in a) to
manage such a rerun (e.g. without a new download of data).

The following command keeps the current reference state in the input directory.    
        ```
        asv.prep_arnoldi_dirs(clean_input_dir = False, keep_Q = False)
        ```   

#### v) Start A-SV Generation.
        ```
        asv.arnoldi(amplitude)
        ```      
      
#### vi) Forecast of perturbed states
Afterwards you have the option to start a forecast of the generated perturbed states:

        ```
        leadtime=72     # in hours.
        asv.forecast_states(amplitude, leadtime) 
        ```

 
To run the file, saving the output in a logfile might be suitable. E.g.

        ```
        python run_asvtools.py > some_logfile.log 
        ```
Prefer a run as batch job.


You will find the output_data in 'iodir/output'
  

### d) Notes

Note 1: filenames and SV numbering is null-indexed. (e.g. 0 for the first SV)

Note 2: The generated perturbations, which can be found in the output folder are no increments,
they are "reference + SV_PERT" (perturbed reference states). Hence one can directy start a run
based on these files. In order to use (e.g. for plots) the pure pert (e.g. plot it) substract 
asv_ref_state (in iodir/input/ref) from it by your own.  
After doing this, for the areas and keys outside your chosen setup these increments 
will be zero of course.

Note 3: Here we do not provide plotting routines. You have to do the plotting by your own. 
If your are not familiar with plotting of meteorological data, we recommand the python packages 
matplotlib and cartopy for a start.


IODIR structure:
------------------------------------
       
       |iodir
             |input
                  |ref
             |run
                  |iter
                  |input
                  |output
             |output 
------------------------------------
'iodir' is inside your workdir. Structure will be created automatically. 

## License

Copyright 2025, Deutscher Wetterdienst

BSD-3-Clause License

Redistribution and use in source and binary forms, with or without modification, are permitted 
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions 
and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of 
conditions and the following disclaimer in the documentation and/or other materials provided with 
the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to 
endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR 
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND 
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY 
WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Note: This holds only for asvtools itself. Since it is based in particular on Machine Learing 
Weather Prediction (MLWP) models and analysis states data, different license conditions may apply. 
Currently the package is based on [Pangu-Weather](https://github.com/198808xc/Pangu-Weather), where
commercial use is forbidden (see their LICENSE). Since MLWP models are in most cases based on the 
ERA5 dataset provided by ECMWF, [their policy](https://apps.ecmwf.int/datasets/licences/copernicus/)
needs to be taken into account.


## Acknowlegdement and Funding

We thank colleges at DWD, Stefan Dahlke, Bernhard A. Schmitt (both University Marburg), Julian 
Quinting (KIT; now University of Cologne) and Kevin Gramlich (KIT) for their useful suggestions, 
advise, ideas and the professional exchange in general.

We also thank the "BMV-Forschungsnetzwerk" for the funding of the project.



## References

If you use this resource or any fork in your research, we would be glad if you cite our work.
Thanks in advance.


First Article (Theoretical basis and results in systems of medium comlexity)

	@article{winkler_krylov_2020,
		title = {Krylov methods for adjoint‐free singular vector based perturbations in dynamical systems},
		volume = {146},
		issn = {0035-9009, 1477-870X},
		url = {https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3668},
		doi = {10.1002/qj.3668},
		number = {726},
		journal = {Quarterly Journal of the Royal Meteorological Society},
		author = {Winkler, Jens and Denhard, Michael and Schmitt, Bernhard A.},
		month = jan,
		year = {2020},
		pages = {225--239},
	}


Second Article (The one direcly corresponding to this repository)

	@misc{winkler_asv_2025,
		title = {Arnoldi {Singular} {Vector} perturbations for machine learning weather prediction},
		url = {http://arxiv.org/abs/2506.22450},
		doi = {10.48550/arXiv.2506.22450},
		publisher = {arXiv},
		author = {Winkler, Jens and Denhard, Michael},
		month = jun,
		year = {2025},
		keywords = {Physics - Atmospheric and Oceanic Physics, Computer Science - Machine Learning},
	}
(Article submitteded to peer-reviewed journal.)

Thanks you very much for your interest.







