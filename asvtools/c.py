###################################################################################################
#    
#    asvtools - Copyright 2025 Deutscher Wetterdienst (DWD)
#    Licenced under BSD-3-Clause License
#
#    Redistribution and use in source and binary forms, with or without modification, are permitted 
#    provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright notice, this list of 
#	conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright notice, this list of 
#    	conditions and the following disclaimer in the documentation and/or other materials 
#    	provided with the distribution.
#
#    3. Neither the name of the copyright holder nor the names of its contributors may be used to 
#    	endorse or promote products derived from this software without specific prior written 
#	permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR 
#    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY 
#    AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
#    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR  
#    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR  
#    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY  
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR  
#    OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
#    POSSIBILITY OF SUCH DAMAGE.
#
###################################################################################################



###################################################################################################
### ASV NAMELIST - (almost) bash style.
### 
###
###
import numpy as np
import os
###################################################################################################
        ###### For configuration do NOT change anything here   #######
###################################################################################################
# ATH VAR NAMES

T, U, V, T_2M, U_10M, V_10M = 'T', 'U', 'V', 'T_2M', 'U_10M', 'V_10M'
PMSL, FI, QV = 'PMSL', 'FI', 'QV'

PANGU_VARS= [PMSL, U_10M,V_10M,T_2M, FI, T,U,V, QV]
SFCVARS = [PMSL, U_10M,V_10M,T_2M]
PVARS = [ FI, T, U, V, QV ]

LAT, LON, LEVEL = 'lat', 'lon', 'plev'

# other variables.
# required level order for pangu in Pacal (not hPa)
PANGU_PLEVELS= np.array([100000,92500,85000,70000,60000,50000,40000,30000,
                                    25000,20000,15000,10000,5000])


###################################################################################################
# AREA specific lat / lon borders

EUR_LON, EUR_LAT  = [336.50000, 62.500000] , [29.500000, 70.500000]
NORTH_HEMI_LAT, SOUTH_HEMI_LAT = 35, -35

###################################################################################################
## Total Energy Norm - Parameters

LOG2E = 1.4426950408889634 	#log_2 (e)

T_R = 270       # reference temperature 
C_P = 1000      # reference pressure    
#R_D = -         # for PSLM + Total Energ
#P_R = -         # 
###################################################################################################
#for pangu level choice.
ALL_LEVEL_INDIZES=np.array([*range(0,len(PANGU_PLEVELS))])        # level indizes.
###################################################################################################


############################################################################################################
############################################################## CHOOSE YOUR CONFIG FOR ASV RUNS HERE 
###################################################################################################
# ASV RUN SPECIFIC PARAMATERS
    # FORMATS TO USE FOR SV_PERT STATES

# OUTPUT at end.

# define which formats the states of the output should have.
# the generated REF + SV PERT states.
OUT_FORMATS=['npy','grib','nc']               # choose 1,2 or 3 out of: 'npy','nc','grib'
# the forecasts of these states. (if you use the 'forecast_states' function.
OUT_FORECAST_FORMATS=['npy']                  # choose 1,2 or 3 out of: 'npy','nc','grib'
###################################################################################################
# ASV vars

# ASV runs on loops and blocks which determine in combination a Krylov Subspace. 
# The larger the Krylov Subpace, the better the approximation, the longer the run.
# The block size defines the number of parallelization, the loops define the number of iteration.

# We recommand not to choose LOOPS below 2 (unless for testing) rather >=4
# Suitable BLOCK_SIZE Values depend on your machine. A number equal your .. ()..

LOOPS=2                                         # multiply of BLOCK_SIZE and LOOPS defines the
BLOCK_SIZE=2                                    # size of Krylov-Subspace
                                                # choose for real experiments a at least 
                                                # BLOCK_SIZE*LOOPS > 30
                                                

SV_CUTOF=21                                     # de facto => min( (LOOPS*BLOCK_SIZE) ; SV_CUTOF )
SV_NRS=[0,1,8]                                  # Which SV to be forecasted afterwards?(null index)
                                                # Indices larger the subspace-size or SV_CUTOF
                                                #   value are ignored.

T_OPT=48                                        # needs to be a mult of n (for pangu_n)
                                                # 

SV_VARS=[T,U,V,T_2M,U_10M,V_10M]
SV_INITIAL_VAR='random'                         # 'ens_diff' | 'random'  # limited func: 'ens_diff'
FACTOR_AMPLITUDE=0.1                            # only relevant for 'ens_diff' amplitude.

    #NOTE: LIMITED functionality (currently) PLEASE USE ALL_LEVEL_INDIZES
SV_LEVELS=ALL_LEVEL_INDIZES                     # level indizes SV PERTS 
                                                # a)e.g.:  np.array([1,2,3]) b) ALL_LEVEL_INDIZES

NORM_VARIANT='energy'                           # 'energy' 'energy_red' or 'original'
                    # 'energy_red' means total energy excluding surface pressure
                    # NOTE: currently 'energy' = 'energy_red', hence without surface pressure
AREA='north_hemi'                               # 'north-hemi', 'south-hemi', 'earth', 'europe'
MODEL='pangu_24'                                # 'pangu_24','pangu_6'


################################################
# model running parameter

PROCESSUNIT='GPU'                              # 'CPU' / 'GPU' 
 
RUN_MODE='local'                                
                        # 'local' - run all the inferences on same machine as whole program.  
                        # 'submit'- submit the inferences to a seperate batch system.
			# Note: 'submit' will probably not work on your machine since its needed 
			# to be adjust to your environment ( but it is not nessecary to be used). 
			# We recommand to run just everything 
			# ('run_asvtools.py') as batch job / submit with 
			# qsub. Than 'local' will be your choice. 

#################################################						

                                                


###################################################################################################
########################## END OF ASV CONFIG
###################################################################################################

############################################################################################################
        ###### For configuration do NOT change anything from here on   #######
###################################################################################################
# path of python venv for model inference / run ONLY for 'submit' mode.
PY_VENV_PATH=""                                 # python envir. for onnx AI inference run
                                                #   only! if RUN_MODE=='submit'
                                                #   leave empty if not required

# ASV-Path-Variables     #no change needed here
SV_WORK_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SV_IO_DIR = os.path.join(SV_WORK_DIR, 'iodir')
SV_INPUT_DIR=os.path.join(SV_IO_DIR,'input')
SV_RUN_DIR= os.path.join(SV_IO_DIR,'run')
SV_OUTPUT_DIR= os.path.join(SV_IO_DIR, 'output')
SV_OUTPUT_FORECASTS_DIR=os.path.join(SV_OUTPUT_DIR, 'forecasts')
SV_RUN_INPUT_DIR= os.path.join(SV_RUN_DIR, 'input')
SV_RUN_OUTPUT_DIR=os.path.join(SV_RUN_DIR,'output')
LATLON_FILE= os.path.join(SV_WORK_DIR, 'll_025_latlon.npy')



# Filenames
RUN_INPUT_FILENAME='asv_state_input'
#RUN_OUTPUT_FILENAME=''                          #currently not used
SV_FILENAME='SV'
SV_PERT_FILENAME='Full_SV_pert'
SV_REF_STATE_BEGIN_NAME='asv_ref_state'
SV_REF_STATE_END_NAME='asv_ref_state_end'


#ASV specific other
META_KEYS = (['md:area','time',LAT,LON, LEVEL, 'gptype',
             'md:norm_euclid','md:norm_energy',  
             'md:state_type', 'md:leadtime', 'md:ref_file', 'lat_area', 'lon_area']) 


###################################################################################################
###################################################################################################











