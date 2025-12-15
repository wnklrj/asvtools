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
#
#    run run_pangu_24.py to start a pangu forecast of a given leadtime based on data of ONE
#                       netcdf analysis file
#    
#    python3 ./run_pangu_24.py an_file leadtime True/False            all/npy/nc/grib   [0-n]
#                                               save_interim_results    out_forms     gpu_device_nr
#    or use as python module via 'import asvtools.run_pangu_24'
#
#    AN_FILE  - file (and its full path) with surface and upper air data.
#       
###################################################################################################

import os
import numpy as np
import xarray as xr

import argparse

#  ensures that the local asvtools module is found.
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from asvtools import c
from asvtools import io_func as asv

import onnx
#import torch
import onnxruntime as ort


import datetime



###################################################################################################
def main(an_file, leadtime:int, save_interim_results:bool, 
                        out_forms:str or list=['npy'],gpu_dev:int=-1):
    """ function description:
        start pangu 24h model run / iterference.
        
        returns: directly nothing.
                 creates the forecasted states in c.SV_RUN_OUTPUT_DIR

    """

    #print('start')
    #now = datetime.datetime.now()
    #print(now)
    #print('computation device')
    #print( ort.get_device() )           #GPU or CPU
    #print('gpu_dev: '+str(gpu_dev) )
    

     



    if str(ort.get_device()) != c.PROCESSUNIT:
        raise TypeError("You chose '" + str(c.PROCESSUNIT)+"' as processing unit (in 'c.py'). "+
                        "But for some reason your current device is '"+str(ort.get_device())+"'")


    #c.LEADTIME=int(c.LEADTIME) 
    if (leadtime % 24 != 0 ):
        raise ValueError("leadtime needs to be a multiply of 24")

    ###############################################################################################

    # Set the behavier of onnxruntime
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    # Increase the number for faster inference and more memory consumption
    options.intra_op_num_threads = 1

    # Set the behavier of cuda provider - for GPUs 
    if isinstance(gpu_dev,int) and gpu_dev >= 0:
        cuda_provider_options = {'device_id': str(gpu_dev), 
                                    'arena_extend_strategy':'kSameAsRequested',}
    else:
        cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}



    # set to number of 
    iter24 = int(np.floor( (leadtime/24 )))

    pangu_24_file =c.SV_WORK_DIR+'/pangu_weather_24.onnx'



    print('start interference')

    if iter24 >0:
        if (c.PROCESSUNIT == 'CPU'):    
            ort_session_24 = ort.InferenceSession(pangu_24_file, sess_options=options, 
                                providers=['CPUExecutionProvider'])
        elif (c.PROCESSUNIT == 'GPU'):
            try:
                ort.preload_dlls(cuda=True,cudnn=True,msvc=False,directory=None)
            except:
                print("Warning: could not load onnxruntime preloaded cuda/cudnn. Try without.")
            ort_session_24 = (ort.InferenceSession(pangu_24_file, sess_options=options, 
                                providers=[('CUDAExecutionProvider', cuda_provider_options)]))                
        model_24 = onnx.load(pangu_24_file)
    #endif

    if isinstance(out_forms ,str):
        out_forms=[out_forms]
    

    head, an_file_name = os.path.split(an_file) 
    an_file_name, an_file_name_ending = os.path.splitext(an_file_name)
    
    input_state = asv.open_full_state(an_file, return_xarr=False)
    
    ###############################################################################################
    """
    # check level order
    if input_state[c.LEVEL][0] != c.PANGU_PLEVELS[0]:
        print(input_state[c.LEVEL][0])
        print(c.PANGU_PLEVELS[0])
        raise ValueError('wrong level order for pangu')
    
    print('Info: Latorder starts with: '+ input_state[c.LAT][0] )
    print('it has to start with 90 in dec-order to -90')
    print('Info: Lonorder starts with: '+ input_state[c.LON][0] )
    print('it has to start with 0 in asc-order to 360 (or equivalent)')    

    # check lat order
    
    # check lon order 
    """
    ###############################################################################################
    #seperate input data
    input_meta, input_surface , input = asv.seperate_state(input_state)

    #pangu requires single prec.
    input=input.astype(np.float32)
    input_surface=input_surface.astype(np.float32)

    ###############################################################################################
    # do pangu forecast

    if (iter24 == 0):                        # forecast leadtime=0; mainly for tests
        output, output_surface = input, input_surface
        curr_leadtime=0
        output_state = asv.merge_state(output_surface, output, input_meta)  
        out_path = asv.get_filepath(c.SV_RUN_OUTPUT_DIR, '.',
                             'fc_'+str(curr_leadtime)+'_'+an_file_name, file_ending=out_forms[0])
        asv.write_full_state(output_state, out_path)
        
    for it in range(0,iter24):
        output, output_surface = ort_session_24.run(None,
                                        {'input':input, 'input_surface':input_surface})
        input, input_surface = output, output_surface


        curr_leadtime = 24*(it+1)
        if save_interim_results or curr_leadtime == leadtime:  
            if len(out_forms) ==1:
                output_state = asv.merge_state(output_surface, output, input_meta)  
                out_path = asv.get_filepath(c.SV_RUN_OUTPUT_DIR, '.',
                             'fc_'+str(curr_leadtime)+'_'+an_file_name, file_ending=out_forms[0])
                asv.write_full_state(output_state, out_path)

            else:
                out_paths=[]    
                for index in range(0,out_forms):
                    output_state = asv.merge_state(output_surface, output, input_meta)  
                    out_paths.append( asv.get_filepath(c.SV_RUN_OUTPUT_DIR, out_forms[index],
                        'fc_'+str(curr_leadtime)+'_'+an_file_name, file_ending=out_forms[index])) 
                    asv.write_full_state(output_state, out_paths)


    print('done with pangu24 forecast')
    
    

#end main
###################################################################################################



if __name__ == '__main__':

    
    # get arguments from an bash like request.
    parser = argparse.ArgumentParser(description='Process some information.')
    parser.add_argument('an_file', type=str,
                        help='forecast state')
    parser.add_argument('leadtime', type=str)

    parser.add_argument('save_interim_results', type=bool)
    parser.add_argument('out_forms', type=str)
    parser.add_argument('gpu_dev', type=int)
    args = parser.parse_args()

    an_file = args.an_file
    leadtime = int(args.leadtime)

    save_interim_results = args.save_interim_results 
    out_forms = args.out_forms
    gpu_dev = args.gpu_dev

    print(save_interim_results)
    print(type(save_interim_results))

    

    # get the output fomat variant.
    #TODO: This should not be the final solution for output paths.
    if out_forms in ['all', 'all-formats', 'all_formats']:
        out_forms = ['npy','nc','grib']
    elif out_forms in ['npy', 'grib', 'nc']:
        out_forms = [out_forms]
    else:
        raise ValueError('no vaild option')

    # start the runfunction
    main(an_file, leadtime, bool(save_interim_results), out_forms, gpu_dev)
   
###################################################################################################

