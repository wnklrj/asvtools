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



import netCDF4 as nC
import numpy as np
from asvtools import c
from asvtools import io_func as asv
from asvtools import run_pangu_6
from asvtools import run_pangu_24
import os
import subprocess

#import threading
import concurrent.futures

import sys
import re
import shutil
import time
  

__all__ = ["orthonormalize", "fill_dir_with_pert_states", "create_initial_vectors", 
            "prep_arnoldi_dirs", "write_run", "submit_run", "prepare_ref", "are_fc_states_avail",
            "forecast_states", "getFreeGPUId"]

###################################################################################################
def orthonormalize(state_list:list, variant:str='gram-schmidt', amplitude:int or float=1):
    """ funcion description:
        pass a state list of incremental states and get these states orthonormalized back.
        
        currently only orth via 'gram-schmitt'
        
        optionally the states can be scaled by a given amplitude.
        
        returns: a list of states.
    """

    for j in range(0,len(state_list)):

        if state_list[j]['gptype'] not in ['inc',201]:
            raise TypeError("states need to be of type 'increment' for orthonormalisation") 




    if variant in ['gram-schmitt', 'gram-schmidt', 'gram_schmitt', 'gram_schmidt', 
                    'gramschmitt','gramschmidt' ]:
        
        for j in range(0,len(state_list)):            

            for i in range(0,j):
                
                #print('scalarprod')
                #print(asv.dot_prod( state_list[i], state_list[j]))

                state_list[j] = asv.sub( state_list[j], 
                                (asv.smult( state_list[i] , 
                                    asv.dot_prod( state_list[i], state_list[j])
                                                                , copy = True )), copy = False)
            #endfor            
            factor = 1/ asv.norm(state_list[j])     
            
            state_list[j] = asv.smult(state_list[j] , factor, copy=False)
    


    elif variant in ['QR','qr']:
        raise NotImplementedError('QR for orthogonalisation is not yet implemented')

    else:
        raise ValueError("unknown value for variant in orthonormalise function. " +
                                "Use 'qr' or 'gram-schmidt' ")
    

    return state_list
   
    

    
    

################################################################################################### 
def create_initial_vectors(x_ref_org, variant:str = c.SV_INITIAL_VAR, get_amplitude:bool = True):
    """ funcion description:
        asv iteration is using a (block) arnoldi algorithm.
        
        in order to use a (block) arnoldi method for SV approx, one needs initial states to 
        start arnoldi. for c.BLOCK_SIZE=n ; n initial states are needed.
        this is done by using increments within an existing ensemble.
        
        returns: a list of orthonormalized states for arnoldi
    """


    #x_ref_org = asv.get_data(asv.get_filepath('ref_full_begin_npy'))
    Q=[]

    # check if enough Q are already avail.
    if not get_amplitude:
        Q_paths=asv.get_filepath( os.path.join(c.SV_RUN_DIR,'iter'),variant='glob')
        if len(Q_paths) == c.BLOCK_SIZE:
            for i in range(0,len(Q_paths)):
                Q.append({})
                Q[i] = asv.transform_to_energy(asv.get_data(Q_paths[i]))
            print('using existing Q files.')
            return Q

        elif len(Q_paths) == 0:
            pass
        elif len(Q_paths) > c.BLOCK_SIZE:
            raise OSError("Too many Q-Files left in iter folder. Please check.")
        elif len(Q_paths) < c.BLOCK_SIZE:
            raise OSError("Too few Q-Files left in iter folder. please provide an empty dir or "+
                    "the number of Q-Files which equals c.BLOCK_SIZE.  Please check.")
        


    if variant in ['ens_diff']:
        print('create Q with ens differences')
        mem_list = asv.get_filepath('input_states')

        # Kombinatorik-check if block size fits to given state list for initial vectors. 
        if c.BLOCK_SIZE > 0.5*((len(mem_list)+1)*len(mem_list)):
            raise ValueError("BLOCK_SIZE too large for available initial states data.")

        

        for i in range(0,min(c.BLOCK_SIZE, len(mem_list))):

                Q.append({})
                Q[i] = asv.get_data(mem_list[i])

        #endfor
        in_1 = len(Q)-1


        if (i < c.BLOCK_SIZE-1):
            print('Warning: You are demanding da block size which is quite high. Hence, '+ 
                    'differences of provided memberstates will be used. But in this case ' +
                    'there might be some issues with xarray metadata.')
            #
            for j in range(0, in_1 ):        # variante: len(tmp)
                for k in range(j+1, in_1+1 ):
                    i = i+1
                    if (i < c.BLOCK_SIZE):      
                            Q.append({})
                            Q[i] = asv.sub(Q[j], Q[k], copy=True) 
                            
        #endif                        
        if not len(Q) == c.BLOCK_SIZE:
                raise RuntimeError('can not create accurate number of inital states.')
              
        for index in range(0, in_1+1 ):
                Q[index] = asv.sub( x_ref_org, Q[index], copy=True )
                Q[index] = asv.smult( Q[index], -1, copy = False )    # in order to keep xarray 
                                                                      # data of xref  
                #print(Q[index]['gptype'])
        del index
        

    elif variant in ['random']:

        #TODO an levelind anpassen.
        print('create random start vectors for arnoldi')

        if get_amplitude:
            raise NotImplementedError("amplitude has to be passed for random inital states. ")
        
        for j in range(0,c.BLOCK_SIZE):
            Q.append(asv.copy_state(x_ref_org))
            Q[j]['gptype']=201

            for var in c.SV_VARS:
                if var in c.SFCVARS:
                    size= ( len(Q[j][var][:,0]), len(Q[j][var][0,:]) )
                    Q[j][var] = np.random.uniform(-1,1, size) 
                elif var in c.PVARS:
                    size = ( len(Q[j][var][0,:,0]),  len(Q[j][var][0,0,:]) )
                    for l in range(0,len(Q[j][var][:,0,0])): 
                        Q[j][var][l,:,:] = np.random.uniform(-1,1,size)
                else:
                    raise KeyError('unvalid key')
                
    #end random
        
    if get_amplitude:
        # save norms of "not-orthonormalized" state list.
        #       in both: original and energy space. 
        ampls_normal = ampls_energy = np.zeros((len(Q),))
        
        for index in range(0,len(Q)):
            Q[index]['md:norm_euclid'] = ampls_normal[index] = asv.norm(Q[index])
            Q[index] = asv.transform_to_energy(Q[index])
            Q[index]['md:norm_energy'] = ampls_energy[index] = asv.norm(Q[index])
    else:
        for index in range(0,len(Q)):
            Q[index] = asv.transform_to_energy(Q[index])


    Q = orthonormalize(Q)
    Qfn=[]
    #write Q to file.
    for i in range(0,len(Q)):
        Qfn.append(asv.get_filepath('Q',ens_mem=i))
        Q[i]= asv.write_athpy_state( Qfn[i], Q[i]) 


    ## if get_amplitude is true 
    if get_amplitude:
            # multiply with an factor (prob <1). This is used to get an adequate amplitude for 
            #       asv perts.
            ampl_normal = c.FACTOR_AMPLITUDE * np.average(np.array(ampls_normal))
            ampl_energy = c.FACTOR_AMPLITUDE * np.average(np.array(ampls_energy))
            
            return Q, int(ampl_normal), int(ampl_energy)
    else:
        return Q
        



###################################################################################################
def fill_dir_with_pert_states(Qe, amplitude:int or float, 
            state_out_filename:str = c.RUN_INPUT_FILENAME, 
            dest_dir:str=c.SV_RUN_INPUT_DIR, include_ref:bool =False, out_forms:str='npy'):
    """ funcion description:
        in order to run a loop in arnoldi a run dir is used. this run_dir has to be filled with 
        states, which are 
                ref_state_begin + Q[i]
        while Q[i] "only" contains information on SV_AREA, SV_LEVEL and SV_VARS
        
        return: directly nothing.
                but fills the c.SV_RUN_INPUT_DIR        
    """


    ref_norm_full_state   = asv.open_full_state(asv.get_filepath('ref_full_begin_npy'))
                                                                                                
    ref_norm_state = asv.get_data( ref_norm_full_state )                       

    asv.clean_dir(dest_dir)

    # if it is the first loop, also the reference state needs to be included to the run.
    if include_ref:
        #run_member_path= asv.get_filepath('run_input',ens_mem=0, file_ending='.npy')
        run_member_ref_path = os.path.join(dest_dir, c.SV_REF_STATE_BEGIN_NAME+'.npy')        
        os.symlink(asv.get_filepath('ref_full_begin_npy'), run_member_ref_path)        
        #asv.write_full_state(ref_norm_full_state, run_member_ref_path )             #['nc']
        print(run_member_ref_path)
        
    
    for i in range(0,len(Qe)):
        print('set input dir files')

        if asv.is_state(Qe[i]): 
            pass
        elif os.path.isfile(Qe[i]):
            Qe[i]= asv.get_data(Qe[i])
        else: 
            print(type(Qe[i]))
            print(os.path.isfile(Qe[i]))   
            raise NotImplementedError('not implemented')


        #TODO: more efficent variant might be beneficial
        aQ = asv.smult(Qe[i],amplitude, copy=True)
        
        #if c.NORM_VARIANT in ['energy','energy_red']:
        aQ = asv.transform_to_original(aQ)
        factor_amplitude = amplitude / asv.norm( aQ )
        aQ = asv.smult(  aQ, factor_amplitude , copy=False)
        

        ref_aQ = asv.add(ref_norm_state,aQ, copy=True)  
        pert_full_state = asv.set_data( ref_aQ ,ref_norm_full_state,  
                        file_type='npy')                                              #or nc

        #####
        if isinstance(out_forms, str):
            out_forms=[out_forms]
        
        run_member_paths=[]
        for index in range(0, len(out_forms)):
            run_member_paths.append(asv.get_filepath(dest_dir,'.', state_out_filename, ens_mem=i, 
                                    file_ending= out_forms[index]))

                                    
        asv.write_full_state(pert_full_state, run_member_paths, leadtime=0)           #['nc']

        
###################################################################################################

def prep_arnoldi_dirs(clean_input_dir:bool=False, keep_Q:bool=True):
    """ funcion description:
    
        create req dirs, if existend, clean these.
        
        return: directly nothing.
    
     DIRECTORY STRUCTURE:
     --------------------------------------
     /workdir
     /iodir
           /input
                /ref
           /run
                /iter/
                /input
                /output
           /output 
                
    --------------------------------------
        
    """
    
   
    # if io_dir does not exist create it. 
    if not os.path.exists(c.SV_IO_DIR):
        os.makedirs(c.SV_IO_DIR)
    else:
        if os.path.exists(c.SV_OUTPUT_DIR):
            asv.clean_dir(c.SV_OUTPUT_DIR)
        if os.path.exists(c.SV_RUN_DIR) and not keep_Q:
            asv.clean_dir(c.SV_RUN_DIR)
        if os.path.exists(c.SV_INPUT_DIR) and clean_input_dir:
            asv.clean_dir(c.SV_INPUT_DIR)
    
    if not os.path.exists(c.SV_INPUT_DIR):
        os.makedirs(c.SV_INPUT_DIR)
    if not os.path.exists(c.SV_OUTPUT_DIR):
        os.makedirs(c.SV_OUTPUT_DIR)
    if not os.path.exists(c.SV_RUN_DIR):    
        os.makedirs(c.SV_RUN_DIR)

    try:
        os.makedirs( os.path.join(c.SV_INPUT_DIR,'ref'))
    except:
        pass

    try:
        os.remove(asv.get_filepath('ref_full_end_npy'))
    except:
        pass


    if keep_Q:
        print('keep Q-Files of first block')
        if os.path.exists(os.path.join(c.SV_RUN_DIR,'input')):
            asv.clean_dir(os.path.join(c.SV_RUN_DIR,'input'))
        if os.path.exists(os.path.join(c.SV_RUN_DIR,'output')):
            asv.clean_dir(os.path.join(c.SV_RUN_DIR,'output'))

    if not os.path.exists(os.path.join(c.SV_RUN_DIR,'input')):
        os.makedirs( os.path.join(c.SV_RUN_DIR,'input') )
    if not os.path.exists(os.path.join(c.SV_RUN_DIR,'output')):        
        os.makedirs( os.path.join(c.SV_RUN_DIR,'output'))

    
    if not keep_Q:
        if not os.path.exists(os.path.join(c.SV_RUN_DIR,'iter')):     
            os.makedirs( os.path.join(c.SV_RUN_DIR,'iter'))
    else: 
        Q_paths=asv.get_filepath( os.path.join(c.SV_RUN_DIR,'iter'),variant='glob')
        for i in range(c.BLOCK_SIZE,len(Q_paths)):
            try:
                os.remove(Q_paths[i])
            except Exception as e:
                raise

    #os.makedirs( os.path.join(c.SV_OUTPUT_DIR,'nc'))
    #os.makedirs( os.path.join(c.SV_OUTPUT_DIR,'npy'))
    #os.makedirs( os.path.join(c.SV_OUTPUT_DIR,'grib'))
    
    print('all dirs created')


###################################################################################################
def prepare_ref(desig_ref_file_path, timepoint:str, ref_copy=False):
    """ funcion description:
        
        moves! ref state to ~/input/ref/
        creates also a athpy/.npy copy  if it is the ref state at first timepoint / initial time.
        
        returns: ref_state athpy/.npy path
        
    """

    #TODO: move nochmal überlegen. evt. Unterscheidung zwischen input dir oder anderem.


    if timepoint not in ['begin','end']:
        raise ValueError("needs to be 'begin' or 'end'")



    ref_athpy_path = asv.get_filepath('ref_full_'+timepoint+'_npy')

    if timepoint == 'begin':
        try:
            asv.clean_dir( os.path.join(c.SV_INPUT_DIR,'ref'))
        except:
            pass
        try:
            os.makedirs( os.path.join(c.SV_INPUT_DIR,'ref'))
        except:
            pass
            
        ref_nc_path , file_ending = os.path.splitext(ref_athpy_path)
        ref_nc_path = ref_nc_path+'.nc'
        if ref_copy:
            shutil.copyfile(desig_ref_file_path, ref_nc_path)         
        else:
            os.rename(desig_ref_file_path, ref_nc_path)         # os.rename = 'mv' in bash
       
        original_ref_path = desig_ref_file_path
        desig_ref_file_path = ref_nc_path                  # imported order. 
                                                # otherwise a non existent file is opened
                                                # and the md:ref_file path is wrong
    

    print(desig_ref_file_path)

    
    # create numpy full state.
    ref_state=asv.open_full_state(desig_ref_file_path)
    asv.write_full_state(ref_state, ref_athpy_path)

    #if ref_copy:
    #    print('copy ref back to original path')
    #    original_ref_path_athpy , file_ending = os.path ... +'npy' ..
    #    shutil.copyfile(ref_athpy_path, original_ref_path)

    return ref_athpy_path



###################################################################################################
def getFreeGPUId():
    """ funcion description:
        
        returns: a list of free GPU device IDs
    """

    #if 'pycuda.autoinit' not in sys.modules:
    if 'cuda' not in sys.modules:
            import pycuda.driver as cuda
            import pycuda.autoinit
    

    # Get the number of available GPU devices
    num_gpus = cuda.Device.count()

    # Initialize a list to store the free GPU devices
    free_gpu_devices = []

    # Iterate over the available GPU devices
    for i in range(num_gpus):
        # Get the current GPU device
        device = cuda.Device(i)

        # Check if the GPU device is available
        if device.compute_capability() != (0, 0):
            free_gpu_devices.append(i)


    # Print the free GPU devices
    return free_gpu_devices




###################################################################################################



def submit_run(leadtime:int,
                        save_interim_results:bool = False, run_mode:str = 'local',
                        job_name:str="eim_pang", out_forms:str='npy', include_ref = False):
                                                  # 'nc','npy','grib','all'
    """ function description:
    
        start inference/runs.
        
    """


    #python3 ./run_pangu_6.py an_file leadtime output_data_dir True/False all/npy/nc/grib
    #                                                       save_interim_results

    #an_file = os.path.join(c.SV_RUN_INPUT_DIR, c.RUN_INPUT_FILENAME)
    an_files = asv.get_filepath(c.SV_RUN_INPUT_DIR, variant='glob')
    
    #do not submit ; compute on this maschine / if the whole job is submitted to a machine this 
    #                                           is the right choice.
    if run_mode=='local':
        


        for i_file in range(len(an_files)):
        #for member in range(ens_start, ens_end+1):


            free_gpus=[]

            if c.PROCESSUNIT in ['GPU']:
                # Define the number of threads to use
                free_gpus = getFreeGPUId()
                print('free_gpus: '+str(free_gpus))

            # zero gpus
            if len(free_gpus)==0 and c.PROCESSUNIT in ['GPU']:
                print('wait for free gpu nodes')
                time.sleep(4)

                            
            # one gpu or just cpu
            elif len(free_gpus) ==1 or c.PROCESSUNIT in ['CPU']:
                
                if c.MODEL in ['pangu_6']:
                    run_pangu_6.main(an_files[i_file], leadtime,
                        save_interim_results,out_forms=out_forms, 
                                gpu_dev=-1)
                elif c.MODEL in ['pangu_24']:
                    run_pangu_24.main(an_files[i_file], leadtime,
                        save_interim_results,out_forms=out_forms, 
                                gpu_dev=-1)
                else:
                    raise NotImplementedError("please use 'pangu_6' or 'pangu_24'")
                #time.sleep(1)

            #two or more gpus.
            else:

                n_threads = min(len(free_gpus) , len(an_files)-i_file ) 

                # Use a ThreadPoolExecutor to run the function in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
                    # Submit tasks to the executor
                    if c.MODEL in ['pangu_6']:
                        futures = [executor.submit( run_pangu_6.main, 
                                        *[an_files[i_file + gpu_i],leadtime, save_interim_results,
                                        out_forms, free_gpus[gpu_i] ] 
                                        ) for gpu_i in range(0,n_threads) ]
                    elif c.MODEL in ['pangu_24']:
                        futures = [executor.submit( run_pangu_24.main, 
                                        *[an_files[i_file + gpu_i],leadtime, save_interim_results,
                                        out_forms, free_gpus[gpu_i] ] 
                                        ) for gpu_i in range(0,n_threads) ]
                    else:
                        raise NotImplementedError("please use 'pangu_6' or 'pangu_24'")
                    
                    i_file = i_file+n_threads

                    
                    # Wait for all threads to complete
                    #for future in concurrent.futures.as_completed(futures):
                    #    pass

                # Print a message after all threads have completed
                print("Current threads have completed.")
                                
                #time.sleep(1)


    # do an extra submit to a specific system. This will probably not your preffered choice.
    # Try to submit the whole job. If you intend to use that feature, you have to adapt it to your
    # system by your own.
    elif run_mode == 'submit':
    
        print("WARNING: 'submit' does an extra qsub of the runs to your batch system. "+ 
                "This will probably not your preffered prefered choice. "+ 
                "Rather submit the whole job and use 'local'. "+ 
                "If you intend to use that option anyway, you have to adapt it to your system"+ 
                "by your own, in particular the 'qsub_inference_cpu.default' file.")
                
        if c.MODEL in ['pangu_24', 'pangu_6']:
            raise NotImplementedError("please use currently RUN_MODEL='local'")

        if c.PROCESSUNIT in ['CPU']:

            #TODO: Übergabe verbessern

            an_file = os.path.join(c.SV_RUN_INPUT_DIR, c.RUN_INPUT_FILENAME)
            #TODO: if out_forms= .. dann 'all'
    
            batch_block = 1

            ens_end = c.BLOCK_SIZE
            if include_ref:
                ens_start=0
            else:
                ens_start=1

            j=ens_start


            qsub_file_default = os.path.join(c.SV_WORK_DIR, 'qsub_inference_cpu.default')


            while j <= ens_end:
            
                ens_start_i=j
                ens_end_i = min(ens_end , ens_start_i+batch_block)    
                if j == 0:

                    job_name_curr = job_name+'_det'
                    qsub_file_curr = os.path.join(c.SV_WORK_DIR,'qsub_inference_cpu_det.sh')
                    ens_start_i=ens_end_i=0                    
                    j=1
                    an_det_file = os.path.join(c.SV_RUN_INPUT_DIR, c.SV_REF_STATE_BEGIN_NAME)
                    replace=([an_det_file,   str(leadtime),   c.PY_VENV_PATH,   c.SV_WORK_DIR, 
                            job_name_curr, str(ens_start_i),  str(ens_end_i), 
                                str(int(save_interim_results)),  out_forms])

                else:
                    batch_block_str = str(ens_start_i)+'_'+str(ens_end_i)

                    job_name_curr = job_name+ batch_block_str
                    qsub_file_curr = os.path.join(c.SV_WORK_DIR, 
                                            "qsub_inference_cpu_"+batch_block_str+".sh")
                    j = ens_end_i+1
                    replace=([an_file,   str(leadtime),   c.PY_VENV_PATH,   c.SV_WORK_DIR, 
                            job_name_curr, str(ens_start_i),  str(ens_end_i), 
                                str(int(save_interim_results)),  out_forms])


                #      qsub_file_curr = os.path.join(c.SV_WORK_DIR, 'qsub_interference_cpu.sh')

                
                search=(['%AN_FILE_LL%','%LEADTIME%','%PY_VENV_PATH%','%WORK_DIR%',     
                            '%JOB_NAME%', '%ENS_START%',  '%ENS_END%' , 
                                '%SAVE_INTERIM_RES%', '%OUT_FORMS%'])

                
                _multi_sed(search,replace, qsub_file_default, qsub_file_curr)


                process = subprocess.run(['qsubw', '-h', '-E', '2', qsub_file_curr]) 
                #process = subprocess.Popen(['qsubw', '-h', '-E', '2', qsub_file_curr],shell=True) 


                time.sleep(3)


		        #echo $j
            #endloop


        
        elif c.PROCESSUNIT in ['GPU']:
            raise NotImplementedError("PROCESSUNT=GPU is not implemented for 'submit'. "+
                                "Please Use 'local'.")
    else:
        raise ValueError("unvalid value. choose 'submit' or 'local'")




###################################################################################################
def are_fc_states_avail(expect_num:int=c.BLOCK_SIZE, wait_min=30):
    """ funcion description:
        waits until runs (of loops) are finished or until a allowed period of waiting is over.
        
        returns True
        
    """

    

    t_end = time.time() + 60 * wait_min

    while time.time() < t_end:
        paths = asv.get_filepath('run_output')
        
        if len(paths) == expect_num:
            print('All required states are avail.')
            
            break
            
        elif len(paths) < expect_num:
            print('loop not ready. wait 15 seconds')
            if c.RUN_MODE == 'submit':
                subprocess.run(['qstat'])
            time.sleep(15)
        elif len(paths) > expect_num:
            print("WARNING: unexpected high number of forecasted states")
            time.sleep(4)
            break
    #endwhile
    if time.time() < t_end:
        return True
    else:    
        raise RuntimeError("Error while running the forecasting. Within "+str(wait_min)+
            " minutes the system was not able to create all nessecary states.")




################################################################################################### 
def _multi_sed(pattern:list, replace:list, source, dest=None):
    """ funcion description:
        
        an adaption of the commonly known shell 'sed' method to python.
        
        returns: directly nothing.
                 modifies an document or creates a modified copy
    """

    fin = open(source, 'r')
    
    n_pat = len(pattern)
    if n_pat != len(replace):
        raise ValueError("'pattern' and 'replace' need to be the same size")

    if dest:
        fout = open(dest, 'w')
    else:
        from tempfile import mkstemp
        fd, name = mkstemp()
        fout = open(name, 'w')

    for line in fin:
        out=line
        for index in range(0,n_pat):
            out = re.sub(pattern[index], replace[index], out)
        fout.write(out)

       
    #try:
    #    fout.writelines(fin.readlines())
    #except Exception as E:
    #    raise E
    
    fin.close()
    fout.close()

    if not dest:
        shutil.move(name, source) 

    print('multi_sed done')


################################################################################################### 
def write_run(states_energy , amplitude:int or float, leadtime= c.T_OPT, 
                                                save_interim_results:bool=False): 
    """ funcion description:
    
        - takes a state_list
        - transforms it back to original (if req)
        - fills the c.SV_RUN_INPUT_DIR
        - evt. modifes a qsub batch script.
        - submits the run
        - reads the new states.
        
        returns these to main arnoldi routine - a list of states
    """


    #ens_range=[1,c.BLOCK_SIZE]

    # is it the first run?
    if not (os.path.exists( asv.get_filepath('ref_full_end_npy') )):
        #does not exist? --> frist run            
        first_loop=True
        #ens_range[0]=0 
    else:
        first_loop=False
        
    # create the pert files for iterration runs.
    fill_dir_with_pert_states(states_energy, amplitude=amplitude, include_ref =first_loop)
    asv.clean_dir(c.SV_RUN_OUTPUT_DIR)
     
    submit_run(leadtime = leadtime , save_interim_results = save_interim_results, 
                run_mode=c.RUN_MODE, job_name="eim_pang_", out_forms='npy', include_ref=first_loop)
    # RUN STARTED
    
    # waits for finished runloop.
    are_fc_states_avail(c.BLOCK_SIZE+int(first_loop))       

    # gets run_output paths.
    paths_run_output = asv.get_filepath('run_output')
            
    if first_loop:
        n_out = len(paths_run_output)
        for index in range(0,n_out):
            if c.SV_REF_STATE_BEGIN_NAME in paths_run_output[index]:
                print('out_ref_state_gefunden')

                ref_norm_state_end = prepare_ref(paths_run_output[index], 'end')
                del(paths_run_output[index])
                break

        if n_out != len(paths_run_output)+1:
            raise RuntimeError("Evolved reference state not found in output paths.")  

    else:
        ref_norm_state_end = asv.get_filepath('ref_full_end_npy')
        

    
    ref_norm_state_end = asv.get_data(ref_norm_state_end)
    ref_energy_state_end = asv.transform_to_energy(ref_norm_state_end)

    state_list_new=[]
    
    for i in range(0,len(paths_run_output)):
        state_list_new.append({})
        state_list_new[i] = asv.transform_to_energy( asv.get_data( paths_run_output[i]) )
        state_list_new[i] = asv.sub( ref_energy_state_end, state_list_new[i], copy=True)
        state_list_new[i] = asv.smult(state_list_new[i], -1, copy=False)
        # why so complicated? because this way the copy comes from the first argument, 
        #   which is the reference. Better to avaid errors, but should work the other way 
        #   either. should be improved later.
        # for np.ndarray (as currently used) not relevant.

    #currently outcomment
    #W = asv.get_filepath('W',ens_mem=[*range(0,c.BLOCK_SIZE)] )
    #for i in range(0, len(W)):
    #    asv.write_athpy_state(W[i],state_list_new[i] ) 

    # option: return only paths
    # return W

    print('run of loop finished')
    return state_list_new





###################################################################################################
def forecast_states(amplitude:int,leadtime:int, an_date:int,state_list='sv1'):
    """ function description:
    
        after generation of ASV-Perts this funcion allows you to start forecasts of the generated
        perturbed reference state.
        
        forecasts:  ref_state
                    ref_state + amplitude*Q_0
                    state_1
                    state_list
        over given leadtime with given amplitude.
        Note: state_1 and state_list need to be some perturbed reference states 
                with the correct amplitude

    """
        
    asv.clean_dir(asv.c.SV_RUN_INPUT_DIR)
    asv.clean_dir(asv.c.SV_RUN_OUTPUT_DIR)

    if state_list == 'sv1':
        state_list=([asv.get_filepath('output',name=c.SV_PERT_FILENAME, 
                            ens_mem=[0],file_ending='.npy')])
        print(state_list)
        #os.path.join(asv.c.SV_OUTPUT_DIR, 'Full_SV_pert_m001.npy')
    
    for state in state_list:
        if not os.path.exists(state):
            raise ValueError("path to state does not exist: "+str(state))


    
    
    print('start forecast states.')
    #FORECAST

    # 1. ref state
    # 2. asv initial iteration start state + ref  ( e.g. random or ens_diff) 

    Q=[asv.get_filepath('Q',ens_mem=0)]
    file_Q= Q[0]

    fill_dir_with_pert_states(Q, amplitude=amplitude, 
            state_out_filename='Q', 
            dest_dir=c.SV_RUN_INPUT_DIR, include_ref=True,out_forms='npy')
    # NOTE: after fill_dir_(..) the enumeration will always start with 0, no matter which state 
    #           was choosen before.



    temp_dir, q_filename = os.path.split(file_Q)
    q_filename_new='random_pert'
    
    os.rename(os.path.join(c.SV_RUN_INPUT_DIR, q_filename) , 
                        os.path.join(c.SV_RUN_INPUT_DIR, q_filename_new+'.npy') )


    for state in state_list:    
        shutil.copy(state, asv.c.SV_RUN_INPUT_DIR)    


    # calc the expected number of output_files.
    if c.MODEL in ['pangu_6']:
        n_iter = int(np.floor( (leadtime/6 )))
    elif c.MODEL in ['pangu_24']:
        n_iter = int(np.floor( (leadtime/24 )))       

    nr_fc_states = len( asv.get_filepath(c.SV_RUN_INPUT_DIR, variant='glob') )

    if isinstance(c.OUT_FORECAST_FORMATS, str):
        n_types = 1
    elif isinstance(c.OUT_FORECAST_FORMATS, list):
        n_types = len(c.OUT_FORECAST_FORMATS)
    else:
        raise TypeError("'OUT_FORECAST_FORMATS' in 'c.py' is of wrong type.")
    expected_states = n_iter * nr_fc_states * n_types
    ####################

    print('') 
    submit_run(leadtime = leadtime , save_interim_results = True, run_mode=c.RUN_MODE,
                        job_name="eim_pang_", out_forms=c.OUT_FORECAST_FORMATS, include_ref=True)
    
    # RUN STARTED
    
    are_fc_states_avail(expected_states)

    # mk dir output/forecasts

    
    out_fc_dir=c.SV_OUTPUT_FORECASTS_DIR

    if not os.path.exists(out_fc_dir):
        os.makedirs( out_fc_dir )
    else:
        asv.clean_dir( out_fc_dir)

    fc_file_paths = asv.get_filepath('run_output')

    for i in range(0,len(fc_file_paths)):
        head, file_name = os.path.split(fc_file_paths[i])
        dest_file_path = os.path.join(out_fc_dir, file_name)
        os.rename(fc_file_paths[i] , dest_file_path )
    

    print('finished')
   
###################################################################################################

###################################################################################################
###################################################################################################

