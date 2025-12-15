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
import os
import subprocess
import xarray as xr
import glob    
import shutil
import math
import copy


__all__ = (["open_full_state", "get_data", "set_data", "get_lat_ind", "get_lon_ind", "get_lev_ind",
            "get_filepath", "write_full_state","write_athpy_state", "open_athpy_state",
            "is_increment","seperate_state","get_meta", "merge_state","copy_state", "delete_state",
            "clean_dir", "is_state","transform_to_energy", "transform_to_original","add","sub",
            "smult", "dot_prod", "is_computable","norm","lk_of_states", "mm_mult", 
            "transform_from_nc_to_npy"])




###################################################################################################
# functions                
def open_full_state(filepath, return_xarr:bool=False):
    """function description:

       open file from filepath and prepare inputdata

       within asvtools meta- surface- and upper air data (an athpy state) is stored within ONE 
       npy-file (very fast and easy).
       pangu for e.g. uses two seperate files for upper-air and surface without metadata.

       Hence, pangu requirements are the basis for a "full_state".
       pangu needs pressure / upper-air variables. It is a numpy array shaped (5,13,721,1440) 
       where the first dimension represents the 5 surface variables
       (FI (Z), QV (Q), T, U and V) in the exact order

       surface variables. It is a numpy array shaped (4,721,1440) 
       where the first dimension represents the 4 surface variables 
       (MSLP, U10, V10, T2M in the exact order).


       13 pressure levels (1000hPa, 925hPa, 850hPa, 700hPa, 600hPa, 500hPa, 400hPa, 300hPa, 250hPa, 
                               200hPa, 150hPa, 100hPa and 50hPa in the exact order).

       lat range is [90,-90] degree and lon range [0,359.75] degree
       
       NOTE:
       For ICON input the lat-data will be inverted. Hence THIS program expects lat data in the 
       range [-90,90]. 
    """


    

    if not os.path.exists(filepath):
        raise IOError("no such file: "+filepath)

    # get filetypes from fileending
    filename, file_type = os.path.splitext(filepath)


    nlat=721
    nlon=1440
    nlev=13


    # variables, lat, lon       # reverse lat
    state = {}
    for var in c.SFCVARS:
        state[var] = np.zeros((nlat,nlon))

    for var in c.PVARS:
        state[var] = np.zeros((nlev,nlat,nlon))
    

    ###############################################################################################
    ## open dateset
    ##################
    try:
        if file_type in ['.grib','.grb','.grib2','.grb2','grib','grb','grib2','grb2']:
            raise NotImplementedError("Handling for grib files is not directly provided. Please"+
                    " use netcdf4 (nc) files. You can use cdo:'cdo -f nc copy 'in.grib' 'out.nc''") 
        elif file_type in ['.npy', 'npy']:
            state_tmp = np.load(filepath, allow_pickle=True)

            for var in c.META_KEYS:
                try:
                    state[var] = state_tmp.item().get(var)
                except:
                    print(var+' is not available.')
                    
            for var in c.SFCVARS:
                try:            
                    state[var] = state_tmp.item().get(var)
                except:
                    print(var+' is not available.')
                    
            for var in c.PVARS:
                try:
                    state[var] = state_tmp.item().get(var)  
                except:
                    print(var+' is not available.')
                    
            del state_tmp

            return state      

        else: 
            ## assuming netCDF4 file.   
            ds = xr.open_dataset(filepath, decode_times=False)
            ds = ds.reindex(plev=c.PANGU_PLEVELS) 
            # ds = ds.reindex(c.LAT=
            #TODO: dasselbe mit lat lon
    except IOError:
        raise


    ###############################################################################################
    # set meta data
    ########################

    # META_KEYS = (['md:area','time',LAT,LON, LEVEL, 'gptype',
    #         'md:norm_euclid','md:norm_energy',  
    #         'md:state_type', 'md:leadtime', 'md:ref_file')
    
    state['md:area'] = 'earth'
    state['time'] = ds.coords['time'].values
    
    state[c.LAT] = ds.coords[c.LAT].values[::-1]	# reverse order of lats
    state[c.LON] = ds.coords[c.LON].values[:]
    state['lat_area'] = 'earth'
    state['lon_area'] = 'earth'

    state[c.LEVEL] = ds.coords[c.LEVEL].values[:]      #[::-1]
    state['gptype'] = 0
    state['md:norm_euclid'] = None
    state['md:norm_energy'] = None
    state['md:state_type'] = 'original'
    state['md:leadtime'] = None     #0
    state['md:ref_file'] = filepath
    
   

    ###############################################################################################
    # surface data
    ##################

    
    #input_sfc = np.zeros((4,721,1440))

    try:
        # 1. PMSL     # 2. U10M    # 3. V10M    # 4. T2M
        state['PMSL'] = ds['PMSL'].values[0,::-1,:].astype(np.float32)
        state['U_10M'] = ds['U_10M'].values[0,0,::-1,:].astype(np.float32)
        state['V_10M'] = ds['V_10M'].values[0,0,::-1,:].astype(np.float32)
        state['T_2M'] = ds['T_2M'].values[0,0,::-1,:].astype(np.float32)
    except (KeyError, IndexError):
        ds.close()
        print("Key- or Indexerror occurs. The given file doesn't contain all nessecary keys. "+
            "These are 'FI', 'QV'/'RELHUM', 'T', 'U', 'V', 'PSML', 'U_10M', 'V_10M', 'T_2M'. "+ 
            "Maybe the shape of the data is wrong.")    
        raise


    ###############################################################################################
    # upper air data    
    ####################
    
    #variables , levels, lat, lon       # reverse: lat 
    #input_p = np.zeros((5,13,721,1440))


    try:
        # Z / FI - geopotential
                        
        state['FI'] = ds['FI'].values[0,:,::-1,:].astype(np.float32)       #[0,::-1,::-1,:]
        # T - temperature
        state['T'] = ds['T'].values[0,:,::-1,:].astype(np.float32)         #[0,::-1,::-1,:]
        # U - U-wind
        state['U'] = ds['U'].values[0,:,::-1,:].astype(np.float32)         #[0,::-1,::-1,:]
        # V - V-wind
        state['V'] = ds['V'].values[0,:,::-1,:].astype(np.float32)         #[0,::-1,::-1,:]

    except (KeyError, IndexError):
        ds.close()
        print("Key- or Indexerror occurs. The given file doesn't contain all nessecary keys. "+
            "These are 'FI', 'QV'/'RELHUM', 'T', 'U', 'V', 'PSML', 'U_10M', 'V_10M', 'T_2M'. "+ 
            "Maybe the shape of the data is wrong.")    
        raise


    ## open Q
    try:
        # Q - specific humidity
        state['QV'] = ds['QV'].values[0,:,::-1,:].astype(np.float32)   #[0,::-1,::-1,:]
 
    except KeyError:
        from metpy.calc import dewpoint_from_relative_humidity, specific_humidity_from_dewpoint
        from metpy.units import units
        import metpy

        
        print('Datafile does not contain QV. Compute QV from RELHUM')
    
        try:
            # compute dew point
            ds['TD']= dewpoint_from_relative_humidity(ds['T']*units.K, ds['RELHUM']*units.percent)
        
        except (KeyError, IndexError):
            ds.close()
            print("Key- or Indexerror occurs. The given file doesn't contain all nessecary keys."+
             "These are 'FI', 'QV'/'RELHUM', 'T', 'U', 'V'. Maybe the shape of the data is wrong.")    
            raise
        #endtry

        ds['QV'] = specific_humidity_from_dewpoint(ds['plev'], ds['TD'])#.astype(np.float32)
        ds['TD'] = ds['TD'].metpy.dequantify()
        ds['QV'] = ds['QV'].metpy.dequantify()
        ds['QV'].values = ds['QV'].values.astype(np.float32)
        #################
        # modify original state

        #ds.QV.attrs['_Fillvalue']=NaNf ?
        ds.QV.attrs['standard_name'] = 'specific_humidity'
        ds.QV.attrs['long_name'] = 'Specific humidity'
        ds.QV.attrs['units'] = '%'
        #ds.QV.attrs['param'] = '?'

        ds=ds.drop_vars(['TD'])
        ds=ds.drop_vars(['RELHUM'])
        #write to file
        print('rewrite nc file')
                
        os.remove(filepath)
        ds.to_netcdf(path=filepath, mode='w', format='NetCDF4')
        
        
        ################
        # QV - specific humidity
        state['QV'] = ds['QV'].values[0,:,::-1,:]  #[0,::-1,::-1,:]  

    
    if return_xarr:
        return state, ds
    else:
        ds.close()
        return state


###################################################################################################
def transform_from_nc_to_npy(file_source:str, file_dest:str, use_vars=c.PANGU_VARS):
    """ funcion description:
    """               
    
    raise NotImplementedError('not impl.')
    
    


###################################################################################################
def get_data(state:str or dict, keyvars:list or str=c.SV_VARS,area=c.AREA,                              
                    use_level=c.SV_LEVELS, file_type='npy'):                                         
    """ funcion description:
        
        get the relevant data for asv run.
        This means only the required c.SV_VARS, c.SV_LEVELS, c.SV_AREA
        
        if a state path is passed that data is read from disk, otherwise a passed state is used.
        
        return: the relevant part of that state.
        
    """               


    if isinstance(state,dict):
        pass
    elif os.path.isfile(state):
        
        state = open_full_state(state)
    
    if isinstance(keyvars,str):
        keyvars=[keyvars]

    
    result = {} 

    if state['md:area'] != area:

        if state['md:area'] in ['earth']:
            lat_ind = get_lat_ind(area,state[c.LAT])
            lon_ind = get_lon_ind(area,state[c.LON])
        else:
            lat_ind = get_lat_ind(area,state['lat_area'])
            lon_ind = get_lon_ind(area,state['lon_area'])


        lev_ind=c.SV_LEVELS    #TODO: get_level_ind(state[level], use_level) 
        

        for var in keyvars:
            if var in c.SFCVARS or var in ['KE_10M']:
                result[var] = state[var][np.ix_(lat_ind,lon_ind)]
            elif var in c.PVARS or var in ['kinetic_energy','KE']:
                result[var] = state[var][np.ix_(lev_ind,lat_ind,lon_ind)] 
                
            else:
                raise KeyError(str(var)+': unvalid key')

        for key in state.keys():
            if key in c.META_KEYS:                  # get_meta_state?!
                    result[key] = state[key]
                

        result['lat_area'] = state[c.LAT][lat_ind]
        result['lon_area'] = state[c.LON][lon_ind]
        #result[c.LEVEL] = result[c.LEVEL][lev_ind]
        result['md:area'] = area

        
    else:
        for key in c.META_KEYS:
            result[key] = state[key]
        for key in keyvars:
            result[key] = state[key][:]
    #endif

    return result

###################################################################################################
def set_data(newdata_state:dict,full_state, keyvars:list or str = c.SV_VARS, 
                        file_type='nc', leadtime=0):
    """ funcion description:
        
        in order to perturb a state the perturbation data has to be added to a full state.
        
        uses data of a passed state, adds it to a defined given state.
        
        returns: if full state was passed: that reseted state.
                 if path of full state was passed: boolean True
                        but that state will be rewritten.   
        
    """               
    #TODO:  pass an increment ADD inc
    #       pass an absolute state RESET full state.
                
    #TODO: copy option?

    from_file=False
    result ={}
    if not is_state(full_state) and isinstance(full_state,str):
        if os.path.isfile(full_state):
            full_state_path=full_state        
            full_state = open_full_state(full_state,file_type=file_type)                
            from_file=True
        else:
            raise ValueError('no valid file(path)')

    if newdata_state['md:state_type'] == 'energy':
        raise TypeError('data is in energyspace. This can not set to a file.')
    if newdata_state['gptype'] in [201, 'inc']:
        raise NotImplementedError('state is an incremental state. Use an absolute state. '+
                            'This function is not yet implemented.')

    area = newdata_state['md:area']
    

    if full_state['md:area'] not in ['earth']:
        raise TypeError("dest state is not a valid state to write at.")
            #TODO: Funktioniert Funktion nicht dennoch (i.W.)?
    
    # = get_area_data(state,keyvars,area,lev_ind)
    
    lat_ind = get_lat_ind(area,full_state[c.LAT])
    lon_ind = get_lon_ind(area,full_state[c.LON])
    
    lev_ind = c.SV_LEVELS       #get_lev_ind(...)

    if isinstance(keyvars,str):
        keyvars=[keyvars]

    # only write the values (not the full xarray)
    for var in keyvars:
        if var in c.SFCVARS:
            #full_state[var][lat_ind,lon_ind].values = newdata_state[var][:].values
            full_state[var][np.ix_(lat_ind,lon_ind)] = newdata_state[var][:]
        elif var in c.PVARS:
            #full_state[var][lev_ind,lat_ind,lon_ind].values = newdata_state[var][:].values
            full_state[var][np.ix_(lev_ind,lat_ind,lon_ind)] = newdata_state[var][:]
        else:
            raise KeyError('unvalid key')
    
    

    if from_file:
        write_full_state(full_state, full_state_path, leadtime=leadtime)
        return True
    else:
        return full_state


################################################################################################### 
def get_lat_ind(area:str or list,lats):
    """ funcion description:
    
        pass a area defined by name and or a latitude numpy array.
        
        returns: the indices of that lats-array which denote lats within the given area.
    """               
    

    if area == 'earth':
        return np.array([*range(0,len(lats))])
    elif area == 'europe':
        lat_b0= c.EUR_LAT[0]
        lat_b1 = c.EUR_LAT[1]
    elif area == 'north_hemi':
        lat_b0 = c.NORTH_HEMI_LAT       #smaller border number
        lat_b1 = 91                     #larger
    elif area == 'south_hemi':
        lat_b0= -91            
        lat_b1= c.SOUTH_HEMI_LAT
    elif area[0] in ['custom']:
        lat_b0 = area[1]
        lat_b1 = area[2]        
    else:
        raise NotImplementedError("no valid area / not yet implemented.")


    ## lats
    ##############
    # test if both lat_borders are numbers
    if isinstance(lat_b0, (int, float)) and isinstance(lat_b1, (int, float)):
        cond = np.logical_and(lats >= (lat_b0 ) , lats <= (lat_b1 ) )
        
        index_lat = np.array(np.where(cond), dtype=int).reshape(-1)
        del cond
        
    return index_lat

        
###################################################################################################
def get_lon_ind(area:str or list,lons):
    """ funcion description:
    
        pass a area defined by name and a longitude numpy array.
        
        returns: the indices of that lons-array which denote lons within the given area.
    """               
    


    if area in ['earth','north_hemi' ,'south_hemi']:
        return np.array([*range(0,len(lons))])
    elif area == 'europe':                                                                                      
        lon_b0 = c.EUR_LON[0]
        lon_b1 = c.EUR_LON[1]
    elif area[0] in ['custom']:
        lon_b0 = area[3]
        lon_b1 = area[4] 
    else:
        raise NotImplementedError("other areas not implemented yet")

    
    
    ## lons
    ###############
    ### attention:
    ### antimeridan (or prime meridan) can be part of the area. --> two cases 
    ### 

    if lon_b0 < lon_b1:
            cond = np.logical_and(lons >= (lon_b0 ) , lons <= (lon_b1 ) )
        ### part of 
    else:
            # detailed but not nessesary (?!!)
            # cond = np.logical_or( np.logical_and(lons > (lon_border[0] ) , lons < 361) 
            #                       , np.logical_and(lons >= 0 , lons < lon_border[1]) )
            cond = np.logical_or( lons >= lon_b0 , lons <= lon_b1 )
        
    index_lon = np.array(np.where(cond), dtype=int).reshape(-1)
    del cond
    
    
    return index_lon    
  

###################################################################################################
def get_lev_ind( avail_level, req_level):
    """ funcion description:
    
        pass array of available level and array of required levels
        
        returns: an array of indices which denote the levels of the available-level-array, 
                    which are required. 
    """               
    

    raise NotImplementedError('Is not implemented, yet. Please use all levels.')
     

###################################################################################################
def get_filepath(folder:str,add_folder:str='.',name:str='', ens_mem: int or str or list='', 
            leadtime: int or str='',file_ending:str='', variant:str='const'):
    """ function description:
    
    returns: a filepath or a list of filepaths to handle all (state)paths.
    
    
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



    ###############################################################################################
    
    if leadtime != '':    
            leadtime = 'fc_'+str(leadtime).zfill(3)
        
    add_folder='.'
           
    ###############################################################################################

    if file_ending in ['npy','grib','nc','grb','grb2','grib2']:
        file_ending = '.'+file_ending


    if isinstance(ens_mem, int):
        ens_mem=[ens_mem]
    elif ens_mem in ['det','']:
        pass
    elif not isinstance(ens_mem, list):
        raise ValueError("ens_mem needs to be 'det' an non negative int or a list of such ints")


    # leadtime and ens_mem irr
    if folder in ['ref_full_begin_npy']: #full state?!
        path=os.path.join(c.SV_INPUT_DIR, 'ref', c.SV_REF_STATE_BEGIN_NAME + '.npy')
        return path
    elif folder in ['ref_full_end_npy']:
        path=os.path.join(c.SV_INPUT_DIR, 'ref', c.SV_REF_STATE_END_NAME + '.npy')        
        return path
    #elif f in ['ref_energy_state_begin']: ?
    #elif f in ['ref_energy_state_end']  ?
    elif folder in ['singular_values_file']:
        path= os.path.join(c.SV_OUTPUT_DIR, 'singular_values_file.txt')    
        return path
    elif folder in ['H_file']:
        path= os.path.join(c.SV_OUTPUT_DIR, 'H.txt')    
        return path
    
    # ens_mem leadtime relevant
    elif folder in ['Q']:
        path = os.path.join(c.SV_RUN_DIR,'iter','Q'+leadtime)
        
        #glob.. not possible here
    elif folder in ['W']:
        path = os.path.join(c.SV_RUN_DIR,'iter','W')
        ens_mem=[*range(0,c.BLOCK_SIZE)]
        #auch glob..            
            
    elif folder in ['input_states']:

        # for create initial vectors only
        #path = os.path.join(c.SV_INPUT_DIR, add_folder,c.INPUT_FILENAME) # +leadtime)
        paths = sorted(glob.glob(os.path.join(c.SV_INPUT_DIR, '*_*')))
        #TODO: anders aussortieren.    
        return paths

    elif folder in ['run_input']:
        #Glob does not work here. Paths have to be returned even if files not avail. 
        #NOT: paths = sorted(glob.glob(os.path.join(c.SV_RUN_INPUT_DIR, '*.*')))    
        #return paths 
        path = os.path.join(c.SV_RUN_INPUT_DIR, '.', add_folder, c.RUN_INPUT_FILENAME)
    elif folder in ['run_output']:
        
            paths = sorted(glob.glob(os.path.join(c.SV_RUN_OUTPUT_DIR, '*_*')))    
            return paths        
        
    elif folder in ['singular_vectors']:
        path = os.path.join(c.SV_OUTPUT_DIR,c.SV_FILENAME)
    
    elif folder in ['output']:
        path = os.path.join(c.SV_OUTPUT_DIR, add_folder,c.SV_PERT_FILENAME)
        
    else:       #use folder and name
        path=os.path.join(folder,add_folder,name+leadtime)

    # option allows to return the paths of EXISTING files.
    if variant == 'glob':
        paths = sorted(glob.glob(os.path.join(folder,add_folder, '*_*')))
        return paths


    if (folder in ['W','Q','run_input','run_output','singular_vectors'] and file_ending == '') :
        file_ending='.npy'

    
    if (ens_mem =='det' or ens_mem =='') and (folder not in ['Q','W']):
        path = path + file_ending
        return path
    else:
        paths=[]
        if ens_mem[0]=='det' and (folder not in ['Q','W']):
            ens_mem = np.delete(ens_mem, 0)
            paths.append( path + file_ending)
        
        for mem in ens_mem:
            paths.append( path +'_m'+str(mem).zfill(3) +file_ending)


    if len(paths)==1:
        paths=paths[0]

    return paths         
    
###################################################################################################
def write_full_state(output_state, out_paths:list or str, leadtime:int=0, ds=None):
    """ funcion description:
    
        write a full state to file. One can pass more than one output destination path in order to
        get outputs in different fileformats. These can be 'npy','nc','grib' ; format is implizitly
        defined by output file path ending.
        
    """               
    
    if isinstance(out_paths,str):
        out_paths=[out_paths]


    out_forms=[]
    for path in out_paths:
        
        fname, file_ext = os.path.splitext(path)
        if file_ext in ['npy','.npy']:
            out_forms.append('npy')
        elif file_ext in ['grib','grib2','grb','grb2','.grib','.grib2','.grb','.grb2']:
            out_forms.append('grib')
        elif file_ext in ['nc','.nc']:
            out_forms.append('nc')
        else:
            print(file_ext)
            raise ValueError("unknown file_type")
            #out_forms.append('nc')
        

    #filedir, an_file_name   = os.path.split(output_state['md:ref_file']) 
    #an_file_name, file_type = os.path.splitext(an_file_name)


    for index in range(0,len(out_forms)):

        if out_forms[index] in ['npy']:
            
            #meta, sfc, press = seperate_state(output_state)
            #write_state = merge_dict(sfc,press)
            #np.save( out_paths[index] , write_state )

            np.save( out_paths[index] , output_state ) 
        #endif npy
        
        elif out_forms[index] in ['grib','nc']:            

            if output_state['md:area'] not in ['earth']:
                raise ValueError("output state needs to contain the full earth as area."+ 
                    "Use set_data first or write_athpy_state.")

            if ds is None:
                
                ds = xr.open_dataset(output_state['md:ref_file'],decode_times=False)
                ds = ds.reindex(plev=c.PANGU_PLEVELS) 
            
                                    
            try:
                # pressure
                # 1. QV specific humidity
                ds['QV'].values[0,:,:,:] = output_state['QV'][:,::-1,:]         #[::-1,::-1,:]
            except KeyError:
                ds.close()
                state , ds = open_full_state(output_state['md:ref_file'], return_xarr=True)            
                ds['QV'].values[0,:,:,:] = output_state['QV'][:,::-1,:]         #[::-1,::-1,:]

            #######################################################################################
            #Pressure
            # 2. Z / FI - geopotential  # 3. T - temperature # 4. U-wind  # 5. V-wind
            ds['FI'].values[0,:,:,:]= output_state['FI'][:,::-1,:]          #[::-1,::-1,:]
            ds['T'].values[0,:,:,:] = output_state['T'][:,::-1,:]           #[::-1,::-1,:]
            ds['U'].values[0,:,:,:] = output_state['U'][:,::-1,:]           #[::-1,::-1,:]
            ds['V'].values[0,:,:,:] = output_state['V'][:,::-1,:]           #[::-1,::-1,:]

            ####################################################################################### 
            # Surface
            # 1. PMSL     # 2. U10M    # 3. V10M    # 4. T2M
            ds['PMSL'].values[0,:,:] = output_state['PMSL'][::-1,:]
            ds['U_10M'].values[0,0,:,:] = output_state['U_10M'][::-1,:]
            ds['V_10M'].values[0,0,:,:] = output_state['V_10M'][::-1,:]
            ds['T_2M'].values[0,0,:,:] = output_state['T_2M'][::-1,:]
           

            # if 'RELHUM' exists? delete it.  
            try:
                #ds.QV.attrs['standard_name'] = 'specific_humidity'
                #ds.QV.attrs['long_name'] = 'Specific humidity'
                #ds.QV.attrs['units'] = '%'
                ##ds.QV.attrs['param'] = '?'
                ds=ds.drop_vars(['RELHUM'])
                ds=ds.drop_vars(['TD'])

            except:
                pass
            
            #######################################################################################
            # set leadtime metadata
            time_attrs = ds.time.attrs
            fc = ds.coords['time'].values
            fc = fc + leadtime
            units, reference_date = time_attrs['units'].split('since')
            #units, reference_date = ds.time.attrs['units'].split('since')
            units = 'hours'
            time_attrs['units']=units+' since'+reference_date
            
            ds = ds.assign_coords(time=fc)
            ds.time.attrs = time_attrs
            #######################################################################################
            
            
            # netcdf? simply write     
            if out_forms[index] in ['nc']:            
                ds.to_netcdf(path=out_paths[index], mode='w', format='NetCDF4')
            # grib? 1. write a nc file 2. transform            
            else:
                temp_nc_path , ext=  os.path.splitext(out_paths[index])
                temp_nc_path=temp_nc_path+'.nc'

                ds.to_netcdf(path=temp_nc_path, mode='w', format='NetCDF4')
            
                cdo_command = "cdo -f grb2 copy "+temp_nc_path+" "+out_paths[index]
                cdo_out = subprocess.run(cdo_command, capture_output=True, shell=True)
                print(cdo_out.stdout.decode())

                try:
                    os.remove(temp_nc_path)
                except OSError as e:
                    raise
                # maybe delete folder ./nc/ 
 
    #endpathsloop  
#end
    
###################################################################################################
def write_athpy_state(filepath:str, state):  #, return_only_meta:bool=False):
    """ funcion description:
    
        within asv program used states in npy format are written to disk.
       
        returns: filepath 
    """               
    
    #Info: might be not a full state (means: area=earth).
    np.save( filepath, state )
    return filepath




###################################################################################################
def open_athpy_state(state_path):
    """ funcion description:
    
        open within asv program used states in npy format from disk.
        
        returns: that state.
    """               
    state = get_data(state_path)
    #state = open_full_state(state_path)
    return state



###################################################################################################
def is_increment(state):
    """ funcion description:
        'gptype' is the metadata name for increments or abolute states.
        
        returns: boolean
    """               
    
    if state['gptype'] in [201, 'inc']:      #increment
        return True
    elif state['gptype'] in [0, 'abs']:      # absstate
        return False
    else:
        raise ValueError('no valid gptype')


###################################################################################################
def get_meta(p_state:dict):
    """ funcion description:
        
        get only the metadata part of a state.
        
        returns: that metadata only state.
    """               
    
    input_meta={}

    for key in c.META_KEYS:
        try:
            input_meta[key] = p_state[key]
        except (KeyError):
            print('WARNING: Missing MetaKey')
            input_meta[key] = None
    return input_meta


###################################################################################################
def seperate_state(state:dict, only_meta=False):
    """ funcion description:
    
        seperate a state into surface, upperair and meta data.
        is needed for pangu input.
        
        returns: metadata, surface, upper-air
        
    """               
    

    input_meta=get_meta(state)
    input_sfc = np.zeros((4,721,1440))

    try:
        # 1. PMSL     # 2. U10M    # 3. V10M    # 4. T2M
        input_sfc[0,:,:] = state['PMSL'][:]
        input_sfc[1,:,:] = state['U_10M'][:]
        input_sfc[2,:,:] = state['V_10M'][:]
        input_sfc[3,:,:] = state['T_2M'][:]
    except (KeyError, IndexError):
        ds.close()
        print("Key- or Indexerror occurs. The given file doesn't contain all nessecary keys. "+
            "These are 'FI', 'QV'/'RELHUM', 'T', 'U', 'V', 'PSML', 'U_10M', 'V_10M', 'T_2M'. "+ 
            "Maybe the shape of the data is wrong.")    
        raise


    #variables , levels, lat, lon       # reverse: lat 
    input_p = np.zeros((5,13,721,1440))


    try:
        # Z / FI - geopotential
        input_p[0,:,:,:] = state['FI'][:]
        # QV - specific humidity
        input_p[1,:,:,:] = state['QV'][:]
        # T - temperature
        input_p[2,:,:,:] = state['T'][:]
        # U - U-wind
        input_p[3,:,:,:] = state['U'][:]
        # V - V-wind
        input_p[4,:,:,:] = state['V'][:]


    except (KeyError, IndexError):
        ds.close()
        print("Key- or Indexerror occurs. The given file doesn't contain all nessecary keys. "+
            "These are 'FI', 'QV'/'RELHUM', 'T', 'U', 'V', 'PSML', 'U_10M', 'V_10M', 'T_2M'. "+ 
            "Maybe the shape of the data is wrong.")    
        raise

    del state

    # important for pangu: single prec.
    input_sfc = input_sfc.astype(np.float32)
    input_p = input_p.astype(np.float32)

    return input_meta, input_sfc, input_p

###################################################################################################
def merge_state(state_sfc:np.ndarray, state_p:np.ndarray, meta:dict):
    """ funcion description:
    
        reverse of seperate_state
        
        returns: that state
    """               
    result={}


    for i in meta.keys():
        result[i]=meta[i]
    

    try:
        # 1. PMSL     # 2. U10M    # 3. V10M    # 4. T2M
        result['PMSL'] = state_sfc[0,:,:] 
        result['U_10M'] = state_sfc[1,:,:] 
        result['V_10M'] = state_sfc[2,:,:] 
        result['T_2M'] = state_sfc[3,:,:]
    except (KeyError, IndexError):
        print("Key- or Indexerror occur. Given surface state doesn't contain all nessecary keys.")
        raise

    try:
        # Z / FI - geopotential
        result['FI'] = state_p[0,:,:,:]
        # QV - specific humidity
        result['QV'] = state_p[1,:,:,:] 
        # T - temperature
        result['T'] = state_p[2,:,:,:] 
        # U - U-wind
        result['U'] = state_p[3,:,:,:] 
        # V - V-wind
        result['V'] = state_p[4,:,:,:]


    except (KeyError, IndexError):
        print("Key- or Indexerror occur. Given surface state doesn't contain all nessecary keys.")
        raise


    
    return result

###################################################################################################
def copy_state(state:dict):
    """ funcion description:
    
        returns: a deep copy of given state.
    """               
    
    # currently just a deepcopy of a dict.
    result = {}
    

    for key in state.keys():
        if key in c.META_KEYS:
            result[key] = state[key]
        else:
            try:
                result[key] = np.array(state[key][:])
            except Exception as e:
                print(key + 'did not work')
    return result

###################################################################################################
def delete_state(state:dict or str):
    """ funcion description:
    
        delete a state.
        
        returns: nothing.
    """               
    
    #TODO allow list?!

    if not is_state(state) and os.path.exists(state):
            os.remove(state)
    elif is_state(state):
        # what to do exactly?! delete referenced file?
        raise NotImplementedError('Please pass a path to a file.'+
                                ' Other functionability is not available, yet.')
            

###################################################################################################
def clean_dir(directory:str):
    """ funcion description:
    
        delete the files of a given dir, if existend.
        
        returns: nothing.
    """               
    
    if os.path.isdir(directory):
        
        shutil.rmtree(directory)
        os.makedirs(directory)

    else:
        print("Can not delete directory. Not a valid directory. Do nothing.")
        
###################################################################################################
def is_state(state):
    """ funcion description:
    
        is the given variable a state?
        
        returns: boolean
    """               
    
    # internpy, full, or None
    # first idea
    #TODO: Improvement.
    # print("Warning:'is_state' is not fully implemented.")
    if type(state) is dict:
        return True
    else:
        return False

###################################################################################################
def transform_to_energy(state:dict):
    """ funcion description:               
    
    Transformation of state into an 'energy space'.
    
    Note: A total energy norm does not exist form a strict mathematical point of view, because its 
        measurement will not be a norm, with the required norm settings. 
        Nevertheless a transformation into a room can be done in which 
        the normal euclidian norm is equivalent to np.sqrt(|| x ||_E), where ||x||_E is the
        "energy norm". This should be done to be mathematical consitend. Transforming the state 
        into the "energy space" will provide a Hilbert space (a normed space, where the norm is 
        induced by the scalar product of that space). The consitency of scalar product and norm is 
        importend since the arnoldi iteration relies an scalar products and norms. These should be 
        consistend.
        This transformation into that 'enery space' is done here. 
    
    returns: the state in energy space.
    
    ###############################################################################################
    
    ### Theory part for TE transform
    
    ##########################
    ## (dry) Total Energy Norm (TEN):
    ##
    ## ||z||_E := ẑ^T ẑ = 0.5 * ( \sum_{p_1}^p_n \sum_{A} 
    ##                              (u_{i,j}² + v_{i,j}² + c_p / T_r * t_{i,j}²)) /\s /\p_j           
    ##                                  +  (0.5 R_d  T_r  p_r \sum_{A} (log²(ps_{i})) /\s
    ##                      sum of pressurelevels and sum within area   u, v, t                 
    ##                      'meshsize' 'level_meshsize_j'+                 
    ##                      sum within area log(ps) 'meshsize'
    ##
    ##  c_p     =
    ##  T_r(ef) = 300
    ##  R_d     =
    ##  p_r     =
    ##  
    ##  /\s = meshsize (area)
    ##  /\p_j = level_meshsize of level j (distance zu j-1 / j+1)
    ##
    ##
    ##########################
    ##  Full Transformation
    ##
    ##  ẑ := (^u, ^v, ^t, ^ps)
    ##
    ##  ^u_{i,j} := sqrt(0.5 /\s /\p_j)           * u_{i,j}  
    ##  ^v_{i,j} := sqrt(0.5 /\s /\p_j)           * v_{i,j}
    ##  ^t_{i,j} := sqrt(0.5 /\s /\p_j c_p/ T_r ) * t_{i,j}
    ##  ^ps_{i}  := sqrt(0.5 /\s R_d T_r p_r )    * log(ps_{i})  
    ##
    ##
    ##########################
    ##  easier Transformation 
    ##  (wrong by a constant factor (problem is? larger for absolute messurements ?!) 
    ##              assures realtive messurements) and assumes /\p_i all equal)
    ##
    ##  ẑ := (^u, ^v, ^t, ^ps)
    ##
    ##  (transform 'to_energy')
    ##
    ##  ^u_{i,j} := sqrt(0.5 )                  * u_{i,j}  
    ##  ^v_{i,j} := sqrt(0.5 )                  * v_{i,j}
    ##  ^t_{i,j} := sqrt(0.5 * c_p / T_r )      * t_{i,j}
    ##  ^ps_{i}  := sqrt(0.5 * R_d * T_r * p_r )* log(ps_{i})  
    ##
    ##  (transform 'to_variables' / transform back)
    ##
    ##  u_{i,j}     := sqrt(2 )                     * ^u_{i,j}  
    ##  v_{i,j}     := sqrt(2 )                     * ^v_{i,j}
    ##  t_{i,j}     := sqrt(2 *  T_r / c_p )        * ^t_{i,j}
    ##  log(ps_{i}) := sqrt(2 / (R_d * T_r * p_r) ) *  ^ps_{i}) 
    ##
    ##  => <ẑ,ẑ> = (||ẑ||_2)² = ||z||_E  <=> ||ẑ||_2 = sqrt( ||z||_E )
    ###########################
    ##
    ## Note: Currently a reduced transformation without surface pressure is done.
    ##      norm = 'energy_red' = 'energy'
    ###############################################################################################
    """

    
    
    if c.NORM_VARIANT not in [ 'energy', 'energy_red']:
        return state

    if state['md:state_type'] == 'energy':
        return state

    if not (c.T in c.SV_VARS and c.U in c.SV_VARS and c.V in c.SV_VARS):
        raise ValueError("T,U,V need be the variables to perturb if 'energy'norm is used.")


    state_energy = copy_state(state)  

    #check if energy_keys are avail.     
    #state_keys = state.keys()

    # transform
    # change: T, U, V not  PS, yet

    ## 'f_'  means "factor".
    
    #sqrt(0.5 )
    f_uv = np.sqrt(0.5)
    
    #sqrt(0.5 * c_p / T_r )
    f_temp = np.sqrt(0.5* c.C_P/c.T_R)
    
    #sqrt(0.5 * R_d * T_r * p_r )
    #f_ps = np.sqrt(0.5* c.R_D * c.T_R * c.P_R)
    
    
    t_vars=[c.T]
    if c.T_2M in c.SV_VARS:
        t_vars.append(c.T_2M)
    

    uv_vars=[c.U,c.V]
    if c.U_10M in c.SV_VARS:
        uv_vars.append(c.U_10M)
    
    if c.V_10M in c.SV_VARS:
        uv_vars.append(c.V_10M)
    


    ##########
    ## u and v
    state_energy = smult(state_energy, f_uv, use_vars= uv_vars, copy=False) 
    ##########
    ## t
    state_energy = smult(state_energy,  f_temp, use_vars= t_vars, copy=False) 

    """  
    NotYetImplemented: the full Total Energy Measurement             
    #if norm_variant in ['energy']:
    ##########
    ## ps
        
        #state = 
        log(state, copy = False, use_variables = [asv.c.PS] )
        ##state=
        smult(state, '*',f_ps, copy = False, use_variables = [asv.c.PS] ) 
    
    
    #endif
    """
    state_energy['md:state_type']='energy'
    
    
    return state_energy
    
    




###################################################################################################
def transform_to_original(state:dict):
    """ funcion description:
        
        reverse of transform_to_energy
        read description there.
        
        returns: state in original space.
    """               


    if state['md:state_type'] == 'original':
        return state

    if not (c.T in c.SV_VARS and c.U in c.SV_VARS and c.V in c.SV_VARS):
        raise ValueError("T,U,V need be the variables to perturb if 'energy'norm is used.")


    state_org = copy_state(state)  
    #state_keys = state.keys()

    #sqrt(2 )
    f_uv = np.sqrt(2)
    #sqrt(2 * T_r / c_p )
    f_temp = np.sqrt(2* c.T_R/ c.C_P)
    #sqrt(2 / (R_d * T_r * p_r ))
    #f_ps = np.sqrt(2 / (c.R_D * c.T_R * c.P_R))
    
    t_vars=[c.T]
    if c.T_2M in c.SV_VARS:
        t_vars.append(c.T_2M)
    

    uv_vars=[c.U,c.V]
    if c.U_10M in c.SV_VARS:
        uv_vars.append(c.U_10M)
    
    if c.V_10M in c.SV_VARS:
        uv_vars.append(c.V_10M)

    ##########
    ## u and v 
    state_org = smult(state_org, f_uv, use_vars=uv_vars, copy=False)                             
    ##########
    ## t
    state_org = smult(state_org,  f_temp, use_vars=t_vars, copy=False) 

   
    """
    NotYetImplemented: the full Total Energy Measurement             
    if norm_variant in ['energy'] ..:
    ##########
    ## ps
        ##state=
        state_org[c.PS] = smult(state_org[c.PS],f_ps ) 
                                                #, copy = False, use_variables = [c.PS] ) 
        #state = 
        state_org[c.PS] = exp(state_org[c.PS])
                                        #, 'exp', None, copy = False, use_variables = [c.PS] )
    #endif
    """

    state_org['md:state_type']='original'

    return state_org

###################################################################################################
def _get_new_gptype(a:dict,b:dict, operator:str=''):
    """ funcion description:
    
        if doing some computations the gptype changes. 
        
        returns: new gptype of two states and an math operation
    """               
    
    if operator not in ['+','-']:
        raise ValueError("unvalid operator")

    if not (is_state(b) and is_state(a)):
        raise ValueError("Two states need to be passed")
    

    if operator in ['-']:
        if a['gptype'] in [0,'abs'] and b['gptype'] in [0,'abs']:  
            return 201
    if operator in ['-','+']:
        if a['gptype'] in [0,'abs'] and b['gptype'] in [201,'inc']: 
            return 0
        if a['gptype'] in [201,'inc'] and b['gptype'] in [201,'inc']:
            return 201
    if operator in ['+']:
        if a['gptype'] in [201,'inc'] and b['gptype'] in [0,'abs']:  
            return 0
        if a['gptype'] in [201,'inc'] and b['gptype'] in [201,'inc']:  
            return 201
        if a['gptype'] in [201,'inc'] and b['gptype'] in [0,'abs']:  
            return 0
    
    
    raise ValueError("something went wrong with gptype. given data: a['gptype'] =" +
                        str(a['gptype'])+ " b['gptype']="+str(b['gptype']) + 
                        " operator:"+str(operator))
     


###################################################################################################
def add(state_a:dict,state_b:dict,copy:bool=True, use_vars=c.SV_VARS):
    """ funcion description:
    
        add two states.
        
        returns: the result thereof
    """               
    
    is_computable(state_a,state_b,'+')

    if copy:
        result = copy_state(state_a)
    else:
        result = state_a

    for var in use_vars:
        if not isinstance(result[var], np.ndarray):
            raise TypeError("values of dict need to be of np.ndarray type."+ 
                        "Var: "+str(var) + " Type: "+type(result[var]) )
        if not isinstance(state_b[var], np.ndarray):
            raise TypeError("values of dict need to be of np.ndarray type."+ 
                        "Var: "+str(var) + " Type: "+type(state_b[var]) )


        result[var] = result[var] + state_b[var]

    result['gptype'] = _get_new_gptype(state_a,state_b,'+')

    return result


###################################################################################################
def sub(state_a:dict,state_b:dict,copy=True, use_vars=c.SV_VARS):
    """ funcion description:
    
        substract two states.
        
        returns: the result thereof
    """                              
    
    is_computable(state_a,state_b,'-')

    if copy:
        result = copy_state(state_a)
    else:
        result = state_a

    for var in use_vars:
        if not isinstance(result[var], np.ndarray):
            raise TypeError("values of dict need to be of np.ndarray type."+ 
                        "Var: "+str(var) + " Type: "+type(result[var]) )
        if not isinstance(state_b[var], np.ndarray):
            raise TypeError("values of dict need to be of np.ndarray type."+ 
                        "Var: "+str(var) + " Type: "+type(state_b[var]) )


        result[var] = result[var] - state_b[var]

    result['gptype'] = _get_new_gptype(state_a,state_b,'-')

    return result

###################################################################################################
def smult(state,scalar:int or float,copy=True, use_vars=c.SV_VARS):
    """ funcion description:
    
        compute a scalar multiplication a state and a scalar.
        
        returns: the result thereof
    """                 
    

    
    if not is_state(state) and os.path.exists(state):
        result = open_athpy_state(state)
    elif is_state(state):
        if copy:
            result = copy_state(state)
        else:
            result=state

    is_computable(result,scalar,'*')

    for var in use_vars:
        if not isinstance(result[var], np.ndarray):
            raise TypeError("values of dict need to be of np.ndarray type."+ 
                        "Var: "+str(var) + " Type: "+type(result[var]) )


        result[var] =scalar*result[var]
    return result


###################################################################################################
def dot_prod(state_a,state_b, use_vars=c.SV_VARS, use_levels=c.SV_LEVELS):
    """ funcion description:
    
        compute a dot product of two states.
        
        returns: the result thereof
    """               
    

    if not is_state(state_a) and os.path.exists(state_a):
        state_a = open_athpy_state(state_a)
    
    if not is_state(state_b) and os.path.exists(state_b):
        state_b = open_athpy_state(state_b)


    is_computable(state_a,state_b,'.*')

    
    dot=0
    for var in use_vars:
        #print(type(state_a[var]))
        if not isinstance(state_a[var], np.ndarray):
            raise TypeError("values of dict need to be of np.ndarray type."+ 
                        "Var: "+str(var) + " Type: "+type(state_a[var]) )
        if not isinstance(state_b[var], np.ndarray):
            raise TypeError("values of dict need to be of np.ndarray type."+ 
                        "Var: "+str(var) + " Type: "+type(state_b[var]) )

        
        if var in c.SFCVARS:

            curr_dot = np.dot(state_a[var].reshape(-1) , state_b[var].reshape(-1) )                
            dot = dot + curr_dot

        elif var in c.PVARS:        #TODO: beware at full levelimplementation-curr:only all levels.
            for l in range(0,len(state_a[var][use_levels,0,0])):                       
                                
                curr_dot = np.dot(state_a[var][l,:,:].reshape(-1) , 
                                            state_b[var][l,:,:].reshape(-1) )                
                                
                dot = dot +curr_dot

    return dot


################################################################################################### 
def is_computable(a,b,operator:str=''):
    """ funcion description:
    
        is a computation of two states semantically valid? Because not all operations make sense.
        
        returns: boolean True
    """               
    
    #TODO: merge with _get_new_gptype ?
    
    if operator not in ['+','-','*','.*']:
        raise ValueError("unvalid operator")

    if not is_state(a):
        return False
    if operator in ['+','-','.*']:
        if not (is_state(b) and is_state(a)):
            raise ValueError("Two states need to be passed for add and sub.")
        if b['md:state_type'] != a['md:state_type']:
            raise ValueError("Both states need to be of same type (original or energy). a:"+
                        str(a['md:state_type'])+' b: '+ str(b['md:state_type'])  )
    

        if operator in ['+']:
            if a['gptype'] in [0,'abs'] and b['gptype'] in [0,'abs']:  
                raise TypeError("Two abs states can not be added. a: "+
                                str(a['gptype'])+ " b:"+ str(b['gptype']) )
        if operator in ['-']:
            #if a['gptype'] in [201,'inc'] and b['gptype'] in [0,'abs']: 
            #    raise TypeError("Absolute state can not be substracted from incremental state.") 
            # rule too strict. should be logical.            
            pass        

        if operator in ['.*']:
            if a['gptype'] in [0,'abs'] or b['gptype'] in [0,'abs']:  
                raise TypeError("dot prod will be computed only if both states are incremental.")

    elif operator in ['*']:
        if not is_state(a):
            raise ValueError("First argument needs to be a state for scalar mult.")
        if not (isinstance(b,int) or isinstance(b,float)):
            raise ValueError("second argument needs to be a number for scalar mult.")
        #if a['gptype'] not in [201,'inc']:
        #    raise TypeError("absolute states can not be multipled.")    # do not mult an abs state
        # NOTE: needs to be possible for transform_to_*()
    
    return True
        

###################################################################################################
def norm(state:dict, use_vars=c.SV_VARS, use_levels=c.SV_LEVELS):
    """ funcion description:
    
        computes the norm of a state
        
        return: norm in double
    """               
    

            
    if not is_state(state) and os.path.exists(state):
        state = open_athpy_state(state)

    #TODO use levels ...
    norm=0
    for var in use_vars:
        if not isinstance(state[var], np.ndarray):
            raise TypeError("values of dict need to be of np.ndarray type."+ 
                        "Var: "+str(var) + " Type: "+type(state[var]) )

        if var in c.SFCVARS:

            curr_norm = np.dot( state[var].reshape(-1) , state[var].reshape(-1) )
            norm = norm + curr_norm


        elif var in c.PVARS:
            for l in range(0,len(state[var][use_levels,0,0])):    
                    #TODO: levelimplementation- curr: only all levels.

                curr_norm = np.dot(state[var][l,:,:].reshape(-1) , 
                                                        state[var][l,:,:].reshape(-1) ) 
                norm = norm+curr_norm
                if math.isnan(curr_norm):
                    raise ValueError("One field of data is Nan (or at least a part of it)."+
                            "Something with the data (or data processing) must be wrong.")
                #print(norm)                 
    #endfor    
   
    norm = np.sqrt(norm)
    #print(norm)
    return norm


################################################################################################### 
def lk_of_states(Q,vek):
    """ funcion description:
        
        compute a linear combination a list of states
        
        returns: that linear combined state.
    """               
    


    if len(Q) != len(vek):
        raise TypeError("unsuitable length of states and vektor for linear combination.")
    
    for i in range(0,len(Q)):
        if not is_state(Q[i]) and os.path.exists(Q[i]):
            Q[i] = open_athpy_state(Q[i])

        if i==0:
            result = smult(Q[0],vek[0], copy=True)        
        else:
            result = add( result ,smult(Q[i],vek[i], copy=True), copy=False) 

    return result

###################################################################################################    
def mm_mult(Q, MAT ):
    """ function description:
        
        computes a list of / several linear combinations of a list of states.
        this is at the end of a day a matrix matrix multiplication
        
        returns: the result thereof, which is a list of states.

    """
  
    if len(Q) != len(MAT[:,0]):
        raise TypeError("unsuitable length of states and matrix-col for linear combination.")
    
    R=[]

    #loop over Qs
    for i in range(0,len(Q)):
        if not is_state(Q[i]) and os.path.exists(Q[i]):
            Q[i] = open_athpy_state(Q[i])
        if i==0:
            for c in range(0,len(MAT[0,:])):
                R.append(smult(Q[0],MAT[0,c], copy=True))
        else:
            for c in range(0,len(MAT[0,:])):
                R[c] = add(R[c], smult( Q[i], MAT[i,c], copy=True),copy=False)
  
    return R

###################################################################################################
def mix_states(pert_list ):
    



    if not isinstance(pert_list,list):
        raise TypeError("Wrong type of pert_inc_list. Has to be list.")

    n_perts = len(pert_list)

    
    rand_koeff = np.random.rand( n_perts, n_perts)
    rand_koeff += 0.2*np.ones(( n_perts, n_perts))    



    mixed_pert_list = mm_mult(pert_list, rand_koeff)

    return mixed_pert_list

