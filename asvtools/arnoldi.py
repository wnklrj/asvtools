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
import os
from asvtools import c
import asvtools as asv



__all__ = ["arnoldi"]

###################################################################################################
def arnoldi(amplitude:int=-1):
    """ function description:
        main asv function, hence the arnoldi alg. for it.
        
        computes the SV and creates some states which are perturbed be these and ready for 
        forecasting.
        
        return: directly nothing, but the above described data is generated.

    ###############################################################################################
    
        the basis for the following variant is the Block Arnoldi Algorithm in Ruhes Variant    
    
        ALGORITHM -  Block Arnoldi – Ruhe’s variant

                  p = c.BLOCK_SIZE
                  m = c.LOOPS

        1.       Choose p initial orthonormal vectors {q_i}  i=1,...,p.
        2.       For j = p, p + 1, . . . , m * p :
        3.           Set k := j − p + 1;
        4.           Compute w := Aq_k ;
        5.           For i = 1, 2, . . . , j:
        6.               h_{i,k} := dot(w, q_i )
        7.               w := w − h_{i,k} q_i
        8.           EndFor
        9.           Compute h_{j+1,k} := ||w||_2 and q_{j+1} := w/h_{j+1,k} 
        10.      EndFor
    
        see: ALG 6.24 in Saad, Y. (1996) Iterative Methods for Sparse Linear Systems. Boston, MA:
              PWS, A. RUHE , Implementation aspects of band Lanczos algorithms for computation of
                   eigenvalues of large sparse symmetric matrices, Mathematics of Computation, 33
                   (1979), pp. 680–687.
    
    
        Ruhe's Arnoldi Variant will be combined with SV Algorithm (denoted as BAP there) from 
        'Krylov Methods for adoint-free perturbations in Dynamical Systems" 
        [Winkler et.al, QJRMS, 2021]
    
        and see 'Arnoldi Singular Vector perturbations for machine learning weather prediction' 
          article as refered in the given asvtools README.
    
    ############################################################################################### 
    """
	

    print('Start Arnoldi routine') 
   


    print(asv.get_filepath('ref_full_begin_npy'))

    x_ref_org = asv.get_data(asv.get_filepath('ref_full_begin_npy'))
    x_ref_energy = asv.transform_to_energy(x_ref_org)
    
    ml = c.LOOPS * c.BLOCK_SIZE
   
    
    # generate initial states for arnoldi iteration.
    # in energyspace 
    if amplitude == -1:
        Q, ampli_normal, ampli_energy = asv.create_initial_vectors(x_ref_org, get_amplitude=True)

        if c.NORM_VARIANT in ['energy','energy_red']:    
            amplitude = ampli_energy
        elif c.NORM_VARIANT in ['original']:
            amplitude = ampli_normal
    else:
        Q = asv.create_initial_vectors(x_ref_org, get_amplitude=False)
        ampli_energy = ampli_normal = amplitude

   

    print('length Q: '+ str(len(Q)))


    Qfn=[]
   
    for i in range(0,len(Q)):
        Qfn.append(asv.get_filepath('Q',ens_mem=i))


    

    ###############################################################################################


    H =  np.zeros(shape=(ml , ml))

    #print(H)
    W = []
  
    j= c.BLOCK_SIZE -1

    #TODO: optimize speed / storage
    while (j <= ml-1 ):     
        

        K = np.arange(j - c.BLOCK_SIZE +1, j +1  )        
      
        W = asv.write_run(                                                                    
                Qfn[K[0]:K[-1]+1] , amplitude)        

        

        for ki in range(0, c.BLOCK_SIZE): 
            for i in range(0,j+1):
                if (i <= ml -1):

                    H[i,K[ki]] = asv.dot_prod(W[ki],Qfn[i])

                    W[ki] = asv.sub(W[ki], 
                                        asv.smult(Qfn[i],H[i,K[ki]],  copy=True),copy=False )                     
            #endfor
            
            if (j <= ml - 2):
                H[j+1,K[ki]] = asv.norm(W[ki])   # "np.norm(W[K[ki]])"        
                ####################
                ## compute and write Q[j+1]              

                Qfn.append({})
                Qfn[j+1] = asv.get_filepath('Q',ens_mem=j+1)                

                #Q[j+1]= 
                Qj1 = asv.smult(W[ki], (1 / H[j+1,K[ki]])) #copy
                Q.append({})
                Q[j+1]=asv.write_athpy_state(Qfn[j+1], Qj1 )

                
                ###################

                ##print('end block')                
                #print(H.round(decimals=1))
                #print('')
            j += 1
            #important j+=1 at ki loop.
        #endfor ki     
    #endwhile
    ###############################################################################################
    ########## end arnoldi loop







    print('H Matrix computed')
    #print(H.round(decimals=2))
    #print('H Matrix / divided by amplitude')
    H = H/amplitude
    print('H Matrix divided by amplitude')
    #print(H.round(decimals=2))
    print('for check: max entry of H-Matrix')
    print(np.max(H))
    
    U,S,V = np.linalg.svd(H, full_matrices=True) 
    print(V)
    #print(V[:,0:int(c.SV_CUTOF)])

    del W

    
                
    P2=[]

    print('Approx. Singular Vectors:')    
    print(S)

    cutof=int(min(ml,c.SV_CUTOF))
    

    SV = asv.mm_mult( Q, V[:,0:cutof] )
    # In order to get an undistinguishable ensemble, one may mix the generated singular vectors.
    #if c.MIX_STATES:    
    #   SV = asv.mix_states(SV)
    
    print('NR of Q: ' + str(len(Q)))

    print('NR of SV: '+ str(len(SV)))

    # del Q states
    #asv.delete_state(Q)    
    ## 
    
    # write SV to file
    #SV_paths=asv.get_filepath('singular_vectors', ens_mem=[*range(0,len(SV))])
    #print(SV_paths)
    #print(type(SV[0]))
    # maybe not properly working:
    #for i in range(0,len(SV)):
    #        asv.write_athpy_state(SV_paths[i],SV[i])
    
    
    ###############################################################################################
    # create and write SV perturbed full states.                                  
    asv.fill_dir_with_pert_states(SV,ampli_normal, state_out_filename= c.SV_PERT_FILENAME, 
            dest_dir=c.SV_OUTPUT_DIR, out_forms=c.OUT_FORMATS)

    with open(asv.get_filepath('singular_values_file'), 'w') as f:
        f.write('\n')                        
        f.write(str(S))

    with open(asv.get_filepath('H_file'), 'w') as f:
        f.write('H-Matrix (Note: It is divided by choosen amplitude.)\n\n') 
                       
        f.write("BLOCK_SIZE: " +str(c.BLOCK_SIZE) + "    LOOPS: " + str(c.LOOPS) + "    "+ 
                "AREA: " +str(c.AREA) + "    T_OPT: " + str(c.T_OPT) + "    ")
        f.write('\n')
        for index in range(0,len(H[:,0])):
            out_string_line = ''.join('%5.6f  ;  ' % i for i in H[index,:] )
            f.write(out_string_line)            
            f.write('\n')        

        
    np.save(os.path.join(c.SV_OUTPUT_DIR,'H.npy') , H)
    #np.save( ?, state )



    print('end')
###################################################################################################


