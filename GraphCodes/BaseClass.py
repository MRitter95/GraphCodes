'''
        ###########
        #automated construction, saving, loading
        ##########
        populate
        save
        load

        #######
        #resonator lattice get /view functions
        #######
        get_xs DONE
        get_ys DONE
        draw_resonator_lattice DONE
        draw_resonator_end_points DONE
        get_all_resonators DONE
        get_coords DONE   need to add roundDepth property in euclideanlayout

        ########
        #functions to generate effective JC-Hubbard lattice (semiduals)
        ######## 
        generate_semiduals SKIP
        _azimuthal_links_full_general SKIP
        _radial_links_full_general SKIP
        _radial_links_non_triangle (defunct) SKIP
        
        #######
        #get and view functions for the JC-Hubbard (semi-dual lattice)
        #######
        draw_SDlinks
        get_semidual_points (semi-defunct)
        get_all_semidual_points

        ######
        #Hamiltonian related methods
        ######
        generate_Hamiltonian DONE
        get_sub_Hamiltonian
        get_eigs DONE

        ##########
        #methods for calculating/looking at states and interactions
        #########
        get_SDindex (removed for now. Needs to be reimplemented in sensible fashion)
        build_local_state DONE
        V_int DONE
        V_int_map DONE
        plot_layout_state DONE
        plot_map_state DONE
        get_end_state_plot_points DONE
        plot_end_layout_state DONE

        ##########
        #methods for calculating things about the root graph
        #########
        generate_root_Hamiltonian DONE
        plot_root_state DONE
'''
import re
import numpy
import scipy
import pylab
import pickle
from scipy.sparse import coo_matrix

class BaseLayout(object):
    def __init__(self):
        pass
    
    def load(self, file_path):
        '''
        laod structure from pickle file
        '''
        pickledict = pickle.load(open(file_path, "rb" ) )
        
        for key in list(pickledict.keys()):
            setattr(self, key, pickledict[key])
           
        #handle the case of old picle files that do not have a mode type property  
        #they are all calculated for the full wave
        if not 'modeType' in list(self.__dict__.keys()):
            print('Old pickle file. Pre FW-HW.')
            self.modeType = 'FW'
        return

    #######
    #resonator lattice get /view functions
    #######
    def get_xs(self):
        '''
        return x coordinates of all the resonator end points
        '''
        return self.coords[:,0]
    
    def get_ys(self):
        '''
        return y coordinates of all the resonator end points
        '''
        return self.coords[:,1]

    def get_all_resonators(self, maxItter = -1):
        '''
        function to get all resonators as a pair of end points
        
        each resontator returned as a row with four entries.
        (orientation is important to TB calculations)
        x0,y0,x1,y1
        
        '''
        return self.resonators
    
    def get_coords(self, resonators, roundDepth = 3):
        '''
        take in a set of resonators and calculate the set of end points.
        
        Will round all coordinates the the specified number of decimals.
        
        Should remove all redundancies.
        '''
        
        coords_overcomplete = numpy.zeros((resonators.shape[0]*2, 1)).astype('complex')
        coords_overcomplete =  numpy.concatenate((resonators[:,0], resonators[:,2])) + 1j * numpy.concatenate((resonators[:,1], resonators[:,3]))
        
        coords_complex = numpy.unique(numpy.round(coords_overcomplete, roundDepth))
    
        coords = numpy.zeros((coords_complex.shape[0],2))
        coords[:,0] = numpy.real(coords_complex)
        coords[:,1] = numpy.imag(coords_complex)
        
        return coords

    def draw_resonator_lattice(self, ax, color = 'g', alpha = 1 , linewidth = 0.5, extras = False, zorder = 1):
        if extras == True:
            resonators = self.extraResonators
        else:
            resonators = self.resonators
            
        for res in range(0,resonators.shape[0] ):
            [x0, y0, x1, y1]  = resonators[res,:]
            ax.plot([x0, x1],[y0, y1] , color = color, alpha = alpha, linewidth = linewidth, zorder = zorder)
        return
            
    def draw_resonator_end_points(self, ax, color = 'g', edgecolor = 'k',  marker = 'o' , size = 10, zorder = 1):
        '''will double draw some points'''
        x0s = self.resonators[:,0]
        y0s = self.resonators[:,1]
        
        x1s = self.resonators[:,2]
        y1s = self.resonators[:,3]
        
        pylab.sca(ax)
        pylab.scatter(x0s, y0s ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder)
        pylab.scatter(x1s, y1s ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder)
        return  
    
    def get_semidual_points(self):
        '''
        get all the semidual points in a given itteration.
        
        Mostly vestigial for compatibility
        '''
        return[self.SDx, self.SDy]

    def draw_SD_points(self, ax, color = 'g', edgecolor = 'k',  marker = 'o' , size = 10,  extra = False, zorder = 1):
        '''
        draw the locations of all the semidual sites
        
        if extra is True it will draw only the edge sites required to fix the edge of the tiling
        '''
        if extra == True:
            xs = self.extraSDx
            ys = self.extraSDy
        else:
            xs = self.SDx
            ys = self.SDy
        
        pylab.sca(ax)
        pylab.scatter(xs, ys ,c =  color, s = size, marker = marker, edgecolors = edgecolor, zorder = zorder)
        
        return
    ######
    #Hamiltonian related methods
    ######
    def generate_Hamiltonian(self, t = 1, internalBond = 1000):
        '''
        create the effective tight-binding Hamiltonian
        
        Also calculated as stores eigenvectors and eigenvalues for that H
        
        
        Will use FW or HW TB coefficients depending on self.modeType
        '''
        

        self.t = t
        self.internalBond = 1000*self.t
        
        totalSize = len(self.SDx)
            
        self.H = numpy.zeros((totalSize, totalSize))
        self.H_HW = numpy.zeros((totalSize*2, totalSize*2)) #vestigial
        
        #loop over the links and fill the Hamiltonian
        for link in range(0, self.SDlinks.shape[0]):
            [sourceInd, targetInd] = self.SDlinks[link, :]
            [sourceEnd, targetEnd] = self.SDHWlinks[link, 2:]
            source = int(sourceInd)
            target = int(targetInd)
            sourceEnd = int(sourceEnd)
            targetEnd = int(targetEnd)
            
            
            if self.modeType == 'FW':
                self.H[source, target] = self.t
            elif self.modeType == 'HW':
                polarity = sourceEnd^targetEnd #xor of the two ends. Will be one when the two ends are different
                signum =(-1.)**(polarity) #will be zero when  two ends are same, and minus 1 otherwise
                self.H[source, target] = self.t * signum
            else:
                raise ValueError('You screwed around with the mode type. It must be FW or HW.')
            self.H_HW[2*source + sourceEnd, 2*target+targetEnd] = 2*self.t
                
        #fix the bonds between the two ends of the same site
        for site in range(0, totalSize):
            self.H_HW[2*site, 2*site+1] = self.internalBond
            self.H_HW[2*site+1, 2*site] = self.internalBond
                
        self.Es, self.Psis = scipy.linalg.eigh(self.H)
        self.Eorder = numpy.argsort(self.Es)
        
        return
    
    def get_eigs(self):
        '''
        returns eigenvectors and eigenvalues
        '''
        return [self.Es, self.Psis, self.Eorder]
    
    #def get_SDindex(self,num, itt, az = True):
    #    '''
    #    get the index location of a semidual point. 
    #    
    #    Point spcified by
    #    something TBD
    #    
    #    (useful for making localized states at specific sites)
    #    '''
    #    
    #    return currInd

    def build_local_state(self, site):
        '''
        build a single site state at any location on the lattice.
        
        site is the absolute index coordinate of the lattice site
        (use get_SDindex to obtain this in a halfway sensible fashion)
        '''
        if site >= len(self.SDx):
            raise ValueError('lattice doesnt have this many sites')
            
        state = numpy.zeros(len(self.SDx))*(0+0j)
        
        state[site] = 1.
        
        return state
    
    def V_int(self, ind1, ind2, states):
        '''
        Calculate total interaction enegery of two particles at lattice sites
        indexed by index 1 and index 2
        
        states is the set of eigenvectors that you want to include e.g. [0,1,2,3]
        '''
        psis_1 = self.Psis[ind1,states]
        psis_2 = self.Psis[ind2,states]
        
        return numpy.dot(numpy.conj(psis_2), psis_1)

    def V_int_map(self, source_ind, states = []):
        '''
        calculate a map of the interaction energy for a given location of the first
        qubit.
        Lattice sites specified by index in semidual points array
        
        must also specify which igenstates to include. default = all
        '''
        if states == []:
            states = numpy.arange(0, len(self.Es),1)
        
        int_vals = numpy.zeros(len(self.SDx))
        for ind2 in range(0, len(self.SDx)):
            int_vals[ind2] = self.V_int(source_ind, ind2,states)
        
        return int_vals
    
    def plot_layout_state(self, state_vect, ax, title = 'state weight', colorbar = False, plot_links = False, cmap = 'Wistia', zorder = 1):
        '''
        plot a state (wavefunction) on the graph of semidual points
        '''
        Amps = state_vect
        Probs = numpy.abs(Amps)**2
        mSizes = Probs * len(Probs)*30
        mColors = numpy.angle(Amps)/numpy.pi
        
        cm = pylab.cm.get_cmap(cmap)
        
        pylab.sca(ax)
        pylab.scatter(self.SDx, self.SDy,c =  mColors, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -0.5, vmax = 1.5, zorder = zorder)
        if colorbar:
            cbar = pylab.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('phase (pi radians)', rotation=270)
              
        if plot_links:
            self.draw_SDlinks(ax, linewidth = 0.5, color = 'firebrick', zorder = zorder)
        
        pylab.title(title, fontsize=8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_aspect('equal')
        return

    def plot_map_state(self, map_vect, ax, title = 'ineraction weight', colorbar = False, plot_links = False, cmap = 'winter', autoscale = False, scaleFactor = 0.5, zorder = 1):
        '''plot an interaction map on the graph
        '''
        Amps = map_vect
        
        mSizes = 100
        mColors = Amps
        
        #cm = pylab.cm.get_cmap('seismic')
        cm = pylab.cm.get_cmap(cmap)
        #cm = pylab.cm.get_cmap('RdBu')
        
        
        vals = numpy.sort(mColors)
        peak = vals[-1]
        second_biggest = vals[-2]
        
        if autoscale:
            vmax = peak
            if self.modeType == 'HW':
                vmin = -vmax
            else:
                vmin = vals[0]
        else:
            vmax = second_biggest
            vmin = -second_biggest
        
        if self.modeType == 'FW':
            pylab.sca(ax)
            pylab.scatter(self.SDx, self.SDy,c =  mColors, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmax = vmax, vmin = vmin, zorder = zorder)
        elif self.modeType == 'HW':
            #build full state with value on both ends of the resonators
            mColors_end = numpy.zeros(len(Amps)*2)
            mColors_end[0::2] = mColors
            mColors_end[1::2] = -mColors
            
            #get coordinates for the two ends of the resonator
            plotPoints = self.get_end_state_plot_points(scaleFactor = scaleFactor)
            xs = plotPoints[:,0]
            ys = plotPoints[:,1]
            
            #mColors_end = numpy.arange(1.,len(Amps)*2+1,1)/300
            #print mColors_end.shape
            #print Amps.shape
            
            #plot
            pylab.sca(ax)
            pylab.scatter(xs, ys,c =  mColors_end, s = mSizes/1.4, marker = 'o', edgecolors = 'k', cmap = cm, vmax = vmax, vmin = vmin, zorder = zorder)
        else:
            raise ValueError('You screwed around with the mode type. It must be FW or HW.')
            
        
        if colorbar:
            cbar = pylab.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('interaction energy (AU)', rotation=270)
              
        if plot_links:
            self.draw_SDlinks(ax, linewidth = 0.5, color = 'firebrick', zorder = zorder)
        
        pylab.title(title, fontsize=8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_aspect('equal')
        return
    
    def get_end_state_plot_points(self,scaleFactor = 0.5):
        '''
        find end coordinate locations part way along each resonator so that
        they can be used to plot the field at both ends of the resonator.
        (Will retun all values up to specified itteration. Default is the whole thing)
        
        Scale factor says how far appart the two points will be: +- sclaeFactor.2 of the total length
        
        returns the polt points as collumn matrix
        '''
        if scaleFactor> 1:
            raise ValueError('scale factor too big')
            
            
        size = len(self.SDx)
        plot_points = numpy.zeros((size*2, 2))
        
        resonators = self.get_all_resonators()
        for ind in range(0, size):
            [x0, y0, x1, y1]  = resonators[ind, :]
            xmean = (x0+x1)/2
            ymean = (y0+y1)/2
            
            xdiff = x1-x0
            ydiff = y1-y0
            
            px0 = xmean - xdiff*scaleFactor/2
            py0 = ymean - ydiff*scaleFactor/2
            
            px1 = xmean + xdiff*scaleFactor/2
            py1 = ymean + ydiff*scaleFactor/2
            
            
            plot_points[2*ind,:] = [px0,py0]
            plot_points[2*ind+1,:] = [px1,py1]
            ind = ind+1
            
        return plot_points
    
    def plot_end_layout_state(self, state_vect, ax, title = 'state weight', colorbar = False, plot_links = False, cmap = 'Wistia', scaleFactor = 0.5, zorder = 1):
        '''
        plot a state (wavefunction) on the graph of semidual points, but with a 
        value plotted for each end of the resonator
        
        If you just want a single value for the resonator use plot_layout_state
        
        Takes states defined on only one end of each resonator. Will autogenerate 
        the value on other end based on mode type.
        
        '''
        Amps = state_vect
        Probs = numpy.abs(Amps)**2
        mSizes = Probs * len(Probs)*30
        mColors = numpy.angle(Amps)/numpy.pi
        
        #build full state with value on both ends of the resonators
        mSizes_end = numpy.zeros(len(Amps)*2)
        mSizes_end[0::2] = mSizes
        mSizes_end[1::2] = mSizes
        
        mColors_end = numpy.zeros(len(Amps)*2)
        mColors_end[0::2] = mColors
        if self.modeType == 'FW':
            mColors_end[1::2] = mColors
        elif self.modeType == 'HW':
            #put opposite phase on other side
            oppositeCols = mColors + 1
            #rectify the phases back to between -0.5 and 1.5 pi radians
            overflow = numpy.where(oppositeCols > 1.5)[0]
            newCols = oppositeCols
            newCols[overflow] = oppositeCols[overflow] - 2
            
            mColors_end[1::2] = newCols
        else:
            raise ValueError('You screwed around with the mode type. It must be FW or HW.')
        
        cm = pylab.cm.get_cmap(cmap)
        
        #get coordinates for the two ends of the resonator
        plotPoints = self.get_end_state_plot_points(scaleFactor = scaleFactor)
        xs = plotPoints[:,0]
        ys = plotPoints[:,1]
        
        pylab.sca(ax)
        pylab.scatter(xs, ys,c =  mColors_end, s = mSizes_end, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -0.5, vmax = 1.5, zorder = zorder)
        if colorbar:
            cbar = pylab.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('phase (pi radians)', rotation=270)
              
        if plot_links:
            self.draw_SDlinks(ax,linewidth = 0.5, color = 'firebrick', zorder = zorder)
        
        pylab.title(title, fontsize=8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_aspect('equal')
        
        #return plotPoints
        return mColors
    
    def generate_root_Hamiltonian(self, roundDepth = 3, t = 1, verbose = False, sparse = False, flags = 5):
        '''
        custom function so I can get vertex dict without having to run the full populate of general layout
        and thereby having to also diagonalize the effective Hamiltonian.
        
        Will process the resonator matrix to get the layout Hamiltonian.
        
        Will return a regular matrix of sparse  = false, and a sparse matrix data type if sparse  = true
        
        Does not need to SD Hamiltonian made first.
        
        
        '''
        resonators = self.get_all_resonators()
        resonators = numpy.round(resonators, roundDepth)
        
        numVerts = self.coords.shape[0]
        if sparse:
            rowVec = numpy.zeros(numVerts*4+flags)
            colVec = numpy.zeros(numVerts*4+flags)
            Hvec = numpy.zeros(numVerts*4+flags)
        else:
            Hmat = numpy.zeros((numVerts, numVerts))
        
        coords_complex = numpy.round(self.coords[:,0] + 1j*self.coords[:,1], roundDepth)
        
        currInd = 0
        for rind in range(0, resonators.shape[0]):
            resPos = resonators[rind,:]
            startPos = numpy.round(resPos[0],roundDepth)+ 1j*numpy.round(resPos[1],roundDepth)
            stopPos = numpy.round(resPos[2],roundDepth)+ 1j*numpy.round(resPos[3],roundDepth)
            
            startInd = numpy.where(startPos == coords_complex)[0][0]
            stopInd = numpy.where(stopPos == coords_complex)[0][0]
    
            if sparse:
                rowVec[currInd] = startInd
                colVec[currInd] = stopInd
                Hvec[currInd] = t #will end up adding t towhatever this entry was before.
                currInd = currInd +1
                
                rowVec[currInd] = stopInd
                colVec[currInd] = startInd
                Hvec[currInd] = t #will end up adding t towhatever this entry was before.
                currInd = currInd +1
                
            else:
                Hmat[startInd, stopInd] = Hmat[startInd, stopInd] + t
                Hmat[stopInd, startInd] = Hmat[stopInd, startInd] + t
        
        #finish making the sparse matrix if we are in sparse matrix mode.
        if sparse:
            #pad the end of the matrix with values so that I can see if one of those is the missing one
            for ind in range(0, flags):
                rowVec[currInd] = numVerts+ind
                colVec[currInd] = numVerts+ind
                Hvec[currInd] = -7.5 #will end up adding t towhatever this entry was before.
                currInd = currInd +1
    
            Hmat = coo_matrix((Hvec,(rowVec,colVec)), shape = (numVerts+flags,numVerts+flags), dtype = 'd')
            Hmat.eliminate_zeros() #removed the unused spots since this is not a regular graph
            
        if verbose:
            temp = numpy.sum(Hmat)/numVerts
            print('average degree = ' + str(temp))
        
        self.rootHamiltonian = Hmat
        
        if not sparse:
            self.rootEs, self.rootPsis = numpy.linalg.eigh(self.rootHamiltonian)
            
        return
    
    def plot_root_state(self, state_vect, ax, title = 'state weight', colorbar = False, plot_links = False, cmap = 'Wistia', zorder = 1):
        '''
        plot a state (wavefunction) on the root graph of original vertices
        '''
        Amps = state_vect
        Probs = numpy.abs(Amps)**2
        mSizes = Probs * len(Probs)*30
        mColors = numpy.angle(Amps)/numpy.pi
        
        cm = pylab.cm.get_cmap(cmap)
        
        pylab.sca(ax)
        pylab.scatter(self.coords[:,0], self.coords[:,1],c =  mColors, s = mSizes, marker = 'o', edgecolors = 'k', cmap = cm, vmin = -0.5, vmax = 1.5, zorder = zorder)
        if colorbar:
            cbar = pylab.colorbar(fraction=0.046, pad=0.04)
            cbar.set_label('phase (pi radians)', rotation=270)
              
        if plot_links:
            self.draw_SDlinks(ax, linewidth = 0.5, color = 'firebrick')
        
        pylab.title(title, fontsize=8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_aspect('equal')
        return