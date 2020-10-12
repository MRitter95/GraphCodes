#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:22:52 2018

@author: kollar2

starting Kollarlab version 1-20-20

modified from LayoutGnerator5 which makes hyperbolic lattices
Tried to keep as muchof the structure and syntax consistent.
Many many things are the same, but some things had to change because this one 
is build unit cell by unit sell and not itteration shell by itteration shell

v0 - basic Huse lattice only

v2 - adding kagome and composite Huse/kagome. UnitCell class will get a lot fo 
    new functions, including Bloch theory calculations
    
    Also added alternate versions on the Huse idea trying to insert other types of polygons.
    Tried 7,4 and 12,3 and 8,4.
    Composite cell function can handle original 7,5 Huse and the new variants.
    (Note: 7,4 is not 3-regular, but it is still triangle protected.)
    
    7-25-18 Added zorder as optional argument to all the plot functions
    
    8-14-18 adding funtions that allow new unit cells to be made either by subdividing an old cell
    or taking its line graph
    
6-18-20 AK added code to compute for the root graph, and not just the effective line graph.
    

EuclideanLayout Class
    Chose your UnitCell type, wave type, and number of unit cells and make a lattice
     
     Methods:
        ###########
        #automated construction, saving, loading
        ##########
        populate (autoruns at creation time)
        save
        load
        
        ########
        #functions to generate the resonator lattice
        #######
        generateLattice
        _fix_edge_resonators (already stores some SD properties of fixed edge)
         
        #######
        #resonator lattice get /view functions
        #######
        get_xs
        get_ys
        draw_resonator_lattice
        draw_resonator_end_points
        get_all_resonators
        get_coords
        
        ########
        #functions to generate effective JC-Hubbard lattice (semiduals)
        ######## 
        generate_semiduals
        _fix_SDedge
        
        #######
        #get and view functions for the JC-Hubbard (semi-dual lattice)
        #######
        draw_SD_points
        draw_SDlinks
        get_semidual_points (semi-defunct)
        
        ######
        #Hamiltonian related methods
        ######
        generate_Hamiltonian
        get_eigs
        
        ##########
        #methods for calculating/looking at states and interactions
        #########
        get_SDindex (removed for now. Needs to be reimplemented in sensible fashion)
        build_local_state
        V_int
        V_int_map
        plot_layout_state
        plot_map_state
        get_end_state_plot_points
        plot_end_layout_state
        
        ##########
        #methods for calculating things about the root graph
        #########
        generate_root_Hamiltonian
        plot_root_state
        
        
    Sample syntax:
        #####
        #loading precalculated layout
        #####
        from EuclideanLayoutGenerator import EuclideanLayout
        testLattice = EuclideanLayout(file_path = 'Huse_4x4_FW.pkl')
        
        #####
        #making new layout
        #####
        from EuclideanLayoutGenerator import EuclideanLayout
        #from built-in cell
        testLattice = EuclideanLayout(xcells = 4, ycells = 4, lattice_type = 'Huse', side = 1, file_path = '', modeType = 'FW')
        
        #from custom cell
        testCell = UnitCell(lattice_type = 'name', side = 1, resonators = resonatorMat, a1 = vec1, a2 = vec2)
        testLattice = EuclideanLayout(xcells = 4, ycells = 4, modeType = 'FW', resonatorsOnly=False, initialCell = testCell)
        
        #####
        #saving computed layout
        #####
        testLattice.save( name = 'filename.pkl') #filename can be a full path, but must have .pkl extension
"""
import re
import scipy
import pylab
import numpy
import time

import pickle
import datetime
import os
import sys

import scipy.linalg
from scipy.sparse import coo_matrix
from GraphCodes.UnitCell import UnitCell
from GraphCodes.BaseClass import BaseLayout

class EuclideanLayout(BaseLayout):
    def __init__(self, xcells = 4, ycells = 4, lattice_type = 'Huse', side = 1, file_path = '', modeType = 'FW', resonatorsOnly=False, initialCell = ''):
        '''
        
        '''
        
        if file_path != '':
            self.load(file_path)
        else:
            #create plank planar layout object with the bare bones that you can build on
            self.xcells = xcells
            self.ycells = ycells
            self.side = side*1.0

            self.lattice_type = lattice_type
            
            if type(initialCell) == UnitCell:
                #use the unit cell object provided
                self.unitcell = initialCell
            else:
                #use a built in unit cell specified by keyword
                #starting unit cell
                self.unitcell = UnitCell(self.lattice_type, self.side)

            
            if not ((modeType == 'FW') or (modeType  == 'HW')):
                raise ValueError('Invalid mode type. Must be FW or HW')
            self.modeType = modeType
            
            self.populate(resonatorsOnly)
            
    ###########
    #automated construction, saving, loading
    ##########
    def populate(self, resonatorsOnly=False, Hamiltonian = True, save = False, save_name = ''):
        '''
        fully populate the structure up to itteration = MaxItter
        
        if Hamiltonian = False will not generate H
        save is obvious
        '''
         
        #make the resonator lattice
        self.generate_lattice(self.xcells, self.ycells)
        
        if not resonatorsOnly:
            #make the JC-Hubbard lattice
            self.generate_semiduals()
            
            if Hamiltonian:
                self.generate_Hamiltonian()
            
            if save:
                self.save(save_name)
            
        return
    
    def save(self, name = ''):
        '''
        save structure to a pickle file
        
        if name is blank, will use dafualt name
        '''
        if self.modeType == 'HW':
            waveStr = 'HW'
        else:
            waveStr = ''
            
        if name == '':
            name = str(self.lattice_type) + '_' + str(self.xcells) + 'x ' + str(self.ycells) + '_' + waveStr + '.pkl'
        
        savedict = self.__dict__
        pickle.dump(savedict, open(name, 'wb'))
        return
    
    ########
    #functions to generate the resonator lattice
    #######
    def generate_lattice(self, xsize = -1, ysize = -1):
        '''
        Hopefully will become a general function to fill out the lattice. Has some issues
        with the edges right now
        
        it is important not to reverse the order of the endpoints of the lattice. These indicate
        the orinetation of the site, and will be needed to fill in the extra links to fix the edge of the lattice
        in HW mode
        '''
        if xsize <0:
            xsize = self.xcells
        if ysize <0:
            ysize = self.ycells
            
        #make sure that the object has the right size recorded
        self.xcells = xsize
        self.ycells = ysize
        
        self.resonators = numpy.zeros((xsize*ysize*self.unitcell.numSites, 4))
        
        #need a place to store extra resonators that live on the edge of the lattice
        self.extraResonators = numpy.zeros((xsize*ysize*self.unitcell.numSites, 4)) 
        self.extraSDx = numpy.zeros(xsize*ysize*self.unitcell.numSites) #these are easier to calculate here rather than later even though they don't fit thematically
        self.extraSDy = numpy.zeros(xsize*ysize*self.unitcell.numSites)
        
        xmask = numpy.zeros((self.unitcell.numSites,4))
        ymask = numpy.zeros((self.unitcell.numSites,4))
        
        xmask[:,0] = 1
        xmask[:,2] = 1
        
        ymask[:,1] = 1
        ymask[:,3] = 1
        
        ind = 0
        extraInd = 0
        for indx in range(0,xsize):
            for indy in range(0,ysize):
                xOffset = indx*self.unitcell.a1[0] + indy*self.unitcell.a2[0]
                yOffset = indx*self.unitcell.a1[1] + indy*self.unitcell.a2[1]
                self.resonators[ind:ind+self.unitcell.numSites, :] = self.unitcell.resonators + xOffset*xmask + yOffset*ymask
                
                if indy ==0:
                    #bottom row of lattice sites
                    xx = 0
                    yy = -1
                    indOut = self._fix_edge_resonators(extraInd,indx, indy, xx, yy)
                    extraInd = indOut
                    
                    xx = 1
                    yy = -1
                    indOut = self._fix_edge_resonators(extraInd,indx, indy, xx, yy)
                    extraInd = indOut
                
                if indx ==(xsize-1):
                    #right hand edge
                    xx = 1
                    yy = 0
                    indOut = self._fix_edge_resonators(extraInd,indx, indy, xx, yy)
                    extraInd = indOut
                    
                    xx = 1
                    yy = 1
                    indOut = self._fix_edge_resonators(extraInd,indx, indy, xx, yy)
                    extraInd = indOut
                    
                    if indy != 0:
                        xx = 1
                        yy = -1
                        indOut = self._fix_edge_resonators(extraInd,indx, indy, xx, yy)
                        extraInd = indOut
                
                if indy ==(ysize-1):
                    #top row of lattice sites
                    xx = 0
                    yy = 1
                    indOut = self._fix_edge_resonators(extraInd,indx, indy, xx, yy)
                    extraInd = indOut
                    
                    xx = -1
                    yy = 1
                    indOut = self._fix_edge_resonators(extraInd,indx, indy, xx, yy)
                    extraInd = indOut
                    
                if indy ==0:
                    #left0-hand edge
                    xx = -1
                    yy = 0
                    indOut = self._fix_edge_resonators(extraInd,indx, indy, xx, yy)
                    extraInd = indOut
                    
                    xx = -1
                    yy = -1
                    indOut = self._fix_edge_resonators(extraInd,indx, indy, xx, yy)
                    extraInd = indOut
                
                ind = ind + self.unitcell.numSites
        
        #clean the blank resonators away
        self.extraResonators = self.extraResonators[~numpy.all(self.extraResonators == 0, axis=1)]
        self.extraSDx = self.extraSDx[0:extraInd] 
        self.extraSDy = self.extraSDy[0:extraInd]
        
        #combine regular unit cell resonators and the extra edge ones
        self.resonators = numpy.concatenate((self.resonators, self.extraResonators))
        
        self.coords = self.get_coords(self.resonators)
        
        return
    
    def _fix_edge_resonators(self, extraInd,indx, indy, xx, yy):
        ''' very dirty function for fixing and adding missing resonators at the edge of the lattice
        mess with it at your own peril.
        
        Should only be called internally from generat_lattice
        
        also generates the extra center points of the resonators, because it's easier to do it all together.
        '''
        xmask = numpy.zeros((self.unitcell.numSites,4))
        ymask = numpy.zeros((self.unitcell.numSites,4))
        
        xmask[:,0] = 1
        xmask[:,2] = 1
        
        ymask[:,1] = 1
        ymask[:,3] = 1
        
        tempSites = self.unitcell.closure[(xx,yy)]
        xOffset = (indx+xx)*self.unitcell.a1[0] + (indy+yy)*self.unitcell.a2[0]
        yOffset = (indx+xx)*self.unitcell.a1[1] + (indy+yy)*self.unitcell.a2[1]
        for site in tempSites:
            self.extraResonators[extraInd, :] = self.unitcell.resonators[site,:] + xOffset*xmask[0,:] + yOffset*ymask[0,:]
            
            self.extraSDx[extraInd] = self.unitcell.SDx[site] + xOffset
            self.extraSDy[extraInd] = self.unitcell.SDy[site] + yOffset
            extraInd = extraInd + 1
            
        return extraInd
        
    ########
    #functions to generate effective JC-Hubbard lattice
    ########
    def generate_semiduals(self):
        '''
        main workhorse function to generate the JC-Hubbard lattice.
        This is the one you shold call. All the others are workhorses that it uses.
        
        Will loop through the existing and create attributes for the 
        JC-Hubbard lattice (here jokingly called semi-dual) and fill them
        '''
        xsize = self.xcells
        ysize = self.ycells
        
        self.SDx = numpy.zeros(xsize*ysize*self.unitcell.numSites)
        self.SDy = numpy.zeros(xsize*ysize*self.unitcell.numSites)
        
        #self.SDlinks = numpy.zeros((xsize*ysize*self.unitcell.numSites*4, 2))
        
        if self.lattice_type == 'square':
            self.SDHWlinks = numpy.zeros((xsize*ysize*self.unitcell.numSites*6, 4))
        
            self.extraSDHWlinks = numpy.zeros((xsize*ysize*self.unitcell.numSites*6, 4))
        else:
            #self.SDHWlinks = numpy.zeros((xsize*ysize*self.unitcell.numSites*4, 4))
            #self.extraSDHWlinks = numpy.zeros((xsize*ysize*self.unitcell.numSites*4, 4))
            
            #temporary hack to allow playing with larger coordination numbers. Otherwise
            #there was not enough space allocated
            #what is really needed is something where the max coordination number of the unit
            #cell is used to do this properly.
            self.SDHWlinks = numpy.zeros((xsize*ysize*self.unitcell.numSites*8, 4))
            
            self.extraSDHWlinks = numpy.zeros((xsize*ysize*self.unitcell.numSites*8, 4))
            
        
        
        #set up for getting the positions of the semidual points
        xmask = numpy.zeros((self.unitcell.numSites,4))
        ymask = numpy.zeros((self.unitcell.numSites,4))
        
        xmask[:,0] = 1
        xmask[:,2] = 1
        
        ymask[:,1] = 1
        ymask[:,3] = 1

        #links will be done by site index, which will include the unit cell number
        latticelinkInd = 0
        ind = 0
        for indx in range(0,xsize):
            for indy in range(0,ysize):
                currCell = [indx, indy]
                
                xOffset = indx*self.unitcell.a1[0] + indy*self.unitcell.a2[0]
                yOffset = indx*self.unitcell.a1[1] + indy*self.unitcell.a2[1]
                self.SDx[ind:ind+self.unitcell.numSites] = self.unitcell.SDx + xOffset
                self.SDy[ind:ind+self.unitcell.numSites] = self.unitcell.SDy + yOffset
                
                ind = ind + self.unitcell.numSites
                
                for link in range(0, self.unitcell.SDlinks.shape[0]):
                    [startSite, targetSite, deltaA1, deltaA2, startEnd, targetEnd]  = self.unitcell.SDHWlinks[link,:]
                    targetCell = [indx + deltaA1, indy + deltaA2]
                    #print [startSite, targetSite, deltaA1, deltaA2, startEnd, targetEnd]
                    #print currCell
                    #print targetCell
                    if (targetCell[0]<0) or (targetCell[1]<0) or (targetCell[0]>xsize-1) or (targetCell[1]>ysize-1):
                        #this cell is outside of the simulation. Leave it
                        #print 'passing by'
                        pass
                    else:
                        startInd = startSite + currCell[0]*self.unitcell.numSites*ysize + currCell[1]*self.unitcell.numSites
                        targetInd = targetSite + targetCell[0]*self.unitcell.numSites*ysize + targetCell[1]*self.unitcell.numSites
                        self.SDHWlinks[latticelinkInd,:] = [startInd, targetInd, startEnd, targetEnd]
                        #print [startInd, targetInd, startEnd, targetEnd]
                        latticelinkInd = latticelinkInd +1  
                    #print '   '
        
        #fix the edge
        self._fix_SDedge()
        
        #clean the skipped links away 
        self.SDHWlinks = self.SDHWlinks[~numpy.all(self.SDHWlinks == 0, axis=1)]  
        self.extraSDHWlinks = self.extraSDHWlinks[~numpy.all(self.extraSDHWlinks == 0, axis=1)] 
        
        #add the extra links to the lattice
        self.SDHWlinks = numpy.concatenate((self.SDHWlinks , self.extraSDHWlinks))
        
        #make the truncated SD links
        self.SDlinks = self.SDHWlinks[:,0:2]
        
        #add the extra sites to the lattice
        self.SDx = numpy.concatenate((self.SDx, self.extraSDx))
        self.SDy = numpy.concatenate((self.SDy, self.extraSDy))
        
        return
    
    def _fix_SDedge(self):
        '''function to loop over the extra edge resonators and add their links in '''
        originalLatticeSize = self.xcells*self.ycells*self.unitcell.numSites
        
        #round the coordinates to prevent stupid mistakes in finding the connections
        plusEnds = numpy.round(self.resonators[:,0:2],3)
        minusEnds = numpy.round(self.resonators[:,2:4],3)
        
        extraLinkInd = 0
        for resInd in range(0,self.extraResonators.shape[0]):
            res = numpy.round(self.extraResonators[resInd,:],3)
            x1 = res[0]
            y1 = res[1]
            x0 = res[2]
            y0 = res[3]

            plusPlus = numpy.where((plusEnds == (x1, y1)).all(axis=1))[0]
            minusMinus = numpy.where((minusEnds == (x0, y0)).all(axis=1))[0]
            
            plusMinus = numpy.where((minusEnds == (x1, y1)).all(axis=1))[0] #plus end of new res, minus end of old
            minusPlus = numpy.where((plusEnds == (x0, y0)).all(axis=1))[0]
            
            for ind in plusPlus:
                if ind == originalLatticeSize+ resInd:
                    #self link
                    pass
                else:
                    self.extraSDHWlinks[extraLinkInd,:] = [originalLatticeSize+ resInd, ind,  1,1]
                    extraLinkInd = extraLinkInd+1
                    
                    #reverse link
                    self.extraSDHWlinks[extraLinkInd,:] = [ind, originalLatticeSize+ resInd,  1,1]
                    extraLinkInd = extraLinkInd+1
            for ind in minusMinus:
                if ind == originalLatticeSize+ resInd:
                    #self link
                    pass
                else:
                    self.extraSDHWlinks[extraLinkInd,:] = [originalLatticeSize+ resInd, ind,  0,0]
                    extraLinkInd = extraLinkInd+1
                    
                    #reverse link
                    self.extraSDHWlinks[extraLinkInd,:] = [ind, originalLatticeSize+ resInd,  0,0]
                    extraLinkInd = extraLinkInd+1
                
            for ind in plusMinus:
                self.extraSDHWlinks[extraLinkInd,:] = [originalLatticeSize+ resInd, ind,  1,0]
                extraLinkInd = extraLinkInd+1
                
                #reverse link
                self.extraSDHWlinks[extraLinkInd,:] = [ind, originalLatticeSize+ resInd,  0,1]
                extraLinkInd = extraLinkInd+1
                
            for ind in minusPlus:
                self.extraSDHWlinks[extraLinkInd,:] = [originalLatticeSize+ resInd, ind,  0,1]
                extraLinkInd = extraLinkInd+1
                
                #reverse link
                self.extraSDHWlinks[extraLinkInd,:] = [ind, originalLatticeSize+ resInd,  1,0]
                extraLinkInd = extraLinkInd+1
            
        return

    def draw_SDlinks(self, ax, color = 'firebrick', linewidth = 0.5, extra = False, minus_links = False, minus_color = 'goldenrod', zorder = 1, alpha = 1):
        '''
        draw all the links of the semidual lattice
        
        if extra is True it will draw only the edge sites required to fix the edge of the tiling
        
        set minus_links to true if you want the links color coded by sign
        minus_color sets the sign of the negative links
        '''
        if extra == True:
            xs = self.SDx
            ys = self.SDy
            links = self.extraSDHWlinks[:]
        else:
            xs = self.SDx
            ys = self.SDy
            links = self.SDHWlinks[:]
        
        for link in range(0, links.shape[0]):
            [startInd, endInd]  = links[link,0:2]
            startInd = int(startInd)
            endInd = int(endInd)
            
            [x0,y0] = [xs[startInd], ys[startInd]]
            [x1,y1] = [xs[endInd], ys[endInd]]
            
            if  minus_links == True and self.modeType == 'HW':
                ends = links[link,2:4]
                if ends[0]==ends[1]:
                    #++ or --, use normal t
                    ax.plot([x0, x1],[y0, y1] , color = color, linewidth = linewidth, zorder = zorder, alpha = alpha)
                else:
                    #+- or -+, use inverted t
                    ax.plot([x0, x1],[y0, y1] , color = minus_color, linewidth = linewidth, zorder = zorder, alpha = alpha)
            else :
                ax.plot([x0, x1],[y0, y1] , color = color, linewidth = linewidth, zorder = zorder, alpha = alpha)
                
        return

if __name__=="__main__":  
    Cell = True
    #Cell = False

    #Lattice = True
    Lattice = False    


    ####Cell mode sub options
    K0States = True #display or not 


    ####Lattice mode sub options
    LatticeHamiltonian = False #display or not 
    LatticeInteractionMap = True #display or not 
    
    
    
    
    ##################################
    ##################################
    ##################################
    ##################################
    ###########
    #lattice testing  and examples
    ##################################
    ##################################
    ##################################
    ##################################
    if Cell:
        testCell = UnitCell('Huse')
        
        #pylab.rcParams.update({'font.size': 14})
        #pylab.rcParams.update({'font.size': 8})
        
        modeType = 'FW'
        #modeType = 'HW'
        
        #testCell = UnitCell('Huse')
        #testCell = UnitCell('74Huse')
        #testCell = UnitCell('84Huse')
        #testCell = UnitCell('123Huse')
        #testCell = UnitCell('kagome')
       
        #testCell = UnitCell('Huse2_1')
        #testCell = UnitCell('Huse2_2')
        #testCell = UnitCell('Huse3_1')
        #testCell = UnitCell('Huse3_3')
        
        #testCell = UnitCell('84Huse2_1')
        
        #testCell = UnitCell('PeterChain')
        #testCell = UnitCell('PaterChain_tail')
        
        ######
        #test the unit cell
        #######
        pylab.figure(1)
        pylab.clf()
        ax = pylab.subplot(1,2,1)
        testCell.draw_sites(ax)
        pylab.title('Sites of Huse Cell')
        
        ax = pylab.subplot(1,2,2)
        testCell.draw_sites(ax,color = 'goldenrod', edgecolor = 'k',  marker = 'o' , size = 20)
        testCell.draw_resonators(ax, color = 'cornflowerblue', linewidth = 1)
        testCell.draw_SDlinks(ax, color = 'firebrick', linewidth = 1)
        testCell.draw_resonator_end_points(ax, color = 'deepskyblue', edgecolor = 'k',  marker = 'o' , size = 20)
        pylab.title('Links of Unit Cell')
        pylab.show()
        
        
        ######
        #show the orientations
        ######
        #alternate version
        fig = pylab.figure(2)
        pylab.clf()
        ax = pylab.subplot(1,1,1)
        testCell.draw_resonators(ax, color = 'cornflowerblue', linewidth = 1)
        testCell.draw_resonator_end_points(ax, color = 'indigo', edgecolor = 'indigo',  marker = '+' , size = 20)
        testCell.draw_site_orientations(ax, title = 'unit cell convention', colorbar = False, plot_links = False, cmap = 'jet', scaleFactor = 0.5)
        testCell.draw_SDlinks(ax, linewidth = 1.5, HW = True , minus_color = 'goldenrod')
        pylab.title('site orientations : ' + testCell.type)
        #ax.set_aspect('auto')
        ax.set_aspect('equal')
        #    fig.savefig('HW.png', dpi = 200)
        
        pylab.show()
        

        
        #####
        #testing bloch theory
        ####
        
        Hmat = testCell.generate_Bloch_matrix(0,0,  modeType = modeType)
        pylab.figure(3)
        pylab.clf()
        ax = pylab.subplot(1,2,1)
        pylab.imshow(numpy.abs(Hmat))
        pylab.title('|H|')
        
        ax = pylab.subplot(1,2,2)
        pylab.imshow(numpy.real(Hmat - numpy.transpose(numpy.conj(Hmat))))
        pylab.title('H - Hdagger')
        
        pylab.show()
        
        
        
        #kx_x, ky_y, cutx = testCell.compute_band_structure(-2*numpy.pi, 0, 2*numpy.pi, 0, numsteps = 100, modeType = modeType)
        #kx_y, ky_y, cuty = testCell.compute_band_structure(0, -8./3*numpy.pi, 0, 8./3*numpy.pi, numsteps = 100, modeType = modeType)
        kx_x, ky_y, cutx = testCell.compute_band_structure(-2*numpy.pi, 0, 2*numpy.pi, 0, numsteps = 100, modeType = modeType)
        kx_y, ky_y, cuty = testCell.compute_band_structure(0, -2.5*numpy.pi, 0, 2.5*numpy.pi, numsteps = 100, modeType = modeType)
        
        fig2 = pylab.figure(4)
        pylab.clf()
        ax = pylab.subplot(1,2,1)
        testCell.plot_band_cut(ax, cutx)
        pylab.title('xcut')
        
        ax = pylab.subplot(1,2,2)
        testCell.plot_band_cut(ax, cuty)
        pylab.title('ycut')
        
        titleStr = testCell.type + ', modeType: ' + modeType + ' (Made with UnitCell class)' 
        pylab.suptitle(titleStr)
        
        pylab.show()
        

        
        #####
        #look at above gap state at k= 0
        #####
        if K0States:
            Es, Psis = scipy.linalg.eigh(Hmat)
            
            stateInd = 0
            aboveGap = Psis[:,stateInd]
            print(Es[stateInd])
            print(aboveGap)
            
            pylab.figure(5)
            pylab.clf()
            
            ax = pylab.subplot(1,1,1)
            #testCell.draw_sites(ax,color = 'goldenrod', edgecolor = 'k',  marker = 'o' , size = 20)
            testCell.draw_SDlinks(ax, color = 'firebrick', linewidth = 1)
            testCell.draw_resonators(ax, color = 'cornflowerblue', linewidth = 1)
            testCell.draw_resonator_end_points(ax, color = 'deepskyblue', edgecolor = 'k',  marker = 'o' , size = 20)
            testCell.plot_bloch_wave(aboveGap*2, ax, title = 'state weight', colorbar = False, plot_links = False, cmap = 'Wistia')
            #temp = testCell.plot_bloch_wave_end_state(aboveGap*2, ax,modeType = modeType,  title = modeType + '_' + str(stateInd), colorbar = False, plot_links = False, cmap = 'Wistia')
            ax.set_aspect('equal')
            pylab.show()
            
            
            ####try to plot all the unit cell wave functions. Doesn't work very well. You can't see anything
            #pylab.figure(6)
            #pylab.clf()
            #for ind in range(0, testCell.numSites):
            #    ax = pylab.subplot(1,testCell.numSites,ind+1)
            #    testCell.draw_SDlinks(ax, color = 'firebrick', linewidth = 1)
            #    testCell.draw_resonators(ax, color = 'cornflowerblue', linewidth = 1)
            #    testCell.draw_resonator_end_points(ax, color = 'deepskyblue', edgecolor = 'k',  marker = 'o' , size = 20)
            ##    testCell.plot_bloch_wave(Psis[:,ind], ax, title = 'state weight', colorbar = False, plot_links = False, cmap = 'Wistia')
            #    testCell.plot_bloch_wave_end_state(Psis[:,ind], ax,modeType = modeType,  title = str(ind), colorbar = False, plot_links = False, cmap = 'Wistia')
            #    ax.set_aspect('equal')
            #pylab.show()
        else:
            pylab.figure(5)
            pylab.clf()
            
            pylab.figure(6)
            pylab.clf()
        
    
    
    
    
    
    ##################################
    ##################################
    ##################################
    ##################################
    ###########
    #lattice testing  and examples
    ##################################
    ##################################
    ##################################
    ##################################
    if Lattice:
        #testLattice = EuclideanLayout(4,3,lattice_type = 'Huse', modeType = 'FW')
        #testLattice = EuclideanLayout(2,1,lattice_type = 'Huse', modeType = 'FW')
        
        #testLattice = EuclideanLayout(4,3,lattice_type = 'Huse', modeType = 'HW')
        #testLattice = EuclideanLayout(4,2,lattice_type = 'Huse', modeType = 'HW')
        #testLattice = EuclideanLayout(2,2,lattice_type = 'Huse', modeType = 'HW')
        #testLattice = EuclideanLayout(1,1,lattice_type = 'Huse', modeType = 'HW')
    
    
        #testLattice = EuclideanLayout(4,3,lattice_type = 'Huse', modeType = 'FW', side = 500)
        
        testLattice = EuclideanLayout(4,4,lattice_type = 'kagome', modeType = 'FW')
        
        #testLattice = EuclideanLayout(2,3,lattice_type = 'Huse2_1', modeType = 'FW')
    
        #testLattice = EuclideanLayout(1,1,lattice_type = '84Huse2_1', modeType = 'FW')
        
        #testLattice = EuclideanLayout(2,1,lattice_type = '84Huse', modeType = 'FW')
        #testLattice = EuclideanLayout(4,3,lattice_type = '74Huse', modeType = 'FW')
        #testLattice = EuclideanLayout(4,3,lattice_type = '123Huse', modeType = 'FW')

        #testLattice = EuclideanLayout(3,3,lattice_type = 'square', modeType = 'FW')
    
        ######
        #test the unit cell
        #######
        pylab.figure(1)
        pylab.clf()
    
        ######
        #test the generate functions
        #######
        #testLattice.generate_lattice()
        #testLattice.generate_semiduals()
        #testLattice.generate_Hamiltonian()
    
        debugMode = False
        #debugMode = True
        
        ######
        #test the lattice and SD lattice constructions
        #######
        pylab.figure(2)
        pylab.clf()
        ax = pylab.subplot(1,2,1)
        testLattice.draw_resonator_lattice(ax, color = 'cornflowerblue', linewidth = 1)
        testLattice.draw_resonator_end_points(ax, color = 'deepskyblue', edgecolor = 'k',  marker = 'o' , size = 20)
        
        if debugMode:
            testLattice.draw_resonator_lattice(ax, color = 'indigo', linewidth = 1, extras = True)
            [x0, y0, x1, y1]  = testLattice.extraResonators[0,:]
            #ax.plot([x0, x1],[y0, y1] , color = 'firebrick', alpha = 1, linewidth = 1)
            [x0, y0, x1, y1]  = testLattice.resonators[6,:]
            #ax.plot([x0, x1],[y0, y1] , color = 'indigo', alpha = 1, linewidth = 1)
        
        pylab.title('Resonators of Huse Lattice')
        
        ax = pylab.subplot(1,2,2)
        testLattice.draw_SD_points(ax, color = 'deepskyblue', edgecolor = 'k',  marker = 'o' , size = 20)
        testLattice.draw_SDlinks(ax, color = 'firebrick', linewidth = 1)
        
        if debugMode:
            testLattice.draw_SD_points(ax, color = 'indigo', edgecolor = 'k',  marker = 'o' , size = 20, extra = True)
            testLattice.draw_SDlinks(ax, color = 'cornflowerblue', linewidth = 1, extra = True)
            #pylab.scatter(testLattice.extraSDx,testLattice.extraSDy ,c =  'indigo', s = 25, marker ='o', edgecolors = 'k')
        pylab.title('Links of the Huse Lattice')
        pylab.show()
        
        
        ######
        #test the Hamiltonian
        #######
        eigNum = 168
        eigNum = 167
        eigNum = 0
        if LatticeHamiltonian:
            pylab.figure(3)
            pylab.clf()
            ax = pylab.subplot(1,2,1)
            pylab.imshow(testLattice.H,cmap = 'winter')
            pylab.title('Hamiltonian')
            
            ax = pylab.subplot(1,2,2)
            pylab.imshow(testLattice.H - numpy.transpose(testLattice.H),cmap = 'winter')
            pylab.title('H - Htranspose')
            pylab.show()
            
    
            
            xs = numpy.arange(0,len(testLattice.Es),1)
            eigAmps = testLattice.Psis[:,testLattice.Eorder[eigNum]]
            
            pylab.figure(4)
            pylab.clf()
            ax1 = pylab.subplot(1,2,1)
            pylab.plot(testLattice.Es, 'b.')
            pylab.plot(xs[eigNum],testLattice.Es[testLattice.Eorder[eigNum]], color = 'firebrick' , marker = '.', markersize = '10' )
            pylab.title('eigen spectrum')
            pylab.ylabel('Energy (t)')
            pylab.xlabel('eigenvalue number')
            
            ax2 = pylab.subplot(1,2,2)
            titleStr = 'eigenvector weight : ' + str(eigNum)
            testLattice.plot_layout_state(eigAmps, ax2, title = titleStr, colorbar = True, plot_links = True, cmap = 'Wistia')
            
            pylab.show()
        else:
            pylab.figure(3)
            pylab.clf()
            
            pylab.figure(4)
            pylab.clf()
        
        
        ######
        #test the layout plotters (center dot)
        #######
        
        pylab.figure(5)
        pylab.clf()
        stateInd = eigNum
        state1 = testLattice.Psis[:,stateInd]
        if testLattice.xcells < 4 and testLattice.ycells <3:
            state2 = testLattice.build_local_state(7)
        else:
            #state2 = testLattice.build_local_state(47)
            state2 = testLattice.build_local_state(4)
        
        
        ax = pylab.subplot(1,2,1)
        testLattice.plot_layout_state(state1, ax, title = 'eigenstate', colorbar = False, plot_links = True, cmap = 'Wistia')
        
        ax = pylab.subplot(1,2,2)
        testLattice.plot_layout_state(state2/10, ax, title = 'local state', colorbar = False, plot_links = True, cmap = 'Wistia')
        
        pylab.show()
        
        
        ######
        #test the interaction funtions
        #######
        if LatticeInteractionMap:
            #    interactionStates = numpy.arange(0,len(testLattice.Es),1)
            if testLattice.xcells < 4 and testLattice.ycells <3:
                interactionStates = numpy.arange(0,4,1)
                site1 = 1
                site2 = 5
            else:
                interactionStates = numpy.arange(0,47,1)
                site1 = 10
                site2 = 54
            
            
            
            V0 = testLattice.V_int(site1, site1, interactionStates)
            VV = testLattice.V_int(site1, site2, interactionStates)
            print(V0)
            print(VV)
            
            Vmap0 = testLattice.V_int_map(site2, interactionStates)
            Vmap1 = testLattice.V_int_map(site2, interactionStates[0:4])
            
            pylab.figure(6)
            pylab.clf()
            ax = pylab.subplot(1,2,1)
            testLattice.plot_map_state(Vmap0, ax, title = 'ineraction weight: all FB states, hopefully', colorbar = True, plot_links = True, cmap = 'winter', autoscale = False)
            pylab.scatter([testLattice.SDx[site2]], [testLattice.SDy[site2]], c =  'gold', s = 150, edgecolors = 'k')
            
            ax = pylab.subplot(1,2,2)
            testLattice.plot_map_state(Vmap1, ax, title = 'ineraction weight: first 4', colorbar = True, plot_links = True, cmap = 'winter', autoscale = False)
            pylab.scatter([testLattice.SDx[site2]], [testLattice.SDy[site2]], c =  'gold', s = 150, edgecolors = 'k')
            
            pylab.show()
        else:
            pylab.figure(6)
            pylab.clf()
        
        
        ######
        #test visualization functions for shwing both ends of the resonators
        #######
        state_uniform = numpy.ones(len(testLattice.SDx))/numpy.sqrt(len(testLattice.SDx))
        
        pylab.figure(7)
        pylab.clf()
        ax = pylab.subplot(1,2,1)
        #testLattice.plot_layout_state(state1, ax, title = 'eigenstate', colorbar = False, plot_links = True, cmap = 'Wistia')
        testLattice.plot_layout_state(state_uniform, ax, title = 'eigenstate', colorbar = False, plot_links = True, cmap = 'Wistia')
        
        ax = pylab.subplot(1,2,2)
        endplot_points = testLattice.get_end_state_plot_points()
        #testLattice.plot_end_layout_state(state1, ax, title = 'end weights', colorbar = False, plot_links = True, cmap = 'Wistia', scaleFactor = 0.5)
        testLattice.plot_end_layout_state(state_uniform, ax, title = 'end weights', colorbar = False, plot_links = True, cmap = 'Wistia', scaleFactor = 0.5)
        
        pylab.show()
        
        
        
    #    #####
    #    #checking conventions
    #    #####
    #    
    #    pylab.figure(17)
    #    pylab.clf()
    #    ax = pylab.subplot(1,2,1)
    #    testLattice.draw_resonator_lattice(ax, color = 'cornflowerblue', linewidth = 1)
    #    testLattice.draw_resonator_end_points(ax, color = 'indigo', edgecolor = 'indigo',  marker = '+' , size = 20)
    ##    testLattice.plot_end_layout_state(state_uniform, ax, title = 'end weights', colorbar = False, plot_links = False, cmap = 'Wistia', scaleFactor = 0.5)
    #    testLattice.plot_end_layout_state(state_uniform*1.4, ax, title = 'unit cell convention', colorbar = False, plot_links = False, cmap = 'jet', scaleFactor = 0.5)
    #    testLattice.draw_SDlinks(ax, linewidth = 1, extra = False, minus_links = True, minus_color = 'goldenrod')
    #    pylab.title('site orientations')
    #    
    #    ax = pylab.subplot(1,2,2)
    #    pylab.imshow(testLattice.H,cmap = 'winter')
    #    pylab.title('Hamiltonian')
    #    pylab.show()
    #    
    #    pylab.figure(19)
    #    pylab.clf()
    #    ax = pylab.subplot(1,1,1)
    #    testLattice.draw_resonator_lattice(ax, color = 'cornflowerblue', linewidth = 1)
    #    testLattice.draw_resonator_end_points(ax, color = 'indigo', edgecolor = 'indigo',  marker = '+' , size = 20)
    ##    testLattice.plot_end_layout_state(state_uniform, ax, title = 'end weights', colorbar = False, plot_links = False, cmap = 'Wistia', scaleFactor = 0.5)
    #    testLattice.plot_end_layout_state(state_uniform*1.4, ax, title = 'unit cell convention', colorbar = False, plot_links = False, cmap = 'jet', scaleFactor = 0.5)
    #    testLattice.draw_SDlinks(ax, linewidth = 1, extra = False, minus_links = True, minus_color = 'goldenrod')
    #    pylab.title('site orientations')
    #    ax.set_aspect('auto')
    #    pylab.show()
        
        #alternate version
        fig = pylab.figure(19)
        pylab.clf()
        ax = pylab.subplot(1,1,1)
        testLattice.draw_resonator_lattice(ax, color = 'cornflowerblue', linewidth = 1)
        testLattice.draw_resonator_end_points(ax, color = 'indigo', edgecolor = 'indigo',  marker = '+' , size = 20)
        testLattice.plot_end_layout_state(state_uniform, ax, title = 'unit cell convention', colorbar = False, plot_links = False, cmap = 'jet', scaleFactor = 0.5)
        testLattice.draw_SDlinks(ax, linewidth = 1.5, extra = False, minus_links = True, minus_color = 'goldenrod')
        pylab.title('site orientations')
        #ax.set_aspect('auto')
        ax.set_aspect('equal')
        #fig.savefig('HW.png', dpi = 200)
        pylab.show()
    
        #show lattice and medial
        fig = pylab.figure(20)
        pylab.clf()
        ax = pylab.subplot(1,1,1)
        #testLattice.draw_resonator_lattice(ax, color = 'cornflowerblue', linewidth = 2)
        testLattice.draw_resonator_lattice(ax, color = 'firebrick', linewidth = 2)
        testLattice.draw_SDlinks(ax, linewidth = 2, extra = False, minus_links = False, color = 'goldenrod')
        pylab.title('site orientations')
        ax.set_aspect('auto')
        #ax.set_aspect('equal')
        ax.axis('off')
        #fig.savefig('HL.png', dpi = 200)
        pylab.show()
    
    #    #show just the medial
    #    fig = pylab.figure(21)
    #    pylab.clf()
    #    ax = pylab.subplot(1,1,1)
    #    testLattice.draw_SDlinks(ax, linewidth = 1.5, extra = False, minus_links = False, color = 'mediumblue')
    ##    ax.set_aspect('auto')
    #    ax.set_aspect('equal')
    #    ax.axis('off')
    ##    fig.savefig('Kagome.png', dpi = 200)
    #    pylab.show()
        
        
        
    #        #show lattice and medial
    #    fig = pylab.figure(21)
    #    pylab.clf()
    #    ax = pylab.subplot(1,2,1)
    #    testLattice.draw_resonator_lattice(ax, color = 'firebrick', linewidth = 2)
    #    testLattice.draw_SDlinks(ax, linewidth = 2, extra = False, minus_links = False, color = 'goldenrod')
    #    pylab.title('original resonators')
    ##    ax.set_aspect('auto')
    #    ax.set_aspect('equal')
    #    ax.axis('off')
    #    
    #    ax = pylab.subplot(1,2,2)
    #    testLattice.draw_SD_points(ax, color = 'dodgerblue', edgecolor = 'k',  marker = 'o' , size = 10)
    #    pylab.title('SD sites')
    #    ax.set_aspect('equal')
    #    ax.axis('off')
    #    
    ##    fig.savefig('HL.png', dpi = 200)
        
        
    #    