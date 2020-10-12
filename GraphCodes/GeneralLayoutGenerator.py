#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:22:52 2018

@author: kollar2

starting Kollarlab version 1-20-20

modified from LayoutGenerator5 which makes hyperbolic lattices
and EuclideanLayoutGenerator2 which makes regular 2D lattices
Tried to keep as muchof the structure and syntax consistent.


GeneralLayout takes as input a set of resonators, and does autoprocessing on that

TreeResonators makes a set of resonators which is a tree

This file also contains some autonomous resonator prcoessing functions

v0 - first pass

    7-25-18 Added zorder optional argument o all the plot functions
pl


6-18-20 AK added code to compute for the root graph, and not just the effective line graph.


        
GeneralLayout Class
    input a set of resoantors (the full lattice/tree/etc) and calculate properties
    v0 - self.coords is wierd, and may not contain all the capacitor points
     
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
        NA
         
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
        generate_vertex_dict
        
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
        from GeneralLayoutGenerator import GeneralLayout
        testLattice = GeneralLayout(file_path = 'name.pkl')
        
        #####
        #making new layout
        #####
        from GeneralLayoutGenerator import GeneralLayout
        from EuclideanLayoutGenerator2 import UnitCell
        from LayoutGenerator5 import PlanarLayout
        from GeneralLayoutGenerator import TreeResonators
        
        #hyperbolic
        test1 = PlanarLayout(gon = 7, vertex = 3, side =1, radius_method = 'lin')
        test1.populate(2, resonatorsOnly=False)
        resonators = test1.get_all_resonators()
        #Euclidean
        test1 = EuclideanLayout(4,3,lattice_type = 'Huse', modeType = 'FW')
        resonators = test1.resonators
        #tree
        Tree = TreeResonators(degree = 3, iterations = 3, side = 1, file_path = '', modeType = 'FW')
        resonators = Tree.get_all_resonators()
        
        #generate full layout with SD simulation
        testLattice = GeneralLayout(resonators , modeType = 'FW', name =  'NameMe')
        
        #####
        #saving computed layout
        #####
        testLattice.save( name = 'filename.pkl') #filename can be a full path, but must have .pkl extension

Resonator Processing Functions
        #######
        #resonator array processing functions
        #######
        split_resonators
        generate_line_graph
        max_degree TBD
        shift_resonators
        rotate_resonators
        get_coordsrrrg
        
    Samples syntax:
        #####
        #split each resonator in two
        #####
        from GeneralLayoutGenerator import split_resonators
        splitGraph = split_resonators(resonators)
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

from GraphCodes.BaseClass import BaseLayout
from GraphCodes.TreeResonators import TreeResonators



class GeneralLayout(BaseLayout):
    def __init__(self, resonators = [0,0,0,0], side = 1, file_path = '', modeType = 'FW', name = 'TBD', vertexDict = True, resonatorsOnly = False, roundDepth = 3):
        '''
        
        '''
        
        if file_path != '':
            self.load(file_path)
        else:
            if numpy.all(numpy.asarray(resonators) == numpy.asarray([0,0,0,0])):
                raise ValueError('need input resonators')
            
            self.name  =  name

            if not ((modeType == 'FW') or (modeType  == 'HW')):
                raise ValueError('Invalid mode type. Must be FW or HW')

            self.modeType = modeType
            
            self.roundDepth = roundDepth
            
            self.resonators = resonators
            self.coords = self.get_coords(self.resonators)

            if not resonatorsOnly:
                self.populate()
                
                if vertexDict:
                    self.generate_vertex_dict()
            
    ###########
    #automated construction, saving, loading
    ##########
    def populate(self, Hamiltonian = True, save = False, save_name = ''):
        '''
        fully populate the structure up to itteration = MaxItter
        
        if Hamiltonian = False will not generate H
        save is obvious
        '''
         
        # #make the resonator lattice
        #NA
        
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
            waveStr = '_HW'
        else:
            waveStr = ''
            
        if name == '':
            name = self.name + waveStr + '.pkl'
        
        savedict = self.__dict__
        pickle.dump(savedict, open(name, 'wb'))
        return
    
    ########
    #functions to generate tlattice properties
    #######
    def generate_semiduals(self):
        '''
        function to autogenerate the links between a set of resonators and itself
        
        
        will return a matrix of all the links [start, target, start_polarity, end_polarity]
        
        
        '''


        ress1 = self.resonators
        len1 = ress1.shape[0]
        
        ress2 = self.resonators

        #place to store the links
        linkMat = numpy.zeros((len1*4+len1*4,4))
        
        #find the links
        
        #round the coordinates to prevent stupid mistakes in finding the connections
        plusEnds = numpy.round(ress2[:,0:2], self.roundDepth)
        minusEnds = numpy.round(ress2[:,2:4],self.roundDepth)
        
        extraLinkInd = 0
        for resInd in range(0,ress1.shape[0]):
            res = numpy.round(ress1[resInd,:], self.roundDepth)
            x1 = res[0]
            y1 = res[1]
            x0 = res[2]
            y0 = res[3]

            plusPlus = numpy.where((plusEnds == (x1, y1)).all(axis=1))[0]
            minusMinus = numpy.where((minusEnds == (x0, y0)).all(axis=1))[0]
            
            plusMinus = numpy.where((minusEnds == (x1, y1)).all(axis=1))[0] #plus end of new res, minus end of old
            minusPlus = numpy.where((plusEnds == (x0, y0)).all(axis=1))[0]
            
            for ind in plusPlus:
                if ind == resInd:
                    #self link
                    pass
                else:
                    linkMat[extraLinkInd,:] = [resInd, ind, 1,1]
                    extraLinkInd = extraLinkInd+1
                    
            for ind in minusMinus:
                if ind == resInd:
                    #self link
                    pass
                else:
                    linkMat[extraLinkInd,:] = [resInd, ind, 0,0]
                    extraLinkInd = extraLinkInd+1
                    
            for ind in plusMinus:
                if ind == resInd: #this is a self loop edge
                    linkMat[extraLinkInd,:] = [resInd, ind,  1,0]
                    extraLinkInd = extraLinkInd+1
                elif ind in plusPlus: #don't double count if you hit a self loop edge 
                    pass 
                elif ind in minusMinus:
                    pass 
                else:
                    linkMat[extraLinkInd,:] = [resInd, ind,  1,0]
                    extraLinkInd = extraLinkInd+1
                
            for ind in minusPlus:
                if ind == resInd:#this is a self loop edge
                    linkMat[extraLinkInd,:] = [ resInd, ind,  0,1]
                    extraLinkInd = extraLinkInd+1
                elif ind in plusPlus: #don't double count if you hit a self loop edge 
                    pass 
                elif ind in minusMinus:
                    pass 
                else:
                    linkMat[extraLinkInd,:] = [ resInd, ind,  0,1]
                    extraLinkInd = extraLinkInd+1
        
        #clean the skipped links away 
        linkMat = linkMat[~numpy.all(linkMat == 0, axis=1)]  
        self.SDHWlinks = linkMat

        xs = numpy.zeros(self.resonators.shape[0])
        ys = numpy.zeros(self.resonators.shape[0])
        for rind in range(0, self.resonators.shape[0]):
            res = self.resonators[rind,:]
            xs[rind] = (res[0] + res[2])/2
            ys[rind] = (res[1] + res[3])/2
        self.SDx = xs
        self.SDy = ys
        self.SDlinks = self.SDHWlinks[:,0:2]
        
        
        return linkMat
    
    def generate_vertex_dict(self):
        plusEnds = numpy.round(self.resonators[:,0:2],self.roundDepth)
        minusEnds = numpy.round(self.resonators[:,2:4],self.roundDepth)
        
        self.vertexDict = {}
        
        #loop over the vertices.
        for vind in range(0, self.coords.shape[0]):
            #vertex = self.coords[vind, :]
            vertex = numpy.round(self.coords[vind, :],self.roundDepth)
            
            startMatch = numpy.where((plusEnds == (vertex[0], vertex[1])).all(axis=1))[0]
            endMatch = numpy.where((minusEnds == (vertex[0], vertex[1])).all(axis=1))[0]
            
            matchList = []
            for rind in startMatch:
                matchList.append(int(rind))
            for rind in endMatch:
                matchList.append(int(rind))
             
            #store the results
            self.vertexDict[vind] = numpy.asarray(matchList)
        
        return self.vertexDict
    
    def draw_SDlinks(self, ax, color = 'firebrick', linewidth = 0.5, extra = False, minus_links = False, minus_color = 'goldenrod', NaNs = True, alpha = 1, zorder = 1):
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
        
        if NaNs:
            if minus_links == True and self.modeType == 'HW':
                plotVecx_plus = numpy.asarray([])
                plotVecy_plus = numpy.asarray([])
                
                plotVecx_minus = numpy.asarray([])
                plotVecy_minus = numpy.asarray([])
                
                for link in range(0, links.shape[0]):
                    [startInd, endInd]  = links[link,0:2]
                    startInd = int(startInd)
                    endInd = int(endInd)
                    
                    ends = links[link,2:4]
                    
                    if ends[0]==ends[1]:
                        plotVecx_plus = numpy.concatenate((plotVecx_plus, [xs[startInd]], [xs[endInd]], [numpy.NaN]))
                        plotVecy_plus = numpy.concatenate((plotVecy_plus, [ys[startInd]], [ys[endInd]], [numpy.NaN]))
                    else:
                        plotVecx_minus = numpy.concatenate((plotVecx_minus, [xs[startInd]], [xs[endInd]], [numpy.NaN]))
                        plotVecy_minus = numpy.concatenate((plotVecy_minus, [ys[startInd]], [ys[endInd]], [numpy.NaN]))
                
                ax.plot(plotVecx_plus,plotVecy_plus, color = color, linewidth = linewidth, alpha = alpha, zorder = zorder)
                ax.plot(plotVecx_minus,plotVecy_minus , color = minus_color, linewidth = linewidth, alpha = alpha, zorder = zorder)
            else:
                plotVecx = numpy.zeros(links.shape[0]*3)
                plotVecy = numpy.zeros(links.shape[0]*3)
                
                for link in range(0, links.shape[0]):
                    [startInd, endInd]  = links[link,0:2]
                    startInd = int(startInd)
                    endInd = int(endInd)
                    
                    plotVecx[link*3:link*3 + 3] = [xs[startInd], xs[endInd], numpy.NaN]
                    plotVecy[link*3:link*3 + 3] = [ys[startInd], ys[endInd], numpy.NaN]
                
                ax.plot(plotVecx,plotVecy , color = color, linewidth = linewidth, alpha = alpha, zorder = zorder)
            
        else:
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
                        ax.plot([x0, x1],[y0, y1] , color = color, linewidth = linewidth, alpha = alpha, zorder = zorder)
                    else:
                        #+- or -+, use inverted t
                        ax.plot([x0, x1],[y0, y1] , color = minus_color, linewidth = linewidth, alpha = alpha, zorder = zorder)
                else :
                    ax.plot([x0, x1],[y0, y1] , color = color, linewidth = linewidth, alpha = alpha, zorder = zorder)
                
        return


#######
#resonator array processing functions
#######
    
#def split_resonators(resMat):
#    '''take in a matrix of resonators, and split them all in half.
#    Return the new resonators
#    (for use in making things like the McLaughlin graph)
#    '''
#    oldNum = resMat.shape[0]
#    newNum = oldNum*2
#    
#    newResonators = numpy.zeros((newNum,4))
#    
#    for rind in range(0, oldNum):
#        oldRes = resMat[rind,:]
#        xstart = oldRes[0]
#        ystart = oldRes[1]
#        xend = oldRes[2]
#        yend = oldRes[3]
#        
#        xmid = (xstart + xend)/2.
#        ymid = (ystart + yend)/2.
#        
#        newResonators[2*rind,:] = [xstart, ystart, xmid, ymid]
#        newResonators[2*rind+1,:] = [xmid, ymid, xend, yend]
#         
#    return newResonators

def split_resonators(resMat, splitIn = 2):
    '''take in a matrix of resonators, and split them all in half.
    Return the new resonators
    (for use in making things like the McLaughlin graph)
    
    set SplitIn > 2 to split the resonators in more than just half
    '''
    oldNum = resMat.shape[0]
    
    if type(splitIn) != int:
        raise ValueError('need an integer split')
    newNum = oldNum*splitIn
    
    newResonators = numpy.zeros((newNum,4))
    
    for rind in range(0, oldNum):
        oldRes = resMat[rind,:]
        xstart = oldRes[0]
        ystart = oldRes[1]
        xend = oldRes[2]
        yend = oldRes[3]
        
        xs = numpy.linspace(xstart, xend, splitIn+1)
        ys = numpy.linspace(ystart, yend, splitIn+1)
        for sind in range(0, splitIn):
            newResonators[splitIn*rind + sind,:] = [xs[sind], ys[sind], xs[sind+1], ys[sind+1]]
            #newResonators[2*rind+1,:] = [xmid, ymid, xend, yend]
         
    return newResonators
    
def generate_line_graph(resMat, roundDepth = 3):
    '''
        function to autogenerate the links between a set of resonators and itself
        will calculate a matrix of all the links [start, target, start_polarity, end_polarity]
        
        then use that to make new resonators that consitute the line graph
        '''


    ress1 = resMat
    len1 = ress1.shape[0]
    
    ress2 = resMat

    #place to store the links
    linkMat = numpy.zeros((len1*4+len1*4,4))
    
    #find the links
    
    #round the coordinates to prevent stupid mistakes in finding the connections
    plusEnds = numpy.round(ress2[:,0:2],roundDepth)
    minusEnds = numpy.round(ress2[:,2:4],roundDepth)
    
    extraLinkInd = 0
    for resInd in range(0,ress1.shape[0]):
        res = numpy.round(ress1[resInd,:],roundDepth)
        x1 = res[0]
        y1 = res[1]
        x0 = res[2]
        y0 = res[3]

        plusPlus = numpy.where((plusEnds == (x1, y1)).all(axis=1))[0]
        minusMinus = numpy.where((minusEnds == (x0, y0)).all(axis=1))[0]
        
        plusMinus = numpy.where((minusEnds == (x1, y1)).all(axis=1))[0] #plus end of new res, minus end of old
        minusPlus = numpy.where((plusEnds == (x0, y0)).all(axis=1))[0]
        
        for ind in plusPlus:
            if ind == resInd:
                #self link
                pass
            else:
                linkMat[extraLinkInd,:] = [resInd, ind, 1,1]
                extraLinkInd = extraLinkInd+1
                
        for ind in minusMinus:
            if ind == resInd:
                #self link
                pass
            else:
                linkMat[extraLinkInd,:] = [resInd, ind, 0,0]
                extraLinkInd = extraLinkInd+1
                
        for ind in plusMinus:
            if ind == resInd: #this is a self loop edge
                linkMat[extraLinkInd,:] = [resInd, ind,  1,0]
                extraLinkInd = extraLinkInd+1
            elif ind in plusPlus: #don't double count if you hit a self loop edge 
                pass 
            elif ind in minusMinus:
                pass 
            else:
                linkMat[extraLinkInd,:] = [resInd, ind,  1,0]
                extraLinkInd = extraLinkInd+1
            
        for ind in minusPlus:
            if ind == resInd:#this is a self loop edge
                linkMat[extraLinkInd,:] = [ resInd, ind,  0,1]
                extraLinkInd = extraLinkInd+1
            elif ind in plusPlus: #don't double count if you hit a self loop edge 
                pass 
            elif ind in minusMinus:
                pass 
            else:
                linkMat[extraLinkInd,:] = [ resInd, ind,  0,1]
                extraLinkInd = extraLinkInd+1
    
    #clean the skipped links away 
    linkMat = linkMat[~numpy.all(linkMat == 0, axis=1)]  
    
    newNum = linkMat.shape[0]/2 #number of resonators in the line graph
    newResonators = numpy.zeros((int(newNum), 4))
    
    

    xs = numpy.zeros(resMat.shape[0])
    ys = numpy.zeros(resMat.shape[0])
    for rind in range(0, resMat.shape[0]):
        res = resMat[rind,:]
        xs[rind] = (res[0] + res[2])/2
        ys[rind] = (res[1] + res[3])/2
    SDx = xs
    SDy = ys
    
    #process into a Hamiltonian because it's a little friendlier to read from and doesn't double count
    totalSize = len(SDx)
    H = numpy.zeros((totalSize, totalSize))
    #loop over the links and fill the Hamiltonian
    for link in range(0, linkMat.shape[0]):
        [sourceInd, targetInd] = linkMat[link, 0:2]
        source = int(sourceInd)
        target = int(targetInd)
        H[source, target] = 1
        
    #loop over one half of the Hamiltonian
    rind = 0
    for sind in range(0, totalSize):
        for tind in range(sind+1,totalSize):
            if H[sind,tind] == 0:
                #no connections
                pass
            else:
                #sites are connected. Need to make a resoantors
                newResonators[rind,:] = numpy.asarray([SDx[sind], SDy[sind], SDx[tind], SDy[tind]])
                rind = rind+1
    
    return newResonators  

def shift_resonators(resonators, dx, dy):
    '''
    take array of resonators and shfit them by dx inthe x direction and dy in the y direction
    
    returns modified resonators
    '''
    newResonators = numpy.zeros(resonators.shape)
    
    newResonators[:,0] = resonators[:,0] + dx
    newResonators[:,1] = resonators[:,1] + dy
    newResonators[:,2] = resonators[:,2] + dx
    newResonators[:,3] = resonators[:,3] + dy
    
    return newResonators

def rotate_resonators(resonators, theta):
    '''
    take matrix of resonators and rotate them by angle theta (in radians)
    
    returns modified resonators 
    '''
    
    newResonators = numpy.zeros(resonators.shape)
    
    newResonators[:,0] = resonators[:,0]*numpy.cos(theta) - resonators[:,1]*numpy.sin(theta)
    newResonators[:,1] = resonators[:,0]*numpy.sin(theta) + resonators[:,1]*numpy.cos(theta)
    
    newResonators[:,2] = resonators[:,2]*numpy.cos(theta) - resonators[:,3]*numpy.sin(theta)
    newResonators[:,3] = resonators[:,2]*numpy.sin(theta) + resonators[:,3]*numpy.cos(theta)
    
    return newResonators

def decorate_layout(layoutResonators, cellResonators):
    '''
    Take a layout and decorate each resonator in it with a cell of resonators.
    
    NOTE: cell must run between (-1/2,0) and (1/2,0) otherwise this will give garbage
    '''
    oldRes = layoutResonators.shape[0]
    cellSites = cellResonators.shape[0]
    newResonators = numpy.zeros((oldRes*cellSites,4))
    
    for rind in range(0, oldRes):
        [xstart,ystart, xend, yend] = layoutResonators[rind,:]
        
        armLength = numpy.sqrt((xend-xstart)**2 + (yend-ystart)**2) #length that the cell has to fit in
        armTheta = numpy.arctan2(yend-ystart, xend-xstart) #angle that the cell has to be at
        
        tempRes = numpy.copy(cellResonators)
        tempRes = shift_resonators(cellResonators, 0.5,0)
        tempRes = tempRes*armLength #rescale to the right length
        tempRes = rotate_resonators(tempRes, armTheta) #rotate into poition
        tempRes = shift_resonators(tempRes, xstart,ystart) #shift into position
        
        #store them away
        newResonators[rind*cellSites:(rind+1)*cellSites,:] = tempRes
    
    return newResonators

def get_coords(resonators, roundDepth = 3):
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

if __name__=="__main__":      
    
    #tree
    Tree = TreeResonators(degree = 3, iterations = 4, side = 1, file_path = '', modeType = 'FW')
    resonators = Tree.get_all_resonators()
#    Tree2 = TreeResonators(file_path = '3regularTree_ 3_.pkl')
    testLattice = GeneralLayout(resonators , modeType = Tree.modeType, name =  'TREEEEE')

    #######split tree
    #Tree = TreeResonators(degree = 3, iterations = 4, side = 1, file_path = '', modeType = 'FW')
    #resonators = Tree.get_all_resonators()
    #splitGraph = split_resonators(resonators)
    #resonators = splitGraph
    #testLattice = GeneralLayout(resonators , modeType = Tree.modeType, name =  'McLaughlinTree')

#    ######non-trivial tree
#    Tree = TreeResonators(cell ='Peter', degree = 3, iterations = 3, side = 1, file_path = '', modeType = 'FW')
#    resonators = Tree.get_all_resonators()
#    testLattice = GeneralLayout(resonators , modeType = Tree.modeType, name =  'PeterTREEEEE')
    ##testLattice = GeneralLayout(Tree.cellResonators , modeType = Tree.modeType, name =  'NameMe')
    ##testLattice = GeneralLayout(rotate_resonators(Tree.cellResonators,numpy.pi/3) , modeType = Tree.modeType, name =  'NameMe')

    
#    #generate full layout with SD simulation
#    testLattice = GeneralLayout(resonators , modeType = Tree.modeType, name =  'NameMe')
    
    showLattice = True
    showHamiltonian = True
    
    
    if showLattice:
    
#        fig1 = pylab.figure(1)
#        pylab.clf()
#        ax = pylab.subplot(1,1,1)
#        Tree.draw_resonator_lattice(ax, color = 'mediumblue', alpha = 1 , linewidth = 2.5)
#        xs = Tree.coords[:,0]
#        ys = Tree.coords[:,1]
#        pylab.sca(ax)
#        #pylab.scatter(xs, ys ,c =  'goldenrod', s = 20, marker = 'o', edgecolors = 'k', zorder = 5)
#        pylab.scatter(xs, ys ,c =  'goldenrod', s = 30, marker = 'o', edgecolors = 'k', zorder = 5)
#        #pylab.scatter(xs, ys ,c =  'goldenrod', s = 40, marker = 'o', edgecolors = 'k', zorder = 5)
#        ax.set_aspect('equal')
#        ax.axis('off')
#        pylab.tight_layout()
#        pylab.show()
#        fig1.set_size_inches(5, 5)


#        fig1 = pylab.figure(1)
#        pylab.clf()
#        ax = pylab.subplot(1,1,1)
#        testLattice.draw_resonator_lattice(ax, color = 'mediumblue', alpha = 1 , linewidth = 1.5)
#        testLattice.draw_SDlinks(ax, color = 'deepskyblue', linewidth = 2.5, minus_links = False, minus_color = 'goldenrod')
#        pylab.scatter(testLattice.SDx, testLattice.SDy,c =  'goldenrod', marker = 'o', edgecolors = 'k', s = 5,  zorder=5)
#        
#        ax.set_aspect('equal')
#        ax.axis('off')
#        pylab.tight_layout()
#        pylab.show()
#        fig1.set_size_inches(5, 5)
        
        fig1 = pylab.figure(1)
        pylab.clf()
        ax = pylab.subplot(1,1,1)
        testLattice.draw_SDlinks(ax, color = 'deepskyblue', linewidth = 1.5, minus_links = True, minus_color = 'goldenrod')
        testLattice.draw_resonator_lattice(ax, color = 'mediumblue', alpha = 1 , linewidth = 2.5)
        xs = testLattice.coords[:,0]
        ys = testLattice.coords[:,1]
        pylab.sca(ax)
        #pylab.scatter(xs, ys ,c =  'goldenrod', s = 20, marker = 'o', edgecolors = 'k', zorder = 5)
        pylab.scatter(xs, ys ,c =  'goldenrod', s = 30, marker = 'o', edgecolors = 'k', zorder = 5)
        #pylab.scatter(xs, ys ,c =  'goldenrod', s = 40, marker = 'o', edgecolors = 'k', zorder = 5)
        ax.set_aspect('equal')
        ax.axis('off')
        pylab.tight_layout()
        pylab.show()
        pylab.title('generalized layout and effective model')
        fig1.set_size_inches(5, 5)
    else:
        pylab.figure(1)
        pylab.clf()
        
        
    if showHamiltonian:
        eigNum = 168
        eigNum = 167
        eigNum = 0
        
        pylab.figure(2)
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
        
        pylab.figure(3)
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
        pylab.figure(2)
        pylab.clf()
        
        pylab.figure(3)
        pylab.clf()
        