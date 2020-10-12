'''
TreeResonators Class
    generates resoantors that form a regular tree of a certain degree
    
    v0 - self.coords is wierd, and may not contain all the capacitor points
     
     Methods:
        ###########
        #automated construction, saving, loading
        ##########
        save
        load
        
        ########
        #functions to generate the resonator lattice
        #######
        generate_lattice
         
        #######
        #resonator lattice get /view functions
        #######
        get_xs
        get_ys
        draw_resonator_lattice
        draw_resonator_end_points
        get_all_resonators
        get_coords

    Sample syntax:
        #####
        #loading precalculated resonator config
        #####
        from GeneralLayoutGenerator import TreeResonators
        testTree = TreeResonators(file_path = 'name.pkl')

        #####
        #making new layout
        #####
        from GeneralLayoutGenerator import TreeResonators
        Tree = TreeResonators(degree = 3, iterations = 3, side = 1, file_path = '', modeType = 'FW')
'''

import re
import pickle
import numpy
import pylab

from GraphCodes.BaseClass import BaseLayout

class TreeResonators(BaseLayout):
    def __init__(self, isRegular = True, degree = 3, iterations = 3, side = 1, file_path = '', modeType = 'FW', cell = '', roundDepth = 3):
        if file_path != '':
            self.load(file_path)
        else:
            #create plank planar layout object with the bare bones that you can build on
            self.isRegular = isRegular
            self.degree = degree
            self.side = side*1.0
            self.iterations = iterations
            
            self.modeType = modeType
            
            self.cell = cell #type of thing to be treed
            
            self.roundDepth = roundDepth

            if self.cell == '':
                self.generate_lattice_basic()
            else:
                self.generate_lattice()
            
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
            name = str(self.degree) + 'regularTree_ ' + str(self.iterations) + '_' + waveStr + '.pkl'
        
        savedict = self.__dict__
        pickle.dump(savedict, open(name, 'wb'))
        return
    
    def generate_lattice_basic(self):
        maxItt = self.iterations

        self.resDict = {}
        
        totalSize = 0
        for itt in range(1, maxItt+1):
            radius = itt*self.side*1.0
            
            if itt==1:
                oldEnds = 1
                newEnds = self.degree
            else:
                #gather the uncapped ends
                oldRes = self.resDict[itt-1]
                oldEnds = oldRes.shape[0]
                newEnds = (self.degree-1)*oldEnds
            
            #thetas = numpy.arange(0,2*numpy.pi,2*numpy.pi/newEnds)
            thetas = numpy.arange(0,newEnds,1)*2*numpy.pi/newEnds
            
            
            if itt == 1:
                #first layer of the tree
                
                xs = radius * numpy.cos(thetas)
                ys = radius * numpy.sin(thetas)
                
                #no old resonators to start with, so make the first set
                newRes = numpy.zeros((newEnds, 4))
                for nrind in range(0, self.degree):
                    newRes[nrind,:] = [0, 0, xs[nrind],  ys[nrind]]
                    
                #store the newly created resonators
                self.resDict[itt] = newRes
                totalSize = totalSize + newEnds
            else:   
                #higher layer of the tree
                
                deltaTheta = thetas[1] - thetas[0]
                
                endInd = 0 #index of the old uncapped ends
                newRes = numpy.zeros((newEnds, 4))
                for orind in range(0, oldEnds):
                    #starting point for the new resonators
                    xstart = oldRes[orind,2]
                    ystart = oldRes[orind,3]
                    oldTheta = numpy.arctan2(ystart, xstart)
                    
                    #loop over teh resonators that need to be attached to each old end
                    for nrind in range(0, self.degree-1):
                        newTheta = oldTheta + deltaTheta*(nrind - (self.degree-2)/2.)
                        
                        xend = radius*numpy.cos(newTheta)
                        yend = radius*numpy.sin(newTheta)
                        newRes[endInd,:] = [xstart, ystart, xend,  yend]
                        endInd = endInd +1
                self.resDict[itt] = newRes
                totalSize = totalSize + newEnds
                 
        #shuffle resoantor dictionary into an array                       
        self.resonators = numpy.zeros((totalSize, 4))   
        currRes = 0
        for itt in range(1, maxItt+1):
            news = self.resDict[itt]
            numNews = news.shape[0]
            self.resonators[currRes:currRes+numNews,:] = news
            currRes = currRes + numNews
            
        
        self.coords = self.get_coords(self.resonators)
        
    def generate_lattice(self):
        maxItt = self.iterations

        self.resDict = {}
        
        if self.cell == 'Peter':
            self.cellSites = 7
            self.cellResonators = numpy.zeros((7,4))
            #set up the poisitions of all the resonators  and their end points
        
            a = self.side/(2*numpy.sqrt(2) + 2)
            b = numpy.sqrt(2)*a
            #xo,yo,x1,y1
            #define them so their orientation matches the chosen one. First entry is plus end, second is minus
            tempRes = numpy.zeros((7,4))
            tempRes[0,:] = [-a-b, 0, -b,  0]
            tempRes[1,:] = [-b, 0, 0,  b]
            tempRes[2,:] = [0, b, b,  0]
            tempRes[3,:] = [-b, 0, 0,  -b]
            tempRes[4,:] = [0, -b, b,  0]
            tempRes[5,:] = [0, -b, 0,  b]
            tempRes[6,:] = [b, 0, a+b, 0]
            
            self.cellResonators = shift_resonators(tempRes, self.side/2,0) #now one end of the cell is at zeo and the other at [self.side,0]
            #self.cellResonators = tempRes
        
        totalSize = 0
        for itt in range(1, maxItt+1):
            radius = itt*self.side*1.0
            
            if itt==1:
                oldEnds = 1
                newEnds = self.degree
            else:
                #gather the uncapped ends
                oldRes = self.resDict[itt-1]
                oldEnds = oldRes.shape[0]/self.cellSites
                #oldEndThetas = self.side*(itt-1)*numpy.arange(0,2*numpy.pi,2*numpy.pi/oldEnds)
                newEnds = (self.degree-1)*oldEnds
            
            thetas = numpy.arange(0,2*numpy.pi,2*numpy.pi/newEnds)
            
            
            if itt == 1:
                #first layer of the tree
                
                #no old resonators to start with, so make the first set
                newRes = numpy.zeros((newEnds*self.cellSites, 4))
                for cind in range(0, self.degree):
                    newRes[cind*self.cellSites:(cind+1)*self.cellSites,:] = rotate_resonators(self.cellResonators,thetas[cind])
                    
                #store the newly created resonators
                self.resDict[itt] = newRes
                totalSize = totalSize + newEnds
                
                #store the polar coordinates of the end points
                oldEndThetas = thetas
            else:   
                #higher layer of the tree
                
                deltaTheta = thetas[1] - thetas[0]
                
                endInd = 0 #index of the old uncapped ends
                newRes = numpy.zeros((newEnds*self.cellSites, 4))
                newEndThetas = numpy.zeros(newEnds) #place to store polar coordinates of the new end points
                for orind in range(0, oldEnds):
                    #starting point for the new resonators
                    xstart = self.side*(itt-1)*numpy.cos(oldEndThetas[orind])
                    ystart = self.side*(itt-1)*numpy.sin(oldEndThetas[orind])
                    oldTheta = numpy.arctan2(ystart, xstart)
                    
                    #loop over the cells that need to be attached to each old end
                    for cind in range(0, self.degree-1):
                        newTheta = oldTheta + deltaTheta*(cind - (self.degree-2)/2.) #polar coordinate of the end point
                        
                        xend = radius*numpy.cos(newTheta)
                        yend = radius*numpy.sin(newTheta)
                        
                        armLength = numpy.sqrt((xend-xstart)**2 + (yend-ystart)**2) #length that the cell has to fit in
                        armTheta = numpy.arctan2(yend-ystart, xend-xstart) #angle that the cell has to be at
                        
                        tempRes = numpy.copy(self.cellResonators)
                        tempRes = tempRes*armLength/self.side #rescale to the right length
                        tempRes = rotate_resonators(tempRes, armTheta) #rotate into poition
                        tempRes = shift_resonators(tempRes, xstart,ystart) #shift into position
                        
                        #store them away
                        newRes[endInd:endInd+1*self.cellSites,:] = tempRes
                        newEndThetas[endInd/self.cellSites] = newTheta #store the absolute polar coorinate of this arm
                        endInd = endInd +self.cellSites
                self.resDict[itt] = newRes
                totalSize = totalSize + newEnds
                oldEndThetas = newEndThetas 
                 
        #shuffle resoantor dictionary into an array                       
        self.resonators = numpy.zeros((totalSize*self.cellSites, 4))   
        currRes = 0
        for itt in range(1, maxItt+1):
            news = self.resDict[itt]
            numNews = news.shape[0]
            self.resonators[currRes:currRes+numNews,:] = news
            currRes = currRes + numNews
            
        
        self.coords = self.get_coords(self.resonators)
