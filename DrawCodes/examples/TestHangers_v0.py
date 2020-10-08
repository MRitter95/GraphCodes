#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:19:45 2020

@author: kollar2
"""

import pylab
import numpy
from scipy import signal
import scipy
import pickle
import sys
import os.path
import matplotlib.gridspec as gridspec
import scipy.io as sio
import fractions
import time
import matplotlib.pyplot as plt


#KollarLabClassPath = r'/Users/kollar2/Documents/KollarLab/MainClasses/'
#if not KollarLabClassPath in sys.path:
    #sys.path.append(KollarLabClassPath)
    
#pkgDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#
#if not pkgDir in sys.path:
#    sys.path.append(pkgDir)
#
#import context

#from CDSconfig import CDSconfig

   
#from GeneralLayoutGenerator import GeneralLayout
#from GeneralLayoutGenerator import TreeResonators
#
#from EuclideanLayoutGenerator2 import UnitCell
#from EuclideanLayoutGenerator2 import EuclideanLayout
#
#from LayoutGenerator5 import PlanarLayout
#
#
#from GeneralLayoutGenerator import split_resonators
#from GeneralLayoutGenerator import rotate_resonators
#from GeneralLayoutGenerator import generate_line_graph
#from GeneralLayoutGenerator import shift_resonators
##from GeneralLayoutGenerator import decorate_layout
#
#from CNT import CNT



#import context

import DrawCodes.sdxf as sdxf

import random

#import ezdxf
from DrawCodes.MaskMakerPro import *
from math import sin,cos,pi,floor,asin,acos,tan,atan,sqrt

import re
from scipy import *
from DrawCodes.alphanum import alphanum_dict
from random import randrange





class layout(Chip):
    
    # chipSize = 24592.
#    chipSize = 22650.
    chipSize = 7000.
    
    def __init__(self,name,size=(chipSize,chipSize),mask_id_loc=(0,0),chip_id_loc=(0,0)):
        
        Chip.__init__(self,name,size,mask_id_loc,chip_id_loc)
        
        #####################################################################################################################
        "Make bounding box and define chip parameters"        
        #####################################################################################################################
        
        smallChipSize = 7000.
        smallChipGap = 350.
        innerEdgeBuffer = 175.
        outerEdgeBuffer = 300.

        "Creates the maskborder box"
        maskborder=350. #How much larger than the chip border        
        border=Structure(self,start=self.bottomleft_corner,color=5,layer="maskborder")
        box=[   (self.bottomleft_corner[0],self.bottomleft_corner[1]),
                (self.bottomright_corner[0],self.bottomright_corner[1]),
                (self.topright_corner[0],self.topright_corner[1]),
                (self.topleft_corner[0],self.topleft_corner[1]),
                (self.bottomleft_corner[0],self.bottomleft_corner[0])
                ]
        border.append(sdxf.PolyLine(box,layer=border.layer,color=border.color))   

        chipSpacing = smallChipSize + smallChipGap

#        chipStarts = [(self.topleft_corner[0]+outerEdgeBuffer+innerEdgeBuffer,self.bottomleft_corner[1]+outerEdgeBuffer+innerEdgeBuffer),
#                    # (self.topleft_corner[0]+outerEdgeBuffer+innerEdgeBuffer,self.bottomleft_corner[1]+outerEdgeBuffer+innerEdgeBuffer),
#                    (self.bottomleft_corner[0]+outerEdgeBuffer+innerEdgeBuffer+chipSpacing,self.bottomleft_corner[1]+outerEdgeBuffer+innerEdgeBuffer),
#                    # (self.bottomleft_corner[0]+outerEdgeBuffer+innerEdgeBuffer+chipSpacing,self.bottomleft_corner[1]+outerEdgeBuffer+innerEdgeBuffer),
#                    (self.bottomleft_corner[0]+outerEdgeBuffer+innerEdgeBuffer+2*chipSpacing,self.bottomleft_corner[1]+outerEdgeBuffer+innerEdgeBuffer),
#                    # (self.bottomleft_corner[0]+outerEdgeBuffer+innerEdgeBuffer+2*chipSpacing,self.bottomleft_corner[1]+outerEdgeBuffer+innerEdgeBuffer),
#                    (self.bottomleft_corner[0]+outerEdgeBuffer+innerEdgeBuffer,self.bottomleft_corner[1]+outerEdgeBuffer+innerEdgeBuffer+chipSpacing),
#                    (self.bottomleft_corner[0]+outerEdgeBuffer+innerEdgeBuffer+chipSpacing,self.bottomleft_corner[1]+outerEdgeBuffer+innerEdgeBuffer+chipSpacing),
#                    (self.bottomleft_corner[0]+outerEdgeBuffer+innerEdgeBuffer+2*chipSpacing,self.bottomleft_corner[1]+outerEdgeBuffer+innerEdgeBuffer+chipSpacing),
#                    (self.bottomleft_corner[0]+outerEdgeBuffer+innerEdgeBuffer,self.bottomleft_corner[1]+outerEdgeBuffer+innerEdgeBuffer+2*chipSpacing),
#                    (self.bottomleft_corner[0]+outerEdgeBuffer+innerEdgeBuffer+chipSpacing,self.bottomleft_corner[1]+outerEdgeBuffer+innerEdgeBuffer+2*chipSpacing),
#                    (self.bottomleft_corner[0]+outerEdgeBuffer+innerEdgeBuffer+2*chipSpacing,self.bottomleft_corner[1]+outerEdgeBuffer+innerEdgeBuffer+2*chipSpacing)]
        
#        chipStarts = [(self.topleft_corner[0]+outerEdgeBuffer+innerEdgeBuffer,self.bottomleft_corner[1]+outerEdgeBuffer+innerEdgeBuffer)]
        chipStarts = [(self.topleft_corner[0],self.bottomleft_corner[1])]
        
        for startPos in chipStarts:
            border=Structure(self,start=startPos,color=3,layer="chipborder")
            box=[   (startPos[0],startPos[1]),
                    (startPos[0]+smallChipSize,startPos[1]),
                    (startPos[0]+smallChipSize,startPos[1]+smallChipSize),
                    (startPos[0],startPos[1]+smallChipSize),
                    (startPos[0],startPos[1])]
            border.append(sdxf.PolyLine(box,layer=border.layer,color=border.color))
        




        pos=[0,0]    #initial position of device
        chipborder=24592  
        chipX=chipborder #Length of device
        chipY=chipborder #Width of device
        cut_border=chipborder + 350
        edgeBufferDistance = 850

    
        """
        CPW Parameters: defines the basic geometry of the resonators 
        """ 
        #default CPW
        default_pinw_Z0 = 11.3
        default_gapw_Z0 = 5.54
        
        pinw_Z0 = default_pinw_Z0/2
        gapw_Z0 = default_gapw_Z0/2
        scalefactor=1.
        pinwLinear=scalefactor*pinw_Z0
        gapwLinear=scalefactor*gapw_Z0
        x = 3000.
        r = 90.
        num_wiggles = 4
        up = 1
        

        "Cap Parameters"     
        cap_length = 80.0
        cap_gap = 2.0

        scalefactor=1.5
        pinw=scalefactor*pinw_Z0
        gapw=scalefactor*gapw_Z0
        stop_gapw=10*gapw
        stop_pinw=8*pinw
        gapw_buffer = 0.0
        cap_gap_out = 100        
        cap_gap_ext = 0 #This values increases the gap for the outer capacitors
        
        "Bondpad Parameters"        
        bond_pad_length=350. #Length of Rectangular portion of bond pad
        launcher_scalefactor = 30
        launcher_pinw=launcher_scalefactor*default_pinw_Z0  #refer to default so that this stays the same size 
        launcher_gapw=launcher_scalefactor*default_gapw_Z0     
#        taper_length= 300.
        taper_length= 300.*2
        launcher_padding = launcher_gapw
        
        # Parameters for Bond Pads
        edgeOffset = 1255
        sideShift = 2158
        centerShift = 11866
        launcher_padding = launcher_gapw
        
        
        

        stub_length = 100.
        extension_length = 30.
        chipCornerToBondPad = [1200,1200]
        
        defaultRadius = 100. #the name of this variable is important. Do not change it. It functions as a global.
        
        
        
        
        #feedline properties
        feedlineRadius = 120.
        feedlinePinw = pinwLinear
        feedlineGapw = gapwLinear
        
        
        
        
        #I think that these belong to the old test chip with 8 hangers
        couplingSpacing = 2.8 #smaller than CW gap
#        couplingSpacing = gapw_Z0 #CPW gap I think. But this makes the gap double
        couplingSpacing = 0.
        couplingGap = 40.
        couplingStraight = 80.
        resonatorSpacing = 500.
        resonatorStraight1 = 1700.
        resonatorStraight2 = 1350.
        targetLength = 4000.
        radius = defaultRadius #the name of this variable is important. Do not change it. It functions as a global.

        showLattice=0

        scalefactorSmall=2.0/10
        pinwSmall=scalefactorSmall*pinw_Z0
        gapwSmall=scalefactorSmall*gapw_Z0
        
        

        scalefactorLarge=3.0/10
        pinwLarge=scalefactorLarge*pinw_Z0
        gapwLarge=scalefactorLarge*gapw_Z0  
        
        
        
        


        def drawTightHanger(structure, start, startDirection, leftRight, pinw, gapw, couplingGap, couplingStraight, resonatorStraight1, meanderSize, turnEdge, targetLength, hangerRadius = 100):
            s = structure
            s.last = start
            s.last_direction = startDirection

            CPWStraight(s,couplingGap,0,gapw+pinw/2)
            CPWStraight(s,couplingStraight, pinw,gapw)
            CPWBend(s,-leftRight*startDirection,pinw,gapw,hangerRadius)
            br_base = 30
            br_width = 50
            CPWStraight(mask,resonatorStraight1,pinw,gapw) 
            accumulatedLength = couplingStraight + (pi/2)*hangerRadius + resonatorStraight1
            isign = -1
            
            ind = 1
            while accumulatedLength<targetLength:
                print(ind)
                #make the first turn to parallel to the feedline
                if (accumulatedLength+pi*hangerRadius/2)>targetLength:
                    angle = isign*((targetLength-accumulatedLength)/hangerRadius)* 180/pi
                else:
                    angle = isign*90
                CPWBend(mask,angle,pinw,gapw,hangerRadius)
                accumulatedLength = accumulatedLength + hangerRadius*abs(angle*pi/180)
                
                #start the meander
                if (accumulatedLength+meanderSize)>targetLength:
                    delta = targetLength-accumulatedLength
                else:
                    delta = meanderSize
                CPWStraight(mask,delta,pinw,gapw)
                accumulatedLength = accumulatedLength + delta
                
                #turn at the end of the first meander
                if (accumulatedLength+pi*hangerRadius/2)>targetLength:
                    angle = -isign*((targetLength-accumulatedLength)/hangerRadius)* 180/pi
                else:
                    angle = -isign*90
                CPWBend(mask,angle,pinw,gapw,hangerRadius)
                accumulatedLength = accumulatedLength + hangerRadius*abs(angle*pi/180)
                print(('after angle:', angle, 'accumulatedLength',accumulatedLength))

                #do the straight in the meander
                isign = isign*-1
                if (accumulatedLength+turnEdge)>targetLength:
                    delta = targetLength-accumulatedLength
                else:
                    delta = turnEdge
                CPWStraight(mask,delta,pinw,gapw) 
                accumulatedLength = accumulatedLength + delta
                print(('after delta:', delta, 'accumulatedLength',accumulatedLength))
                
                ind = ind +1
            print('             ')
        
        def draw_ParallelStraighTestChip(step = 150):
                #parallel straight test chip
#                if idx == 3:
#                    step = 300
#                if idx == 4:
#                    step = 250
#                if idx == 5:
#                    step = 200
#                step = 200
#                step = 150
                
                #draw the feedline
                startPtBondPad = bondPadPositions[stopBondPads[-1]]
                mask.last = current_pos
                endPtBondPad = mask.last
                i = stopBondPads[0]
                j = stopBondPads[1]
                CPWStraight(mask,4000,feedlinePinw,feedlineGapw)
                CPWBend(mask,-90,feedlinePinw,feedlineGapw,feedlineRadius)  
                CPWStraight(mask,bondPadPositions[j][0] - bondPadPositions[i][0]-2*feedlineRadius,feedlinePinw,feedlineGapw)
                CPWBend(mask,-90,feedlinePinw,feedlineGapw,feedlineRadius)  
                CPWStraight(mask,4000,feedlinePinw,feedlineGapw)
                mask.last = (mask.last[0], mask.last[1]-(endPtBondPad[1]-startPtBondPad[1]))
                
                
                #setup for the hangers
                leftRight = 1
                scalefactor= 1.
                pinw=scalefactor*pinw_Z0
                gapw=scalefactor*gapw_Z0
                startDirection = -90
                start=  (endPtBondPad[0]+(feedlinePinw/2+feedlineGapw+pinw/2+couplingSpacing),endPtBondPad[1]+resonatorSpacing+1100)
                
                
                baseLength = 7000/2.
                jog = 400
                
                #resonator 1 : lower left
                resLength = baseLength
                leftRight = 1
                meanderSize = 300.
                turnEdge = 0.
                nbends = 5
                resonatorStraight1 = resLength -couplingStraight - radius * numpy.pi/2 - (nbends-1)*turnEdge - nbends*meanderSize - nbends*2*radius * numpy.pi/2
                start=  (endPtBondPad[0]+(feedlinePinw/2+feedlineGapw+pinw/2+couplingSpacing),endPtBondPad[1]+resonatorSpacing+400)
                drawTightHanger(mask, start, startDirection, leftRight, pinw, gapw, couplingGap, couplingStraight, resonatorStraight1, meanderSize, turnEdge, resLength, hangerRadius = radius)
                
                #resonator 2 : middle left
                resLength = baseLength +step
                leftRight = 1
                meanderSize = 300.
                turnEdge = 12.5
                nbends = 5
                resonatorStraight1 = resLength -couplingStraight - radius * numpy.pi/2 - (nbends-1)*turnEdge - nbends*meanderSize - nbends*2*radius * numpy.pi/2
                start=  (endPtBondPad[0]+(feedlinePinw/2+feedlineGapw+pinw/2+couplingSpacing),endPtBondPad[1]+resonatorSpacing+1400)
                drawTightHanger(mask, start, startDirection, leftRight, pinw, gapw, couplingGap, couplingStraight, resonatorStraight1, meanderSize, turnEdge, resLength, hangerRadius = radius)
                
                #resonator 3 : upper left
                resLength = baseLength +2*step
                leftRight = 1
                meanderSize = 300.
                turnEdge = 25.
                nbends = 5
                resonatorStraight1 = resLength -couplingStraight - radius * numpy.pi/2 - (nbends-1)*turnEdge - nbends*meanderSize - nbends*2*radius * numpy.pi/2
                start=  (endPtBondPad[0]+(feedlinePinw/2+feedlineGapw+pinw/2+couplingSpacing),endPtBondPad[1]+resonatorSpacing+2400)
                drawTightHanger(mask, start, startDirection, leftRight, pinw, gapw, couplingGap, couplingStraight, resonatorStraight1, meanderSize, turnEdge, resLength, hangerRadius = radius)
                
                #resonator 4 : upper left
                resLength = baseLength +3*step
                leftRight = 1
                meanderSize = 300.
                turnEdge = 50.
                nbends = 5
                resonatorStraight1 = resLength -couplingStraight - radius * numpy.pi/2 - (nbends-1)*turnEdge - nbends*meanderSize - nbends*2*radius * numpy.pi/2
                start=  (endPtBondPad[0]+(feedlinePinw/2+feedlineGapw+pinw/2+couplingSpacing),endPtBondPad[1]+resonatorSpacing+3400)
                drawTightHanger(mask, start, startDirection, leftRight, pinw, gapw, couplingGap, couplingStraight, resonatorStraight1, meanderSize, turnEdge, resLength, hangerRadius = radius)
                
                
                #resonator 5 : upper right
                resLength = baseLength +4*step
                leftRight = -1
                meanderSize = 300.
                nbends = 5.
                turnEdge = 100.
                resonatorStraight1 = resLength -couplingStraight - radius * numpy.pi/2 - (nbends-1)*turnEdge - nbends*meanderSize - nbends*2*radius * numpy.pi/2
                if step == 150:
                    start=  (endPtBondPad[0]-(feedlinePinw/2+feedlineGapw+pinw/2)+bondPadPositions[j][0] - bondPadPositions[i][0]-couplingSpacing,endPtBondPad[1]+resonatorSpacing+2900)
                else:
                    start=  (endPtBondPad[0]-(feedlinePinw/2+feedlineGapw+pinw/2)+bondPadPositions[j][0] - bondPadPositions[i][0]-couplingSpacing,endPtBondPad[1]+resonatorSpacing+2200-125)
                drawTightHanger(mask, start, startDirection, leftRight, pinw, gapw, couplingGap, couplingStraight, resonatorStraight1, meanderSize, turnEdge, resLength, hangerRadius = radius)
                
                #resonator 6 : middle right
                resLength = baseLength +5*step
                leftRight = -1
                meanderSize = 300.
                turnEdge = 200.
                nbends = 5
                resonatorStraight1 = resLength -couplingStraight - radius * numpy.pi/2 - (nbends-1)*turnEdge - nbends*meanderSize - nbends*2*radius * numpy.pi/2
                if step == 150:
                    start=  (endPtBondPad[0]-(feedlinePinw/2+feedlineGapw+pinw/2)+bondPadPositions[j][0] - bondPadPositions[i][0]-couplingSpacing,endPtBondPad[1]+resonatorSpacing+1900)
                else:
                    start=  (endPtBondPad[0]-(feedlinePinw/2+feedlineGapw+pinw/2)+bondPadPositions[j][0] - bondPadPositions[i][0]-couplingSpacing,endPtBondPad[1]+resonatorSpacing+1200-125)
                drawTightHanger(mask, start, startDirection, leftRight, pinw, gapw, couplingGap, couplingStraight, resonatorStraight1, meanderSize, turnEdge, resLength, hangerRadius = radius)
                
                #resonator 7 : lower right
                resLength = baseLength +6*step
                leftRight = -1
                meanderSize = 300.
                if step == 150:
                    turnEdge = 250.
                else:
                    turnEdge = 300.
                nbends = 5
                resonatorStraight1 = resLength -couplingStraight - radius * numpy.pi/2 - (nbends-1)*turnEdge - nbends*meanderSize - nbends*2*radius * numpy.pi/2
                if step == 150:
                    start=  (endPtBondPad[0]-(feedlinePinw/2+feedlineGapw+pinw/2)+bondPadPositions[j][0] - bondPadPositions[i][0]-couplingSpacing,endPtBondPad[1]+resonatorSpacing+900)
                else:
                    start=  (endPtBondPad[0]-(feedlinePinw/2+feedlineGapw+pinw/2)+bondPadPositions[j][0] - bondPadPositions[i][0]-couplingSpacing,endPtBondPad[1]+resonatorSpacing+125)
                drawTightHanger(mask, start, startDirection, leftRight, pinw, gapw, couplingGap, couplingStraight, resonatorStraight1, meanderSize, turnEdge, resLength, hangerRadius = radius)
                
                if step == 150:
                    resLength = baseLength +7*step
                    leftRight = -1
                    meanderSize = 300.
                    turnEdge = 75.
                    nbends = 5
                    resonatorStraight1 = resLength -couplingStraight - radius * numpy.pi/2 - (nbends-1)*turnEdge - nbends*meanderSize - nbends*2*radius * numpy.pi/2
                    start=  (endPtBondPad[0]-(feedlinePinw/2+feedlineGapw+pinw/2)+bondPadPositions[j][0] - bondPadPositions[i][0]-couplingSpacing,endPtBondPad[1]+resonatorSpacing-100)
                    drawTightHanger(mask, start, startDirection, leftRight, pinw, gapw, couplingGap, couplingStraight, resonatorStraight1, meanderSize, turnEdge, resLength, hangerRadius = radius)
                                    
                return
            
            
        def draw_quad_hanger_chip(step = 150, targetLength = 4000/2, feedlineOffset = 150, nbends = 3, meanderSize = 300, turnEdge = 100):
            
                
                #draw the feedline
                startPtBondPad = bondPadPositions[stopBondPads[-1]]
                mask.last = current_pos
                endPtBondPad = mask.last
                i = stopBondPads[0]
                j = stopBondPads[1]
#                CPWStraight(mask,4000,feedlinePinw,feedlineGapw)
#                CPWBend(mask,-90,feedlinePinw,feedlineGapw,feedlineRadius)  
#                CPWStraight(mask,bondPadPositions[j][0] - bondPadPositions[i][0]-2*feedlineRadius,feedlinePinw,feedlineGapw)
#                CPWBend(mask,-90,feedlinePinw,feedlineGapw,feedlineRadius)  
#                CPWStraight(mask,4000,feedlinePinw,feedlineGapw)
                CPWStraight(mask,feedlineOffset,feedlinePinw,feedlineGapw)
                CPWBend(mask,90,feedlinePinw,feedlineGapw,feedlineRadius)  
                CPWStraight(mask,bondPadPositions[j][1] - bondPadPositions[i][1]-2*feedlineRadius,feedlinePinw,feedlineGapw)
                CPWBend(mask,90,feedlinePinw,feedlineGapw,feedlineRadius)  
                CPWStraight(mask,feedlineOffset,feedlinePinw,feedlineGapw)
                mask.last = (mask.last[0], mask.last[1]-(endPtBondPad[1]-startPtBondPad[1]))
                
                
                #setup for the hangers
                leftRight = 1
                scalefactor= 1.
                pinw=scalefactor*pinw_Z0
                gapw=scalefactor*gapw_Z0
                startDirection = -90
                start=  (endPtBondPad[0]+(feedlinePinw/2+feedlineGapw+pinw/2+couplingSpacing),endPtBondPad[1]+resonatorSpacing+1100)
                
                
                baseLength = targetLength #targeted resonator size
#                jog = 500 #offset between resonators along the feedline
                
                #resonator 1 : lower left
                resLength = baseLength
                resonatorStraight1 = resLength -couplingStraight - radius * numpy.pi/2 - (nbends-1)*turnEdge - nbends*meanderSize - nbends*2*radius * numpy.pi/2
                start=  (endPtBondPad[0]+feedlineOffset + feedlineRadius + (feedlinePinw/2+feedlineGapw+pinw/2+couplingSpacing),endPtBondPad[1]+resonatorSpacing+200)
                drawTightHanger(mask, start, startDirection, leftRight, pinw, gapw, couplingGap, couplingStraight, resonatorStraight1, meanderSize, turnEdge, resLength, hangerRadius = radius)
                
                #resonator 2 : middle left
                resLength = baseLength +step
                resonatorStraight1 = resLength -couplingStraight - radius * numpy.pi/2 - (nbends-1)*turnEdge - nbends*meanderSize - nbends*2*radius * numpy.pi/2
                start=  (endPtBondPad[0]+feedlineOffset + feedlineRadius+(feedlinePinw/2+feedlineGapw+pinw/2+couplingSpacing),endPtBondPad[1]+resonatorSpacing+1400)
                drawTightHanger(mask, start, startDirection, leftRight, pinw, gapw, couplingGap, couplingStraight, resonatorStraight1, meanderSize, turnEdge, resLength, hangerRadius = radius)
                
                #resonator 3 : upper left
                resLength = baseLength +2*step
                resonatorStraight1 = resLength -couplingStraight - radius * numpy.pi/2 - (nbends-1)*turnEdge - nbends*meanderSize - nbends*2*radius * numpy.pi/2
                start=  (endPtBondPad[0]+feedlineOffset + feedlineRadius+(feedlinePinw/2+feedlineGapw+pinw/2+couplingSpacing),endPtBondPad[1]+resonatorSpacing+2600)
                drawTightHanger(mask, start, startDirection, leftRight, pinw, gapw, couplingGap, couplingStraight, resonatorStraight1, meanderSize, turnEdge, resLength, hangerRadius = radius)
                
                #resonator 4 : upper left
                resLength = baseLength +3*step
                resonatorStraight1 = resLength -couplingStraight - radius * numpy.pi/2 - (nbends-1)*turnEdge - nbends*meanderSize - nbends*2*radius * numpy.pi/2
                start=  (endPtBondPad[0]+feedlineOffset + feedlineRadius+(feedlinePinw/2+feedlineGapw+pinw/2+couplingSpacing),endPtBondPad[1]+resonatorSpacing+3800)
                drawTightHanger(mask, start, startDirection, leftRight, pinw, gapw, couplingGap, couplingStraight, resonatorStraight1, meanderSize, turnEdge, resLength, hangerRadius = radius)
                
                   
                return
            

        #####################################################################################################################
        "Start laying out resonators"
        #####################################################################################################################
        
        mask=Structure(self,start=0)
        mask.defaults = {'pinw':pinw,'gapw':gapw,'bendradius':r} #Set these as defaults for simplified inherited input
   
        # Draw some alignment marks
        edgeBuffer = 530
        # drawAlignmentMarks(mask,20,[520,520],edgeBuffer,chipborder-300,pos)
       
        mid=(1250,-7000+1400-200) 

        mask.last = mid

#        # Parameters for Bond Pads
#        edgeOffset = 1255
#        sideShift = 2158
#        centerShift = 11866
#        launcher_padding = launcher_gapw
#        
#
#        couplingSpacing = 2.8 #smaller than CW gap
##        couplingSpacing = gapw_Z0 #CPW gap I think. But this makes the gap double
#        couplingSpacing = 0.
#        couplingGap = 40.
#        couplingStraight = 80.
#        resonatorSpacing = 500.
#        resonatorStraight1 = 1700.
#        resonatorStraight2 = 1350.
#        targetLength = 4000.
#        radius = defaultRadius #the name of this variable is important. Do not change it. It functions as a global.
#
#        showLattice=0
#
#        scalefactorSmall=2.0/10
#        pinwSmall=scalefactorSmall*pinw_Z0
#        gapwSmall=scalefactorSmall*gapw_Z0
#        
#        feedlineRadius = 120.
#        feedlinePinw = pinwLinear
#        feedlineGapw = gapwLinear
#
#        scalefactorLarge=3.0/10
#        pinwLarge=scalefactorLarge*pinw_Z0
#        gapwLarge=scalefactorLarge*gapw_Z0        

#        for idx in range(9):
        for idx in range(1):
            # startVertices = bondPadConnectionParameters['startVertices']
            
            bufferDistance = 1;
            
#            # define potential bond pad positions
#            bondPadConnectionParameters = {'startVertices':[0,1],'resonatorIndices':[0,2],'stopBondPad':[0,1],'directLines':[1,1],'pinwLinear':pinwLinear,'gapwLinear':gapwLinear}
#            stopBondPads = bondPadConnectionParameters['stopBondPad']
#            directLines = bondPadConnectionParameters['directLines']
#            startResonators = bondPadConnectionParameters['resonatorIndices']
#            lowerLeftChipCorner = chipStarts[idx]
#            bondPadPositions = [[lowerLeftChipCorner[0]+chipCornerToBondPad[0],lowerLeftChipCorner[1]+chipCornerToBondPad[1]],
#                                [lowerLeftChipCorner[0]+smallChipSize-chipCornerToBondPad[0],lowerLeftChipCorner[1]+chipCornerToBondPad[1]],
#                                [lowerLeftChipCorner[0]+smallChipSize-chipCornerToBondPad[0],lowerLeftChipCorner[1]+smallChipSize-chipCornerToBondPad[1]],
#                                [lowerLeftChipCorner[0]+chipCornerToBondPad[0],lowerLeftChipCorner[1]+smallChipSize-chipCornerToBondPad[1]]]
#            bondPadAngles = [90,90,270,270]
#            
#            for idy in range(len(bondPadPositions)):
#
#                # find where that bond pad is
#                bondPadPos = bondPadPositions[idy]
#                bondPadAng = bondPadAngles[idy]
#                bondPadEndPos = [bondPadPos[0]+(bond_pad_length+launcher_padding+taper_length)*numpy.cos(bondPadAng*numpy.pi/180.),bondPadPos[1]+(bond_pad_length+launcher_padding+taper_length)*numpy.sin(bondPadAng*numpy.pi/180.)]
#
#                # figure out if that bond pad is connected
#                if idy in stopBondPads:
#
#                    drawBondPad(mask,bondPadPos,bondPadAng,pinwLinear,gapwLinear,bond_pad_length,launcher_pinw,launcher_gapw,taper_length,launcher_padding)
#                    if idy == stopBondPads[0]:
#                        current_pos = mask.last
#                    index = stopBondPads.index(idy)
#                    # print('idx',idx,'stopBondPads',stopBondPads[index],'startVertex',startVertex)
            
            
            # define potential bond pad positions
            bondPadConnectionParameters = {'startVertices':[0,1],'resonatorIndices':[0,2],'stopBondPad':[0,3],'directLines':[1,1],'pinwLinear':pinwLinear,'gapwLinear':gapwLinear}
            stopBondPads = bondPadConnectionParameters['stopBondPad']
            directLines = bondPadConnectionParameters['directLines']
            startResonators = bondPadConnectionParameters['resonatorIndices']
            lowerLeftChipCorner = chipStarts[idx]
            bondPadPositions = [[lowerLeftChipCorner[0]+chipCornerToBondPad[0] - 350,lowerLeftChipCorner[1]+chipCornerToBondPad[1]],
                                [lowerLeftChipCorner[0]+smallChipSize-chipCornerToBondPad[0],lowerLeftChipCorner[1]+chipCornerToBondPad[1]],
                                [lowerLeftChipCorner[0]+smallChipSize-chipCornerToBondPad[0],lowerLeftChipCorner[1]+smallChipSize-chipCornerToBondPad[1]],
                                [lowerLeftChipCorner[0]+chipCornerToBondPad[0] -350,lowerLeftChipCorner[1]+smallChipSize-chipCornerToBondPad[1]]]
            bondPadAngles = [0,90,270,0]
            
            for idy in range(len(bondPadPositions)):

                # find where that bond pad is
                bondPadPos = bondPadPositions[idy]
                bondPadAng = bondPadAngles[idy]
                bondPadEndPos = [bondPadPos[0]+(bond_pad_length+launcher_padding+taper_length)*numpy.cos(bondPadAng*numpy.pi/180.),bondPadPos[1]+(bond_pad_length+launcher_padding+taper_length)*numpy.sin(bondPadAng*numpy.pi/180.)]

                # figure out if that bond pad is connected
                if idy in stopBondPads:

                    drawBondPad(mask,bondPadPos,bondPadAng,pinwLinear,gapwLinear,bond_pad_length,launcher_pinw,launcher_gapw,taper_length,launcher_padding)
                    if idy == stopBondPads[0]:
                        current_pos = mask.last
                    index = stopBondPads.index(idy)
                    # print('idx',idx,'stopBondPads',stopBondPads[index],'startVertex',startVertex)
                    
                    
            #######################        
            #do each individual 7x7
            #######################
            
            
            targetFreq = 4.*10**9
            epsilonEff = 6.35
            
            targetLength_m = 3*10**8/targetFreq/numpy.sqrt(epsilonEff)/4 
            targetLength_um = targetLength_m*10**6
            #!!!!!!!!!!!
            #I reverse engineered this from an online widget. It should be double checked

            #fill in the chips
#            draw_ParallelStraighTestChip(step = 150)
#            draw_quad_hanger_chip(step = 150, targetLength = targetLength_um , nbends = 7, meanderSize = 450, turnEdge = 200)
            draw_quad_hanger_chip(step = 250, targetLength = targetLength_um , nbends = 7, meanderSize = 450, turnEdge = 50)


if __name__=="__main__":
        chip = layout('A')
        mask=sdxf.Drawing()
        mask.blocks.append(chip)
        mask.append(sdxf.Insert(chip.name,point=(0,0)))
        filename = 'quadHanger_v8.dxf'
        "Name the output file here"
        mask.saveas(filename)


