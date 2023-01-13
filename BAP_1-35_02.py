# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 10:39:21 2021

@author: PerGeos
"""

# pip install mplstereonet
# pip install spherecluster
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import os
import statistics as stat
import itertools
from scipy import stats
import decimal
import ast
import mplstereonet
from matplotlib import cm
import scipy
from scipy.spatial import SphericalVoronoi, geometric_slerp
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import patches
from matplotlib.path import Path
from matplotlib import tri
import math
import random
import time
import mpl_toolkits.mplot3d.art3d as art3d
# from spherecluster import SphericalKMeans

def subcon_init(input_dat):
    """
    Parameters
    ----------
    input_dat : Must be binarized imgdat.
    
    Returns
    -------
    init_output : 
    A structure to feed to connected_melt_calc, with most avizo modules and names of objects that are needed for each. 
    """
    #INP_DATA = hx_project.get('HugeSubvolumeXDirection_1')
    INP_DATA = hx_project.load(input_dat)
    inpdataname = INP_DATA.name
    DATA = hx_project.get(inpdataname)##Grab the name of the loaded data, assign to DATA
    (SUBVOLUME_RED, SUBVOLUME_OUTPUT) = subvolume_creation(DATA)##create an "extract subvolume" module, needed to then make the melt fraction objects and such
    CONRED = axis_concreate(SUBVOLUME_OUTPUT)##Create the axis connectivity module and grab its setting module
    constr = (CONRED.downstream_connections[0].get_owner().name)
    CONOUTPUT = hx_project.get(constr)
    MELTFRACOBJ = melt_fractioncreate(SUBVOLUME_OUTPUT)
    meltfracstr = (MELTFRACOBJ.downstream_connections[0].get_owner().name)##grab the name of the melt fraction output by using downstream connection
    MELTFRACOUTPUT = hx_project.get(meltfracstr)##assign this
    TEMP_SPREADSHEET = spreadsheetcreate(MELTFRACOUTPUT)##make table and attach to subvolume melt fraction (should never be zero so should be fine)
    tablestr = (TEMP_SPREADSHEET.downstream_connections[0].get_owner().name)
    conDF = init_DF()##initialize the dataframe to output
    init_output = (conDF, tablestr, meltfracstr, constr, SUBVOLUME_RED, TEMP_SPREADSHEET, MELTFRACOUTPUT, CONOUTPUT, CONRED, DATA, inpdataname, SUBVOLUME_OUTPUT, MELTFRACOBJ)
    return(init_output)##return these strings, assigned objects, and the dataframe into an object


def subvolume_creation(DATA):
    """
    Parameters
    ----------
    DATA : Input binary img data
        This must be an input binarized imgfile, with units (use unit editor in Avizo if there are none to assign them).

    Returns
    -------
    SUBVOLUME_RED : The red extract subvolume module created within Avizo.
    SUBVOLUME_OUTPUT : The output attached the the extract subvolume module, this is generally what is wanted (the imgdata region)
    subvolstr : The name of the subvolume; this is needed sometimes for saving the subvolume. 

    """
    SUBVOLUME_RED = hx_project.create('HxLatticeAccess')
    SUBVOLUME_RED.ports.data.connect(DATA)##connect the extract subvolume module to the input imgdat
    SUBVOLUME_RED.ports.action.was_hit = True##click 'apply'
    SUBVOLUME_RED.fire()##finish clicking apply
    SUBVOLUME_OUTPUT = hx_project.get((SUBVOLUME_RED.downstream_connections[0].get_owner().name))##grab the object with the name of the output subvolume....
    return(SUBVOLUME_RED, SUBVOLUME_OUTPUT)


def subvolume_update(SUBVOLUME_RED, SUBVOLUME_OUTPUT, xinit1, yinit1, zinit1, x_dim, y_dim, z_dim):
    """
    Parameters
    ----------
    SUBVOLUME_RED : Avizo 'extract subvolume' module
        Output by the subvolume_creation function, this is what extracts a region of the imgset.
    SUBVOLUME_OUTPUT : The region of the imgsample.
        This is also an output from the subvolume_creation function, and is the img/labeldata region.
    xinit1 : Integer.
        This number defines the starting X value for the subvolume (all numbers are in pixel-count units)
    yinit1 : Integer.
        This number defines the starting Y value for the subvolume.
    zinit1 : Integer.
        This number defines the starting Z value for the subvolume.
    x_dim : Integer.
        This is the length of the subvolume in the X direction.
    y_dim : Integer.
        This is the length of the subvolume in the Y direction.
    z_dim : Integer.
        This is the length of the subvolume in the Z direction.

    Returns
    -------
    SUBVOLUME_OUTPUT : This is the now updated (based on xinit, xdim, et cet) subvolume. 
    In general, it doesn't have to be returned, since the name is unchanged and the module was updated.

    """
    port_float_text_n = SUBVOLUME_RED.ports.boxMin##set port_float_texts_n to the starting coordinate boxes.
    port_float_text_n.texts[0].value = xinit1##set each box to the respective value.
    port_float_text_n.texts[1].value = yinit1
    port_float_text_n.texts[2].value = zinit1
    port_float_text_m = SUBVOLUME_RED.ports.boxSizeUnits ##set subvolume size using dims
    port_float_text_m.texts[0].value = x_dim
    port_float_text_m.texts[1].value = y_dim
    port_float_text_m.texts[2].value = z_dim
    SUBVOLUME_RED.ports.action.was_hit = True##This and next line = update the subvolume. 
    SUBVOLUME_RED.fire()
    SUBVOLUME_OUTPUT = hx_project.get(SUBVOLUME_RED.downstream_connections[0].get_owner().name)##grab the output again for prosperity's sake
    SUBVOLUME_RED.viewer_mask=0
    return(SUBVOLUME_OUTPUT)##output this


def melt_fractioncreate(SUBVOLUME_OUTPUT):
    """
    Parameters
    ----------
    SUBVOLUME_OUTPUT : Imgdata (.view file from subvolume_create).
        This is the output of an extract subvolume module in Avizo. 
        This is necessary to attach the volume fraction object to, so melt fraction can be measured. 
    Returns
    -------
    MELTFRACOBJ : Volume fraction module settings.
        This is the 'volume fraction' settings (red) module.  
    """
    meltfracobj = hx_project.create('label_ratio')##create the object
    meltfracobj.ports.inputImage.connect(SUBVOLUME_OUTPUT)##connect it to the subvolume data
    meltfracobj.ports.doIt.was_hit = True##click apply
    meltfracobj.fire()
    return(meltfracobj)


def melt_fracupdate(INPUTREGION, MELTFRACOBJ):
    """
    Parameters
    ----------
    INPUTREGION : Imgdata.
        Desired object to measure pore fraction of (can be connectivity output as well).
    MELTFRACOBJ : Volume fraction red settings module.
        Produced by melt_fractioncreate, this is the object that attaches to regions to compute melt fraction. 
    Returns
    -------
    MELTFRACOBJ : Returns the setting module that was input (redundant).
    """
    MELTFRACOBJ.ports.inputImage.connect(INPUTREGION)##connect module to region
    MELTFRACOBJ.ports.doIt.was_hit = True##click apply
    MELTFRACOBJ.execute()##click apply
    MELTFRACOUTPUT = hx_project.get(MELTFRACOBJ.downstream_connections[0].get_owner().name)
    return(MELTFRACOUTPUT)##return this settings obj
    

def axis_concreate(SUBVOLUME_OUTPUT):
    """
    Parameters
    ----------
    SUBVOLUME_OUTPUT : Imgdata.
        The subvolume to measure connectivity from.
        
    Returns
    -------
    None.

    """
    CONRED = hx_project.create('HxAxisConnectivity')
    CONRED.ports.neighborhood.selected = 0
    CONRED.ports.data.connect(SUBVOLUME_OUTPUT)
    CONRED.ports.doIt.was_hit = True
    CONRED.fire()
    return(CONRED)


def axisconupdate(SUBVOLUME_OUTPUT, CONRED, con_direction):
    """
    Parameters
    ----------
    SUBVOLUME_OUTPUT : TYPE
        DESCRIPTION.
    CONRED : TYPE
        DESCRIPTION.
    constr : TYPE
        DESCRIPTION.
    con_direction : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    CONRED.ports.data.connect(SUBVOLUME_OUTPUT)
    CONRED.ports.orientation.selected = con_direction
    CONRED.ports.doIt.was_hit = True
    CONRED.fire()
    CONOUTPUT = hx_project.get(CONRED.downstream_connections[0].get_owner().name)
    return(CONOUTPUT)

def spreadsheetcreate(MELT_DATA):
    ##with input data, makes spreadsheet and generates an output from that
    temp_spreadsheet = hx_project.create('HxSpreadSheetExtract')
    temp_spreadsheet.ports.data.connect(MELT_DATA)
    temp_spreadsheet.execute()
    return(temp_spreadsheet)

def spreadsheetcreate2():
    ##with no input data, creates a spreadsheet module (no output until later attached to data object)
    temp_spreadsheet = hx_project.create('HxSpreadSheetExtract')
    return(temp_spreadsheet)


def update_spreadsheet2(TEMP_SPREADSHEET, MELTFRACOUTPUT):
    ##With a specified spreadsheet object, and data to grab (named meltfracoutput) this will attach the
    # spreadsheet module and grab the datapoint from it. If the datapoint (melt fraction) is a zero,
    # it cannot attach a spreadsheet(there is actually no output created) so it assumes 
    # a zero value if the spreadsheet table cannot be accessed
    TEMP_SPREADSHEET.ports.data.connect(MELTFRACOUTPUT)
    TEMP_SPREADSHEET.execute()
    melt_table = hx_project.get(TEMP_SPREADSHEET.downstream_connections[0].get_owner().name)
    melt_spreadsheet = melt_table.all_interfaces.HxSpreadSheetInterface
    try:
        MELT = melt_spreadsheet.tables[0].columns[1]
        data = MELT.asarray()
        datapoint = data[-1]
    except:
        datapoint = 0
    return(datapoint)

def connected_meltcalc(x_dim, y_dim, z_dim, xinterval, yinterval, zinterval, savedir, savename, init_output, cylinder_details, EDstructure):
    """
    Parameters
    ----------
    x_dim : Integer.
        Size (all in pixels) in the X direction of each subvolume.
    y_dim : Integer.
        Size (all in pixels) in the Y direction of each subvolume.
    z_dim : Integer.
        Size (all in pixels) in the Z direction of each subvolume.
    interval : Integer.
        How many pixels each subvolume steps (if 10, for example, the start of each subvolume will step by 10px, if this number
        = the dimensions there is no overlap, if this number is less than dimensions there is overlap, if it is greater, there
        is a gap between subvolumes).
    savedir : Full path (string).
        This specifies where the dataframe and such will be saved.
    savename : String, assigned to dataframe.
        This string is the label of the .csv sheet that contains the dataframe, with all coordinate, melt data. The juicy data <3
    init_output : Structure from subcon_init.
        Just pass the structure here, the script grabs everything from it. Note if you changed what is input/ output of that script, it could interfer with this code!!!
    cylinder_details : Structure containing information about the cylindrical datascan.
        One of the more complicated inputs; this requires a tuple with the (cylinder radius, °of x rotation, °of y rotation, and ° of z rotation) that was applied
        to the binarized image scan (zeros can be input). Also, coord_condition assumes the height of the cylinder is 2*(cylinder radius) (true when I wrote this, for scan CQ0705)
        so if this is not true, something must be done to check the Z heights compared against 1/2 cylinder height and this should be passed to the 
        cylinder details inputvar.
    Returns
    -------
    This code saves the dataframe in path specified. If needing to trouble shoot, you can have an outputvar defined, or print things along the way. 
    """

    ##pull everything out of init_output
    (conDF, tablestr, meltfracstr, constr, SUBVOLUME_RED, TEMP_SPREADSHEET, MELTFRACOUTPUT, CONOUTPUT, CONRED, DATA, inpdataname, SUBVOLUME_OUTPUT, MELTFRACOBJ) = init_output
    # (radius_of_cylinder, x_rotation, y_rotation, z_rotation) = cylinder_details##grab cylinder details
    Databbox = DATA.bounding_box##grab total X, Y, Z size of inputdata
    bbox_1 = Databbox[1]
    bbox_0 = Databbox[0]
    center_of_cylinder = ((((bbox_1[0]-bbox_0[0])/(.16)))/2, (((bbox_1[1]-bbox_0[1])/(.16)))/2, (((bbox_1[2]-bbox_0[2])/(.16)))/2)##calculate cylinder center (origin)
    (begincoords) = scan_discretization(DATA, x_dim, y_dim, z_dim, xinterval, yinterval, zinterval)##permute every possible X, Y, Z triplet based on the size of the scan, subvolumes with desired dimensions, limited by the subvolume step interval.
    hyp_subvolume_count = (len(begincoords))##get number of total possible subvolumes
    index = 0##set beginning of loop to zero
    index_2 = index##set index2 to 0 as well
    x_begin_r = []##
    y_begin_r = []
    z_begin_r = []
    if EDstructure[0] == 'Y':
        inlet = hx_project.load(EDstructure[1])
        inlet = hx_project.get(inlet.name)
        outlet = inlet.duplicate()
        inlet = outlet.duplicate()
        hx_project.add(inlet)
        hx_project.add(outlet)
        inlet.name = 'inlet'
        inlet = hx_project.get('inlet')        
        outlet.name = 'outlet'
        outlet = hx_project.get('outlet')
        bboxinlet = inlet.bounding_box
        bboxbegins = bboxinlet[0]
        bboxends = bboxinlet[1]
        inletxlen = round(int(1+((bboxends[0]-bboxbegins[0])/.16)))
        inletylen = round(int(1+((bboxends[1]-bboxbegins[1])/.16)))
        inletzlen = round(int(1+((bboxends[2]-bboxbegins[2])/.16)))
    else:
        pass
    
    while index+1 < hyp_subvolume_count:##while you are still within the realm of possible subvolumes
        index = index_2
        index_2 = index_2 + 1##expand the index for next loop
        coord_iter = begincoords[index]##cram the X, Y, Z value for this iteration from scan_discretization output
        xbeg = coord_iter[0]##grab the x, y, and z val from this...
        ybeg = coord_iter[1]
        zbeg = coord_iter[2]
        xend=xbeg+x_dim
        yend=ybeg+y_dim
        zend=zbeg+z_dim
        bcords = (xbeg, ybeg, zbeg)##reassign this to input tuple
        ecords=(xend, yend, zend)
        bcords, ecords=coord_center_distance_calculator(center_of_cylinder, bcords, ecords)
        
        cylinder_details = np.asarray(cylinder_details)
        radius_of_cylinder=[]
        height_of_cylinder=[]
        x_rotation=[]
        y_rotation=[]
        z_rotation=[]
        number_of_rotations=np.shape(cylinder_details)
        number_of_rotations=number_of_rotations[0]
        for i in range(number_of_rotations):
            radius_of_cylinder.append(cylinder_details[i, 0])
            height_of_cylinder.append(cylinder_details[i,4])
            x_rotation.append(cylinder_details[i, 1])
            y_rotation.append(cylinder_details[i, 2])
            z_rotation.append(cylinder_details[i, 3])
        for i in range(number_of_rotations):
            (x_b, y_b, z_b, x_e, y_e, z_e) = back_rotate(radius_of_cylinder[i], x_rotation[i], y_rotation[i], z_rotation[i], bcords, ecords)
            
            bcords=(x_b, y_b, z_b)
            ecords=(x_e, y_e, z_e)
        
        condition = coord_condition(radius_of_cylinder[i], height_of_cylinder[i], bcords, ecords)
        
        
        ##Above line plugs these into coord_condition, which checks the distance of both Z values, and the vector to each cube corner, to make sure these are less than cylinder_radius away from center_of_cylinder
        if condition == ('keep'):##if we are told to keep the subvolume based on this condition
            x_begin_r.append(xbeg)##append the lovely output vectors with this information
            y_begin_r.append(ybeg)
            z_begin_r.append(zbeg)                
        elif condition == ('discard'):##if we are told to discard this triplet (part of the subvolume lies outside of the imgscan)
            continue##Then, continue. 
    conDF['Xi'] = (x_begin_r)##put X, Y, Z values in respective columns of DF
    conDF['Yi'] = (y_begin_r)
    conDF['Zi'] = (z_begin_r)

    if EDstructure[0] == 'Y':
        ##if you want extended domains, it forms them in the long-axis direction. The statements below 
        ##find the beginning of the larger subvolume the inlet/outlet will be added to, where the extended domain is precisely 
        ## twice the size of the original skinny rectangular prism
        for i in range(len(conDF)):
            if abs(x_dim) > abs(y_dim) and abs(x_dim) > abs(z_dim):
                conDF.loc[i, 'EDYi'] = int(round(conDF.loc[i, 'Yi'] - ((.25)*inletylen)))
                conDF.loc[i, 'EDZi'] = int(round(conDF.loc[i, 'Zi'] - ((.25)*inletzlen)))
                conDF.loc[i, 'EDXi'] = int(round(conDF.loc[i, 'Xi']))
                conDF.loc[i, 'EDdX'] = x_dim
                conDF.loc[i, 'EDdY'] = inletylen+1
                conDF.loc[i, 'EDdZ'] = inletzlen
            elif abs(y_dim) > abs(x_dim) and abs(y_dim) > abs(z_dim):
                conDF.loc[i, 'EDYi'] = int(round(conDF.loc[i, 'Yi']))
                conDF.loc[i, 'EDXi'] = int(round(conDF.loc[i, 'Xi'] - ((.25)*inletxlen)))
                conDF.loc[i, 'EDZi'] = int(round(conDF.loc[i, 'Zi'] - ((.25)*inletzlen)))
                conDF.loc[i, 'EDdX'] = inletxlen+1
                conDF.loc[i, 'EDdY'] = y_dim
                conDF.loc[i, 'EDdZ'] = inletzlen
            elif abs(z_dim) > abs(y_dim) and abs(z_dim) > abs(x_dim):
                conDF.loc[i, 'EDZi'] = int(round(conDF.loc[i, 'Zi']))
                conDF.loc[i, 'EDXi'] = int(round(conDF.loc[i, 'Xi'] - ((.25)*inletxlen)))
                conDF.loc[i, 'EDYi'] = int(round(conDF.loc[i, 'Yi'] - ((.25)*inletylen)))
                conDF.loc[i, 'EDdX'] = inletxlen+1
                conDF.loc[i, 'EDdY'] = inletylen+1
                conDF.loc[i, 'EDdZ'] = z_dim

    
    subvolume_count = (len(x_begin_r))##get # of subvolumes within imgscan that will be selected
    dX = [x_dim] * subvolume_count##create vectors of same length of subvolume count that have X, Y, Z dimension
    dY = [y_dim] * subvolume_count
    dZ = [z_dim] * subvolume_count
    conDF['dX'] = dX##put X, Y, Z lengths in respective columns as well
    conDF['dY'] = dY
    conDF['dZ'] = dZ

    index = 0##set beginning of loop to zero
    index_2 = index
    ##This loop will update subvolume settings, measure melt fraction and connected melt fraction for these, 
    ##and store this data in the vectors created at top of script
    while (index+1<subvolume_count):##while the index is less than desired number of subvolumes
        index = index_2
        index_2 = index_2 + 1
        xinit1 = conDF.iloc[index]['Xi']
        yinit1 = conDF.iloc[index]['Yi']
        zinit1 = conDF.iloc[index]['Zi']
        SUBVOLUME_RED, SUBVOLUME_OUTPUT = subvolume_creation(DATA)
        subvolume_update(SUBVOLUME_RED, SUBVOLUME_OUTPUT, xinit1, yinit1, zinit1, x_dim, y_dim, z_dim)
        ##the subvolume object has been created with default settings; update it with the right init (with one of the inits plus the interval),
        ##the right size, and update the object
        con_direction = 0
        axisconupdate(SUBVOLUME_OUTPUT, CONRED, con_direction)
            ##update the axis connectivity module with THIS iterations subvolume output
        melt_fracupdate(SUBVOLUME_OUTPUT, MELTFRACOBJ)
            ##update the melt fraction settings object to the subvolume
        TOTAL_MELT = update_spreadsheet2(TEMP_SPREADSHEET, MELTFRACOUTPUT)
            ##update the spreadsheet with the new melt fraction for total subvolume- this is total melt!
        melt_fracupdate(CONOUTPUT, MELTFRACOBJ)
            ##calculate this melt fraction    
        xCON = update_spreadsheet2(TEMP_SPREADSHEET, MELTFRACOUTPUT)
            ##calculate connected melt using the melt frac output which has now been assigned to the connectivity label img
        con_direction = 1##set con direction to Y
        axisconupdate(SUBVOLUME_OUTPUT, CONRED, con_direction)##update the connectivity module
        melt_fracupdate(CONOUTPUT, MELTFRACOBJ)##update the melt fraction with this connected melt
        yCON = update_spreadsheet2(TEMP_SPREADSHEET, MELTFRACOUTPUT)##update the spreadsheet to this model and grab the value, now named yCON
        con_direction = 2##repeat for the Z direction
        axisconupdate(SUBVOLUME_OUTPUT, CONRED, con_direction)
        melt_fracupdate(CONOUTPUT, MELTFRACOBJ)
        zCON = update_spreadsheet2(TEMP_SPREADSHEET, MELTFRACOUTPUT)
        sublabel = sub_name(xinit1, yinit1, zinit1, x_dim, y_dim, z_dim, savename)
        SUBVOLUME_OUTPUT.name = sublabel
        if xCON>0:
            bbox(SUBVOLUME_OUTPUT, x_dim, y_dim, z_dim)
        elif yCON>0:
            bbox(SUBVOLUME_OUTPUT, x_dim, y_dim, z_dim)
        conDF.loc[conDF.index == index, 'X Connected Melt'] = xCON##store all these values based on the index
        conDF.loc[conDF.index == index, 'Y Connected Melt'] = yCON
        conDF.loc[conDF.index == index, 'Z Connected Melt'] = zCON
        conDF.loc[conDF.index == index, 'Subvolume Name'] = sublabel
        conDF.loc[conDF.index == index, 'Total Melt'] = TOTAL_MELT
        fsavename = (savedir + savename + '.csv')##append the save directory, savename, and .csv to save the dataframe
        conDF.to_csv(fsavename)##save the dataframe as .csv
        
        ##this command below will create the extended domain X, Y, Z beginning and ending coordinates then check them
        # to ensure these values are within the sample volume. Then, it will create the extended domain region.
        # It then measures the connected melt fraction for this region, stores these values, and removes the extended domain region which does not
        # have the inlet/ outlet merged yet. This leaves the extended domain subvolume, along with the original probe,
        # to later save in a project file.
        if EDstructure[0] == 'Y':
            b_e_1=conDF.iloc[index]['EDXi']
            b_e_2=conDF.iloc[index]['EDYi']
            b_e_3=conDF.iloc[index]['EDZi']
            e_e_1=conDF.iloc[index]['EDXi'] + conDF.iloc[index]['EDdX']
            e_e_2=conDF.iloc[index]['EDYi'] + conDF.iloc[index]['EDdY']
            e_e_3=conDF.iloc[index]['EDZi'] + conDF.iloc[index]['EDdZ']
            begincoords_ED = (b_e_1, b_e_2, b_e_3)
            endcoords_ED=(e_e_1, e_e_2, e_e_3)
            begincoords_ED, endcoords_ED=coord_center_distance_calculator(center_of_cylinder, begincoords_ED, endcoords_ED)
            for j in range(number_of_rotations):
                (b_e_1, b_e_2, b_e_3, e_e_1, e_e_2, e_e_3) = back_rotate(radius_of_cylinder[j], x_rotation[j], y_rotation[j], z_rotation[j], begincoords_ED, endcoords_ED)
                begincoords_ED = (b_e_1, b_e_2, b_e_3)
                endcoords_ED=(e_e_1, e_e_2, e_e_3)
            condition = coord_condition(radius_of_cylinder[j], height_of_cylinder[j], begincoords_ED, endcoords_ED)
            if condition == ('keep'):
                SUBVOLUME_RED, SUBVOLUME_OUTPUT_ED = subvolume_creation(DATA)
                subvolume_update(SUBVOLUME_RED, SUBVOLUME_OUTPUT_ED, conDF.iloc[index]['EDXi'], conDF.iloc[index]['EDYi'], conDF.iloc[index]['EDZi'], conDF.iloc[index]['EDdX'], conDF.iloc[index]['EDdY'], conDF.iloc[index]['EDdZ'])
                ext_domain = ext_domain_sub_create(inlet, outlet, SUBVOLUME_OUTPUT_ED, sublabel)
                
                con_direction = 0
                axisconupdate(ext_domain, CONRED, con_direction)
                melt_fracupdate(CONOUTPUT, MELTFRACOBJ)
                xCON_ED = update_spreadsheet2(TEMP_SPREADSHEET, MELTFRACOUTPUT)
                
                con_direction = 1
                axisconupdate(ext_domain, CONRED, con_direction)
                melt_fracupdate(CONOUTPUT, MELTFRACOBJ)
                yCON_ED = update_spreadsheet2(TEMP_SPREADSHEET, MELTFRACOUTPUT)
                
                con_direction = 2
                axisconupdate(ext_domain, CONRED, con_direction)
                melt_fracupdate(CONOUTPUT, MELTFRACOBJ)
                zCON_ED = update_spreadsheet2(TEMP_SPREADSHEET, MELTFRACOUTPUT)
                
                conDF.loc[conDF.index == index, 'EDXCon'] = xCON_ED
                conDF.loc[conDF.index == index, 'EDYCon'] = yCON_ED
                conDF.loc[conDF.index == index, 'EDZCon'] = zCON_ED
                hx_project.remove(SUBVOLUME_OUTPUT_ED)
                bbox(ext_domain, x_dim, y_dim, z_dim)
                conDF.to_csv(fsavename)##save the dataframe as .csv
            else:
                pass
            
        else:
            pass

    print('Script is terminating- values saved as .csv in folder/ filename specified.\nIf user desires subvolumes saved, call saveaspackngo and they will be placed in associated folder.')
    return

def bbox(SUB_INP, xdim, ydim, zdim):
    SUBFORBBOX = (SUB_INP)
    X = int(xdim)
    Y = int(ydim)
    Z = int(zdim)
    if X > Y and X > Z:
        A = round((120/360), 3)
        B = 75.5/100
        C = 54.5/100
        color=[A, B, C]
        if 'ED' in SUBFORBBOX.name:
            lineW = 3
        else:
            lineW = 1
    elif Y > X and Y > Z:
        A = 1
        B = 0
        C = 0
        color=[A, B, C]
        if 'ED' in SUBFORBBOX.name:
            lineW = 3
        else:
            lineW = 1
    elif Z > X and Z > Y:
        A = 0
        B = 0
        C = 1
        color = [A, B, C]
        if 'ED' in SUBFORBBOX.name:
            lineW = 3
        else:
            lineW = 1
    else:
        A = 1
        B = 0
        C = 0
        color=[A, B, C]
        if 'ED' in SUBFORBBOX.name:
            lineW = 3
        else:
            lineW = 5
    bbox_1 = hx_project.create('HxBoundingBox')
    bbox_1.ports.data.connect(SUBFORBBOX)
    bbox_1.ports.lineWidth.value = lineW
    bbox_1.ports.options.items[1].color = color
    bbox_1.execute()
    return

def saveaspackngo(savename):
    #quick function to save project as a packngo (this allows for individually loading and measuring
    # subvolumes later)
    hx_project.save(savename,pack_and_go = True)
    return()


def binarizeddata(data):
    """
    Parameters
    ----------
    data : Binarized imgdata.
        This data can be any label or imgdata.

    Returns
    -------
    binarized_data : This is a relabeled imgfile, where the porespace has been relabeled as exterior.
    This is since there is a bug where the checkboxes during perm_run cannot be modified.....
    This was probed extensively, and while can be done by hand,
    or in the command terminal, cannot be automated. The easiest solution was to essentially invert the img as done here.
    """
    THRESHOLD_RED = hx_project.create('HxSegmentationOverlayThreshold')
    data.selected = True
    THRESHOLD_RED.ports.data.connect(data)##Connect the just-created Interactive Threshold Overlay module to the inputdat.
    THRESHOLD_RED.selected = True
    THRESHOLD_RED.ports.threshold.clamp_range = [0,0]
    THRESHOLD_RED.execute()##Relabel (twice...) because it literally doesn't work when applied once. 
    THRESHOLD_RED.ports.threshold.clamp_range = [0,0]
    THRESHOLD_RED.execute()
    binarized_data = hx_project.get(THRESHOLD_RED.downstream_connections[0].get_owner().name)##Grab the first object which shows the porelabels as needed when calling perm_run.
    return(binarized_data)
    

def perm_run(DIRECTION, Number_cores, DATA):
    """

    Parameters
    ----------
    DIRECTION : This is what direction the calculation will run in (XYZ). Normally coded automatrically in permeability_calculation.
    Number_cores : How many cores the calculation will use. Keep in mind you need this set up (MPI configuration). Cores are dual threadded, so usually a 12-core processor has 24 logical cores you can use.
    DATA : The subvolume that will be measured. =
    Returns
    -------
    PERM_RED: This is the structure that is now running the calculation (should see a message in the loading bar like "External Command Running")

    """
    PERM_RED = hx_project.create('HxAbsolutePermeabilityLatticeBoltzmann')
    if 'Y' in DIRECTION:
        flow_d = 1
    elif 'X' in DIRECTION:
        flow_d = 0
    elif 'Z' in DIRECTION:
        flow_d = 2
    else:
        print('Invalid permeability calculation selected- select X, Y, or Z and try again')
    PERM_RED.ports.data.connect(DATA)
    PERM_RED.ports.flowDirection.selected = flow_d
    PERM_RED.selected = True
    PERM_RED.ports.nbProcessors.texts[0].value = Number_cores
    PERM_RED.execute()
    return(PERM_RED)


def perm_measuregrab(PERM_OUTPUT, TEMP_SPREADSHEET):
    ##gets the permeability datapoint (the final value, once convergence criterion is reached)
    # using update_spreadsheet2. 
    spread_inp = PERM_OUTPUT
    PERM_DATA = update_spreadsheet2(TEMP_SPREADSHEET, spread_inp)
    return(PERM_DATA)
    

def permeability_calculation(file_directory, dataframepname, Number_cores, savedir):
    """
    This function is the main script to calculate permeability on a series of subvolumes using the Lattice-Boltzmann module 
    in PerGeos.
    
    Parameters
    ----------
    file_directory : This is the folder (full path) where all subvolumes are, which much correspond to names in the .csv file.
    dataframepname : .csv file (string, full pathname)
        This is the .csv file generated from connected_meltcalc, which should have subvolumes, with connected and total melt fraction
    Number_cores : Integer
        This is to specify how many cores to use in the LBM calculation (logical cores, so for dual-threadded processor with 12 cores you can specify up to 24).
        You must have microsoft API correctly configured in PerGeos to use this.
    savedir : Folder path, string
        Where to save each subvolume file that has the associated permeability run. It saves each subvolume as an individual project.
        
    """
    
    
    for filename in os.listdir(file_directory):
        list_of_files = os.listdir(file_directory)##for all files in the folder
        number_of_files = len(list_of_files)##calculate how many folders there are
        TEMP_SPREADSHEET =  spreadsheetcreate2()##create a spreadsheet to grab permeability datapoint from later
        data_sheet = pd.read_csv(dataframepname)##load the dataframe from connected_meltcalc
        data_sheet.set_index('Subvolume Name', inplace=True)##set the index of the dataframe to the subvolume name
        filepath = os.path.join(file_directory, filename)##now, each path is the filename in the folder, plus the folderpath specified
        loadeddata = hx_project.load(filepath)#load the subvolume
        fnamestr = str(filename)#get the string that the subvolume file is
        getdata = hx_project.get(fnamestr)#grab this
        cub_vox_force(getdata)#make sure voxels are cubic (sometimes, this is not the case when importing due to trailing digits)
        labeldata = binarizeddata(getdata)#rebinarize the data so that the pores are selected to calculate permeability by default (does not effect the measurement at all)
        if fnamestr[-2:] == 'ED':##if the subvolume loaded is that of an extended domain, find which direction is longest to run the calculation in
            if pd.isnull(data_sheet.loc[fnamestr[:-3], 'EDKX']) and pd.isnull(data_sheet.loc[fnamestr[:-3], 'EDKY']) and pd.isnull(data_sheet.loc[fnamestr[:-3], 'EDKZ']):                   
                xlen = data_sheet.loc[fnamestr[:-3], 'EDdX']
                ylen = data_sheet.loc[fnamestr[:-3], 'EDdY']
                zlen = data_sheet.loc[fnamestr[:-3], 'EDdZ']
                if xlen > ylen and xlen > zlen:
                    if data_sheet.loc[fnamestr[:-3], 'EDXCon'] != 0:##these statements check connectivity (cannot measure permeability if connectivity is zero)
                        DIRECTION = 'X'
                        PERM_RED = perm_run(DIRECTION, Number_cores, labeldata)##run the calculation in this direction, with specified cores and the relabeled data
                        try: 
                            PERM_OUTPUT = hx_project.get(PERM_RED.downstream_connections[0].get_owner().name)
                            xEDperm = perm_measuregrab(PERM_OUTPUT, TEMP_SPREADSHEET)
                        except:
                            xEDperm = 0#if the calculation fails, permeability is zero
                            print('Calculation failed for subvolume'+str(filename))
                    elif data_sheet.loc[fnamestr[:-3], 'EDXCon'] == 0:
                        xEDperm = 0##if there is no connected melt network then permeability is zero
                    data_sheet.loc[fnamestr[:-3], 'EDKX'] = xEDperm
                    del xEDperm
                elif ylen > xlen and ylen > zlen:
                    if data_sheet.loc[fnamestr[:-3], 'EDYCon'] != 0:
                        DIRECTION = 'Y' 
                        PERM_RED = perm_run(DIRECTION, Number_cores, labeldata)
                        try: 
                            PERM_OUTPUT = hx_project.get(PERM_RED.downstream_connections[0].get_owner().name)
                            yEDperm = perm_measuregrab(PERM_OUTPUT, TEMP_SPREADSHEET)
                        except:
                            yEDperm = 0
                            print('Calculation failed for subvolume'+str(filename))
                    elif data_sheet.loc[fnamestr[:-3], 'EDYCon'] == 0:
                        yEDperm = 0
                    data_sheet.loc[fnamestr[:-3], 'EDKY'] = yEDperm
                    del yEDperm
                elif zlen > xlen and zlen > ylen:
                    if data_sheet.loc[fnamestr[:-3], 'EDZCon'] !=0:
                        DIRECTION = 'Z'
                        PERM_RED = perm_run(DIRECTION, Number_cores, labeldata)
                        try: 
                            PERM_OUTPUT = hx_project.get(PERM_RED.downstream_connections[0].get_owner().name)
                            zEDperm = perm_measuregrab(PERM_OUTPUT, TEMP_SPREADSHEET)
                        except:
                            zEDperm = 0
                            print('Calculation failed for subvolume'+str(filename))
                    elif data_sheet.loc[fnamestr[:-3], 'EDZCon'] == 0:
                        zEDperm = 0
                    data_sheet.loc[fnamestr[:-3], 'EDKZ'] = zEDperm
                    del zEDperm
            else:
                print('already calculated permeability- skipping')
        else:
            pass
        fnamestr = str(filename)
        if fnamestr[-2:] != 'ED':
            ##If the opened filename does not have the suffix ED, it is either a probe or cube. These statements below
            ##run the calculation in the X, Y, and/ or Z; if the axis is not shorter than the other two, and the connectivity is 
            ## non-zero, it will collect this number (meaning for probe or extended domain subvolumes it does not collect the
            ##short axis measurements)
            ##If the calculation cannot run for whatever reason it assigns a zero-value to the permeability measurement
            ##It then stores these values in the dataframe loaded in the beginning of the code
            if data_sheet.loc[fnamestr, 'X Connected Melt'] !=0 and data_sheet.loc[fnamestr, 'dX'] >= data_sheet.loc[fnamestr, 'dY'] and data_sheet.loc[fnamestr, 'dX'] >= data_sheet.loc[fnamestr, 'dZ'] and pd.isnull(data_sheet.loc[fnamestr, 'kX']):
                DIRECTION = 'X'
                PERM_RED = perm_run(DIRECTION, Number_cores, labeldata)
                try: 
                    PERM_OUTPUT = hx_project.get(PERM_RED.downstream_connections[0].get_owner().name)
                    x_perm = perm_measuregrab(PERM_OUTPUT, TEMP_SPREADSHEET)
                except:
                    x_perm = 0
                data_sheet.loc[fnamestr, 'kX'] = x_perm
            elif data_sheet.loc[fnamestr, 'X Connected Melt'] ==0 and data_sheet.loc[fnamestr, 'dX'] >= data_sheet.loc[fnamestr, 'dY'] and data_sheet.loc[fnamestr, 'dX'] >= data_sheet.loc[fnamestr, 'dZ']:
                x_perm = 0
                data_sheet.loc[fnamestr, 'kX'] = x_perm
            else:
                print('already calculated permeability- skipping')
                
            if data_sheet.loc[fnamestr, 'Y Connected Melt'] !=0 and data_sheet.loc[fnamestr, 'dY'] >= data_sheet.loc[fnamestr, 'dX'] and data_sheet.loc[fnamestr, 'dY'] >= data_sheet.loc[fnamestr, 'dZ'] and pd.isnull(data_sheet.loc[fnamestr, 'kY']):    
                DIRECTION = 'Y'
                PERM_RED = perm_run(DIRECTION, Number_cores, labeldata)
                try: 
                    PERM_OUTPUT = hx_project.get(PERM_RED.downstream_connections[0].get_owner().name)
                    y_perm = perm_measuregrab(PERM_OUTPUT, TEMP_SPREADSHEET)
                except:
                    y_perm = 0
                data_sheet.loc[fnamestr, 'kY'] = y_perm
            elif data_sheet.loc[fnamestr, 'Y Connected Melt'] ==0 and data_sheet.loc[fnamestr, 'dY'] >= data_sheet.loc[fnamestr, 'dX'] and data_sheet.loc[fnamestr, 'dY'] >= data_sheet.loc[fnamestr, 'dZ']:
                y_perm = 0
                data_sheet.loc[fnamestr, 'kY'] = y_perm
            else:
                print('already calculated permeability- skipping')
                
            if data_sheet.loc[fnamestr, 'Z Connected Melt'] !=0 and data_sheet.loc[fnamestr, 'dZ'] >= data_sheet.loc[fnamestr, 'dX'] and data_sheet.loc[fnamestr, 'dZ'] >= data_sheet.loc[fnamestr, 'dY'] and pd.isnull(data_sheet.loc[fnamestr, 'kX']):    
                DIRECTION = 'Z'
                PERM_RED = perm_run(DIRECTION, Number_cores, labeldata)
                try: 
                    PERM_OUTPUT = hx_project.get(PERM_RED.downstream_connections[0].get_owner().name)
                    z_perm = perm_measuregrab(PERM_OUTPUT, TEMP_SPREADSHEET)
                except:
                    z_perm = 0
                data_sheet.loc[fnamestr, 'kZ'] = z_perm
            elif data_sheet.loc[fnamestr, 'Z Connected Melt'] ==0 and data_sheet.loc[fnamestr, 'dZ'] >= data_sheet.loc[fnamestr, 'dX'] and data_sheet.loc[fnamestr, 'dZ'] >= data_sheet.loc[fnamestr, 'dY']:
                z_perm = 0
                data_sheet.loc[fnamestr, 'kZ'] = z_perm
            else:
                print('already calculated permeability- skipping')
        else:
            pass
        data_sheet.to_csv(dataframepname)#save the spreadsheet after every calculation
        projsavename = (savedir + fnamestr)#set the savename to the specified directory and subvolume name
        try:
            saveaspackngo(projsavename)#save this as a packandgo
        except:
            print('could not save project, may be duplicate calcultion')
        hx_project.remove_all()#clear the project workspace
    print('Script is terminating')
    return



def cub_vox_force(subvol):
    ##Since occasionally the voxel lengths, when loading a file from Avizo into PerGeos, 
    ##are not perfectly cubic, the perm calc will occasionally break. this finds what the size of the subvolume is and forces the voxels to be
    ##exactly the same size to allow the calculation to proceed. 
    reso = decimal.Decimal('0.160000000000000000000000')
    inlets = subvol.bounding_box[0]
    outlets = subvol.bounding_box[1]
    xin = decimal.Decimal(str(round(inlets[0]*1.000000000000000000000000000, 2)))
    yin = decimal.Decimal(str(round(inlets[1]*1.000000000000000000000000000, 2)))
    zin = decimal.Decimal(str(round(inlets[2]*1.000000000000000000000000000, 2)))
    xin = decimal.Decimal(str(round(xin,24)))
    yin = decimal.Decimal(str(round(yin,24)))
    zin = decimal.Decimal(str(round(zin,24)))
    xout = decimal.Decimal(str(round(outlets[0], 2)))
    yout = decimal.Decimal(str(round(outlets[1], 2)))
    zout = decimal.Decimal(str(round(outlets[2], 2)))
    xout = decimal.Decimal(str(round(xout,24)))
    yout = decimal.Decimal(str(round(yout,24)))
    zout = decimal.Decimal(str(round(zout,24)))
    new_x = decimal.Decimal(str(round(((xout-xin)/reso),24)))
    new_y = decimal.Decimal(str(round(((yout-yin)/reso),24)))
    new_z = decimal.Decimal(str(round(((zout-zin)/reso),24)))
    print(new_x)
    xin=0
    yin=0
    zin=0
    new_xout = decimal.Decimal(str(round(xin+(new_x*reso),24)))
    new_yout = decimal.Decimal(str(round(yin+(new_y*reso),24)))
    new_zout = decimal.Decimal(str(round(zin+(new_z*reso),24)))
    new_outlets = (new_xout, new_yout, new_zout)
    inlets=(xin, yin, zin)    
    subvol.bounding_box = ((inlets), (new_outlets))
    subvol.execute()
    subvol.compute()
    subvol.fire()
    print(subvol.bounding_box)
    return


  
def ext_domain_sub_create(inlet, outlet, subvolume_output, sublabel):
    subvol_inits = subvolume_output.bounding_box[0]##to the original subvolume selected, add the inlet and outlet and label it
    subvol_ends = subvolume_output.bounding_box[1]
    Xlen = subvol_inits[0] - subvol_ends[0]
    Ylen = subvol_inits[1] - subvol_ends[1]
    Zlen = subvol_inits[2] - subvol_ends[2]##do so to the longest axis
    if abs(Xlen) > abs(Ylen) and abs(Xlen) > abs(Zlen):
        long_axis = 'X'
        inlet_begins = ((subvol_inits[0]-(10*.16)), subvol_inits[1], subvol_inits[2])
        inlet_ends = (subvol_inits[0], subvol_ends[1], subvol_ends[2])
        outlet_begins = ((subvol_ends[0]), (subvol_inits[1]), (subvol_inits[2]))
        outlet_ends = ((subvol_ends[0]+(10*.16)), subvol_ends[1], subvol_ends[2])
    elif abs(Ylen) > abs(Xlen) and abs(Ylen) > abs(Zlen):
        long_axis = 'Y'
        inlet_begins = ((subvol_inits[0]), (subvol_inits[1]-(10*.16)), subvol_inits[2])
        inlet_ends = (subvol_ends[0], subvol_inits[1], subvol_ends[2])
        outlet_begins = ((subvol_inits[0]), (subvol_ends[1]), (subvol_inits[2]))
        outlet_ends = ((subvol_ends[0]), (subvol_ends[1]+(10*.16)), subvol_ends[2])
    elif abs(Zlen) > abs(Xlen) and abs(Zlen) > abs(Ylen):
        long_axis = 'Z'
        inlet_begins = (subvol_inits[0], subvol_inits[1], subvol_inits[2]-(10*.16))
        inlet_ends = (subvol_ends[0], subvol_ends[1], subvol_ends[2])
        outlet_begins = (subvol_inits[0], subvol_inits[1], subvol_ends[2])
        outlet_ends = ((subvol_ends[0]), subvol_ends[1], subvol_ends[2]+(10*.16))
    else:
        print('Somehow no axis is the longest and the script is going to break')
    
    inlet_bbox = (inlet_begins, inlet_ends)
    outlet_bbox = (outlet_begins, outlet_ends)
    inlet.bounding_box = inlet_bbox
    outlet.bounding_box = outlet_bbox
    MERGE_RED = hx_project.create('HxMerge')
    MERGE_RED.ports.data.connect(subvolume_output)
    MERGE_RED.ports.lattice1.connect(inlet)
    MERGE_RED.ports.lattice2.connect(outlet)
    MERGE_RED.execute()
    ext_domain_str = (MERGE_RED.downstream_connections[0].get_owner().name)
    ext_domain = hx_project.get(ext_domain_str)
    ext_domain.name = (sublabel + '_ED')
    return(ext_domain)



def folder_pandas_dfstack(file_directory, savename):
    spread_name = (file_directory + '/' + savename + 'CombinedSpreadsheet.csv')
    list_of_dfs = []
    for filename in os.listdir(file_directory):
        filepath = os.path.join(file_directory, filename)
        loaded_file = pd.read_csv(filepath)
        list_of_dfs.append(loaded_file)
    comb_spread = pd.concat(list_of_dfs)
    comb_spread.to_csv(spread_name)
    return
        
def melt_perm_data_combine(perm_spread_filepath, melt_spread_filepath, savedir):
    df1 = pd.read_csv(perm_spread_filepath)
    df2 = pd.read_csv(melt_spread_filepath)
    perm_melt_spread = pd.merge(df1, df2, on='Subvolume Name')
    savename = (savedir + '/MergedDF.csv')
    perm_melt_spread.to_csv(savename)
    return(perm_melt_spread)

def DF_CONCAT(spreadsheetfpaths, savename):
    i = 0
    hugeDF  = {}
    for spreadsheet in spreadsheetfpaths:
        df = pd.read_csv(spreadsheet)
        df.set_index('Subvolume Name')
        hugeDF[i] = df
        print(hugeDF[i])
    finalDF = pd.concat(hugeDF.values(), sort='False')
    finalDF.to_csv(savename)
    print(finalDF)
    return

    
def regression_analysis_log(x_data, y_data):
    ##make sure statistics imported as stat
    x_dat = np.array(np.log10(x_data))
    y_dat = np.array(np.log10(y_data))
    m, b, r_value, p_value, std_err = stats.linregress(x_dat, y_dat)
    y_predict = np.array(10**(m*x_dat +b))
    regression_line = (x_data, y_predict)
    return(regression_line, y_predict, m, b)

def regression_analysis(x_data, y_data):
    ##make sure statistics imported as stat
    x_dat = np.array((x_data))
    y_dat = np.array((y_data))
    m, b, r_value, p_value, std_err = stats.linregtress(x_dat, y_dat)
    y_predict = np.array((m*x_dat +b))
    regression_line = (x_data, y_predict)
    return(regression_line, y_predict, m, b)


def scan_discretization(rotated_scan, x_len, y_len, z_len, xinterval, yinterval, zinterval):
    scanbbox = (rotated_scan.bounding_box)
    scan_end_coords = scanbbox[1]
    scan_begin_coords = scanbbox[0]
    scan_xlen = (scan_end_coords[0]-scan_begin_coords[0])/0.16
    scan_ylen = (scan_end_coords[1]-scan_begin_coords[1])/0.16
    scan_zlen = (scan_end_coords[2]-scan_begin_coords[2])/0.16
    x_beginning = [0]
    y_beginning = [0]
    z_beginning = [0]
    begincoords = []
    while x_beginning[-1] < (scan_xlen-(xinterval+x_len)):
        x_beginning.append(x_beginning[-1]+xinterval)
    while y_beginning[-1] < (scan_ylen-(yinterval+y_len)):
        y_beginning.append(y_beginning[-1]+yinterval)
    while z_beginning[-1] < (scan_zlen-(zinterval+z_len)):
        z_beginning.append(z_beginning[-1]+zinterval)
    for x_beg, y_beg, z_beg in itertools.product(x_beginning, y_beginning, z_beginning):
        begincoords.append((x_beg, y_beg, z_beg))
    return(begincoords)


def sub_name(xinit, yinit, zinit, xlen, ylen, zlen, savename):
    xistr = str(xinit)
    yistr = str(yinit)
    zistr = str(zinit)
    xlstr = str(xlen)
    ylstr = str(ylen)
    zlstr = str(zlen)
    subvolume_name = (savename + 'xi' + xistr + 'yi' + yistr + 'zi' + zistr + 'dx' + xlstr + 'dy' + ylstr + 'dz' + zlstr)
    return(subvolume_name)


def coord_center_distance_calculator(center_of_cylinder, begincoords, endcords):
    x = begincoords[0]-center_of_cylinder[0]
    y = begincoords[1]-center_of_cylinder[1]
    z = begincoords[2]-center_of_cylinder[2]##set the origin to the imageset center. Get the vector for the point here.
    x_e = endcords[0]-center_of_cylinder[0]
    y_e = endcords[1]-center_of_cylinder[1]
    z_e = endcords[2]-center_of_cylinder[2]
    begincords=(x,y,z)
    endcords=(x_e, y_e, z_e)
    return(begincords, endcords)

def back_rotate(radius_of_cylinder, x_0, y_0, z_0, begincoords, endcords):
    x = begincoords[0]
    y = begincoords[1]
    z = begincoords[2]
    x_e = endcords[0]
    y_e = endcords[1]
    z_e = endcords[2]
    x_1, y_1, z_1 = backrotation(x,y,z,x_0, y_0, z_0)
    x_2, y_2, z_2 = backrotation(x_e, y_e, z_e, x_0, y_0, z_0)
    return(x_1, y_1, z_1, x_2, y_2, z_2)


def coord_condition(radius_of_cylinder, height_of_cylinder, bcords, ecords):
    halfheight=0.5*height_of_cylinder
    x_1=bcords[0]
    y_1=bcords[1]
    z_1=bcords[2]
    x_2=ecords[0]
    y_2=ecords[1]
    z_2=ecords[2]
    
    distance_1 = np.sqrt((x_1*x_1)+(y_1*y_1))
    distance_2 = abs(z_1)
    distance_3 = abs(z_2)
    distance_4 = np.sqrt((x_2*x_2)+(y_2*y_2))
    distance_5 = np.sqrt((x_2*x_2)+(y_1*y_1))
    radius_of_cylinder  = radius_of_cylinder
    distance_6 = np.sqrt((x_1*x_1)+(y_2*y_2))
    if distance_1 < radius_of_cylinder and distance_2 < halfheight and distance_3 < halfheight and distance_4 < radius_of_cylinder and distance_5 < radius_of_cylinder and distance_6 < radius_of_cylinder:
        condition = ('keep')
    else:
        condition = ('discard')
    return(condition)

    
def backrotation(x, y, z, xthet, ythet, zthet):
    ztheta = (360-zthet)*((np.pi)/180)    
    ytheta = (360-ythet)*((np.pi)/180)
    xtheta = (360-xthet)*((np.pi)/180)
    x, y, z = ([((((x)*np.cos(ytheta))+((z)*np.sin(ytheta)))), y, ((((-x)*np.sin(ytheta)+((z)*np.cos(ytheta)))))])
    x, y, z = ([x, ((((y)*np.cos(xtheta)-((z)*np.sin(xtheta))))), ((((y)*np.sin(xtheta)+((z)*np.cos(xtheta)))))])
    x, y, z = ([((((x)*np.cos(ztheta))-((y)*np.sin(ztheta)))), ((((x)*np.sin(ztheta)+((y)*np.cos(ztheta))))), z])
    backrotated = (x,y,z)
    return(backrotated)
    

def init_DF():
    column_names = ['Subvolume Name', 'Total Melt', 'X Connected Melt', 'Y Connected Melt', 'Z Connected Melt', 'kX', 'kY', 'kZ', 'Xi', 'Yi', 'Zi', 'dX', 'dY', 'dZ', 'EDXCon', 'EDYCon', 'EDZCon', 'EDKX', 'EDKY', 'EDKZ', 'EDXi', 'EDYi', 'EDZi', 'EDdX', 'EDdY', 'EDdZ']
    conspread = pd.DataFrame(columns = column_names)
    return(conspread)

def fix_datasheet(fpath):
    perm_dataframe = pd.read_csv(fpath)
    len_DF = len(perm_dataframe.index)
    for i in range(len_DF):
        if perm_dataframe.kX[i] == 0:
            perm_dataframe.loc[perm_dataframe.index[i], 'X Connected Melt'] = 0
        else:
            pass
        if perm_dataframe.kY[i] == 0:
            perm_dataframe.loc[perm_dataframe.index[i], 'Y Connected Melt'] = 0
        else:
            pass
        if perm_dataframe.kZ[i] == 0:
            perm_dataframe.loc[perm_dataframe.index[i], 'Z Connected Melt'] = 0
        else:
            pass
    perm_dataframe.to_csv(fpath)
    return  
            
        
def ext_domain_meltfrac(folderpath, spreadsheetpath):
    datasheet = pd.read_csv(spreadsheetpath)
    datasheet.set_index('Subvolume Name', inplace=True)
    for filename in os.listdir(folderpath):
        if (str(filename[-2:])) == 'ED':
            filepath = os.path.join(folderpath, filename)
            loadeddata = hx_project.load(filepath)
            SUBVOLUME = hx_project.get(loadeddata.name)
            MELTFRACOBJ = melt_fractioncreate(SUBVOLUME)
            meltfracstr = (MELTFRACOBJ.downstream_connections[0].get_owner().name)##grab the name of the melt fraction output by using downstream connection
            MELTFRACOUTPUT = hx_project.get(meltfracstr)##assign this
            TEMP_SPREADSHEET = spreadsheetcreate(MELTFRACOUTPUT)##make table and attach to subvolume melt fraction (should never be zero so should be fine)
            tablestr = (TEMP_SPREADSHEET.downstream_connections[0].get_owner().name)        
            melt_fracupdate(SUBVOLUME, MELTFRACOBJ)
            TOTAL_MELT = update_spreadsheet2(TEMP_SPREADSHEET, MELTFRACOUTPUT)
            datasheet.loc[filename[:-3], 'EDTotalMelt'] = TOTAL_MELT
            datasheet.to_csv(spreadsheetpath)
            hx_project.remove_all()
        else:
            pass
                  
        
    
def ext_domain_greaterless_prism(Spreadsheet_Fpath):
    datasheet = pd.read_csv(Spreadsheet_Fpath)
    datasheet.set_index('Subvolume Name', inplace=True)
    for subvolume in datasheet.index:
        ratio_numerator = datasheet.loc[subvolume, 'EDTotalMelt'] - datasheet.loc[subvolume, 'Total Melt']
        if ratio_numerator != 0:
            ratio_denominator = datasheet.loc[subvolume, 'Total Melt']
            ratio = ratio_numerator/ratio_denominator
        else:
            ratio = 0
            print('subvolume' + str(subvolume) + ' has perfectly equal extended domain and prism melt fraction')
        datasheet.loc[subvolume, 'EDPrismMeltRatio'] = ratio_numerator
        datasheet.to_csv(Spreadsheet_Fpath)
    return


        
def get_pore_statistics_pergeos(Folderpath, spreadsheetpath, projpath):
    datasheet = pd.read_csv(spreadsheetpath)
    datasheet.set_index('Subvolume Name', inplace=True)
    for file in os.listdir(Folderpath):
        if (str(file[-2:])) != 'ED':
            filename = os.path.join(Folderpath, file)
            loaddat = hx_project.load(filename)
            SUBVOLUME = hx_project.get(loaddat.name)
            pore_stat = hx_project.create('HxAnalyzeLabels')
            pore_stat.ports.data.connect(SUBVOLUME)
            pore_stat.ports.measures.measure_group_name = ('Orientation2')
            pore_stat.execute()
            statis = hx_project.get(pore_stat.downstream_connections[1].get_owner().name)
            table = statis.all_interfaces.HxSpreadSheetInterface
            for i in range(len(table.tables[0].columns)):
                COLUMNLABEL = ('Column_' + str(i))
                data = (table.tables[0].columns[i]).asarray()
                data = data.tolist()
                if COLUMNLABEL not in datasheet.columns:
                    datasheet[COLUMNLABEL] = ""
                    datasheet[COLUMNLABEL] = datasheet[COLUMNLABEL].astype(object)
                print(COLUMNLABEL, data)
                datasheet.at[file, COLUMNLABEL] = data
            savepaths = (projpath + str(file))
            saveaspackngo(savepaths)
            hx_project.remove_all()
        else:
            pass
        datasheet.to_csv(spreadsheetpath)
    return
            
        
def orientation_weight_calculator(spreadsheetfpath, min_S, max_S):
    datasheet = pd.read_csv(spreadsheetfpath)
    datasheet['Weight List'] = ""
    datasheet['PoreDataSize'] = ""
    maxiter_shape = []
    miniter_shape = []
    weight_list = []
    maxiter_vol = []
    miniter_vol = []
    if len(datasheet.index) > 0:
        datasheet.set_index('Subvolume Name', inplace=True)
        for subvolume in datasheet.index:        
            if pd.notnull(datasheet['Column_2'][subvolume])==True:
                weight_list = []
                list1 = list(ast.literal_eval(datasheet.Column_2[subvolume]))
                maxiter_shape.append(max(list1))
                miniter_shape.append(min(list1))
                for i in range(len(list1)):
                        weight_list.append(list1[i])
                datasheet.at[subvolume, 'Weight List'] = weight_list
    else:
            try:
                list1 = list(ast.literal_eval(datasheet.Column_2[0]))
                maxshape = max(list1)
                minshape = min(list1)
                maxminshape = maxshape-minshape
                sizediff=max_S-min_S
                sizespread=sizediff/(maxminshape)
            except:
                pass
            for i in range(len(list1)):
                if list1[i] < 1:
                    weight_list.append(0)
                else:
                    weight_list.append((list1[i]-minshape)*sizespread + min_S)
                    datasheet.at[0, 'PoreDataSize'] = weight_list
    if datasheet.shape[0] > 0:
        print('Again, datasheet has more than one subvolume')
        maxshape = max(maxiter_shape)
        minshape = min(miniter_shape)
        maxminshape = maxshape-minshape
        sizediff = max_S-min_S
        sizespread = sizediff/(maxminshape)
        print(sizespread)
        for subvolume in datasheet.index:
            if pd.notnull(datasheet['Column_2'][subvolume])==True:
                poredatasize = []
                list1 = list((datasheet.loc[subvolume, 'Weight List']))
                for i in range(len(list1)):
                    if list1[i] == 0:
                        poredatasize.append(min_S)
                    else:
                        poredatasize.append((list1[i]-minshape)*sizespread + min_S)
                datasheet.at[subvolume, 'PoreDataSize'] = poredatasize
                print(max(poredatasize))
    else:
        pass
    datasheet.to_csv(spreadsheetfpath)
    return


def subvoldatasheet(projectfilepath, datasheetpath):
    datasheet=pd.read_csv(datasheetpath)
    datasheet=datasheet.set_index('Subvolume Name')
    init_output=subcon_init(projectfilepath)
    (conDF, tablestr, meltfracstr, constr, SUBVOLUME_RED, TEMP_SPREADSHEET, MELTFRACOUTPUT, CONOUTPUT, CONRED, DATA, inpdataname, SUBVOLUME_OUTPUT, MELTFRACOBJ) = init_output
    for subvolume in datasheet.index:
        x, y, z, xe, ye, ze = datasheet.at[subvolume, 'Xi'], datasheet.at[subvolume, 'Yi'], datasheet.at[subvolume, 'Zi'], datasheet.at[subvolume, 'dX'], datasheet.at[subvolume, 'dY'], datasheet.at[subvolume, 'dZ']
        SUBVOLUME_RED, SUBVOLUME_OUTPUT = subvolume_creation(DATA)
        subvolume_update(SUBVOLUME_RED, SUBVOLUME_OUTPUT, x, y, z, xe, ye, ze)
        bbox(SUBVOLUME_OUTPUT, xe, ye, ze)
    return




def plotporeorientferet(spreadsheetfpath, PORESHAPECOL, POREVOLCOL, savefolder):
    """
    Script designed to plot the orientation of melt pocket (pore) feret  diameter orientations (for information about feret diameter, see Avizo/PerGeos manual).
    
    Give full string of spreadsheet filepath, the label of the datasheet column which you want to use for color (large value of this 
                                                                                                                 label will appear pink,
                                                                                                                 small blue).
    Also provide full folder path (already made) where you want all plots saved.
    
    
    The script will plot several things (some may be disabled in future versions as they are intermediary/ repetitive). 
    First the script plots the orientation of the data. This requires conversion from the theta (-180-->180 degrees from the X axis)
    to radians (0+). Phi (angle from the XZ plane to the XY plane) is left in degrees, and the data is plotted on a polar histogram.
    Then, the data is converted to spherical coordinates, and the associated voronoi cell 
    projection is calculated using scipys prebuilt function (was developed opensource, and is documented).
    A figure of this is shown with the regions drawn in 3d. From here, new vertices are found where these region dilineations 
    intersect with z=0, which is useful to then project the stereographic projection of the regions. This projection is then performed,
    and then mapped back to theta and phi coordinates. The "modified" regions are plotted in 3D, and the stereographic projection of them is plotted
    with each region filled by using patch of the region bounded using a path. These final figures are then saved in the folder initially specified. 
    
    """
    datasheet = pd.read_csv(spreadsheetfpath)
    if len(datasheet.index) > 1:
        ##Initialize the empty lists needed....
        maxiter_vol = []
        miniter_vol = []
        maxiter_shape=[]
        miniter_shape=[]
        tot_vol=[]
        tot_shape=[]
        points=[]
        points_NEW = []
        t_vals=np.linspace(0, 1, 10000)
        pathDF = pd.DataFrame(columns=['XYZPoint','Path', 'Shape Sums', 'Volume Sums'])
        
        ##Segment below this finds the max shape, volume (to scale colormap) for whole datasheet, and then the mean of all these values (to determine which MP has
        ##a volume and shape over the mean)
        for i in range(len(datasheet.index)):
            try:
                list2 = list(ast.literal_eval(datasheet[PORESHAPECOL][i]))
                maxiter_shape.append(max(list2))
                miniter_shape.append(min(list2))
                for k in range(len(list2)):    
                    tot_shape.append(list2[k])
            except:
                pass
            try:
                list3 = list(ast.literal_eval(datasheet[POREVOLCOL][i]))
                for k in range(len(list3)):
                    tot_vol.append(list3[k])
                maxiter_vol.append(max(list3))
                miniter_vol.append(min(list3))
            except:
                pass
        minshape=min(miniter_shape)
        maxshape=max(maxiter_shape)
        maxvol = max(maxiter_vol)
        minvol = min(miniter_vol)
        
        
        ##Normalize the colormap (both to shape and volume)        
        cmap=matplotlib.cm.get_cmap('cool')
        normalcolor_shape=matplotlib.colors.Normalize(vmin=minshape, vmax=maxshape)
        normalcolor_vol=matplotlib.colors.Normalize(vmin=minvol, vmax=maxvol)
        
        ##The bottom segment uses the first subvolume in the datasheet to compute the voronoi cells
        phidat =list(ast.literal_eval(datasheet['Column_1'][0]))
        thetadat = list(ast.literal_eval(datasheet['Column_0'][0]))
        thetadat2 = []
        for i in range(len(thetadat)):
            if thetadat[i]<0:
                thetadat2.append(360+thetadat[i])##Do some conversions since python wants data from 0-360degrees and PerGeos gives it in -180 to 180.
            else:
                thetadat2.append(thetadat[i])
        xdat = np.sin(np.deg2rad(phidat))*np.cos(np.deg2rad(thetadat2))
        ydat = np.sin(np.deg2rad(phidat))*np.sin(np.deg2rad(thetadat2))
        zdat = np.cos(np.deg2rad(phidat))
        for i in range(len(xdat)):
            point = [xdat[i], ydat[i], zdat[i]]
            points.append(point)##Find all X, Y, Z points from the theta, phi computed just above
        shape = np.shape(points)
        lenth = shape[0]
        for i in range(lenth):
            point=points[i]
            if point in points_NEW:
                continue
            elif point not in points_NEW: 
                points_NEW.append(points[i])##This for-loop creates a list with only the unique values (voronoi cannot compute with duplicates)
        
        pointsnewlen = np.shape(points_NEW)
        pointsnewlen=pointsnewlen[0]
        for i in range(pointsnewlen):
            pathDF.at[i, 'XYZPoint']=points_NEW[i]
            
        
        ##Compute the voronoi cell using the unit sphere and unique X, Y, Z points of MPO from the initial subvolume
        radius = 1
        center=[0,0,0]
        sv = SphericalVoronoi((points_NEW), radius, center)
        sv.sort_vertices_of_regions()
        
        
        ##For every region in the voronoi cell computation
        dfind = 0
    
        
        for region in sv.regions:
            ##initialize empty vectors for this individual region...
            path=[]
            verts=[]
            adverts=[]
            fullpath=[]
            vertsadj=[]
            codesadj=[]
            
            ##how many vertices are there in this region
            n=len(region)
            
            # ax_debug.scatter(sv.vertices[region][...,0],sv.vertices[region][...,1],sv.vertices[region][...,2])
            for i in range(n):
                thetaofvorn = []
                phiofvorn = []
                start = sv.vertices[region][i]
                end = sv.vertices[region][(i+1) % n]
                radius=start[0]*start[0]+start[1]*start[1]+start[2]*start[2]
                new_z = np.sqrt(1-start[0]*start[0]-start[1]*start[1])
                if start[2] <0:
                    new_z = new_z*-1
                start=[start[0],start[1],new_z]
                radius=end[0]*end[0]+end[1]*end[1]+end[2]*end[2]
                new_z = np.sqrt(1-end[0]*end[0]-end[1]*end[1])
                if end[2]<0:
                    new_z = new_z*-1
                end=[end[0],end[1],new_z]
                result=geometric_slerp(start, end, t_vals)
                if all(number >= 0 for number in result[..., 2]):
                    if i==0:
                        list1 = (result[0, 0], result[0,1],result[0,2])
                        list2 = (result[-1, 0], result[-1,1],result[-1,2])
                        adverts.append(list1)
                        adverts.append(list2)
                    else:
                        list2 = (result[-1, 0], result[-1,1],result[-1,2])
                        adverts.append(list2)
                    for k in range(len(result[..., 0])):
                        new_x=(result[k, 0]/(1-result[k, 2]))
                        new_y=(result[k, 1]/(1-result[k, 2]))
                        thetaofvorn.append((np.arccos(result[k, 2])))
                        if new_y >= 0:
                            if new_x >= 0:
                                phiofvorn.append((np.arctan(new_y/new_x)))
                            elif new_x < 0:
                                phiofvorn.append((np.arctan(new_y/new_x))+np.pi)
                        elif new_y < 0:
                            if new_x >=0:    
                                phiofvorn.append((np.arctan(new_y/new_x))+2*np.pi)
                            elif new_x < 0:
                                phiofvorn.append((np.arctan(new_y/new_x))+np.pi)
                else:
                    if result[-1, 2]>0:
                        list2 = (result[-1, 0], result[-1, 1], result[-1, 2])
                        for k in range(len(result[..., 2])-1, -1, -1):
                            if result[k, 2]>=0:
                                new_x=(result[k, 0]/(1-result[k, 2]))
                                new_y=(result[k, 1]/(1-result[k, 2]))
                                thetaofvorn.append((np.arccos(result[k, 2])))
                                if new_y >= 0:
                                    if new_x >= 0:
                                        phiofvorn.append((np.arctan(new_y/new_x)))
                                    elif new_x < 0:
                                        phiofvorn.append((np.arctan(new_y/new_x))+np.pi)
                                elif new_y < 0:
                                    if new_x >=0:    
                                        phiofvorn.append((np.arctan(new_y/new_x))+2*np.pi)
                                    elif new_x < 0:
                                        phiofvorn.append((np.arctan(new_y/new_x))+np.pi)
                            else:
                                new_x=(result[k, 0]/(1-result[k, 2]))                                    
                                new_y=(result[k, 1]/(1-result[k, 2]))
                                thetaofvorn.append(1.5708)
                                if new_y >= 0:
                                    if new_x >= 0:
                                        phiofvorn.append((np.arctan(new_y/new_x)))
                                    elif new_x < 0:
                                        phiofvorn.append((np.arctan(new_y/new_x))+np.pi)
                                elif new_y < 0:
                                    if new_x >=0:    
                                        phiofvorn.append((np.arctan(new_y/new_x))+2*np.pi)
                                    elif new_x < 0:
                                        phiofvorn.append((np.arctan(new_y/new_x))+np.pi)
                                break
                        if np.sqrt(new_x*new_x+new_y*new_y) == 1.00000:
                            continue
                        else:
                            new_y =np.sqrt(1-new_x*new_x)
                            if result[k,1]<0:
                                new_y = new_y*-1
                        list1 = (new_x, new_y, 0)        
                        adverts.append(list1)
                        adverts.append(list2)
                    elif result[0,2]>0:
  
                        list1 = (result[0, 0], result[0, 1], result[0, 2])
                        for k in range(len(result[..., 0])):
                            if result[k, 2] >=0:
                                new_x=(result[k, 0]/(1-result[k, 2]))
                                new_y=(result[k, 1]/(1-result[k, 2]))
                                thetaofvorn.append((np.arccos(result[k, 2])))
                                if new_y >= 0:
                                    if new_x >= 0:
                                        phiofvorn.append((np.arctan(new_y/new_x)))
                                    elif new_x < 0:
                                        phiofvorn.append((np.arctan(new_y/new_x))+np.pi)
                                elif new_y < 0:
                                    if new_x >=0:    
                                        phiofvorn.append((np.arctan(new_y/new_x))+2*np.pi)
                                    elif new_x < 0:
                                        phiofvorn.append((np.arctan(new_y/new_x))+np.pi)
                            else:
                                new_x=(result[k, 0]/(1-result[k, 2]))
                                new_y=(result[k, 1]/(1-result[k, 2]))
                                if np.sqrt(new_x*new_x+new_y*new_y) == 1.00000:
                                    continue
                                else:
                                    new_y =np.sqrt(1-new_x*new_x)
                                    if result[k,1]<0:
                                        new_y = new_y*-1
                                thetaofvorn.append(1.5708)
                                if new_y >= 0:
                                    if new_x >= 0:
                                        phiofvorn.append((np.arctan(new_y/new_x)))
                                    elif new_x < 0:
                                        phiofvorn.append((np.arctan(new_y/new_x))+np.pi)
                                elif new_y < 0:
                                    if new_x >=0:    
                                        phiofvorn.append((np.arctan(new_y/new_x))+2*np.pi)
                                    elif new_x < 0:
                                        phiofvorn.append((np.arctan(new_y/new_x))+np.pi)
                                break
                        list2 = (new_x, new_y, 0)
                        adverts.append(list1)
                        adverts.append(list2)

                

            l=[]
            lenadv = np.shape(adverts)
            lenadv = lenadv[0]-1
            thetaofvornadj = []
            phiofvornadj = []
            
    
            for s in range(lenadv):
                start=adverts[s]
                end=adverts[(s+1)]
                if start!=end:
                    result=geometric_slerp(start, end, t_vals)
                    for l in range(len(result[...,0])):
                        pathpoint=(result[l,0], result[l,1], result[l,2])
                        fullpath.append(pathpoint)
                        
            start=adverts[-1]
            end=adverts[0]
            if start!=end:
                result=geometric_slerp(start, end, t_vals)
                for l in range(len(result[..., 0])):
                    pathpoint=(result[l,0],result[l,1],result[l,2])
                    fullpath.append(pathpoint)
            
            
            
            
            lenfpath=np.shape(fullpath)
            lenfpath=lenfpath[0]
                
            
            
            for s in range(lenfpath):
                point=fullpath[s]
                new_x = (point[0]/(1-point[2]))
                new_y = (point[1]/(1-point[2]))
                thetaofvornadj.append(np.arccos(point[2]))
                if new_y >= 0:
                    if new_x >= 0:
                        phiofvornadj.append((np.arctan(new_y/new_x)))
                    elif new_x < 0:
                        phiofvornadj.append((np.arctan(new_y/new_x))+np.pi)
                elif new_y < 0:
                    if new_x >=0:    
                        phiofvornadj.append((np.arctan(new_y/new_x))+2*np.pi)
                    elif new_x < 0:
                        phiofvornadj.append((np.arctan(new_y/new_x))+np.pi)
                x = phiofvornadj[-1]
                y = np.rad2deg(thetaofvornadj[-1])
                if x<0:
                    x = x+(2*np.pi)
                vert = (x,y)
                vertsadj.append(vert)

    
            for s in range(len(phiofvorn)):
                x = phiofvorn[s]
                y = np.rad2deg(thetaofvorn[s])
                if x<1:
                    x = x+(2*np.pi)
                vert = (x, y)
                verts.append(vert)                    
            else:
                pass
            
            for s in range(len(vertsadj)):
                if s == 0:
                    codesadj.append(Path.MOVETO)
                else:
                    codesadj.append(Path.LINETO)
            path = Path(vertsadj, codesadj)            
            pathDF.at[dfind, 'Path'] = path
            dfind=dfind+1

            
        ##Now, there should be a dataframe with the X, Y, Z point from which each region was constructed, and the path surrounding this region, where each 
        ##region terminates at z=0 not z<0.
        
        for i in range(len(pathDF.index)):
            pathDF.at[i, 'XYZPoint'] = tuple(pathDF.at[i, 'XYZPoint'])
        datasheet.set_index('Subvolume Name', inplace=True)
        
        all_yinits = datasheet['Yi']
        uniqueYI=[]
        for value in all_yinits:
            if value in uniqueYI:
                pass
            else:
                uniqueYI.append(value)
                colnamec=(str(value)+'c')
                colnames=(str(value)+'s')
                
                pathDF[colnamec]=np.nan
                pathDF[colnames]=np.nan
        
        for subvolume in datasheet.index:
            ##Subvolume DF initialized
            df_subvol = pd.DataFrame(columns=['XYZPoint','Shape Color', 'Volume Color', 'Volume Sum', 'Shape Sum'])
            thisYI = str(datasheet.at[subvolume, 'Yi'])
            thisYIc = thisYI+'c'
            thisYIs = thisYI+'s'
            
            if datasheet['Column_0'][subvolume]:
                
                ##grab the data from the porestat dataframe
                phidat =list(ast.literal_eval(datasheet['Column_1'][subvolume]))
                thetadat = list(ast.literal_eval(datasheet['Column_0'][subvolume]))
                thetadat2 = []
                points=[]
                points_NEW=[]
                
                ##below
                for i in range(len(thetadat)):
                    if thetadat[i]<0:
                        thetadat2.append(360+thetadat[i])
                    else:
                        thetadat2.append(thetadat[i])
                xdat = np.sin(np.deg2rad(phidat))*np.cos(np.deg2rad(thetadat2))
                ydat = np.sin(np.deg2rad(phidat))*np.sin(np.deg2rad(thetadat2))
                zdat = np.cos(np.deg2rad(phidat))
                
    
                ##use the normalized colorscheme to find each individual color
                shapecolDF = list(ast.literal_eval(datasheet[PORESHAPECOL][subvolume]))
                volcolDF = list(ast.literal_eval(datasheet[POREVOLCOL][subvolume]))
                shapecolDF_shape = np.shape(shapecolDF)
                colordat=[]
                colordat2=[]
                for i in range(shapecolDF_shape[0]):
                    shapenormcol = cmap(normalcolor_shape(shapecolDF[i]))
                    volnormcol=cmap(normalcolor_vol(volcolDF[i]))
                    colordat.append(shapenormcol)
                    colordat2.append(volnormcol)
                
                for i in range(len(xdat)):
                    point = [xdat[i], ydat[i], zdat[i]]
                    points.append(point)
                shape = np.shape(points)
                lenth = shape[0]
                for i in range(lenth):
                    point=points[i]
                    if point in points_NEW:
                        continue
                    elif point not in points_NEW: 
                        points_NEW.append(points[i])
                
                ##Find the color for the max vol or shape from dataframe
                cpatch_shape=[]
                cpatch_volume=[]
                shapesum=[]
                volsum=[]
                
                for point in points_NEW:
                    colorlist=[]
                    colorlist_vol=[]
                    shapecheck=[]
                    volcheck=[]
                    for i in range(len(xdat)):
                        pointcheck=[xdat[i], ydat[i], zdat[i]]
                        if point==pointcheck:
                            volcheck.append(volcolDF[i])
                            shapecheck.append(shapecolDF[i])
                            colorlist.append(colordat[i])
                            colorlist_vol.append(colordat2[i])
                        ##Volume and shape data stored for each point in the volume or shape DF (column labeled the name of subvolume)
                    
                    maxshape=max(shapecheck)
                    maxvol=max(volcheck)
                    maxvol_ind=volcheck.index(maxvol)
                    maxshape_ind=shapecheck.index(maxshape)
                    maxcolorvol=colorlist_vol[maxvol_ind]
                    maxcolorshape=colorlist[maxshape_ind]
                    cpatch_shape.append(maxcolorshape)
                    cpatch_volume.append(maxcolorvol)
                    shapesum.append(sum(shapecheck))
                    volsum.append(sum(volcheck))
                
                print(len(points_NEW), len(pathDF.index))
                
                
                ##info for this subvolume stored
                df_subvol['XYZPoint']=points_NEW
                df_subvol['Shape Color']=cpatch_shape
                df_subvol['Volume Color']=cpatch_volume
                df_subvol['Volume Sum'] = volsum
                df_subvol['Shape Sum'] = shapesum
                ##This DF re-indexed so the rows of this, and that from the voronoi cell path, are in the same order. 
                for i in range(len(df_subvol.index)):
                    df_subvol.at[i, 'XYZPoint'] = tuple(df_subvol.at[i, 'XYZPoint'])
                df_subvol=df_subvol.set_index('XYZPoint')
                df_subvol=df_subvol.reindex(index=pathDF['XYZPoint'])
                df_subvol=df_subvol.reset_index()
                
                
                ##Initialize the plots, set axis nicely
                fig_col =plt.figure()
                ax_col=fig_col.add_subplot(1,1,1, projection='polar')
                ax_col.set_rmax(90)
                ax_col.set_theta_direction(-1)
                ax_col.set_theta_zero_location('N')
                cbar=plt.colorbar(cm.ScalarMappable(norm=normalcolor_vol, cmap=cmap), ax=ax_col)
                
                fig_shape=plt.figure()
                ax_shape=fig_shape.add_subplot(1,1,1, projection='polar')
                ax_shape.set_rmax(90)
                ax_shape.set_theta_direction(-1)
                ax_shape.set_theta_zero_location('N')
                cbar=plt.colorbar(cm.ScalarMappable(norm=normalcolor_shape, cmap=cmap), ax=ax_shape)
                
                
                ##Bottom plots each subvolume's data (color scaled to volume and shape separately)
                df_subvol = df_subvol.set_index('XYZPoint')
                pathDF = pathDF.set_index('XYZPoint')
                for point in df_subvol.index:
                    volcol=df_subvol.at[point, 'Volume Color']
                    volshape=df_subvol.at[point,'Shape Color']
                    path=pathDF.at[point,'Path']
                    if pd.isna(pathDF.at[point, 'Volume Sums']) == False and pd.isna(df_subvol.at[point, 'Volume Sum']) == False:
                        pathDF.at[point, 'Volume Sums'] = pathDF.at[point, 'Volume Sums']+df_subvol.at[point, 'Volume Sum']
                        pathDF.at[point, 'Shape Sums'] = pathDF.at[point, 'Shape Sums']+df_subvol.at[point, 'Shape Sum']
                    elif pd.isna(pathDF.at[point, 'Volume Sums']) == True and pd.isna(df_subvol.at[point, 'Volume Sum']) == False:
                        print('Patch sum is initializing')
                        pathDF.at[point, 'Volume Sums']=df_subvol.at[point, 'Volume Sum']
                        pathDF.at[point, 'Shape Sums']=df_subvol.at[point, 'Shape Sum']
                    elif pd.isna(pathDF.at[point, 'Volume Sums']) == False and pd.isna(df_subvol.at[point, 'Volume Sum']) == True:
                        pass
                    
                    try:
                        patch_col = patches.PathPatch(path, facecolor=volcol)
                        patch_shape=patches.PathPatch(path, facecolor=volshape)
                        ax_col.add_patch(patch_col)
                        ax_shape.add_patch(patch_shape)
                    except:
                        print('~~~Patch does not exist~~~')
                    if pd.isna(pathDF.at[point, thisYIc]) == False:
                        pathDF.at[point, thisYIc] = pathDF.at[point, thisYIc]+df_subvol.at[point, 'Volume Sum']
                        pathDF.at[point, thisYIs] = pathDF.at[point, thisYIs]+df_subvol.at[point, 'Shape Sum']
                    else:
                        pathDF.at[point, thisYIc]=df_subvol.at[point, 'Volume Sum']
                        pathDF.at[point, thisYIs]=df_subvol.at[point, 'Shape Sum']
                    
                    savestrvol = savefolder + str(subvolume) +'VolumeColor'+ '.pdf'
                    savestrshape = savefolder + str(subvolume) +'ShapeColor'+ '.pdf'
                fig_col.savefig(savestrvol)
                fig_shape.savefig(savestrshape)
                plt.close('all')
                pathDF = pathDF.reset_index()
                
        

        shapesums=pathDF['Shape Sums']
        volsums=pathDF['Volume Sums']
        
        normcol_shapemax = matplotlib.colors.Normalize(vmin=min(shapesums), vmax=max(shapesums))
        normcol_volmax = matplotlib.colors.Normalize(vmin=min(volsums), vmax=max(volsums))
        
        
        
        
        fig_shapesum = plt.figure()
        ax_shapesum=fig_shapesum.add_subplot(1,1,1, projection='polar')
        ax_shapesum.set_rmax(90)
        ax_shapesum.set_theta_direction(-1)
        ax_shapesum.set_theta_zero_location('N')
        ax_shapesum.set_title('All Subvolumes- Colored by Shape')
        cbar=plt.colorbar(cm.ScalarMappable(norm=normcol_shapemax, cmap=cmap), ax=ax_shapesum)
        
        fig_volsum = plt.figure()
        ax_volsum=fig_volsum.add_subplot(1,1,1, projection='polar')
        ax_volsum.set_rmax(90)
        ax_volsum.set_theta_direction(-1)
        ax_volsum.set_theta_zero_location('N')
        ax_volsum.set_title('All Subvolumes- Colored by Volume')
        cbar=plt.colorbar(cm.ScalarMappable(norm=normcol_volmax, cmap=cmap), ax=ax_volsum)
        
        pathDF = pathDF.set_index('XYZPoint')
        
        print(volsums, pathDF.index)
        
        for point in pathDF.index:
            shapecolor=cmap(normcol_shapemax(pathDF.at[point,'Shape Sums']))
            volcolor=cmap(normcol_volmax(pathDF.at[point,'Volume Sums']))
            
            path=pathDF.at[point, 'Path']
            patch_volcol = patches.PathPatch(path, facecolor=volcolor)
            patch_shapecol = patches.PathPatch(path, facecolor=shapecolor)
            ax_volsum.add_patch(patch_volcol)
            ax_shapesum.add_patch(patch_shapecol)
          
        for unique_yinit in uniqueYI:
            yisumc = str(unique_yinit)+'c'
            yisums = str(unique_yinit)+'s'
            title_color='Subvolumes with '+str(unique_yinit)+' as Y-Initial--Cmap on Color'
            title_shape='Subvolumes with '+str(unique_yinit)+' as Y-Initial--Cmap on Shape'
            fig = plt.figure()
            ax_volyin=fig.add_subplot(1,1,1, projection='polar')
            fig = plt.figure()
            ax_shapyin=fig.add_subplot(1,1,1, projection='polar')
            ax_volyin.set_rmax(90)
            ax_volyin.set_theta_direction(-1)
            ax_volyin.set_theta_zero_location('N')
            ax_volyin.set_title(title_color)
            ax_shapyin.set_rmax(90)
            ax_shapyin.set_theta_direction(-1)
            ax_shapyin.set_theta_zero_location('N')
            ax_shapyin.set_title(title_shape)
            cbar=plt.colorbar(cm.ScalarMappable(norm=normcol_volmax, cmap=cmap), ax=ax_volyin)
            cbar=plt.colorbar(cm.ScalarMappable(norm=normcol_shapemax, cmap=cmap), ax=ax_shapyin)
            for point in pathDF.index:    
                volcol=cmap(normcol_volmax(pathDF.at[point, yisumc]))
                shapecol=cmap(normcol_volmax(pathDF.at[point, yisums]))
                path=pathDF.at[point, 'Path']
                patch_volcol = patches.PathPatch(path, facecolor=volcol)
                patch_shapecol = patches.PathPatch(path, facecolor=shapecol)
                ax_volyin.add_patch(patch_volcol)
                ax_shapyin.add_patch(patch_shapecol)
            
    return



def plotporeorientinertia(spreadsheetfpath, PORESHAPECOL, POREVOLCOL, savefpath, voronoi_desired='yes', scannum='01'):
    
    datasheet = pd.read_csv(spreadsheetfpath)
    if len(datasheet.index) > 1:
        ##Initialize the empty lists needed....
        tot_vol=[]
        totlen=(0)
        tot_shape=[]
        tot_theta=[]
        tot_phi=[]
        maxiter_vol = []
        miniter_vol = []
        maxiter_shape=[]
        miniter_shape=[]
        tot_xdat=[]
        tot_ydat=[]
        tot_zdat=[]
        x_weightvol=[]
        y_weightvol=[]
        radius_weightvol=[]
        starttime=time.time()

        datasheet.set_index('Subvolume Name')
        for i in range(len(datasheet.index)):
            try:
                list2 = list(ast.literal_eval(datasheet[PORESHAPECOL][i]))
                maxiter_shape.append(max(list2))
                miniter_shape.append(min(list2))
                for k in range(len(list2)):    
                    tot_shape.append(list2[k])
            except:
                pass
            try:
                list3 = list(ast.literal_eval(datasheet[POREVOLCOL][i]))
                for k in range(len(list3)):
                    tot_vol.append(list3[k])
                maxiter_vol.append(max(list3))
                miniter_vol.append(min(list3))
            except:
                pass
        sumvol=np.sum(tot_vol)
        minshape=min(miniter_shape)
        maxshape=max(maxiter_shape)
        maxvol = max(maxiter_vol)
        minvol = min(miniter_vol)
        meanshape=np.mean(tot_shape)
        ##Keep min max volume, shape scaled for all of Adaire's scans##
        maxvol = 1000
        minvol= 0
        minshape = 0
        maxshape = 100
        cmapdesi='Greys'
        levelcount=20
        
        
        ##Normalize the colormap (both to shape and volume)        
        cmap=matplotlib.cm.get_cmap(cmapdesi)
        normalcolor_shape=matplotlib.colors.Normalize(vmin=minshape, vmax=maxshape)
        normalcolor_vol=matplotlib.colors.Normalize(vmin=minvol, vmax=maxvol)
        
        
        
        ##Make a test vector
        teta=60
        ph=35
        # ph=90-ph
        teta = np.deg2rad(teta)
        ph = np.deg2rad(ph)
        x_t =(np.cos(teta)*np.sin(ph))
        y_t =(np.sin(teta)*np.sin(ph))
        z_t =(np.cos(ph))
        # if z_t<0:
        #     x_t = x_t*-1
        #     y_t = y_t*-1
        #     z_t = z_t*-1
        
        
        ###If desired, average the points like in plotporeorientferet using voronoi cells which have patches of the sum of either volume or shape###
        if voronoi_desired == 'yes':
            pathDF = pd.DataFrame(columns=['Cartesian Path', 'Path', 'Shape Sums', 'Volume Sums', 'Patch Area', 'Filtered Volume Sums'])
            dfind=0
            pointsvornin=[]
            pathlist=[]
            zlevel=[]
            # zlevel.append(0)
            zmiddle=[]
            sphericalzonearea=[]
            t_vals=np.linspace(0, 1, 10000)
            numtheta=20
            fig_3ddebug = plt.figure()
            ax_3ddebug=fig_3ddebug.add_subplot(1,1,1,projection='3d')
            
            thetavordisc = np.linspace(0, 2*np.pi, numtheta)
            phivordisc = np.linspace(0, 90, 8)
            print(phivordisc)
            zlevel.append(0)
            zlevel.append(np.sin(np.deg2rad(phivordisc[1]))/2)
            for i in range(len(phivordisc)-1):
                zlevel.append(np.sin(np.deg2rad(phivordisc[1]+(np.rad2deg((np.arcsin(zlevel[-1])))))))
            zlevel[-1]=1
            ##zlevel contains the edge of each voronoi cell dilineation (z coordinate, altitude), which is the spherical zone separation
            for i in range(len(zlevel)-1):
                sphericalzonearea.append(2*np.pi*1*(zlevel[i+1]-zlevel[i]))
                
            print(zlevel, sphericalzonearea)
                
            thetavornin, phivornin=np.meshgrid(thetavordisc, phivordisc)
            thetavornin=np.ravel(thetavornin)
            phivornin=np.ravel(phivornin)
            for i in range(len(thetavornin)):
                xiter=np.sin(np.deg2rad(phivornin[i]))*np.cos(thetavornin[i])
                yiter=np.sin(np.deg2rad(phivornin[i]))*np.sin(thetavornin[i])
                ziter=np.cos(np.deg2rad(phivornin[i]))
                if np.sqrt(yiter*yiter) <=1E-10:
                    yiter=0.0
                if np.sqrt(xiter*xiter) <=1E-10:
                    xiter=0.0
                if np.sqrt(ziter*ziter) <=1E-10:
                    ziter=0.0
                vornpt=[xiter, yiter, ziter]
                ax_3ddebug.scatter(vornpt[0], vornpt[1], vornpt[2])
                if vornpt in pointsvornin:
                    continue
                elif vornpt not in pointsvornin:
                    pointsvornin.append(vornpt)
            avgvor=SphericalVoronoi((pointsvornin))
            regionareas=scipy.spatial.SphericalVoronoi.calculate_areas(avgvor)
            for l in range(len(regionareas)):
                pathDF.at[l, 'Patch Area']=regionareas[l]
            avgvor=SphericalVoronoi((pointsvornin))
            avgvor.sort_vertices_of_regions()
            voronoitime=time.time()
            print('Time to calculate voronoi cells:', voronoitime-starttime)
            
            ax_3ddebug.scatter(avgvor.vertices[:, 0], avgvor.vertices[:, 1], avgvor.vertices[:, 2],c='k')
            
            
            fig_3ddebug2=plt.figure()
            ax_3ddebug2=fig_3ddebug2.add_subplot(1,1,1, projection='3d')
            ax_3ddebug2.set_xlabel('X')
            ax_3ddebug2.set_ylabel('Y')
            ax_3ddebug2.set_zlabel('Z')
            
            fig_polardebug=plt.figure()
            ax_polardebug=fig_polardebug.add_subplot(1,1,1, projection='polar')
            ax_polardebug.set_rmax(90)
            ax_polardebug.set_theta_direction(-1)
            ax_polardebug.set_theta_zero_location('N')
        
            fig_cartdebug = plt.figure()
            ax_cartdebug=fig_cartdebug.add_subplot()
            ax_cartdebug.set_ylim([-1, 1])
            ax_cartdebug.set_xlim([-1, 1])
            
            

            
            
            for region in avgvor.regions:
                path=[]
                verts=[]
                adverts=[]
                fullpath=[]
                vertsadj=[]
                codesadj=[]
                
                ##how many vertices are there in this region
                n=len(region)
                
                # ax_debug.scatter(sv.vertices[region][...,0],sv.vertices[region][...,1],sv.vertices[region][...,2])
                for i in range(n):
                    thetaofvorn = []
                    phiofvorn = []
                    start = avgvor.vertices[region][i]
                    end = avgvor.vertices[region][(i+1) % n]
                    radius=start[0]*start[0]+start[1]*start[1]+start[2]*start[2]
                    new_z = np.sqrt(1-start[0]*start[0]-start[1]*start[1])
                    if start[2] <0:
                        new_z = new_z*-1
                    start=[start[0],start[1],new_z]
                    radius=end[0]*end[0]+end[1]*end[1]+end[2]*end[2]
                    new_z = np.sqrt(1-end[0]*end[0]-end[1]*end[1])
                    if end[2]<0:
                        new_z = new_z*-1
                    end=[end[0],end[1],new_z]
                    result=geometric_slerp(start, end, t_vals)
                    
                    if start != end:
                    
                        if all(number >= 0 for number in result[..., 2]):
                            if i==0:
                                list1 = (result[0, 0], result[0,1],result[0,2])
                                list2 = (result[-1, 0], result[-1,1],result[-1,2])
                                adverts.append(list1)
                                adverts.append(list2)
                            else:
                                list2 = (result[-1, 0], result[-1,1],result[-1,2])
                                adverts.append(list2)
                            for k in range(len(result[..., 0])):
                                new_x=(result[k, 0]/(1-result[k, 2]))
                                new_y=(result[k, 1]/(1-result[k, 2]))
                                thetaofvorn.append((np.arccos(result[k, 2])))
                                if new_y >= 0:
                                    if new_x >= 0:
                                        phiofvorn.append((np.arctan(new_y/new_x)))
                                    elif new_x < 0:
                                        phiofvorn.append((np.arctan(new_y/new_x))+np.pi)
                                elif new_y < 0:
                                    if new_x >=0:    
                                        phiofvorn.append((np.arctan(new_y/new_x))+2*np.pi)
                                    elif new_x < 0:
                                        phiofvorn.append((np.arctan(new_y/new_x))+np.pi)
                        else:
                            if result[-1, 2]>0:
                                list2 = (result[-1, 0], result[-1, 1], result[-1, 2])
                                for k in range(len(result[..., 2])-1, -1, -1):
                                    if result[k, 2]>=0:
                                        new_x=(result[k, 0]/(1-result[k, 2]))
                                        new_y=(result[k, 1]/(1-result[k, 2]))
                                        thetaofvorn.append((np.arccos(result[k, 2])))
                                        if new_y >= 0:
                                            if new_x >= 0:
                                                phiofvorn.append((np.arctan(new_y/new_x)))
                                            elif new_x < 0:
                                                phiofvorn.append((np.arctan(new_y/new_x))+np.pi)
                                        elif new_y < 0:
                                            if new_x >=0:    
                                                phiofvorn.append((np.arctan(new_y/new_x))+2*np.pi)
                                            elif new_x < 0:
                                                phiofvorn.append((np.arctan(new_y/new_x))+np.pi)
                                    else:
                                        new_x=(result[k, 0]/(1-result[k, 2]))                                    
                                        new_y=(result[k, 1]/(1-result[k, 2]))
                                        thetaofvorn.append(1.5708)
                                        if new_y >= 0:
                                            if new_x >= 0:
                                                phiofvorn.append((np.arctan(new_y/new_x)))
                                            elif new_x < 0:
                                                phiofvorn.append((np.arctan(new_y/new_x))+np.pi)
                                        elif new_y < 0:
                                            if new_x >=0:    
                                                phiofvorn.append((np.arctan(new_y/new_x))+2*np.pi)
                                            elif new_x < 0:
                                                phiofvorn.append((np.arctan(new_y/new_x))+np.pi)
                                        break
                                if np.sqrt(new_x*new_x+new_y*new_y) == 1.00000:
                                    continue
                                else:
                                    new_y =np.sqrt(1-new_x*new_x)
                                    if result[k,1]<0:
                                        new_y = new_y*-1
                                list1 = (new_x, new_y, 0)        
                                adverts.append(list1)
                                adverts.append(list2)
                            elif result[0,2]>0:
          
                                list1 = (result[0, 0], result[0, 1], result[0, 2])
                                for k in range(len(result[..., 0])):
                                    if result[k, 2] >=0:
                                        new_x=(result[k, 0]/(1-result[k, 2]))
                                        new_y=(result[k, 1]/(1-result[k, 2]))
                                        thetaofvorn.append((np.arccos(result[k, 2])))
                                        if new_y >= 0:
                                            if new_x >= 0:
                                                phiofvorn.append((np.arctan(new_y/new_x)))
                                            elif new_x < 0:
                                                phiofvorn.append((np.arctan(new_y/new_x))+np.pi)
                                        elif new_y < 0:
                                            if new_x >=0:    
                                                phiofvorn.append((np.arctan(new_y/new_x))+2*np.pi)
                                            elif new_x < 0:
                                                phiofvorn.append((np.arctan(new_y/new_x))+np.pi)
                                    else:
                                        new_x=(result[k, 0]/(1-result[k, 2]))
                                        new_y=(result[k, 1]/(1-result[k, 2]))
                                        if np.sqrt(new_x*new_x+new_y*new_y) == 1.00000:
                                            continue
                                        else:
                                            new_y =np.sqrt(1-new_x*new_x)
                                            if result[k,1]<0:
                                                new_y = new_y*-1
                                        thetaofvorn.append(1.5708)
                                        if new_y >= 0:
                                            if new_x >= 0:
                                                phiofvorn.append((np.arctan(new_y/new_x)))
                                            elif new_x < 0:
                                                phiofvorn.append((np.arctan(new_y/new_x))+np.pi)
                                        elif new_y < 0:
                                            if new_x >=0:    
                                                phiofvorn.append((np.arctan(new_y/new_x))+2*np.pi)
                                            elif new_x < 0:
                                                phiofvorn.append((np.arctan(new_y/new_x))+np.pi)
                                        break
                                list2 = (new_x, new_y, 0)
                                adverts.append(list1)
                                adverts.append(list2)
        
                        
                
                l=[]
                lenadv = np.shape(adverts)
                lenadv = lenadv[0]-1
                thetaofvornadj = []
                phiofvornadj = []
                zvals=[]

                for s in range(lenadv):
                    point=adverts[s]
                    ax_3ddebug2.scatter(point[0], point[1], point[2], color='black')
                
        
                for s in range(lenadv):
                    start=adverts[s]
                    end=adverts[(s+1)]
                    if start!=end:
                        result=geometric_slerp(start, end, t_vals)
                        ax_3ddebug2.plot(result[..., 0], result[..., 1], result[..., 2], color='k')
                        for l in range(len(result[...,0])):
                            pathpoint=(result[l,0], result[l,1], result[l,2])
                            zvals.append(result[l, 2])
                            fullpath.append(pathpoint)
                            
                start=adverts[-1]
                end=adverts[0]
                if start!=end:
                    result=geometric_slerp(start, end, t_vals)
                    ax_3ddebug2.plot(result[..., 0], result[..., 1], result[..., 2], color='k')
                    for l in range(len(result[..., 0])):
                        pathpoint=(result[l,0],result[l,1],result[l,2])
                        fullpath.append(pathpoint)
                        zvals.append(result[l, 2])
                
                zvalmiddle = (max(zvals)+min(zvals))/2
                print(zvalmiddle)
                for s in range(len(sphericalzonearea)):
                    if zvalmiddle < zlevel[s+1] and zvalmiddle > zlevel[s]:
                        pathDF.at[dfind, 'Patch Area']=sphericalzonearea[s]/numtheta
                    if dfind==0:
                        pathDF.at[dfind, 'Patch Area']=sphericalzonearea[-1]

                
                
                
                lenfpath=np.shape(fullpath)
                lenfpath=lenfpath[0]
                    
                cartpath=[]
                
                for s in range(lenfpath):
                    point=fullpath[s]
                    new_x = (point[0]/(1-point[2]))
                    new_y = (point[1]/(1-point[2]))
                    thetaofvornadj.append(np.arccos(point[2]))
                    if new_y >= 0:
                        if new_x >= 0:
                            phiofvornadj.append((np.arctan(new_y/new_x)))
                        elif new_x < 0:
                            phiofvornadj.append((np.arctan(new_y/new_x))+np.pi)
                    elif new_y < 0:
                        if new_x >=0:    
                            phiofvornadj.append((np.arctan(new_y/new_x))+2*np.pi)
                        elif new_x < 0:
                            phiofvornadj.append((np.arctan(new_y/new_x))+np.pi)
                    x = phiofvornadj[-1]
                    y = np.rad2deg(thetaofvornadj[-1])
                    if x<0:
                        x = x+(2*np.pi)
                    vert = (x,y)
                    vertsadj.append(vert)
                    
                    cart_x = (np.sin(thetaofvornadj[-1]))*(np.cos(phiofvornadj[-1]))
                    cart_y = (np.sin(thetaofvornadj[-1]))*(np.sin(phiofvornadj[-1]))
                    cartpathpt=(cart_x, cart_y)
                    cartpath.append(cartpathpt)
        
                for s in range(len(phiofvorn)):
                    x = phiofvorn[s]
                    y = np.rad2deg(thetaofvorn[s])
                    if x<1:
                        x = x+(2*np.pi)
                    vert = (x, y)
                    verts.append(vert)                    
                else:
                    pass
                
                for s in range(len(vertsadj)):
                    if s == 0:
                        codesadj.append(Path.MOVETO)
                    else:
                        codesadj.append(Path.LINETO)
    
                path = Path(vertsadj, codesadj)   
                rand1=random.random()
                rand2=random.random()
                rand3=random.random()
                random_color=(rand1, rand2, rand3)
                patch=patches.PathPatch(path, facecolor=random_color)          
                # ax_polardebug.add_patch(patch)
                
                cartpathcon=Path(cartpath, codesadj)
                pathDF.at[dfind, 'Path'] = path
                pathDF.at[dfind, 'Cartesian Path'] = cartpathcon
                
                dfind=dfind+1
                fullpath=np.asarray(fullpath)
                
                
                
                # fullpath_dupe=fullpath[0::10]
                # print(fullpath)
                
                # patch=patches.PathPatch(cartpathcon, facecolor='white')    
                # ax_cartdebug.add_patch(patch)
                # surfx, surfy=np.meshgrid(fullpath_dupe[:,0], fullpath_dupe[:,1])
                # surfz=np.sqrt(1-(surfx*surfx)+(surfy*surfy))
                # ax_3ddebug2.plot_surface(surfx, surfy, surfz, color=random_color, cstride=1, rstride=1)
            
            
            ax_cartdebug.set_ylabel('Y')
            ax_cartdebug.set_xlabel('X')
            ax_cartdebug.axis('off')
            
            
            shapes = pathDF['Patch Area']
            # print(shapes)
            print(pathDF['Patch Area'])
            pathDF = pathDF.set_index('Cartesian Path')
            for path in pathDF.index:
                if path.contains_point([x_t, y_t]):
                    patch=patches.PathPatch(path, facecolor='k')
                    ax_cartdebug.add_patch(patch)
                else:
                    patch=patches.PathPatch(path, facecolor='white')
                    ax_cartdebug.add_patch(patch)
                
            voronoimappingtime=time.time()
            print('Time to map voronoi cells to cartesian coordinates and store this data:', voronoimappingtime-voronoitime)
            ax_3ddebug2.plot([0, x_t], [0, y_t], [0, z_t], linewidth=6, color='gray')
            # ax_3ddebug2.scatter(x_t, y_t, z_t, s=20, color='red')
        # print(notavar)
        # break   
        # print(yet)
        for subvolume in datasheet.index:
            phidat =list(ast.literal_eval(datasheet['Column_1'][subvolume]))
            thetadat = list(ast.literal_eval(datasheet['Column_0'][subvolume]))
            shapedat=list(ast.literal_eval(datasheet[PORESHAPECOL][subvolume]))
            voldat=list(ast.literal_eval(datasheet[POREVOLCOL][subvolume]))
            thetadat2 = []
            xdat=[]
            ydat=[]
            
            
            totlen=(totlen+(len(thetadat)))
            for i in range(len(thetadat)):
                if thetadat[i]<0:
                    thetadat2.append(360+thetadat[i])##Do some conversions since python wants data from 0-360degrees and PerGeos gives it in -180 to 180.
                else:
                    thetadat2.append(thetadat[i])
                tot_phi.append(phidat[i])
                tot_theta.append(thetadat2[i])
                xdat.append(np.sin(np.deg2rad(phidat[i]))*(np.cos(np.deg2rad(thetadat2[i]))))
                ydat.append(np.sin(np.deg2rad(phidat[i]))*(np.sin(np.deg2rad(thetadat2[i]))))
                tot_xdat.append(np.sin(np.deg2rad(phidat[i]))*(np.cos(np.deg2rad(thetadat2[i]))))
                tot_ydat.append(np.sin(np.deg2rad(phidat[i]))*(np.sin(np.deg2rad(thetadat2[i]))))
                tot_zdat.append(np.cos(np.deg2rad(phidat[i])))
                x_weightvol.append(tot_xdat[-1]*(voldat[i]/maxvol))
                y_weightvol.append(tot_ydat[-1]*(voldat[i]/maxvol))
                radius_weightvol.append((np.sqrt(tot_xdat[-1]*tot_xdat[-1]+tot_ydat[-1]*tot_ydat[-1]))*(voldat[i]/maxvol))
                if voronoi_desired=='yes':
                    onesubpatchstarttime = time.time()
                    for path in pathDF.index:
                        if path.contains_point([xdat[-1], ydat[-1]]):
                            # print(path, ([xdat[-1], ydat[-1]]))
                            if pd.isna(pathDF.at[path, 'Volume Sums']) == False:
                                pathDF.at[path, 'Volume Sums'] = pathDF.at[path, 'Volume Sums'] + voldat[i]
                                pathDF.at[path, 'Shape Sums'] = pathDF.at[path, 'Shape Sums'] + shapedat[i]
                            elif pd.isna(pathDF.at[path, 'Volume Sums']) == True:
                                pathDF.at[path, 'Volume Sums'] = voldat[i]
                                pathDF.at[path, 'Shape Sums'] = shapedat[i]
                            if shapedat[i]>meanshape:
                                if pd.isna(pathDF.at[path, 'Filtered Volume Sums']) == False:
                                    pathDF.at[path, 'Filtered Volume Sums'] = pathDF.at[path, 'Filtered Volume Sums'] + voldat[i]
                                elif pd.isna(pathDF.at[path, 'Filtered Volume Sums']) == True:
                                    pathDF.at[path, 'Filtered Volume Sums'] = voldat[i]
            if voronoi_desired=='yes':
                onesubpatchtime = time.time()
                # print('Time to see what patch every point within one subvolume is in:', onesubpatchtime-onesubpatchstarttime)
            phidat=np.array(phidat)
            thetadat2 = np.deg2rad(thetadat2)
            voldat=np.array(voldat)
            

        
            
            fig_thissubvol=plt.figure()
            ax_thissubvol=fig_thissubvol.add_subplot(1,1,1, projection='polar')
            ax_thissubvol.scatter((thetadat2), phidat, c=cmap(normalcolor_vol(voldat)))
            


            try:
                fig_cartvol = plt.figure()
                ax_cartvol = fig_cartvol.add_subplot()
                plt.xlim([-1, 1])
                plt.ylim([-1, 1])
                ax_cartvol.set_aspect('equal')
                ax_cartvol.tricontourf(xdat, ydat, voldat,levels=levelcount, cmap=cmapdesi, vmin=minvol, vmax=maxvol)
                # ax_cartvol.scatter(xdat, ydat, c=cmap(normalcolor_vol(voldat)))
                ax_polarvol = fig_cartvol.add_axes(ax_cartvol.get_position(), polar=True, frameon=False)
                ax_polarvol.set_rmax(90)
                ax_cartvol.axis('off')
                
                
                
                fig_cartshape = plt.figure()
                ax_cartshape = fig_cartshape.add_subplot()
                plt.xlim([-1, 1])
                plt.ylim([-1, 1])
                ax_cartshape.set_aspect('equal')
                ax_cartshape.tricontourf(xdat, ydat, shapedat,levels=levelcount, cmap=cmapdesi, vmin=minshape, vmax=maxshape)
                # ax_cartshape.scatter(xdat, ydat, c=cmap(normalcolor_shape(shapedat)))
                ax_polarshape = fig_cartshape.add_axes(ax_cartshape.get_position(), polar=True, frameon=False)
                ax_polarshape.set_rmax(90)
                ax_cartshape.axis('off')
                
                
                
                
                
                savestrvol = savefpath + str(subvolume) +'VolumeColor'+ '.pdf'
                savestrshape = savefpath + str(subvolume) +'ShapeColor'+ '.pdf'
                fig_cartvol.savefig(savestrvol)
                fig_cartshape.savefig(savestrshape)
                plt.close('all')
            except:
                pass
            
        
        
        meandirection_weightvol=math.degrees(math.atan2(sum(x_weightvol), sum(y_weightvol)))
        meanradius_weightvol=sum(radius_weightvol)/(len(tot_xdat))
        x_of_weightvol_circular_mean=meanradius_weightvol*(np.cos(np.deg2rad(meandirection_weightvol)))
        y_of_weightvol_circular_mean=meanradius_weightvol*(np.sin(np.deg2rad(meandirection_weightvol)))
        # print(meandirection_weightvol, meanradius_weightvol)
        # print(x_of_weightvol_circular_mean, y_of_weightvol_circular_mean)
        
        
        voldattot=np.array(tot_vol)
        shapedattot=np.array(tot_shape)
        tot_xdat = np.array(tot_xdat)
        tot_ydat = np.array(tot_ydat)
        
        
        logvolmin = math.log10(min(voldattot))
        logvolmax=math.log10(max(voldattot))
        # vollogspace = np.linspace(logvolmin, logvolmax, num=60)
        # vollogspace=10**vollogspace
        
        logshapemin = math.log10(min(shapedattot))
        logshapemax=math.log10(max(shapedattot))
        shapelogspace = np.linspace(logshapemin, logshapemax, num=7)
        shapelogspace=10**shapelogspace
        # for i in range(len(voldattot)):
        #     voldattot[i] = math.log10(voldattot[i])
        
        
        fig_allpointsvol=plt.figure()
        ax_allpointsvol=fig_allpointsvol.add_subplot()
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        ax_allpointsvol.set_aspect('equal')
        volcont=ax_allpointsvol.tricontourf(tot_xdat, tot_ydat, (voldattot), levels=levelcount, cmap=cmapdesi, vmin=minvol, vmax=maxvol)
        cb=fig_allpointsvol.colorbar(volcont)
        cb.set_label('Melt Volume')
        ax_allpointsvol.tricontourf(tot_xdat, tot_ydat, (voldattot), levels=levelcount, cmap=cmapdesi, vmin=minvol, vmax=maxvol)
        # ax_allpointsvol.scatter(tot_xdat, tot_ydat)
        ax_polarvoltot = fig_allpointsvol.add_axes(ax_allpointsvol.get_position(), polar=True, frameon=False)
        ax_polarvoltot.set_rmax(90)
        ax_allpointsvol.set_title('All points (Inertia Measurement) with Filled Contour on Melt Volume')
        ax_allpointsvol.axis('off')
        
        
        fig_allpointsshape=plt.figure()
        ax_allpointsshape=fig_allpointsshape.add_subplot()
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        ax_allpointsshape.set_aspect('equal')
        shapecont=ax_allpointsshape.tricontourf(tot_xdat, tot_ydat, shapedattot, cmap=cmapdesi, levels=levelcount, vmin=minshape, vmax=maxshape)
        cb=fig_allpointsshape.colorbar(shapecont)
        cb.set_label('Melt Pocket Shape')
        ax_allpointsshape.tricontourf(tot_xdat, tot_ydat, shapedattot, cmap=cmapdesi, levels=levelcount, vmin=minshape, vmax=maxshape)
        # ax_allpointsshape.scatter(tot_xdat, tot_ydat)
        ax_polarshapetot = fig_allpointsshape.add_axes(ax_allpointsshape.get_position(), polar=True, frameon=False)
        ax_polarshapetot.set_rmax(90)
        ax_allpointsshape.set_title('All points (Inertia Measurement) with Filled Contour on Melt Shape')
        ax_allpointsshape.axis('off')
        
        pathDF = pathDF.reset_index()
        if voronoi_desired=='yes':
            patcharea=pathDF['Patch Area']
            shapesums=pathDF['Shape Sums']
            volsums=pathDF['Volume Sums']
            filteredvolsums=pathDF['Filtered Volume Sums']
            weightvolsums=[]
            weightshapesums=[]
            weightfvolsums=[]
            for i in range(len(shapesums)):
                weightshapesums.append(shapesums[i]/(patcharea[i]*sumvol))
                weightvolsums.append(volsums[i]/(patcharea[i]*sumvol))
                weightfvolsums.append(filteredvolsums[i]/(patcharea[i]*sumvol))
            # print(volsums)
            # print(weightvolsums)
            normcol_shapemax = matplotlib.colors.Normalize(vmin=(0), vmax=(.8))
            normcol_volmax = matplotlib.colors.Normalize(vmin=(0), vmax=(.7))    
            normcol_volume=matplotlib.colors.Normalize(vmin=min(volsums), vmax=max(volsums))
            normcol_fvolmax=matplotlib.colors.Normalize(vmin=(0), vmax=(.7))
            fig_voronoisumvol = plt.figure()
            ax_voronoisumvol = fig_voronoisumvol.add_subplot(1,1,1, projection='polar')
            ax_voronoisumvol.set_rmax(90)
            ax_voronoisumvol.set_theta_direction(-1)
            ax_voronoisumvol.set_theta_zero_location('N')
            fig_voronoisumshape = plt.figure()
            ax_voronoisumshape = fig_voronoisumshape.add_subplot(1,1,1, projection='polar')
            ax_voronoisumshape.set_rmax(90)
            ax_voronoisumshape.set_theta_direction(-1)
            ax_voronoisumshape.set_theta_zero_location('N')
            ax_voronoisumvol.axes.xaxis.set_ticklabels([])
            ax_voronoisumvol.axes.yaxis.set_ticklabels([])
            ax_voronoisumshape.axes.xaxis.set_ticklabels([])
            ax_voronoisumshape.axes.yaxis.set_ticklabels([])
            pathDF = pathDF.set_index('Path')
            fig_voronoivol=plt.figure()
            ax_voronoivol=fig_voronoivol.add_subplot(1,1,1, projection='polar')
            ax_voronoivol.set_rmax(90)
            ax_voronoivol.set_theta_direction(-1)
            ax_voronoivol.set_theta_zero_location('N')
            ax_voronoivol.axes.xaxis.set_ticklabels([])
            ax_voronoivol.axes.yaxis.set_ticklabels([])
            fig_fvolsums=plt.figure()
            ax_fvolsums=fig_fvolsums.add_subplot(1,1,1, projection='polar')
            ax_fvolsums.set_rmax(90)
            ax_fvolsums.set_theta_direction(-1)
            ax_fvolsums.set_theta_zero_location('N')
            ax_fvolsums.axes.xaxis.set_ticklabels([])
            ax_fvolsums.axes.yaxis.set_ticklabels([])
            i=0
            for path in pathDF.index:
                weightedvol = (weightvolsums[i])
                weightedshape = (weightshapesums[i])
                fvol=(weightfvolsums[i])
                vol=volsums[i]
                patch_volcol=patches.PathPatch(path, facecolor=cmap(normcol_volmax(weightedvol)))
                patch_shapecol=patches.PathPatch(path, facecolor=cmap(normcol_shapemax(weightedshape)))
                patch_fvol=patches.PathPatch(path, facecolor=cmap(normcol_fvolmax(fvol)))
                patchvol=patches.PathPatch(path, facecolor=cmap(normcol_volume(vol)))
                ax_voronoisumvol.add_patch(patch_volcol)
                ax_voronoisumshape.add_patch(patch_shapecol)
                ax_voronoivol.add_patch(patchvol)
                ax_fvolsums.add_patch(patch_fvol)
                i=i+1
            cbar=plt.colorbar(cm.ScalarMappable(norm=normcol_volmax, cmap=cmap), ax=ax_voronoisumvol)
            cbar=plt.colorbar(cm.ScalarMappable(norm=normcol_shapemax, cmap=cmap), ax=ax_voronoisumshape)
            cbar=plt.colorbar(cm.ScalarMappable(norm=normcol_fvolmax, cmap=cmap), ax=ax_fvolsums)
            # ax_voronoisumvol.set_title('All Melt Pockets (Inertia Measurement) Color-map on Sum of Melt Pocket Volume / Patch Area')
            # ax_voronoisumshape.set_title('All Melt Pockets (Inertia Measurement) Color-map on Sum of Melt Pocket Shape Factor / Patch Area')
            # ax_fvolsums.set_title('Melt Pockets where shape>mean(shape)- Colormap shows sum of MP volume / Patch Area')
        volsumstr = savefpath +'VoronoiNormVol_' + scannum + '.svg'
        shapesumstr = savefpath +'VoronoiNormShape_' + scannum + '.svg'
        volfiltstr = savefpath +'VoronoiNormVol_filtered_' + scannum + '.svg'
        fig_voronoisumvol.savefig(volsumstr)
        fig_voronoisumshape.savefig(shapesumstr)
        fig_fvolsums.savefig(volfiltstr)
        volsumstr = savefpath +'VoronoiNormVol_' + scannum + '_2.pdf'
        shapesumstr = savefpath +'VoronoiNormShape_' + scannum + '_2.pdf'
        volfiltstr = savefpath +'VoronoiNormVol_filtered_' + scannum + '_2.pdf'
        fig_voronoisumvol.savefig(volsumstr)
        fig_voronoisumshape.savefig(shapesumstr)
        fig_fvolsums.savefig(volfiltstr)
        
        pathDF.to_csv(savefpath + 'SumPathDataframe_' + scannum + '.csv')
        
        
        DF_to_output=pd.DataFrame(columns=['X Coordinates', 'Y Coordinates', 'Z Coordinates', 'Melt Pocket Volume'])
        DF_to_output['X Coordinates']=tot_xdat
        DF_to_output['Y Coordinates']=tot_ydat
        DF_to_output['Z Coordinates']=tot_zdat
        DF_to_output['Melt Pocket Volume']=tot_vol
        DF_to_output.to_csv(savefpath + 'AllMPCoord_' + scannum + '.csv')
        mpcount=totlen/(len(datasheet.index))
        print('number of melt pockets per subvolume:' + str(mpcount))
        
        
        return
        


def orient_eigencheck(DFLIST):
    ##size order: 20, 50, 100, 200 (increasing)
    colorsum=['red', 'orange', 'yellow', 'pink']
    sizestring=['20-pixel$^3$','50-pixel$^3$','100-pixel$^3$','200-pixel$^3$']
    fig_eigvecsum = plt.figure()
    ax_eigvecsum=fig_eigvecsum.add_subplot(1,1,1, projection='polar')
    ax_eigvecsum.set_rmax(90)
    ax_eigvecsum.set_theta_direction(-1)
    ax_eigvecsum.set_theta_zero_location('N')
    
    for l in range(len(DFLIST)):
        DF=DFLIST[l]
        tot_xdat = DF['X Coordinates']
        tot_ydat = DF['Y Coordinates']
        tot_zdat = DF['Z Coordinates']
        tot_vol = DF['Melt Pocket Volume']
        Orientation_Matrix=np.zeros((3,3))
        totalmarkersize=600
        sumx=[0]
        sumy=[0]
        sumz=[0]
        summelt=sum(tot_vol)
        print(summelt)
        normeig=[]
        e11=(0)
        e12=(0)
        e22=(0)
        alpha=.9
        numpockets=len(tot_xdat)
        maxvol = max(tot_vol)
        minvol = min(tot_vol)
        cmapdesi='Reds'
        cmap=matplotlib.cm.get_cmap(cmapdesi)
        normalcolor_vol=matplotlib.colors.Normalize(vmin=minvol, vmax=maxvol)
        normalcolor_eignorm=matplotlib.colors.Normalize(vmin=0, vmax=1)
        
        
        
        for i in range(len(tot_xdat)):
            Orientation_Matrix[0, 0] = Orientation_Matrix[0, 0] + tot_xdat[i]*tot_xdat[i]*tot_vol[i]
            Orientation_Matrix[0, 1] = Orientation_Matrix[0, 1] + tot_xdat[i]*tot_ydat[i]*tot_vol[i]
            Orientation_Matrix[0, 2] = Orientation_Matrix[0, 2] + tot_xdat[i]*tot_zdat[i]*tot_vol[i]
            Orientation_Matrix[1, 0] = Orientation_Matrix[1, 0] + tot_xdat[i]*tot_ydat[i]*tot_vol[i]
            Orientation_Matrix[1, 1] = Orientation_Matrix[1, 1] + tot_ydat[i]*tot_ydat[i]*tot_vol[i]
            Orientation_Matrix[1, 2] = Orientation_Matrix[1, 2] + tot_ydat[i]*tot_zdat[i]*tot_vol[i]
            Orientation_Matrix[2, 0] = Orientation_Matrix[2, 0] + tot_zdat[i]*tot_xdat[i]*tot_vol[i]
            Orientation_Matrix[2, 1] = Orientation_Matrix[2, 1] + tot_ydat[i]*tot_zdat[i]*tot_vol[i]
            Orientation_Matrix[2, 2] = Orientation_Matrix[2, 2] + tot_zdat[i]*tot_zdat[i]*tot_vol[i]
            sumx=sumx+(tot_vol[i]*tot_xdat[i])
            sumy=sumy+(tot_vol[i]*tot_ydat[i])
            sumz=sumz+(tot_vol[i]*tot_zdat[i])
        
        print(Orientation_Matrix)
        sumxsquare=sumx*sumx
        sumysquare=sumy*sumy
        sumzsquare=sumz*sumz
        [eigval, normeigvec] = np.linalg.eig(Orientation_Matrix)
        
        for i in range(len(eigval)):
            normeig.append(eigval[i]/summelt)
    
        normeig=np.array(normeig)
        idx=normeig.argsort()[::-1]
        normeig=normeig[idx]
        normeigvec=normeigvec[:,idx]
        print(normeig)
        
        
        ##Find rotation needed to put shortest eigenvector at the Z axis
        pitch_shorteig=np.arccos(normeigvec[2,2])
        # print(pitch_shorteig)
        if normeigvec[0,2]>0.0:
            theta_shorteig=np.arctan(normeigvec[1,2]/normeigvec[0,2])
        elif normeigvec[0,2]<0.0:
            if normeigvec[1,2]>=0.0:
                theta_shorteig=np.arctan(normeigvec[1,2]/normeigvec[0,2])+3.14159
            elif normeigvec[1,2]<0.0:
                theta_shorteig=np.arctan(normeigvec[1,2]/normeigvec[0,2])-3.14159
        elif normeigvec[0, 2]==0.0:
            if normeigvec[1, 2]>0.0:
                theta_shorteig=3.14159/2
            elif normeigvec[1, 2]<0.0:
                theta_shorteig=-3.14159/2
            elif normeigvec[1,2]==0.0:
                theta_shorteig=0
            
        
        # print(theta_shorteig)
        theta_adjust=3.14159-(abs(theta_shorteig))
        # print(theta_adjust)
        ##so, each melt vector must be rotated by these amounts. 
        
        
        if normeigvec[2,0]*normeigvec[2,1]<0:
            for i in range(len(normeigvec[0])):
                normeigvec[i,0]=normeigvec[i,0]*-1
            
        ##Below is some initial preparation for statistical tests if a girdle distribution (could easily be modified for bipolar)
        
        
        eigv1=normeigvec[:,2]
        eigv2=normeigvec[:,1]
        eigv3=normeigvec[:,0]
        eigval1=normeig[2]
        eigval2=normeig[1]
        eigval3=normeig[0]
        # print(eigv1, eigv2, eigv3)
        Orientation_Matrix_2=np.zeros((3,3))
        for i in range(len(tot_xdat)):
            meltvector=[tot_xdat[i], tot_ydat[i], tot_zdat[i]]
                    
            dot1 = np.dot(eigv1, meltvector)
            dot2 = np.dot(eigv3, meltvector)
            dot3 = np.dot(eigv2, meltvector)
            e11 = e11+((dot1*dot1)*(dot2*dot2))*(summelt)
            e22 = e22+((dot1*dot1)*(dot3*dot3))*(summelt)
            e12 = e12+((dot2*dot3*(dot1*dot1)))*(summelt)
        
        
        
        
        
        e11 = e11*(1/((summelt)*((eigval1-eigval3)*(eigval1-eigval3))))
        e22 = e22*(1/((summelt)*((eigval1-eigval2)*(eigval1-eigval2))))
        e12 = e12*(1/((summelt)*((eigval1-eigval2)*(eigval1-eigval3))))
        
        # print(e11, e12, e22)
        E_MAT = np.array([[e11, e12], [e12, e22]])
        # print(E_MAT)
        F_MAT = np.linalg.inv(E_MAT)
        rsquare=(sumxsquare+sumysquare+sumzsquare)
        theta=[]
        phi=[]
        eigveccol=normeigvec.shape
        eigveccol=eigveccol[1]
        for i in range(eigveccol):
            x = normeigvec[0, i]
            y = normeigvec[1, i]
            z = normeigvec[2, i]
            
            if z < 0:
                print('z is negative')
                x = x*-1
                y = y*-1
                z= z*-1
            
            new_x=x/(1-z)
            new_y=y/(1-z)
    
            phi.append(np.rad2deg(np.arccos((z))))
            if new_y>=0:
                if new_x>=0:
                    theta.append((np.arctan(new_y/new_x)))
                elif new_x<0:
                    theta.append((np.arctan(new_y/new_x))+np.pi)
            elif new_y<0:
                if new_x>=0:
                    theta.append((np.arctan(new_y/new_x))+2*np.pi)
                elif new_x<0:
                    theta.append((np.arctan(new_y/new_x))+np.pi)
        
            
        vec_length=np.sqrt(rsquare)
        x_res_vec=sumx/vec_length
        y_res_vec=sumy/vec_length
        z_res_vec=sumz/vec_length
        resultant_vector=[x_res_vec, y_res_vec, z_res_vec]
        mean_vec_length=vec_length/summelt
        resvectest=(mean_vec_length*(numpockets))**2
        print(resvectest*3/numpockets)
        scale=summelt/numpockets
        rsquaretest=3*rsquare/(summelt)/scale
        rsquaretest_two=3*rsquare/numpockets
        print(rsquaretest, rsquaretest_two)
        print(mean_vec_length)
        if rsquaretest>7.81:
            print('R-Square Test:' + str(rsquaretest) + '...R-Square test rejects uniform distribution at 95% confidence interval')
        else:
            print('R-Square test accepts uniform distribution at 95% confidence interval compared to unimodal estimation', str(rsquaretest))
        ttestval = .33-(1.038/np.sqrt(100))
        ttestval_2 = .33+(1.038/np.sqrt(100))
        if normeig[2]<ttestval:
            print('Distribution is better fit by a girdle model than uniform model @95% confidence interval', str(ttestval))
        else:
            print('Distribution is better fit by a uniform model than a girdle model @95% confidence interval', str(ttestval))
        if normeig[0]>ttestval_2:
            print('Distribution is better fit by a bimodal model than uniform model @95% confidence interval', str(ttestval_2))
        else:
            print('Distribution is better fit by a uniform model than a bimodal model @95% confidence interval', str(ttestval_2))
        
        
        print(normeig)
        logcheck=(np.log10(normeig[1]/normeig[2]))
        logcheck_two=(np.log10(normeig[0]/normeig[1]))
        print(logcheck, logcheck_two)
        print('shape parameter:' + str(logcheck_two/logcheck))
        print('strength criterion:' + str(np.log10(normeig[0]/normeig[2])))
        
        fig_eigvec = plt.figure()
        ax_eigvec=fig_eigvec.add_subplot(1,1,1, projection='polar')
        labels=[]
        labels.append('Longest Eigen Vector')
        labels.append('Middle Eigen Vector')
        labels.append('Shortest Eigen Vector')
        c=[cmap(normalcolor_eignorm(eigval3)), cmap(normalcolor_eignorm(eigval2)), cmap(normalcolor_eignorm(eigval1))]
        for i in range(len(theta)):
            size=normeig[i]*totalmarkersize
            # ax_eigvec.scatter(theta[i], phi[i], label=labels[i], color=c[i], s=size)
            ax_eigvec.scatter(theta[i], phi[i], label=labels[i], color='darkorange', s=size)
            sumlabel=(sizestring[l]+labels[i])
            ax_eigvecsum.scatter(theta[i], phi[i], label=sumlabel, color=colorsum[l], s=size)
        ax_eigvec.set_rmax(90)
        ax_eigvec.set_theta_direction(-1)
        ax_eigvec.set_theta_zero_location('N')
        # leg=ax_eigvec.legend()
        
        
        fig_3deig=plt.figure()
        ax_3deig=fig_3deig.add_subplot(projection='3d')
        
        for i in range(len(theta)):
            xs=[0, normeigvec[0, i]]
            ys=[0, normeigvec[1, i]]
            zs=[0, normeigvec[2, i]]
            # ax_3deig.plot3D(xs, ys, zs, label=labels[i], color=c[i], linewidth=4)
            # ax_3deig.plot3D(xs, ys, zs, label=labels[i], color='darkorange', linewidth=4)
        ax_3deig.set_xlabel('X Axis')
        ax_3deig.set_ylabel('Y Axis')
        ax_3deig.axes.set_xlim3d(left=-1, right=1) 
        ax_3deig.axes.set_ylim3d(bottom=-1, top=1) 
        ax_3deig.axes.set_zlim3d(bottom=-1, top=1) 
        
        
        ##Below is a series of tests to determine if the shortest eigenvector is indeed the pole to the girdle plane,
        ##and what the confidence ellipse around this pole is! See Fisher, Lewis, and Embleton, pages 35, 162, and 180. 
        
        A_coeff = (F_MAT[0, 0])
        B_coeff = (F_MAT[0, 1])
        C_coeff = (F_MAT[1, 1])
        D_coeff = (-2*np.log10(alpha)/(summelt))
        print(A_coeff, B_coeff, C_coeff, D_coeff)
        Z = np.array([[A_coeff, B_coeff], [B_coeff, C_coeff]])
        print(Z)
        [eigval_test1, eigvect_test1]= np.linalg.eig(Z)
        tcoeff_1=(eigval_test1[0])
        tcoeff_2=(eigval_test1[1])
        
        alpha_COEFF=(A_coeff-C_coeff)/(2*B_coeff)+np.sqrt((A_coeff-C_coeff)*(A_coeff-C_coeff)/(4*B_coeff*B_coeff)+1)
        lower_a = alpha_COEFF/(np.sqrt(1+alpha_COEFF*alpha_COEFF))
        lower_b = 1/(np.sqrt(1+alpha_COEFF*alpha_COEFF))
        
        lower_a=eigvect_test1[0, 0]
        lower_b=eigvect_test1[1, 0]
        print(lower_a, lower_b)
        
        gcoeff_1=np.sqrt(D_coeff/tcoeff_1)
        gcoeff_2=np.sqrt(D_coeff/tcoeff_2)
        
        print(tcoeff_1, tcoeff_2, gcoeff_1)
        confellipse_x=[]
        confellipse_y=[]
        confellipse_z=[]
        t_vals=5
        confellipse_angles = np.linspace(0, 6.28, num=t_vals)
        for angle in confellipse_angles:
            vcoeff_1 = gcoeff_1*np.cos(angle)        
            vcoeff_2 = gcoeff_2*np.sin(angle)
            confellipse_x.append(lower_a*vcoeff_1-lower_b*vcoeff_2)
            confellipse_y.append(lower_b*vcoeff_1+lower_a*vcoeff_2)
            confellipse_z.append(np.sqrt(1-(confellipse_x[-1]*confellipse_x[-1])-(confellipse_y[-1]*confellipse_y[-1])))
                
        # print(confellipse_x, confellipse_y, confellipse_z)
        # for i in range(t_vals):
        #     confellipse_x2=confellipse_x[i]*np.cos(-pitch_shorteig)+confellipse_z[i]*np.sin(-pitch_shorteig)
        #     confellipse_y2=confellipse_y[i]
        #     confellipse_z2=-confellipse_x[i]*np.sin(-pitch_shorteig)+confellipse_z[i]*np.cos(-pitch_shorteig)
        #     confellipse_x3=confellipse_x2*np.cos(theta_adjust)-confellipse_y2*np.sin(theta_adjust)
        #     confellipse_y3=confellipse_x2*np.sin(theta_adjust)+confellipse_y2*np.cos(theta_adjust)
        #     confellipse_z3=confellipse_z2
            
            
        #     ax_3deig.scatter(confellipse_x[i], confellipse_y[i], confellipse_z[i], c='purple', s=.1)
        #     ax_3deig.scatter(confellipse_x3, confellipse_y3, confellipse_z3, c='yellow', s=10)
            
        # for i in range(len(tot_xdat)):
        #     ax_3deig.scatter(tot_xdat[i], tot_ydat[i], tot_zdat[i], color='k')
        
        
        [eigval, normeigvec] = np.linalg.eig(Orientation_Matrix_2)
        
        normeig=[]
        for i in range(len(eigval)):
            normeig.append(eigval[i]/summelt)
    
        normeig=np.array(normeig)
        idx=normeig.argsort()[::-1]
        normeig=normeig[idx]
        normeigvec=normeigvec[:,idx]
        # for i in range(len(normeig)):
        #     xs=[0, normeigvec[0, i]]
        #     ys=[0, normeigvec[1, i]]
        #     zs=[0, normeigvec[2, i]]
        #     ax_3deig.plot3D(xs, ys, zs, label=labels[i], color=c[i], linewidth=4)
        # for i in range(len(tot_xdat)):
        #     ax_3deig.scatter(tot_xdat[i], tot_ydat[i], tot_zdat[i], color=cmap(normalcolor_vol(tot_vol[i])), s=1, alpha=0.50)
            
        
        print('Rotate -' + str(np.rad2deg(pitch_shorteig)) + 'around the Y, then ' + str(np.rad2deg(theta_adjust)) + ' around the Z')
    leg=ax_eigvecsum.legend()
        
    return(Orientation_Matrix, resultant_vector, mean_vec_length, normeig, normeigvec)
            
def check_if_inband(Spreadsheetfpath, YSpreadsheetPath):
    datasheet = pd.read_csv(Spreadsheetfpath)
    datasheet.set_index('Subvolume Name')
    ydatasheet = pd.read_csv(YSpreadsheetPath)
    min_ratio = ydatasheet['EDPrismMeltRatio'].min()
    max_ratio = ydatasheet['EDPrismMeltRatio'].max()
    min_ymelt = ydatasheet['Total Melt'].min()
    max_ymelt = ydatasheet['Total Melt'].max()
    print(str(min_ratio) + str(max_ratio) + str(min_ymelt) + str(max_ymelt))
    for subvolume in datasheet.index:
        if datasheet.loc[subvolume, 'EDPrismMeltRatio'] > min_ratio and datasheet.loc[subvolume, 'EDPrismMeltRatio'] < max_ratio and datasheet.loc[subvolume, 'Total Melt'] < max_ymelt and datasheet.loc[subvolume, 'Total Melt'] > min_ymelt:
            datasheet.loc[subvolume, 'Quadrant'] = 7
        elif datasheet.loc[subvolume, 'EDPrismMeltRatio'] > max_ratio and datasheet.loc[subvolume, 'Total Melt'] < min_ymelt:
            datasheet.loc[subvolume, 'Quadrant'] = 2
        elif datasheet.loc[subvolume, 'EDPrismMeltRatio'] < min_ratio and datasheet.loc[subvolume, 'Total Melt'] > max_ymelt:
            datasheet.loc[subvolume, 'Quadrant'] = 5
        elif datasheet.loc[subvolume, 'EDPrismMeltRatio'] > min_ratio and datasheet.loc[subvolume, 'EDPrismMeltRatio'] < max_ratio and datasheet.loc[subvolume, 'Total Melt'] < min_ymelt:
            datasheet.loc[subvolume, 'Quadrant'] = 1
        elif datasheet.loc[subvolume, 'EDPrismMeltRatio'] > min_ratio and datasheet.loc[subvolume, 'EDPrismMeltRatio'] < max_ratio and datasheet.loc[subvolume, 'Total Melt'] > max_ymelt:
            datasheet.loc[subvolume, 'Quadrant'] = 6
        elif datasheet.loc[subvolume, 'Total Melt'] < max_ymelt and datasheet.loc[subvolume, 'Total Melt'] > min_ymelt and datasheet.loc[subvolume, 'EDPrismMeltRatio'] > max_ratio:
            datasheet.loc[subvolume, 'Quadrant'] = 3
        elif datasheet.loc[subvolume, 'Total Melt'] < max_ymelt and datasheet.loc[subvolume, 'Total Melt'] > min_ymelt and datasheet.loc[subvolume, 'EDPrismMeltRatio'] < min_ratio:
            datasheet.loc[subvolume, 'Quadrant'] = 4
        elif datasheet.loc[subvolume, 'EDPrismMeltRatio'] > max_ratio and datasheet.loc[subvolume, 'Total Melt'] > max_ymelt:
            datasheet.loc[subvolume, 'Quadrant'] = 9
        elif datasheet.loc[subvolume, 'EDPrismMeltRatio'] < min_ratio and datasheet.loc[subvolume, 'Total Melt'] < min_ymelt:
            datasheet.loc[subvolume, 'Quadrant'] = 8
        datasheet.to_csv(Spreadsheetfpath)
    return       
        
        
    
    
    
def ext_domain_greaterless_prism(Spreadsheet_Fpath):
    datasheet = pd.read_csv(Spreadsheet_Fpath)
    datasheet.set_index('Subvolume Name', inplace=True)
    for subvolume in datasheet.index:
        ratio_numerator = datasheet.loc[subvolume, 'EDTotalMelt'] - datasheet.loc[subvolume, 'Total Melt']
        datasheet.loc[subvolume, 'EDPrismMeltRatio'] = ratio_numerator
        datasheet.to_csv(Spreadsheet_Fpath)
    return

    
    
    
    
    
def dominant_melt(spreadsheetfpath):
    datasheet = pd.read_csv(spreadsheetfpath)
    datasheet.set_index('Subvolume Name', inplace=True)
    for subvolume in datasheet.index:
        if datasheet.loc[subvolume, 'Quadrant'] == 1 or datasheet.loc[subvolume, 'Quadrant'] == 2 or  datasheet.loc[subvolume, 'Quadrant'] == 3:
            datasheet.loc[subvolume, 'Dominant Melt'] = datasheet.loc[subvolume, 'EDTotalMelt']
        else:
            datasheet.loc[subvolume, 'Dominant Melt'] = datasheet.loc[subvolume, 'Total Melt']
    datasheet.to_csv(spreadsheetfpath)
    return

    
    
    
def relativeconnectivity(spreadsheetfpath, meltstep):
    datasheet = pd.read_csv(spreadsheetfpath)
    datasheet.set_index('Subvolume Name', inplace=True)
    minmelt = 0
    maxmelt = max(datasheet['Total Melt'])
    binning = np.arange(minmelt, maxmelt+meltstep, meltstep).tolist()
    for i in (range(len(binning)-1)):
        val1 = binning[i]
        val2 = binning[i+1]
        for subvolume in datasheet.index:
            if datasheet.loc[subvolume, 'Total Melt'] < val2 and datasheet.loc[subvolume, 'Total Melt'] > val1:
                datasheet.loc[subvolume, 'MeltRange'] = i+1
                datasheet.loc[subvolume, 'XConnectedMeltRatio'] = datasheet.loc[subvolume, 'X Connected Melt'] / datasheet.loc[subvolume, 'Total Melt']
                datasheet.loc[subvolume, 'YConnectedMeltRatio'] = datasheet.loc[subvolume, 'Y Connected Melt'] / datasheet.loc[subvolume, 'Total Melt']
                datasheet.loc[subvolume, 'ZConnectedMeltRatio'] = datasheet.loc[subvolume, 'Z Connected Melt'] / datasheet.loc[subvolume, 'Total Melt']
            elif datasheet.loc[subvolume, 'Total Melt'] == 0:
                datasheet.loc[subvolume, 'MeltRange'] = 1
                datasheet.loc[subvolume, 'XConnectedMeltRatio'] = 0
                datasheet.loc[subvolume, 'YConnectedMeltRatio'] = 0
                datasheet.loc[subvolume, 'ZConnectedMeltRatio'] = 0
            else:
                pass
            
    datasheet.to_csv(spreadsheetfpath)
    return



        
def logdat(spreadsheetfpath, logmeltstep):
    datasheet = pd.read_csv(spreadsheetfpath)#read the spreadsheet
    datasheet.set_index('Subvolume Name', inplace=True)#set index to subvolume name
    logminmelt = np.log10(0.01)#begin binning at log10(.01) melt (because zeroes cannot be permitted for log units)
    logmaxmelt = np.log10(max(datasheet['Total Melt']))#find the log of the max melt 
    binning = np.arange(logminmelt, logmaxmelt+logmeltstep, logmeltstep).tolist()#make a list containing all dilineations of log(melt fraction)
    EDcheck = 'EDMeltRatio' in datasheet
    for i in (range(len(binning)-1)):#for each bin
        val1 = binning[i]#value one is the first, val2 is the next value
        val2 = binning[i+1]
        for subvolume in datasheet.index:#for each subvolume
            logtotmelt = np.log10(datasheet.loc[subvolume, 'Total Melt'])#calculate the log10 of the melt
            if logtotmelt < val2 and logtotmelt > val1:#if the log of the melt is between the two values (greater than val2, smaller than val1 since values are negative and approaching zero)
                datasheet.loc[subvolume, 'LogMeltRange'] = i+1#assign this bin number to the subvolume (plus one because zero indexing in python)
            else:
                pass
        datasheet.to_csv(spreadsheetfpath)# save the spreadsheet
    if EDcheck == True:
        for i in (range(len(binning)-1)):#for each bin
            val1 = binning[i]#value one is the first, val2 is the next value
            val2 = binning[i+1]
            for subvolume in datasheet.index:#for each subvolume
                logEDmelt = np.log10(datasheet.loc[subvolume, 'EDTotalMelt'])
                if logEDmelt < val2 and logEDmelt > val1:
                    datasheet.loc[subvolume, 'EDLogMeltRange'] = i+1
                elif datasheet.loc[subvolume, 'Total Melt Fraction'] == 0:
                    datasheet.loc[subvolume, 'EDLogMeltRange'] =1
                else:
                    pass
        datasheet.to_csv(spreadsheetfpath)
    return

        
        
        
def degree_of_meltsegregation(datasheetfilepath):
    datasheet=pd.read_csv(datasheetfilepath)
    unique_initial_xi=[]
    unique_initial_yi=[]
    unique_initial_zi=[]

    for value in datasheet['Xi']:
        if value in unique_initial_xi:
            pass
        else:
            unique_initial_xi.append(value)
            
    for value in datasheet['Yi']:
        if value in unique_initial_yi:
            pass
        else:
            unique_initial_yi.append(value)        
            
    for value in datasheet['Zi']:
        if value in unique_initial_zi:
            pass
        else:
            unique_initial_zi.append(value)
    
    X_MeltSumDF = pd.DataFrame(columns=['Initial X Value', 'Melt Fraction Sum', 'Subvolume Count', 'Melt Fraction Average'])
    Y_MeltSumDF = pd.DataFrame(columns=['Initial Y Value', 'Melt Fraction Sum', 'Subvolume Count', 'Melt Fraction Average'])
    Z_MeltSumDF = pd.DataFrame(columns=['Initial Z Value', 'Melt Fraction Sum', 'Subvolume Count', 'Melt Fraction Average'])
    
    X_MeltSumDF['Initial X Value']=unique_initial_xi
    Y_MeltSumDF['Initial Y Value']=unique_initial_yi
    Z_MeltSumDF['Initial Z Value']=unique_initial_zi
    
    X_MeltSumDF['Melt Fraction Sum']=0.0
    Y_MeltSumDF['Melt Fraction Sum']=0.0
    Z_MeltSumDF['Melt Fraction Sum']=0.0
    
    X_MeltSumDF['Subvolume Count']=0
    Y_MeltSumDF['Subvolume Count']=0
    Z_MeltSumDF['Subvolume Count']=0
    
    X_MeltSumDF = X_MeltSumDF.set_index('Initial X Value')
    Y_MeltSumDF = Y_MeltSumDF.set_index('Initial Y Value')
    Z_MeltSumDF = Z_MeltSumDF.set_index('Initial Z Value')
    
    datasheet = datasheet.set_index('Subvolume Name')
    

    for subvolume in datasheet.index:
        xi=datasheet.at[subvolume, 'Xi']
        yi=datasheet.at[subvolume, 'Yi']
        zi=datasheet.at[subvolume, 'Zi']
        
        
        X_MeltSumDF.at[xi, 'Melt Fraction Sum'] = X_MeltSumDF.at[xi, 'Melt Fraction Sum'] + datasheet.at[subvolume, 'Total Melt']
        X_MeltSumDF.at[xi, 'Subvolume Count'] += 1
        
        Y_MeltSumDF.at[yi, 'Melt Fraction Sum'] += datasheet.at[subvolume, 'Total Melt']
        Y_MeltSumDF.at[yi, 'Subvolume Count'] += 1
        
        Z_MeltSumDF.at[zi, 'Melt Fraction Sum'] += datasheet.at[subvolume, 'Total Melt']
        Z_MeltSumDF.at[zi, 'Subvolume Count'] += 1
            
    
    for Xi in X_MeltSumDF.index:
        X_MeltSumDF.at[Xi,'Melt Fraction Average'] = X_MeltSumDF.at[Xi,'Melt Fraction Sum'] / X_MeltSumDF.at[Xi,'Subvolume Count']
    print('Max melt in X direction:', max(X_MeltSumDF['Melt Fraction Average']),'...Min melt in X direction:', min(X_MeltSumDF['Melt Fraction Average']))
            
    for Yi in Y_MeltSumDF.index:
        Y_MeltSumDF.at[Yi,'Melt Fraction Average'] = Y_MeltSumDF.at[Yi,'Melt Fraction Sum'] / Y_MeltSumDF.at[Yi,'Subvolume Count']
    print('Max melt in Y direction:', max(Y_MeltSumDF['Melt Fraction Average']),'...Min melt in X direction:', min(Y_MeltSumDF['Melt Fraction Average']))
            
    for Zi in Z_MeltSumDF.index:
        Z_MeltSumDF.at[Zi,'Melt Fraction Average'] = Z_MeltSumDF.at[Zi,'Melt Fraction Sum'] / Z_MeltSumDF.at[Zi,'Subvolume Count']
    print('Max melt in Z direction:', max(Z_MeltSumDF['Melt Fraction Average']),'...Min melt in Z direction:', min(Z_MeltSumDF['Melt Fraction Average']))
    
    print('Mean Melt Fraction X Direction:', np.mean(X_MeltSumDF['Melt Fraction Average']), 'Mean Melt Fraction Y Direction:', np.mean(Y_MeltSumDF['Melt Fraction Average']),'Mean Melt Fraction Z Direction:', np.mean(Z_MeltSumDF['Melt Fraction Average']))
                        
    maxvals = (max(X_MeltSumDF.index), max(Y_MeltSumDF.index), max(Z_MeltSumDF.index))                        
    fig, ax = plt.subplots()
    ax.set_xlim([0, max(maxvals)])
    xvals=X_MeltSumDF.index
    yvals=Y_MeltSumDF.index
    zvals=Z_MeltSumDF.index
    
    xmelts=X_MeltSumDF['Melt Fraction Average']
    ymelts=Y_MeltSumDF['Melt Fraction Average']
    zmelts=Z_MeltSumDF['Melt Fraction Average']
    
    plt.scatter(xvals, xmelts, color='forestgreen')
    plt.scatter(yvals, ymelts, color='blue')
    plt.scatter(zvals, zmelts, color='red')
    ax.set_xlabel('Initial Coordinate')
    ax.set_ylabel('Average \u03C6'+'$_{t}$')
    ax.set_xlim([0, 3000])
    ax.set_ylim([0, 0.20])
    
    
    return
    
    
    
    
def grayscale_lineplots(rawimgcsv, smoothedimgcsv, gradientcsv, grayscalemax_img, grayscalemax_gradient, xlim):
    
    original_grayscale = pd.read_csv(rawimgcsv, names=['Distance', 'Grayscale'])
    smoothed_grayscale = pd.read_csv(smoothedimgcsv, names=['Distance', 'Grayscale'])
    gradient_grayscale = pd.read_csv(gradientcsv, names=['Distance', 'Grayscale'])
    
    
    fig, ax=plt.subplots()
    ax.set_xlim([0, xlim])
    ax.set_ylim([0, grayscalemax_img])
    plt.plot(original_grayscale['Distance'], original_grayscale['Grayscale'], linestyle='-', color='black')
    plt.plot(smoothed_grayscale['Distance'], smoothed_grayscale['Grayscale'], linestyle='-', color=[.5, .5, .5])
    
    ax_gradient=ax.twinx()
    ax_gradient.set_ylim([0, grayscalemax_gradient])
    plt.plot(gradient_grayscale['Distance'], gradient_grayscale['Grayscale'], linestyle='-', color='red')
    
    
    
    return
    


def grayscale_histogramplot(solidcsv, aircsv, meltcsv, xmax, ymax):
    bincount=100
    
    solidsheet = pd.read_csv(solidcsv, names=['Distance', 'Grayscale'])
    airsheet = pd.read_csv(aircsv, names=['Distance', 'Grayscale'])
    meltsheet = pd.read_csv(meltcsv, names=['Distance', 'Grayscale'])
    
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_ylim(1, ymax)
    ax.set_xlim(0, xmax)
    ax.hist(solidsheet['Grayscale'], bins=bincount, alpha=.5, color='black')
    ax.hist(airsheet['Grayscale'], bins=bincount, alpha=.1, color='black')
    ax.hist(meltsheet['Grayscale'], bins=bincount, alpha=.1, color='black')
    

    return
    
    

def quadrant_percentile(xspreadsheetfilepath, yspreadsheetfilepath, zspreadsheetfilepath, percentile_vals=[30, 70]):
    x_datasheet=pd.read_csv(xspreadsheetfilepath)
    y_datasheet=pd.read_csv(yspreadsheetfilepath)
    z_datasheet=pd.read_csv(zspreadsheetfilepath)
    x_edmeltfrac=np.asarray(x_datasheet['Dominant Melt'])
    y_edmeltfrac=np.asarray(y_datasheet['Dominant Melt'])
    z_edmeltfrac=np.asarray(z_datasheet['Dominant Melt'])
    x_edmeltdiff=np.asarray(x_datasheet['EDPrismMeltRatio'])
    y_edmeltdiff=np.asarray(y_datasheet['EDPrismMeltRatio'])
    z_edmeltdiff=np.asarray(z_datasheet['EDPrismMeltRatio'])
    upper_dmelt_x=np.percentile(x_edmeltfrac, percentile_vals[1])
    upper_dmelt_y=np.percentile(y_edmeltfrac, percentile_vals[1])
    upper_dmelt_z=np.percentile(z_edmeltfrac, percentile_vals[1])
    lower_dmelt_x=np.percentile(x_edmeltfrac, percentile_vals[0])
    lower_dmelt_y=np.percentile(y_edmeltfrac, percentile_vals[0])
    lower_dmelt_z=np.percentile(z_edmeltfrac, percentile_vals[0])
    
    
    
    print(upper_dmelt_z, lower_dmelt_z)
    
    
    
    
    for subvolume in x_datasheet.index:
        if x_datasheet.at[subvolume, 'Dominant Melt'] < lower_dmelt_x:
            x_datasheet.at[subvolume, 'Percentile Range'] = 1
        if x_datasheet.at[subvolume, 'Dominant Melt'] > upper_dmelt_x:
            x_datasheet.at[subvolume, 'Percentile Range'] = 3
        elif x_datasheet.at[subvolume, 'Dominant Melt'] < upper_dmelt_x and x_datasheet.at[subvolume, 'Dominant Melt'] > lower_dmelt_x:
            x_datasheet.at[subvolume, 'Percentile Range'] = 2
        
    for subvolume in y_datasheet.index:
        if y_datasheet.at[subvolume, 'Dominant Melt'] < lower_dmelt_y:
            y_datasheet.at[subvolume, 'Percentile Range'] = 1
        if y_datasheet.at[subvolume, 'Dominant Melt'] > upper_dmelt_y:
            y_datasheet.at[subvolume, 'Percentile Range'] = 3
        elif y_datasheet.at[subvolume, 'Dominant Melt'] < upper_dmelt_y and y_datasheet.at[subvolume, 'Dominant Melt'] > lower_dmelt_y:
            y_datasheet.at[subvolume, 'Percentile Range'] = 2
    
    for subvolume in z_datasheet.index:
        if z_datasheet.at[subvolume, 'Dominant Melt'] < lower_dmelt_z:
            z_datasheet.at[subvolume, 'Percentile Range'] = 1
        if z_datasheet.at[subvolume, 'Dominant Melt'] > upper_dmelt_z:
            z_datasheet.at[subvolume, 'Percentile Range'] = 3
        elif z_datasheet.at[subvolume, 'Dominant Melt'] < upper_dmelt_z and z_datasheet.at[subvolume, 'Dominant Melt'] > lower_dmelt_z:
            z_datasheet.at[subvolume, 'Percentile Range'] = 2

    x_datasheet.to_csv(xspreadsheetfilepath)
    y_datasheet.to_csv(yspreadsheetfilepath)
    z_datasheet.to_csv(zspreadsheetfilepath)
    xinband_tot=x_datasheet[x_datasheet['Percentile Range'] ==3]
    zinband_tot=z_datasheet[z_datasheet['Percentile Range'] ==3]
    xmix_tot=x_datasheet[x_datasheet['Percentile Range'] == 2]
    zmix_tot=z_datasheet[z_datasheet['Percentile Range'] == 2]
    xoutband_tot=x_datasheet[x_datasheet['Percentile Range'] ==1]
    zoutband_tot=z_datasheet[z_datasheet['Percentile Range'] ==1]
    
    xz_inband_tot=pd.concat([xinband_tot, zinband_tot])
    xz_mix_tot=pd.concat([xmix_tot, zmix_tot])
    xz_outband_tot=pd.concat([xoutband_tot, zoutband_tot])
    xz_inband_tot.to_csv(xspreadsheetfilepath[:-4]+ str(percentile_vals[1]) + 'inbandtotal.csv')
    xz_mix_tot.to_csv(xspreadsheetfilepath[:-4]+ str(percentile_vals[1]) + 'mixtotal.csv')
    xz_outband_tot.to_csv(xspreadsheetfilepath[:-4]+ str(percentile_vals[1]) + 'outbandtotal.csv')
    
    
    
    
    upper_dmelt_x=np.percentile(x_edmeltdiff, percentile_vals[1])
    upper_dmelt_y=np.percentile(y_edmeltdiff, percentile_vals[1])
    upper_dmelt_z=np.percentile(z_edmeltdiff, percentile_vals[1])
    lower_dmelt_x=np.percentile(x_edmeltdiff, percentile_vals[0])
    lower_dmelt_y=np.percentile(y_edmeltdiff, percentile_vals[0])
    lower_dmelt_z=np.percentile(z_edmeltdiff, percentile_vals[0])
    
    for subvolume in x_datasheet.index:
        if x_datasheet.at[subvolume, 'EDPrismMeltRatio'] < lower_dmelt_x:
            x_datasheet.at[subvolume, 'Percentile Range'] = 1
        if x_datasheet.at[subvolume, 'EDPrismMeltRatio'] > upper_dmelt_x:
            x_datasheet.at[subvolume, 'Percentile Range'] = 3
        elif x_datasheet.at[subvolume, 'EDPrismMeltRatio'] < upper_dmelt_x and x_datasheet.at[subvolume, 'EDPrismMeltRatio'] > lower_dmelt_x:
            x_datasheet.at[subvolume, 'Percentile Range'] = 2
        
    for subvolume in y_datasheet.index:
        if y_datasheet.at[subvolume, 'EDPrismMeltRatio'] < lower_dmelt_y:
            y_datasheet.at[subvolume, 'Percentile Range'] = 1
        if y_datasheet.at[subvolume, 'EDPrismMeltRatio'] > upper_dmelt_y:
            y_datasheet.at[subvolume, 'Percentile Range'] = 3
        elif y_datasheet.at[subvolume, 'EDPrismMeltRatio'] < upper_dmelt_y and y_datasheet.at[subvolume, 'EDPrismMeltRatio'] > lower_dmelt_y:
            y_datasheet.at[subvolume, 'Percentile Range'] = 2
    
    for subvolume in z_datasheet.index:
        if z_datasheet.at[subvolume, 'EDPrismMeltRatio'] < lower_dmelt_z:
            z_datasheet.at[subvolume, 'Percentile Range'] = 1
        if z_datasheet.at[subvolume, 'EDPrismMeltRatio'] > upper_dmelt_z:
            z_datasheet.at[subvolume, 'Percentile Range'] = 3
        elif z_datasheet.at[subvolume, 'EDPrismMeltRatio'] < upper_dmelt_z and z_datasheet.at[subvolume, 'EDPrismMeltRatio'] > lower_dmelt_z:
            z_datasheet.at[subvolume, 'Percentile Range'] = 2
    
    xinband_tot=x_datasheet[x_datasheet['Percentile Range'] ==3]
    zinband_tot=z_datasheet[z_datasheet['Percentile Range'] ==3]
    xmix_tot=x_datasheet[x_datasheet['Percentile Range'] == 2]
    zmix_tot=z_datasheet[z_datasheet['Percentile Range'] == 2]
    xoutband_tot=x_datasheet[x_datasheet['Percentile Range'] ==1]
    zoutband_tot=z_datasheet[z_datasheet['Percentile Range'] ==1]
    
    xz_inband_tot=pd.concat([xinband_tot, zinband_tot])
    xz_mix_tot=pd.concat([xmix_tot, zmix_tot])
    xz_outband_tot=pd.concat([xoutband_tot, zoutband_tot])
    xz_inband_tot.to_csv(xspreadsheetfilepath[:-4]+ str(percentile_vals[1]) + 'inbandiff.csv')
    xz_mix_tot.to_csv(xspreadsheetfilepath[:-4]+ str(percentile_vals[1]) + 'mixdiff.csv')
    xz_outband_tot.to_csv(xspreadsheetfilepath[:-4]+ str(percentile_vals[1]) + 'outbanddiff.csv')
    
    
    return
        

def quadrant_percentile_y(xspreadsheetfilepath, yspreadsheetfilepath, zspreadsheetfilepath, percentile_vals=[5, 95]):
    x_datasheet=pd.read_csv(xspreadsheetfilepath)
    y_datasheet=pd.read_csv(yspreadsheetfilepath)
    z_datasheet=pd.read_csv(zspreadsheetfilepath)
    y_edmeltfrac=np.asarray(y_datasheet['Dominant Melt'])
    y_edmeltdiff=np.asarray(y_datasheet['EDPrismMeltRatio'])
    upper_dmelt_y=np.percentile(y_edmeltfrac, percentile_vals[1])
    lower_dmelt_y=np.percentile(y_edmeltfrac, percentile_vals[0])
    
    
    for subvolume in x_datasheet.index:
        if x_datasheet.at[subvolume, 'Dominant Melt'] < lower_dmelt_y:
            x_datasheet.at[subvolume, 'Percentile Range'] = 1
        if x_datasheet.at[subvolume, 'Dominant Melt'] > upper_dmelt_y:
            x_datasheet.at[subvolume, 'Percentile Range'] = 3
        elif x_datasheet.at[subvolume, 'Dominant Melt'] < upper_dmelt_y and x_datasheet.at[subvolume, 'Dominant Melt'] > lower_dmelt_y:
            x_datasheet.at[subvolume, 'Percentile Range'] = 2
        
    for subvolume in y_datasheet.index:
        if y_datasheet.at[subvolume, 'Dominant Melt'] < lower_dmelt_y:
            y_datasheet.at[subvolume, 'Percentile Range'] = 1
        if y_datasheet.at[subvolume, 'Dominant Melt'] > upper_dmelt_y:
            y_datasheet.at[subvolume, 'Percentile Range'] = 3
        elif y_datasheet.at[subvolume, 'Dominant Melt'] < upper_dmelt_y and y_datasheet.at[subvolume, 'Dominant Melt'] > lower_dmelt_y:
            y_datasheet.at[subvolume, 'Percentile Range'] = 2
    
    for subvolume in z_datasheet.index:
        if z_datasheet.at[subvolume, 'Dominant Melt'] < lower_dmelt_y:
            z_datasheet.at[subvolume, 'Percentile Range'] = 1
        if z_datasheet.at[subvolume, 'Dominant Melt'] > upper_dmelt_y:
            z_datasheet.at[subvolume, 'Percentile Range'] = 3
        elif z_datasheet.at[subvolume, 'Dominant Melt'] < upper_dmelt_y and z_datasheet.at[subvolume, 'Dominant Melt'] > lower_dmelt_y:
            z_datasheet.at[subvolume, 'Percentile Range'] = 2

    x_datasheet.to_csv(xspreadsheetfilepath)
    y_datasheet.to_csv(yspreadsheetfilepath)
    z_datasheet.to_csv(zspreadsheetfilepath)
    xinband_tot=x_datasheet[x_datasheet['Percentile Range'] ==3]
    zinband_tot=z_datasheet[z_datasheet['Percentile Range'] ==3]
    xmix_tot=x_datasheet[x_datasheet['Percentile Range'] == 2]
    zmix_tot=z_datasheet[z_datasheet['Percentile Range'] == 2]
    xoutband_tot=x_datasheet[x_datasheet['Percentile Range'] ==1]
    zoutband_tot=z_datasheet[z_datasheet['Percentile Range'] ==1]
    
    xz_inband_tot=pd.concat([xinband_tot, zinband_tot])
    xz_mix_tot=pd.concat([xmix_tot, zmix_tot])
    xz_outband_tot=pd.concat([xoutband_tot, zoutband_tot])
    xz_inband_tot.to_csv(xspreadsheetfilepath[:-4]+ str(percentile_vals[1]) + 'inbandtotal.csv')
    xz_mix_tot.to_csv(xspreadsheetfilepath[:-4]+ str(percentile_vals[1]) + 'mixtotal.csv')
    xz_outband_tot.to_csv(xspreadsheetfilepath[:-4]+ str(percentile_vals[1]) + 'outbandtotal.csv')
    
    
    
    
    
    upper_dmelt_y=np.percentile(y_edmeltdiff, percentile_vals[1])
    lower_dmelt_y=np.percentile(y_edmeltdiff, percentile_vals[0])
    
    
    for subvolume in x_datasheet.index:
        if x_datasheet.at[subvolume, 'EDPrismMeltRatio'] < lower_dmelt_y:
            x_datasheet.at[subvolume, 'Percentile Range'] = 1
        if x_datasheet.at[subvolume, 'EDPrismMeltRatio'] > upper_dmelt_y:
            x_datasheet.at[subvolume, 'Percentile Range'] = 3
        elif x_datasheet.at[subvolume, 'EDPrismMeltRatio'] < upper_dmelt_y and x_datasheet.at[subvolume, 'EDPrismMeltRatio'] > lower_dmelt_y:
            x_datasheet.at[subvolume, 'Percentile Range'] = 2
        
    for subvolume in y_datasheet.index:
        if y_datasheet.at[subvolume, 'EDPrismMeltRatio'] < lower_dmelt_y:
            y_datasheet.at[subvolume, 'Percentile Range'] = 1
        if y_datasheet.at[subvolume, 'EDPrismMeltRatio'] > upper_dmelt_y:
            y_datasheet.at[subvolume, 'Percentile Range'] = 3
        elif y_datasheet.at[subvolume, 'EDPrismMeltRatio'] < upper_dmelt_y and y_datasheet.at[subvolume, 'EDPrismMeltRatio'] > lower_dmelt_y:
            y_datasheet.at[subvolume, 'Percentile Range'] = 2
    
    for subvolume in z_datasheet.index:
        if z_datasheet.at[subvolume, 'EDPrismMeltRatio'] < lower_dmelt_y:
            z_datasheet.at[subvolume, 'Percentile Range'] = 1
        if z_datasheet.at[subvolume, 'EDPrismMeltRatio'] > upper_dmelt_y:
            z_datasheet.at[subvolume, 'Percentile Range'] = 3
        elif z_datasheet.at[subvolume, 'EDPrismMeltRatio'] < upper_dmelt_y and z_datasheet.at[subvolume, 'EDPrismMeltRatio'] > lower_dmelt_y:
            z_datasheet.at[subvolume, 'Percentile Range'] = 2
    
    xinband_tot=x_datasheet[x_datasheet['Percentile Range'] ==3]
    zinband_tot=z_datasheet[z_datasheet['Percentile Range'] ==3]
    xmix_tot=x_datasheet[x_datasheet['Percentile Range'] == 2]
    zmix_tot=z_datasheet[z_datasheet['Percentile Range'] == 2]
    xoutband_tot=x_datasheet[x_datasheet['Percentile Range'] ==1]
    zoutband_tot=z_datasheet[z_datasheet['Percentile Range'] ==1]
    
    xz_inband_tot=pd.concat([xinband_tot, zinband_tot])
    xz_mix_tot=pd.concat([xmix_tot, zmix_tot])
    xz_outband_tot=pd.concat([xoutband_tot, zoutband_tot])
    xz_inband_tot.to_csv(xspreadsheetfilepath[:-4]+ str(percentile_vals[1]) + 'inbandiff.csv')
    xz_mix_tot.to_csv(xspreadsheetfilepath[:-4]+ str(percentile_vals[1]) + 'mixdiff.csv')
    xz_outband_tot.to_csv(xspreadsheetfilepath[:-4]+ str(percentile_vals[1]) + 'outbanddiff.csv')
    
    
    return
        
    
def simple_melt_intersect_test_2D(phirange, width_scale):
    length=abs((2*width_scale)/(math.tan(2*np.deg2rad(phirange))))
    print('Minimum distance for these two melt pockets to meet (assuming angle from 0 of ' + str(phirange) + ' degrees and subvolume width of ' + str(width_scale)+ ' pixels) = ' +str(length))
    return

def melt_intersect_test_3D(spreadsheet_filepath, width_scale):
    number_tests=100
    datasheet=pd.DataFrame(columns=['Intersection Points', 'X Traveled Distance', 'Y Traveled Distance', 'Z Traveled Distance', 'X Tort', 'Y Tort', 'Z Tort', 'X Coordinates', 'Y Coordinates', 'Z Coordinates', 'Step Number'])
    
    for k in range(number_tests):
        x_distances=[]
        y_distances=[]
        z_distances=[]
        intersections=[]
        step_number=[]
        number_runs=1000
        
        # fig=plt.figure()
        
        # ax=fig.add_subplot(1,1,1, projection='3d')
        melt_point_one=[]
        melt_point_two=[]
        
        spreadsheet=pd.read_csv(spreadsheet_filepath)
        theta_measurements=[]
        phi_measurements=[]
        for i in range(len(spreadsheet.index)):
            theta=list(ast.literal_eval(spreadsheet.loc[i, 'Column_0']))
            phi=list(ast.literal_eval(spreadsheet.loc[i, 'Column_1']))
            for j in range(len(theta)):
                theta_measurements.append(theta[j])
                phi_measurements.append(phi[j])
        indexing_list=random.sample(range(0, len(theta_measurements)), number_runs)
        indexing_list_2=random.sample(range(0, len(theta_measurements)), number_runs)
        
        melt_one_coordinate_one=[0, 0, 0]
        melt_two_coordinate_one=[0, 0+width_scale, 0]
        
        for i in range(number_runs):
            j=indexing_list[i]
            k=indexing_list_2[i]
            
            melt_one_coordinate_two=melt_one_coordinate_one
            melt_two_coordinate_two=melt_two_coordinate_one
            melt_point_one.append(melt_one_coordinate_two)
            melt_point_two.append(melt_two_coordinate_two)
    
            
            theta=theta_measurements[j]
            phi=phi_measurements[j]
            theta_two=theta_measurements[k]
            phi_two=phi_measurements[k]
            
            x_one=width_scale*np.sin(np.deg2rad(phi))*np.cos(np.deg2rad(theta))
            y_one=width_scale*np.sin(np.deg2rad(phi))*np.sin(np.deg2rad(theta))
            z_one=width_scale*np.cos(np.deg2rad(phi))
            
            x_two=width_scale*np.sin(np.deg2rad(phi_two))*np.cos(np.deg2rad(theta_two))
            y_two=width_scale*np.sin(np.deg2rad(phi_two))*np.sin(np.deg2rad(theta_two))
            z_two=width_scale*np.cos(np.deg2rad(phi_two))
            
            melt_one_coordinate_one=[x_one+melt_one_coordinate_one[0], y_one+melt_one_coordinate_one[1], z_one+melt_one_coordinate_one[2]]
            melt_two_coordinate_one=[x_two+melt_two_coordinate_one[0], y_two+melt_two_coordinate_one[1], z_two+melt_two_coordinate_one[2]]
    
            # x1=[melt_one_coordinate_one[0], melt_one_coordinate_two[0]]
            # y1=[melt_one_coordinate_one[1], melt_one_coordinate_two[1]]
            # z1=[melt_one_coordinate_one[2], melt_one_coordinate_two[2]]
            # x2=[melt_two_coordinate_one[0], melt_two_coordinate_two[0]]
            # y2=[melt_two_coordinate_one[1], melt_two_coordinate_two[1]]
            # z2=[melt_two_coordinate_one[2], melt_two_coordinate_two[2]]
            
            # ax.plot(x1, y1, z1, color='red')
            # ax.plot(x2, y2, z2, color='blue')
            
            if (i/10).is_integer():
                x_distances.append(melt_one_coordinate_one[0])
                y_distances.append(melt_one_coordinate_one[1])
                z_distances.append(melt_one_coordinate_one[2])
                step_number.append(i)
            
            
        # xmin, xmax=ax.get_xlim()
        # ymin, ymax=ax.get_ylim()
        # zmin, zmax=ax.get_zlim()
        # new_min=min(xmin, ymin, zmin)
        # new_max=max(xmax, ymax, zmax)
        # ax.set_xlim([new_min, new_max])
        # ax.set_ylim([new_min, new_max])
        # ax.set_zlim([new_min, new_max])
        # ax.set_xlabel('X axis')
        # ax.set_zlabel('Z axis')
        # ax.set_ylabel('Y axis')
            
            
        traveled_distance_x=[0]
        traveled_distance_y=[0]
        traveled_distance_z=[0]
        

        
        
        for i in range(number_runs):
            melt_point_first=melt_point_one[i]
            if abs(melt_point_first[0])>=(width_scale*3) and traveled_distance_x==[0]:
                traveled_distance_x=i*width_scale
            if abs(melt_point_first[1])>=(width_scale*3) and traveled_distance_y==[0]:
                traveled_distance_y=i*width_scale
            if abs(melt_point_first[2])>=(width_scale*3) and traveled_distance_z==[0]:
                traveled_distance_z=i*width_scale
            
        #     for j in range(number_runs):
        #         melt_point_second=melt_point_two[j]
        #         if abs(melt_point_first[0]-melt_point_second[0])<5:
        #             if abs(melt_point_first[1]-melt_point_second[1])<5:
        #                 if abs(melt_point_first[2]-melt_point_second[2])<5:
        #                     print('These melt pockets intersected after ' + str(i) +' points ')
        #                     intersections.append(i)
        # datasheet.at[k, 'Intersection Points'] = intersections
        print(width_scale*3, traveled_distance_x, traveled_distance_y, traveled_distance_z)
      
        datasheet.at[k, 'X Traveled Distance']=traveled_distance_x
        datasheet.at[k, 'Y Traveled Distance']=traveled_distance_y
        datasheet.at[k, 'Z Traveled Distance']=traveled_distance_z
        datasheet.at[k, 'X Tort']=traveled_distance_x/(width_scale*3)
        datasheet.at[k, 'Y Tort']=traveled_distance_y/(width_scale*3)
        datasheet.at[k, 'Z Tort']=traveled_distance_z/(width_scale*3)
        datasheet.at[k, 'X Coordinates']=x_distances
        datasheet.at[k, 'Y Coordinates']=y_distances
        datasheet.at[k, 'Z Coordinates']=z_distances
        datasheet.at[k, 'Step Number']=step_number
        
    nbins=10
    fig_hist=plt.figure()
    ax_hist=fig_hist.add_subplot()
    ax_hist.hist(datasheet['Y Tort'], bins=nbins, label='Y Direction')
    ax_hist.hist(datasheet['X Tort'], bins=nbins, label='X Direction', alpha=0.7)
    ax_hist.hist(datasheet['Z Tort'], bins=nbins, label='Z Direction', alpha=0.7)
    ax_hist.set_xlabel('Melt Pocket Length / Traveled Distance')
    ax_hist.set_ylabel('Frequency')
    ax_hist.legend()
    print(datasheet.iloc[0])
    
    xcord=datasheet.iloc[0]['X Coordinates']
    for i in range(len(xcord)):
        xcoords=[]
        ycoords=[]
        zcoords=[]
        for subvolume in range(len(datasheet.index)):
            xcord=datasheet.iloc[subvolume]['X Coordinates']
            xcoords.append(xcord[i])
            ycord=datasheet.iloc[subvolume]['Y Coordinates']
            ycoords.append(ycord[i])
            zcord=datasheet.iloc[subvolume]['Z Coordinates']
            zcoords.append(zcord[i])
        fig_hist_2=plt.figure()
        ax_hist_2=fig_hist_2.add_subplot()
        ax_hist_2.hist(xcoords, bins=nbins, label='X Direction', alpha=0.7)
        ax_hist_2.hist(zcoords, bins=nbins, label='Z Direction', alpha=0.7)
        ax_hist_2.hist(ycoords, bins=nbins, label='Y Direction', alpha=0.7)
        ax_hist_2.set_xlabel('Coordinate after'+ str(i*10)+'steps')
        ax_hist_2.set_ylabel('Frequency')
        ax_hist_2.legend()
    
    return(datasheet)





    