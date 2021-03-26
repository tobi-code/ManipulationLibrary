#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2020

@author: Tobias Strübing
"""
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from itertools import combinations
import os
import progressbar
import pandas as pd
import copy
from operator import xor
import math
import json
import shutil
import sys
import random
from scipy.spatial.distance import pdist

from ManipulationLibrary.cython_filter_new import filter_cython_new as filter_cython
#from cython_filter_new import filter_cython_new as filter_cython

##define global variables
#o1,o2,o3 are the three main objects of the manipulation
o1 = None
o2 = None
o3 = None
#count variables esnure o1, o2, o3 are only once defined
#count1 is assigned to object1 and so on
count1, count2, count3 = 0, 0, 0

#define variables for labels of o1, o2 and o3
o1_label = 0
o2_label = 0
o3_label = 0

#define previous frame for DSR
previous_array = None

#define variables for the internal of the algorithm
internal_count = 0
total_unique_labels = 0
hand_label_inarray = -1
count_esec = 0
ground = 0
count_ground = 0
absent_o1, absent_o2, absent_o3 = False, False, False
first_o1, first_o2, first_03 = None, None, None
ao1, ao2, ao3 = 0,0,0


def esec_to_e2sec(esec_array, relations):
    '''
    Takes an eSEC matrix as input and outputs the calculated e2SEC matrix.
    
    Parameters:
        * esec_array: eSEC array
        
    Returns:
        * eSEC array: array that contains the eSEC matrix
        * e2SEC array: array that contains the e2SEC matrix
    '''

    liste_array = esec_array
    #----------------------------------------------------------------  
    
    #uppercase every entry
    for i in range(liste_array.shape[0]):
        for k in range(liste_array.shape[1]):
            liste_array[i][k] = liste_array[i][k].upper()
    
    #----------------------------------------------------------------  
    
    #copy esec matrix and delete first column becuase first column consits of "-" due to the algorithm
    e2sec_array = copy.deepcopy(liste_array)     
    e2sec_array = np.delete(e2sec_array, 0, 1)

    #----------------------------------------------------------------  
    #remove rows 4 and 10 in T/N, SSR, DSR
    if relations == 3:
        e2sec_array = np.delete(e2sec_array, [3,9,13,19,23,29], 0)
    if relations == 2:
        e2sec_array = np.delete(e2sec_array, [3,9,13,19], 0)
    if relations == 1:
        e2sec_array = np.delete(e2sec_array, [3,9], 0)
    
    #----------------------------------------------------------------  
    
    #find columns that are same due to e2SEC and remove them
    k = []
    for j in range(e2sec_array.shape[1]-1):
        if(np.array_equal(e2sec_array[:,j], e2sec_array[:,j+1])):
            k.append(j+1)

    e2sec_array = np.delete(e2sec_array, k, 1)
    #----------------------------------------------------------------  
    
    #return e2sec and esec dict
    return e2sec_array, liste_array

def _replace_labels(label_resized, old, new):
    '''
    Takes a label array as input and replaces the old labels with the new ones and returns this array.
    
    Parameters:
        * label_resized: array that contains labels
        * old: old labels that should be replaced [int]
        * new: new labels that will be assigned [int]
        
    Returns:
        * new_labels: label array with new labels
    '''
    #deepcopy old labels
    old_label_file = copy.deepcopy(label_resized)

    #replace old labels with new ones
    for i in range(len(old)):
        if i == 0:
            new_labels = np.where(old_label_file == old[i], new[i], old_label_file) 
        else:
            new_labels = np.where(new_labels == old[i], new[i], new_labels) 
            
    return new_labels

def _distance(p1, p2):
    '''
    Calculates the euclidean distance between two points.
    '''

    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)


def _rotateSceneNewNew(pcd_file, label_file, ground_label):
    '''
    Calculates the rotation of the scene by using oriented bounding boxes 
    from the RASNAC ground.
    
    Parameters:
        * pcd_file: pcd file to process (.pcd)
        * label_file: label file corresponding to pcd file (.dat)
        * ground_label: label of the ground (int)
        
    Returns:
        * rotation: rotation of the scene
    '''
    
    #load pcd file with nan points to map lables 
    pcd = o3d.io.read_point_cloud(pcd_file, remove_nan_points=False)
    
    #cast cloud to numpy array and replace nan values with int value -100
    cloud = np.asarray(pcd.points)
    cloud = np.nan_to_num(cloud, nan=-100)
    
    #resize labels to point cloud size
    my_mat = np.zeros((640, 480))
    label = np.loadtxt(label_file)
    label_resized = cv2.resize(label, my_mat.shape, interpolation = cv2.INTER_NEAREST)
    label_resized = label_resized.flatten()
    
    #add labels to points from the cloud
    pcd_array =  np.column_stack((cloud, label_resized))
    
    #sort point cloud array by labels and calculate maximal label
    pcd_array_sorted = pcd_array[pcd_array[:, 3].argsort()]
   
    #delete -100 values
    result_1 = np.where(pcd_array_sorted == -100)
    pcd_array_sorted = np.delete(pcd_array_sorted, np.unique(result_1[0]), 0)

    #assign unique labels 
    unique_labels = np.unique(label)
    
    #define index_0 as lower limit of ground cloud(i.e. cloud[lower_limit:upper_limit])
    #this loop is nessesary to find the lower limit, sometimes the ground_label-1 is not
    #assigned in labels so it finds the next lower label from ground_label
    #i.e. best case gound_point_cloud = raw_point_cloud[ground_label-1:ground_label]
    i = 1
    index_0 = None
    while(index_0 == None):
        if(float(unique_labels[np.where(unique_labels == ground_label)[0][0]-i]) in pcd_array_sorted[:,3]):
            index_0 = np.max(np.where(pcd_array_sorted[:,3] == float(unique_labels[np.where(unique_labels == ground_label)[0][0]-i])))
        else:
            i += 1
    
    #define index_1 as upper limit of ground cloud(i.e. cloud[lower_limit:upper_limit])
    index_1 = np.max(np.where(pcd_array_sorted[:,3] == float(ground_label)))

    
    #delete the labels from cloud array
    pcd_array_sorted = np.delete(pcd_array_sorted, 3,1)
    
    #create objects dict which contains ground point cloud
    if(ground_label == 0):
        objects = pcd_array_sorted[0:index_1]
    else:
        objects = pcd_array_sorted[index_0:index_1]  
    
    #obejects array to pointclouds
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(objects)
    
    #use RANSAC algorithm to extract ground plane, create obb and extract the rotation to return
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=100)
    [a, b, c, d] = plane_model

    filtered_pcd_voxel = pcd.select_by_index(inliers)
    filtered_pcd_voxel = filtered_pcd_voxel.voxel_down_sample(voxel_size=0.02)
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(filtered_pcd_voxel.points)
    rotation = obb.R
    
    return rotation

def _getTranslation(ground_cloud):
    '''
    Calculates the translation of the scene by using ICP Registration.
    
    Parameters:
        * ground_cloud: point cloud of the ground
        
    Returns:
        * translation: translation of the scene
    '''
    #load ground cloud and voxel it for faster processing
    pcd = ground_cloud
    pcd_voxel = pcd.voxel_down_sample(voxel_size=0.02)

    #calculate obb around ground cloud and get the extent
    pcd_voxel_box = o3d.geometry.OrientedBoundingBox.create_from_points(pcd_voxel.points)
    pcd_voxel_box_extend = pcd_voxel_box.extent

    #calculate the center and the mean z value of the ground
    middle = pcd_voxel.get_center()
    pcd_z = np.mean(np.asarray(pcd_voxel.points)[:,2])

    #plane_array descibes the corner points of the new plane
    plane_array = np.array([[middle[0] + pcd_voxel_box_extend[0]/2, middle[1] + pcd_voxel_box_extend[1]/2, pcd_z], 
                            [middle[0] + pcd_voxel_box_extend[0]/2, middle[1] - pcd_voxel_box_extend[1]/2, pcd_z],
                            [middle[0] - pcd_voxel_box_extend[0]/2, middle[1] - pcd_voxel_box_extend[1]/2, pcd_z],
                            [middle[0] - pcd_voxel_box_extend[0]/2, middle[1] + pcd_voxel_box_extend[1]/2, pcd_z]])
    
    #get max and min x,y values from corner points
    plain = np.zeros_like(np.asarray(pcd_voxel.points))
    pcd_array_length = len(np.asarray(pcd_voxel.points))
    min_x = np.min(plane_array[:,0])
    max_X = np.max(plane_array[:,0])
    min_y = np.min(plane_array[:,1])
    max_y = np.max(plane_array[:,1])

    #fill the space between corners with random points and save new plane as plane_cloud
    for i in range(pcd_array_length):
        number_x = random.uniform(min_x, max_X)
        number_y = random.uniform(min_y, max_y)
        number_z = random.uniform(pcd_z-0.001, pcd_z+0.001)
        added_array = [number_x, number_y, number_z]
        plain[i] = added_array
    plane_cloud = o3d.geometry.PointCloud()
    plane_cloud.points = o3d.utility.Vector3dVector(plain)

    #define initial translation and use ICP Registration to get translation to map the ground to the plane
    trans_init = np.asarray([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_voxel, plane_cloud, 0.2, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    #define roll that turns the ground upside down
    roll = [[-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]]

    return reg_p2p.transformation, roll

def _getGroundiNew(pcd_file, label_file, ground_label, rotation):
    #load pcd file with nan points to map lables 
    pcd = o3d.io.read_point_cloud(pcd_file, remove_nan_points=False)
    #cast cloud to numpy array and replace nan values with int value -100
    cloud = np.asarray(pcd.points)
    cloud = np.nan_to_num(cloud, nan=-100)
    
    #resize labels to point cloud size
    my_mat = np.zeros((640, 480))
    label = np.loadtxt(label_file)
    label_resized = cv2.resize(label, my_mat.shape, interpolation = cv2.INTER_NEAREST)
    label_resized = label_resized.flatten()

    #load pcd file with nan points to map lables  
    
    pcd = o3d.io.read_point_cloud(pcd_file, remove_nan_points=False)
    #cast cloud to numpy array and replace nan values with int value -100
    cloud = np.asarray(pcd.points)
    cloud = np.nan_to_num(cloud, nan=-100)
   
    
    
    #add labels to points from the cloud
    pcd_array = np.column_stack((cloud, label_resized))
    pcd_array_sorted = pcd_array[pcd_array[:, 3].argsort()]

    #search for -100 values and delete them
    result_1 = np.where(pcd_array_sorted == -100)
    pcd_array_sorted = np.delete(pcd_array_sorted, np.unique(result_1[0]), 0)

    #assign unique labels 
    unique_labels = np.unique(label)
    
    #define index_0 as lower limit of ground cloud(i.e. cloud[lower_limit:upper_limit])
    #this loop is nessesary to find the lower limit, sometimes the ground_label-1 is not
    #assigned in labels so it finds the next lower label from ground_label
    #i.e. best case gound_point_cloud = raw_point_cloud[ground_label-1:ground_label]
    i = 1
    index_0 = None
    while(index_0 == None):
        if(float(unique_labels[np.where(unique_labels == ground_label)[0][0]-i]) in pcd_array_sorted[:,3]):
            index_0 = np.max(np.where(pcd_array_sorted[:,3] == float(unique_labels[np.where(unique_labels == ground_label)[0][0]-i])))
        else:
            i += 1
                
    #define index_1 as upper limit of ground cloud(i.e. cloud[lower_limit:upper_limit])
    index_1 = np.max(np.where(pcd_array_sorted[:,3] == float(ground_label)))
    
    #delete the labels from cloud array
    pcd_array_sorted = np.delete(pcd_array_sorted, 3,1)
    
    
    #rotate the point cloud with calculated rotation
    rotated_cloud = o3d.geometry.PointCloud()
    rotated_cloud.points = o3d.utility.Vector3dVector(pcd_array_sorted)
    rotated_cloud = rotated_cloud.rotate(R=rotation.T, center=rotated_cloud.get_center())

    #extract points from rotated cloud
    pcd_array_sorted = np.asarray(rotated_cloud.points)

    #create objects dict
    objects = pcd_array_sorted[index_0:index_1]

    #empty dict for point clouds of single objects, 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(objects)

      
    #use RANSAC algorithm to extract ground plane, create obb and extract the rotation to return
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=100)
    [a, b, c, d] = plane_model

    #select inliers as ground cloud and voxel down the ground plane
    filtered_pcd_voxel = pcd.select_by_index(inliers)
    filtered_pcd_voxel = filtered_pcd_voxel.voxel_down_sample(voxel_size=0.02)

    return filtered_pcd_voxel

def _getGroundiNewNew(pcd_file, label_file, ground_label):
    #load pcd file with nan points to map lables 
    pcd = o3d.io.read_point_cloud(pcd_file, remove_nan_points=False)
    #cast cloud to numpy array and replace nan values with int value -100
    cloud = np.asarray(pcd.points)
    cloud = np.nan_to_num(cloud, nan=-100)
    
    #resize labels to point cloud size
    my_mat = np.zeros((640, 480))
    label = np.loadtxt(label_file)
    label_resized = cv2.resize(label, my_mat.shape, interpolation = cv2.INTER_NEAREST)
    label_resized = label_resized.flatten()

    #load pcd file with nan points to map lables  
    
    pcd = o3d.io.read_point_cloud(pcd_file, remove_nan_points=False)
    #cast cloud to numpy array and replace nan values with int value -100
    cloud = np.asarray(pcd.points)
    cloud = np.nan_to_num(cloud, nan=-100)
   
    
    
    #add labels to points from the cloud
    pcd_array = np.column_stack((cloud, label_resized))
    pcd_array_sorted = pcd_array[pcd_array[:, 3].argsort()]

    #search for -100 values and delete them
    result_1 = np.where(pcd_array_sorted == -100)
    pcd_array_sorted = np.delete(pcd_array_sorted, np.unique(result_1[0]), 0)

    #assign unique labels 
    unique_labels = np.unique(label)
    
    #define index_0 as lower limit of ground cloud(i.e. cloud[lower_limit:upper_limit])
    #this loop is nessesary to find the lower limit, sometimes the ground_label-1 is not
    #assigned in labels so it finds the next lower label from ground_label
    #i.e. best case gound_point_cloud = raw_point_cloud[ground_label-1:ground_label]
    i = 1
    index_0 = None
    while(index_0 == None):
        if(float(unique_labels[np.where(unique_labels == ground_label)[0][0]-i]) in pcd_array_sorted[:,3]):
            index_0 = np.max(np.where(pcd_array_sorted[:,3] == float(unique_labels[np.where(unique_labels == ground_label)[0][0]-i])))
        else:
            i += 1
                
    #define index_1 as upper limit of ground cloud(i.e. cloud[lower_limit:upper_limit])
    index_1 = np.max(np.where(pcd_array_sorted[:,3] == float(ground_label)))
    
    #delete the labels from cloud array
    pcd_array_sorted = np.delete(pcd_array_sorted, 3,1)
    
    
    #rotate the point cloud with calculated rotation
    rotated_cloud = o3d.geometry.PointCloud()
    rotated_cloud.points = o3d.utility.Vector3dVector(pcd_array_sorted)

    #extract points from rotated cloud
    pcd_array_sorted = np.asarray(rotated_cloud.points)

    #create objects dict
    objects = pcd_array_sorted[index_0:index_1]

    #empty dict for point clouds of single objects, 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(objects)

      
    #use RANSAC algorithm to extract ground plane, create obb and extract the rotation to return
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=100)
    [a, b, c, d] = plane_model

    #select inliers as ground cloud and voxel down the ground plane
    filtered_pcd_voxel = pcd.select_by_index(inliers)
    filtered_pcd_voxel = filtered_pcd_voxel.voxel_down_sample(voxel_size=0.02)

    return filtered_pcd_voxel

def _fillTN_absent(hand, ground, thresh, table):
    '''
    Creates the T/N relations with pre-defined point clouds of hand and ground as input. Thresh defines the distance to recognize touching.
    
    Parameters:
        * hand: point cloud of the hand
        * ground: point cloud of the ground
        * thresh: threshold distance of touching
        * table: chararray table   
    '''
#     T/N
#     H, 1
#     H, 2
#     H, 3
#     H, G
#     1, 2
#     1, 3
#     1, G
#     2, 3
#     2, G
#     3, G

    #get global objects 1, 2 and 3
    global o1, o2, o3, absent_o1, absent_o2, absent_o3 
        
    #fill in table with U,T or N corresponding to threshold distances
    if(o1 == None):
        table[0][0] = 'U'
    elif(len(hand.points) > 0):
        if(len(o1.points) == 0 or absent_o1 == True):
            table[1][0] = 'A'
            absent_o1 = True
        elif(len(o1.points) == 0):
            table[0][0] = 'N'
        elif(np.min(hand.compute_point_cloud_distance(o1)) < thresh):
            table[0][0] = 'T'
        else:
            table[0][0] = 'N'
        
    if(o2 == None):
        table[1][0] = 'U'
    elif(len(hand.points) > 0):
        if(len(o2.points) == 0 or absent_o2 == True):
            table[1][0] = 'A'
            absent_o2 = True
        elif(len(o2.points) == 0):
            table[1][0] = 'N'
        elif(np.min(hand.compute_point_cloud_distance(o2)) < thresh):
            table[1][0] = 'T'
        else:
            table[1][0] = 'N'
        

    if(o3 == None):
        table[2][0] = 'U'
    elif(len(hand.points) > 0):
        if(len(o3.points) == 0 or absent_o3 == True):
            table[2][0] = 'A'
            absent_o3 = True
        elif(len(o3.points) == 0):
            table[2][0] = 'N'
        elif(np.min(hand.compute_point_cloud_distance(o3)) < thresh):
            table[2][0] = 'T'
        else:
            table[2][0] = 'N'


    if(len(hand.points) > 0):
        if(np.min(hand.compute_point_cloud_distance(ground)) < thresh):
            table[3][0] = 'T'
        else:
            table[3][0] = 'N'

    if(o1 == None or o2 == None):
        table[4][0] = 'U'
    elif(len(o1.points) == 0 or len(o2.points) == 0 or absent_o1 == True or absent_o2 == True):
        table[4][0] = 'A'
    elif(np.min(o1.compute_point_cloud_distance(o2)) < thresh):
        table[4][0] = 'T'
    else:
        table[4][0] = 'N'

    if(o1 == None or o3 == None):
        table[5][0] = 'U'
    elif(len(o1.points) == 0 or len(o3.points) == 0 or absent_o1 == True or absent_o3 == True):
        table[5][0] = 'A'
    elif(np.min(o1.compute_point_cloud_distance(o3)) < thresh):
        table[5][0] = 'T'
    else:
        table[5][0] = 'N'

    if(o1 == None):
        table[6][0] = 'U'
    elif(len(o1.points) == 0 or absent_o1 == True):
        table[6][0] = 'A'
    elif(np.min(o1.compute_point_cloud_distance(ground)) < thresh):
        table[6][0] = 'T'
    else:
        table[6][0] = 'N'

    if(o2 == None or o3 == None):
        table[7][0] = 'U'
    elif(len(o2.points) == 0 or len(o3.points) == 0 or absent_o2 == True or absent_o3 == True):
        table[7][0] = 'A'
    elif(np.min(o2.compute_point_cloud_distance(o3)) < thresh):
        table[7][0] = 'T'
    else:
        table[7][0] = 'N'

    if(o2 == None):
        table[8][0] = 'U'
    elif(len(o2.points) == 0 or absent_o2 == True):
        table[8][0] = 'A'
    elif(np.min(o2.compute_point_cloud_distance(ground)) < thresh):
        table[8][0] = 'T'
    else:
        table[8][0] = 'N'

    if(o3 == None):
        table[9][0] = 'U'
    elif(len(o3.points) == 0 or absent_o3 == True):
        table[9][0] = 'A'
    elif(len(o3.points) > 0 and np.min(o3.compute_point_cloud_distance(ground)) < thresh):
        table[9][0] = 'T'
    else:
        table[9][0] = 'N'

def _fillSSR_2(hand, ground, table):
    '''
    Creates the SSR with pre-defined point clouds of hand and ground.
    
    Parameters:
        * hand: point cloud of the hand
        * ground: point cloud of the ground
        * table: chararray table   
    '''
#     SSR
#     H, 1
#     H, 2
#     H, 3
#     H, G
#     1, 2
#     1, 3
#     1, G
#     2, 3
#     2, G
#     3, G

    #get global objects 1, 2 and 3
    global o1, o2, o3
    
    #create AABB around object1 if it is defined and not absecent in the scene
    if(o1 != None and table[0][0] != b'A'):
        o1_box = o1.get_axis_aligned_bounding_box()
        points = np.asarray(o1_box.get_box_points())
        o1_max_x, o1_max_y, o1_max_z = np.max(points[:,0]), np.max(points[:,1]), np.max(points[:,2])
        o1_min_x, o1_min_y, o1_min_z = np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])
        
    #create AABB around object2 if it is defined and not absecent in the scene   
    if(o2 != None and table[1][0] != b'A'):
        o2_box = o2.get_axis_aligned_bounding_box()
        points = np.asarray(o2_box.get_box_points())
        o2_max_x, o2_max_y, o2_max_z = np.max(points[:,0]), np.max(points[:,1]), np.max(points[:,2])
        o2_min_x, o2_min_y, o2_min_z = np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])
    
    #create AABB around object2 if it is defined and not absecent in the scene
    if(o3 != None and table[2][0] != b'A'):
        o3_box = o3.get_axis_aligned_bounding_box()
        points = np.asarray(o3_box.get_box_points())
        o3_max_x, o3_max_y, o3_max_z = np.max(points[:,0]), np.max(points[:,1]), np.max(points[:,2])
        o3_min_x, o3_min_y, o3_min_z = np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])
        
    #create AABB around hand if it has a point cloud
    if(len(hand.points) > 0):
        hand_box = hand.get_axis_aligned_bounding_box()
        points = np.asarray(hand_box.get_box_points())
        hand_max_x, hand_max_y, hand_max_z = np.max(points[:,0]), np.max(points[:,1]), np.max(points[:,2])
        hand_min_x, hand_min_y, hand_min_z = np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])
    
    #create AABB around ground
    ground_box = ground.get_axis_aligned_bounding_box()
    points = np.asarray(ground_box.get_box_points())
    ground_max_x, ground_max_y, ground_max_z = np.max(points[:,0]), np.max(points[:,1]), np.max(points[:,2])
    ground_min_x, ground_min_y, ground_min_z = np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])
    

    
    if(o1 == None):
        table[10][0] = 'U'
    elif(table[0][0] == b'A'):
        table[10][0] = 'A'   
    elif(len(hand.points) > 0):
        if((hand_min_y > o1_min_y and hand_min_y < o1_max_y) and
           (hand_min_x > o1_min_x and hand_max_x < o1_max_x) and
           (hand_min_z > o1_min_z and hand_max_z < o1_max_z)):
            table[10][0] = 'In'   
        elif((hand_min_y < o1_min_y and hand_min_y > o1_max_y) and
           (hand_min_x < o1_min_x and hand_max_x > o1_max_x) and
           (hand_min_z < o1_min_z and hand_max_z > o1_max_z)):
            table[10][0] = 'Sa'       
        elif(((hand_min_y < o1_min_y and o1_max_y > hand_max_y) or
            (hand_min_y > o1_min_y and o1_max_y < hand_max_y))and
           ((o1_min_x < hand_min_x and o1_max_x < hand_max_x) or
            (o1_min_x > hand_min_x and o1_max_x < hand_max_x) or
            (o1_min_x > hand_min_x and o1_max_x > hand_max_x) or
            (o1_min_x < hand_min_x and o1_max_x > hand_max_x))and
            ((o1_min_z < hand_min_z and o1_max_z < hand_max_z)or
            (o1_min_z > hand_min_z and o1_max_z < hand_max_z) or
            (o1_min_z > hand_min_z and o1_max_z > hand_max_z) or
            (o1_min_z < hand_min_z and o1_max_z > hand_max_z))):
            if table[0][0] == b'T':
                table[10][0] = 'VArT'
            else:
                table[10][0] = 'VAr'
        else:
            if table[0][0] == b'T':
                table[10][0] = 'HArT'
            else:
                table[10][0] = 'HAr'
        
        
    if(o2 == None):
        table[11][0] = 'U'
    elif(table[1][0] == b'A'):
        table[11][0] = 'A'
    elif(len(hand.points) > 0):
        if((hand_min_y > o2_min_y and hand_min_y < o2_max_y) and
           (hand_min_x > o2_min_x and hand_max_x < o2_max_x) and
           (hand_min_z > o2_min_z and hand_max_z < o2_max_z)):
            table[11][0] = 'In'     
        elif((hand_min_y < o2_min_y and hand_min_y > o2_max_y) and
           (hand_min_x < o2_min_x and hand_max_x > o2_max_x) and
           (hand_min_z < o2_min_z and hand_max_z > o2_max_z)):
            table[11][0] = 'Sa'      
        elif((hand_min_y < o2_min_y and o2_max_y > hand_max_y)and
           ((o2_min_x < hand_min_x and o2_max_x < hand_max_x) or
            (o2_min_x > hand_min_x and o2_max_x < hand_max_x) or
            (o2_min_x > hand_min_x and o2_max_x > hand_max_x) or
            (o2_min_x < hand_min_x and o2_max_x > hand_max_x))and
            ((o2_min_z < hand_min_z and o2_max_z < hand_max_z)or
            (o2_min_z > hand_min_z and o2_max_z < hand_max_z) or
            (o2_min_z > hand_min_z and o2_max_z > hand_max_z) or
            (o2_min_z < hand_min_z and o2_max_z > hand_max_z))):
            if table[1][0] == b'T':
                table[11][0] = 'VArT'
            else:
                table[11][0] = 'VAr'
        else: 
            if table[1][0] == b'T':
                table[11][0] = 'HArT'
            else:
                table[11][0] = 'HAr'

    #print(table[11][0])

    if(o3 == None):
        table[12][0] = 'U'
    elif(table[2][0] == b'A'):
        table[12][0] = 'A'
    elif(len(hand.points) > 0):
        if((hand_min_y > o3_min_y and hand_min_y < o3_max_y) and
           (hand_min_x > o3_min_x and hand_max_x < o3_max_x) and
           (hand_min_z > o3_min_z and hand_max_z < o3_max_z)):
            table[12][0] = 'In'   
        if((hand_min_y < o3_min_y and hand_min_y > o3_max_y) and
           (hand_min_x < o3_min_x and hand_max_x > o3_max_x) and
           (hand_min_z < o3_min_z and hand_max_z > o3_max_z)):
            table[12][0] = 'Sa' 
        elif((hand_min_y < o3_min_y and o3_max_y > hand_max_y)and
           ((o3_min_x < hand_min_x and o3_max_x < hand_max_x) or
            (o3_min_x > hand_min_x and o3_max_x < hand_max_x) or
            (o3_min_x > hand_min_x and o3_max_x > hand_max_x) or
            (o3_min_x < hand_min_x and o3_max_x > hand_max_x))and
            ((o3_min_z < hand_min_z and o3_max_z < hand_max_z)or
            (o3_min_z > hand_min_z and o3_max_z < hand_max_z) or
            (o3_min_z > hand_min_z and o3_max_z > hand_max_z) or
            (o3_min_z < hand_min_z and o3_max_z > hand_max_z))):
            if table[2][0] == b'T':
                table[12][0] = 'VArT'
            else:
                table[12][0] = 'VAr'
        else: 
            if table[2][0] == b'T':
                table[12][0] = 'HArT'
            else:
                table[12][0] = 'HAr'
        
    if(len(hand.points) > 0): 
        if((hand_min_y > ground_min_y and hand_min_y < ground_max_y) and
           (hand_min_x > ground_min_x and hand_max_x < ground_max_x) and
           (hand_min_z > ground_min_z and hand_max_z < ground_max_z)):
            table[13][0] = 'In' 
        if((hand_min_y < ground_min_y and hand_min_y > ground_max_y) and
           (hand_min_x < ground_min_x and hand_max_x > ground_max_x) and
           (hand_min_z < ground_min_z and hand_max_z > ground_max_z)):
            table[13][0] = 'Sa'    
        elif((hand_min_y < ground_min_y and ground_max_y > hand_max_y)and
           ((ground_min_x < hand_min_x and ground_max_x < hand_max_x) or
            (ground_min_x > hand_min_x and ground_max_x < hand_max_x) or
            (ground_min_x > hand_min_x and ground_max_x > hand_max_x) or
            (ground_min_x < hand_min_x and ground_max_x > hand_max_x))and
            ((ground_min_z < hand_min_z and ground_max_z < hand_max_z)or
            (ground_min_z > hand_min_z and ground_max_z < hand_max_z) or
            (ground_min_z > hand_min_z and ground_max_z > hand_max_z) or
            (ground_min_z < hand_min_z and ground_max_z > hand_max_z))):
            if table[3][0] == b'T':
                table[13][0] = 'VArT'
            else:
                table[13][0] = 'VAr'
        else: 
            if table[3][0] == b'T':
                table[13][0] = 'HArT'
            else:
                table[13][0] = 'HAr'
    else:
        table[13][0] = 'U'

    if(o1 == None or o2 == None):
        table[14][0] = 'U'
    elif(table[4][0] == b'A'):
        table[14][0] = 'A'
    elif((o1_min_y > o2_min_y and o1_min_y < o2_max_y) and
           (o1_min_x > o2_min_x and o1_max_x < o2_max_x) and
           (o1_min_z > o2_min_z and o1_max_z < o2_max_z)):
            table[14][0] = 'In'
    elif((o1_min_y < o2_min_y and o1_min_y > o2_max_y) and
       (o1_min_x < o2_min_x and o1_max_x > o2_max_x) and
       (o1_min_z < o2_min_z and o1_max_z > o2_max_z)):
        table[14][0] = 'Sa'       
    elif((o1_min_y < o2_min_y and o2_max_y > o1_max_y)and
           ((o2_min_x < o1_min_x and o2_max_x < o1_max_x) or
            (o2_min_x > o1_min_x and o2_max_x < o1_max_x) or
            (o2_min_x > o1_min_x and o2_max_x > o1_max_x) or
            (o2_min_x < o1_min_x and o2_max_x > o1_max_x))and
            ((o2_min_z < o1_min_z and o2_max_z < o1_max_z)or
            (o2_min_z > o1_min_z and o2_max_z < o1_max_z) or
            (o2_min_z > o1_min_z and o2_max_z > o1_max_z) or
            (o2_min_z < o1_min_z and o2_max_z > o1_max_z))):
            if table[4][0] == b'T':
                table[14][0] = 'VArT'
            else:
                table[14][0] = 'VAr'
    else: 
            if table[4][0] == b'T':
                table[14][0] = 'HArT'
            else:
                table[14][0] = 'HAr'
    
 
    if(o1 == None or o3 == None):
        table[15][0] = 'U'
    elif(table[5][0] == b'A'):
        table[15][0] = 'A'
    elif((o1_min_y > o3_min_y and o1_min_y < o3_max_y) and
           (o1_min_x > o3_min_x and o1_max_x < o3_max_x) and
           (o1_min_z > o3_min_z and o1_max_z < o3_max_z)):
            table[15][0] = 'In' 
    elif((o1_min_y < o3_min_y and o1_min_y > o3_max_y) and
           (o1_min_x < o3_min_x and o1_max_x > o3_max_x) and
           (o1_min_z < o3_min_z and o1_max_z > o3_max_z)):
            table[15][0] = 'Sa'       
    elif((o1_min_y < o3_min_y and o3_max_y > o1_max_y)and
           ((o3_min_x < o1_min_x and o3_max_x < o1_max_x) or
            (o3_min_x > o1_min_x and o3_max_x < o1_max_x) or
            (o3_min_x > o1_min_x and o3_max_x > o1_max_x) or
            (o3_min_x < o1_min_x and o3_max_x > o1_max_x))and
            ((o3_min_z < o1_min_z and o3_max_z < o1_max_z)or
            (o3_min_z > o1_min_z and o3_max_z < o1_max_z) or
            (o3_min_z > o1_min_z and o3_max_z > o1_max_z) or
            (o3_min_z < o1_min_z and o3_max_z > o1_max_z))):
        if table[5][0] == b'T':
            table[15][0] = 'VArT'
        else:
            table[15][0] = 'VAr'
    else:
        if table[5][0] == b'T':
            table[15][0] = 'HArT'
        else:
            table[15][0] = 'HAr'

    if(o1 == None):
        table[16][0] = 'U'
    elif(table[6][0] == b'A'):
        table[16][0] = 'A'
    elif((o1_min_y > ground_min_y and o1_min_y < ground_max_y) and
           (o1_min_x > ground_min_x and o1_max_x < ground_max_x) and
           (o1_min_z > ground_min_z and o1_max_z < ground_max_z)):
            table[16][0] = 'In'
    elif((o1_min_y < ground_min_y and o1_min_y > ground_max_y) and
           (o1_min_x < ground_min_x and o1_max_x > ground_max_x) and
           (o1_min_z < ground_min_z and o1_max_z > ground_max_z)):
            table[16][0] = 'Sa'        
    elif((o1_min_y < ground_min_y and ground_max_y > o1_max_y)and
           ((ground_min_x < o1_min_x and ground_max_x < o1_max_x) or
            (ground_min_x > o1_min_x and ground_max_x < o1_max_x) or
            (ground_min_x > o1_min_x and ground_max_x > o1_max_x) or
            (ground_min_x < o1_min_x and ground_max_x > o1_max_x))and
            ((ground_min_z < o1_min_z and ground_max_z < o1_max_z)or
            (ground_min_z > o1_min_z and ground_max_z < o1_max_z) or
            (ground_min_z > o1_min_z and ground_max_z > o1_max_z) or
            (ground_min_z < o1_min_z and ground_max_z > o1_max_z))):
        if table[6][0] == b'T':
            table[16][0] = 'VArT'
        else:
            table[16][0] = 'VAr'
    else:
        if table[6][0] == b'T':
                table[16][0] = 'HArT'
        else:
            table[16][0] = 'HAr'
        
    if(o2 == None or o3 == None):
        table[17][0] = 'U'
    elif(table[7][0] == b'A'):
        table[17][0] = 'A'
    elif((o2_min_y > o3_min_y and o2_min_y < o3_max_y) and
           (o2_min_x > o3_min_x and o2_max_x < o3_max_x) and
           (o2_min_z > o3_min_z and o2_max_z < o3_max_z)):
            table[17][0] = 'In'
    elif((o2_min_y < o3_min_y and o2_min_y > o3_max_y) and
           (o2_min_x < o3_min_x and o2_max_x > o3_max_x) and
           (o2_min_z < o3_min_z and o2_max_z > o3_max_z)):
            table[17][0] = 'Sa'        
    elif((o2_min_y < o3_min_y and o3_max_y > o2_max_y)and
           ((o3_min_x < o2_min_x and o3_max_x < o2_max_x) or
            (o3_min_x > o2_min_x and o3_max_x < o2_max_x) or
            (o3_min_x > o2_min_x and o3_max_x > o2_max_x) or
            (o3_min_x < o2_min_x and o3_max_x > o2_max_x))and
            ((o3_min_z < o2_min_z and o3_max_z < o2_max_z)or
            (o3_min_z > o2_min_z and o3_max_z < o2_max_z) or
            (o3_min_z > o2_min_z and o3_max_z > o2_max_z) or
            (o3_min_z < o2_min_z and o3_max_z > o2_max_z))):
        if table[7][0] == b'T':
            table[17][0] = 'VArT'
        else:
            table[17][0] = 'VAr'
    else:
        if table[7][0] == b'T':
                table[17][0] = 'HArT'
        else:
            table[17][0] = 'HAr'
    
    if(o2 == None):
        table[18][0] = 'U'
    elif(table[8][0] == b'A'):
        table[18][0] = 'A'
    elif((o2_min_y > ground_min_y and o2_min_y < ground_max_y) and
           (o2_min_x > ground_min_x and o2_max_x < ground_max_x) and
           (o2_min_z > ground_min_z and o2_max_z < ground_max_z)):
            table[18][0] = 'In'
    elif((o2_min_y < ground_min_y and o2_min_y > ground_max_y) and
           (o2_min_x < ground_min_x and o2_max_x > ground_max_x) and
           (o2_min_z < ground_min_z and o2_max_z > ground_max_z)):
            table[18][0] = 'Sa'        
    elif((o2_min_y < ground_min_y and ground_max_y > o2_max_y)and
           ((ground_min_x < o2_min_x and ground_max_x < o2_max_x) or
            (ground_min_x > o2_min_x and ground_max_x < o2_max_x) or
            (ground_min_x > o2_min_x and ground_max_x > o2_max_x) or
            (ground_min_x < o2_min_x and ground_max_x > o2_max_x))and
            ((ground_min_z < o2_min_z and ground_max_z < o2_max_z)or
            (ground_min_z > o2_min_z and ground_max_z < o2_max_z) or
            (ground_min_z > o2_min_z and ground_max_z > o2_max_z) or
            (ground_min_z < o2_min_z and ground_max_z > o2_max_z))):
        if table[8][0] == b'T':
            table[18][0] = 'VArT'
        else:
            table[18][0] = 'VAr'
    else:
        if table[8][0] == b'T':
                table[18][0] = 'HArT'
        else:
            table[18][0] = 'HAr'
            
    if(o3 == None):
        table[19][0] = 'U'
    elif(table[9][0] == b'A'):
        table[19][0] = 'A'
    elif((o3_min_y > ground_min_y and o3_min_y < ground_max_y) and
           (o3_min_x > ground_min_x and o3_max_x < ground_max_x) and
           (o3_min_z > ground_min_z and o3_max_z < ground_max_z)):
            table[19][0] = 'In'
    elif((o3_min_y < ground_min_y and o3_min_y > ground_max_y) and
           (o3_min_x < ground_min_x and o3_max_x > ground_max_x) and
           (o3_min_z < ground_min_z and o3_max_z > ground_max_z)):
            table[19][0] = 'Sa'        
    elif((o3_min_y < ground_min_y and ground_max_y > o3_max_y)and
           ((ground_min_x < o3_min_x and ground_max_x < o3_max_x) or
            (ground_min_x > o3_min_x and ground_max_x < o3_max_x) or
            (ground_min_x > o3_min_x and ground_max_x > o3_max_x) or
            (ground_min_x < o3_min_x and ground_max_x > o3_max_x))and
            ((ground_min_z < o3_min_z and ground_max_z < o3_max_z)or
            (ground_min_z > o3_min_z and ground_max_z < o3_max_z) or
            (ground_min_z > o3_min_z and ground_max_z > o3_max_z) or
            (ground_min_z < o3_min_z and ground_max_z > o3_max_z))):
        if table[9][0] == b'T':
            table[19][0] = 'VArT'
        else:
            table[19][0] = 'VAr'
        
    else:
        if table[9][0] == b'T':
                table[19][0] = 'HArT'
        else:
            table[19][0] = 'HAr'

def _fillSSR_3(hand, ground, table):
    '''
    Creates the SSR with pre-defined point clouds of hand and ground.
    
    Parameters:
        * hand: point cloud of the hand
        * ground: point cloud of the ground
        * table: chararray table   
    '''
#     SSR
#     H, 1
#     H, 2
#     H, 3
#     H, G
#     1, 2
#     1, 3
#     1, G
#     2, 3
#     2, G
#     3, G

    #get global objects 1, 2 and 3
    global o1, o2, o3
    #--------------------------------------------------
    #y-z replaced 
    # |y
    # |
    # |    - z
    # |   -
    # |  -
    # |-
    #  - - - - - - - - - - - - - -x
    #--------------------------------------------------
    #create AABB around object1 if it is defined and not absecent in the scene
    if(o1 != None and table[0][0] != b'A'):
        o1_box = o1.get_axis_aligned_bounding_box()
        points = np.asarray(o1_box.get_box_points())
        o1_max_x, o1_max_z, o1_max_y = np.max(points[:,0]), np.max(points[:,1]), np.max(points[:,2])
        o1_min_x, o1_min_z, o1_min_y = np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])
        
    #create AABB around object2 if it is defined and not absecent in the scene   
    if(o2 != None and table[1][0] != b'A'):
        o2_box = o2.get_axis_aligned_bounding_box()
        points = np.asarray(o2_box.get_box_points())
        o2_max_x, o2_max_z, o2_max_y = np.max(points[:,0]), np.max(points[:,1]), np.max(points[:,2])
        o2_min_x, o2_min_z, o2_min_y = np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])
    
    #create AABB around object2 if it is defined and not absecent in the scene
    if(o3 != None and table[2][0] != b'A'):
        o3_box = o3.get_axis_aligned_bounding_box()
        points = np.asarray(o3_box.get_box_points())
        o3_max_x, o3_max_z, o3_max_y = np.max(points[:,0]), np.max(points[:,1]), np.max(points[:,2])
        o3_min_x, o3_min_z, o3_min_y = np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])
        
    #create AABB around hand if it has a point cloud
    if(len(hand.points) > 0):
        hand_box = hand.get_axis_aligned_bounding_box()
        points = np.asarray(hand_box.get_box_points())
        hand_max_x, hand_max_z, hand_max_y = np.max(points[:,0]), np.max(points[:,1]), np.max(points[:,2])
        hand_min_x, hand_min_z, hand_min_y = np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])
    
    #create AABB around ground
    ground_box = ground.get_axis_aligned_bounding_box()
    points = np.asarray(ground_box.get_box_points())
    ground_max_x, ground_max_z, ground_max_y = np.max(points[:,0]), np.max(points[:,1]), np.max(points[:,2])
    ground_min_x, ground_min_z, ground_min_y = np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])
    

    
    if(o1 == None):
        table[10][0] = 'U'
    elif(table[0][0] == b'A'):
        table[10][0] = 'A'   
    elif(len(hand.points) > 0):
        if((hand_min_y > o1_min_y and hand_min_y < o1_max_y) and
        #    (hand_min_x > o1_min_x and hand_max_x < o1_max_x) and
           (hand_min_z > o1_min_z and hand_max_z < o1_max_z)):
            table[10][0] = 'In'   
        elif((hand_min_y < o1_min_y and hand_min_y > o1_max_y) and
           (hand_min_x < o1_min_x and hand_max_x > o1_max_x) and
           (hand_min_z < o1_min_z and hand_max_z > o1_max_z)):
            table[10][0] = 'Sa'       
        elif(((hand_min_y < o1_box.get_center()[2]  and o1_max_y > hand_max_y) or
            (hand_min_y > o1_box.get_center()[2]  and o1_max_y < hand_max_y))and
           ((o1_min_x < hand_min_x and o1_max_x < hand_max_x) or
            (o1_min_x > hand_min_x and o1_max_x < hand_max_x) or
            (o1_min_x > hand_min_x and o1_max_x > hand_max_x) or
            (o1_min_x < hand_min_x and o1_max_x > hand_max_x))and
            ((o1_min_z < hand_min_z and o1_max_z < hand_max_z)or
            (o1_min_z > hand_min_z and o1_max_z < hand_max_z) or
            (o1_min_z > hand_min_z and o1_max_z > hand_max_z) or
            (o1_min_z < hand_min_z and o1_max_z > hand_max_z))):
            if table[0][0] == b'T':
                table[10][0] = 'VArT'
            else:
                table[10][0] = 'VAr'
        else:
            if table[0][0] == b'T':
                table[10][0] = 'HArT'
            else:
                table[10][0] = 'HAr'
        
        
    if(o2 == None):
        table[11][0] = 'U'
    elif(table[1][0] == b'A'):
        table[11][0] = 'A'
    elif(len(hand.points) > 0):
        if((hand_min_y > o2_min_y and hand_min_y < o2_max_y) and
        #    (hand_min_x > o2_min_x and hand_max_x < o2_max_x) and
           (hand_min_z > o2_min_z and hand_max_z < o2_max_z)):
            table[11][0] = 'In'     
        elif((hand_min_y < o2_min_y and hand_min_y > o2_max_y) and
           (hand_min_x < o2_min_x and hand_max_x > o2_max_x) and
           (hand_min_z < o2_min_z and hand_max_z > o2_max_z)):
            table[11][0] = 'Sa'      
        elif(((hand_min_y < o2_box.get_center()[2]  and o2_max_y > hand_max_y) or
            (hand_min_y > o2_box.get_center()[2]  and o2_max_y < hand_max_y)) and
           ((o2_min_x < hand_min_x and o2_max_x < hand_max_x) or
            (o2_min_x > hand_min_x and o2_max_x < hand_max_x) or
            (o2_min_x > hand_min_x and o2_max_x > hand_max_x) or
            (o2_min_x < hand_min_x and o2_max_x > hand_max_x))and
            ((o2_min_z < hand_min_z and o2_max_z < hand_max_z)or
            (o2_min_z > hand_min_z and o2_max_z < hand_max_z) or
            (o2_min_z > hand_min_z and o2_max_z > hand_max_z) or
            (o2_min_z < hand_min_z and o2_max_z > hand_max_z))):
            if table[1][0] == b'T':
                table[11][0] = 'VArT'
            else:
                table[11][0] = 'VAr'
        else: 
            if table[1][0] == b'T':
                table[11][0] = 'HArT'
            else:
                table[11][0] = 'HAr'

    #print(table[11][0])

    if(o3 == None):
        table[12][0] = 'U'
    elif(table[2][0] == b'A'):
        table[12][0] = 'A'
    elif(len(hand.points) > 0):
        if((hand_min_y > o3_min_y and hand_min_y < o3_max_y) and
        #    (hand_min_x > o3_min_x and hand_max_x < o3_max_x) and
           (hand_min_z > o3_min_z and hand_max_z < o3_max_z)):
            table[12][0] = 'In'   
        if((hand_min_y < o3_min_y and hand_min_y > o3_max_y) and
           (hand_min_x < o3_min_x and hand_max_x > o3_max_x) and
           (hand_min_z < o3_min_z and hand_max_z > o3_max_z)):
            table[12][0] = 'Sa' 
        elif(((hand_min_y < o3_box.get_center()[2]  and o3_max_y > hand_max_y)or
            (hand_min_y > o3_box.get_center()[2]  and o3_max_y < hand_max_y)) and
           ((o3_min_x < hand_min_x and o3_max_x < hand_max_x) or
            (o3_min_x > hand_min_x and o3_max_x < hand_max_x) or
            (o3_min_x > hand_min_x and o3_max_x > hand_max_x) or
            (o3_min_x < hand_min_x and o3_max_x > hand_max_x))and
            ((o3_min_z < hand_min_z and o3_max_z < hand_max_z)or
            (o3_min_z > hand_min_z and o3_max_z < hand_max_z) or
            (o3_min_z > hand_min_z and o3_max_z > hand_max_z) or
            (o3_min_z < hand_min_z and o3_max_z > hand_max_z))):
            if table[2][0] == b'T':
                table[12][0] = 'VArT'
            else:
                table[12][0] = 'VAr'
        else: 
            if table[2][0] == b'T':
                table[12][0] = 'HArT'
            else:
                table[12][0] = 'HAr'
        
    if(len(hand.points) > 0): 
        if((hand_min_y > ground_min_y and hand_min_y < ground_max_y) and
        #    (hand_min_x > ground_min_x and hand_max_x < ground_max_x) and
           (hand_min_z > ground_min_z and hand_max_z < ground_max_z)):
            table[13][0] = 'In' 
        if((hand_min_y < ground_min_y and hand_min_y > ground_max_y) and
           (hand_min_x < ground_min_x and hand_max_x > ground_max_x) and
           (hand_min_z < ground_min_z and hand_max_z > ground_max_z)):
            table[13][0] = 'Sa'    
        elif(((hand_min_y < ground_box.get_center()[2]  and ground_max_y > hand_max_y) or
            (hand_min_y > ground_box.get_center()[2]  and ground_max_y < hand_max_y)) and
           ((ground_min_x < hand_min_x and ground_max_x < hand_max_x) or
            (ground_min_x > hand_min_x and ground_max_x < hand_max_x) or
            (ground_min_x > hand_min_x and ground_max_x > hand_max_x) or
            (ground_min_x < hand_min_x and ground_max_x > hand_max_x))and
            ((ground_min_z < hand_min_z and ground_max_z < hand_max_z)or
            (ground_min_z > hand_min_z and ground_max_z < hand_max_z) or
            (ground_min_z > hand_min_z and ground_max_z > hand_max_z) or
            (ground_min_z < hand_min_z and ground_max_z > hand_max_z))):
            if table[3][0] == b'T':
                table[13][0] = 'VArT'
            else:
                table[13][0] = 'VAr'
        else: 
            if table[3][0] == b'T':
                table[13][0] = 'HArT'
            else:
                table[13][0] = 'HAr'
    else:
        table[13][0] = 'U'

    # if(o1 != None and o2 != None):
    #     print("\n","(",o1_min_y < o2_min_y, "and", o2_max_y > o1_max_y,")", "or","\n",
    #         "(",o1_min_y > o2_min_y, "and", o2_max_y < o1_max_y,"))", "and","\n",
    #         "((",o2_min_x < o1_min_x, "and", o2_max_x < o1_max_x,")", "or","\n",
    #         "(",o2_min_x > o1_min_x, "and", o2_max_x < o1_max_x,")" ,"or","\n",
    #         "(",o2_min_x > o1_min_x, "and", o2_max_x > o1_max_x,")", "or","\n",
    #         "(",o2_min_x < o1_min_x ,"and" ,o2_max_x > o1_max_x,"))","and","\n",
    #         "((",o2_min_z < o1_min_z, "and" ,o2_max_z < o1_max_z,")","or","\n",
    #         "(",o2_min_z > o1_min_z ,"and", o2_max_z < o1_max_z,")", "or","\n",
    #         "(",o2_min_z > o1_min_z, "and", o2_max_z > o1_max_z,")", "or","\n",
    #         "(",o2_min_z < o1_min_z,"and", o2_max_z > o1_max_z,")))")

    if(o1 == None or o2 == None):
        table[14][0] = 'U'
    elif(table[4][0] == b'A'):
        table[14][0] = 'A'
    elif((o1_min_y > o2_min_y and o1_min_y < o2_max_y) and
        #    (o1_min_x > o2_min_x and o1_max_x < o2_max_x) and
           (o1_min_z > o2_min_z and o1_max_z < o2_max_z)):
            table[14][0] = 'In'
    elif((o1_min_y < o2_min_y and o1_min_y > o2_max_y) and
       (o1_min_x < o2_min_x and o1_max_x > o2_max_x) and
       (o1_min_z < o2_min_z and o1_max_z > o2_max_z)):
        table[14][0] = 'Sa'       
    elif(((o1_min_y < o2_box.get_center()[2]  and o2_max_y > o1_max_y) or
        (o1_min_y > o2_box.get_center()[2]  and o2_max_y < o1_max_y)) and
           ((o2_min_x < o1_min_x and o2_max_x < o1_max_x) or
            (o2_min_x > o1_min_x and o2_max_x < o1_max_x) or
            (o2_min_x > o1_min_x and o2_max_x > o1_max_x) or
            (o2_min_x < o1_min_x and o2_max_x > o1_max_x))and
            ((o2_min_z < o1_min_z and o2_max_z < o1_max_z)or
            (o2_min_z > o1_min_z and o2_max_z < o1_max_z) or
            (o2_min_z > o1_min_z and o2_max_z > o1_max_z) or
            (o2_min_z < o1_min_z and o2_max_z > o1_max_z))):
            if table[4][0] == b'T':
                table[14][0] = 'VArT'
            else:
                table[14][0] = 'VAr'
    else: 
            if table[4][0] == b'T':
                table[14][0] = 'HArT'
            else:
                table[14][0] = 'HAr'
    
 
    if(o1 == None or o3 == None):
        table[15][0] = 'U'
    elif(table[5][0] == b'A'):
        table[15][0] = 'A'
    elif((o1_min_y > o3_min_y and o1_min_y < o3_max_y) and
        #    (o1_min_x > o3_min_x and o1_max_x < o3_max_x) and
           (o1_min_z > o3_min_z and o1_max_z < o3_max_z)):
            table[15][0] = 'In' 
    elif((o1_min_y < o3_min_y and o1_min_y > o3_max_y) and
           (o1_min_x < o3_min_x and o1_max_x > o3_max_x) and
           (o1_min_z < o3_min_z and o1_max_z > o3_max_z)):
            table[15][0] = 'Sa'       
    elif(((o1_min_y < o3_box.get_center()[2]  and o3_max_y > o1_max_y) or
        (o1_min_y > o3_box.get_center()[2]  and o3_max_y < o1_max_y)) and
           ((o3_min_x < o1_min_x and o3_max_x < o1_max_x) or
            (o3_min_x > o1_min_x and o3_max_x < o1_max_x) or
            (o3_min_x > o1_min_x and o3_max_x > o1_max_x) or
            (o3_min_x < o1_min_x and o3_max_x > o1_max_x))and
            ((o3_min_z < o1_min_z and o3_max_z < o1_max_z)or
            (o3_min_z > o1_min_z and o3_max_z < o1_max_z) or
            (o3_min_z > o1_min_z and o3_max_z > o1_max_z) or
            (o3_min_z < o1_min_z and o3_max_z > o1_max_z))):
        if table[5][0] == b'T':
            table[15][0] = 'VArT'
        else:
            table[15][0] = 'VAr'
    else:
        if table[5][0] == b'T':
            table[15][0] = 'HArT'
        else:
            table[15][0] = 'HAr'

    if(o1 == None):
        table[16][0] = 'U'
    elif(table[6][0] == b'A'):
        table[16][0] = 'A'
    elif((o1_min_y > ground_min_y and o1_min_y < ground_max_y) and
        #    (o1_min_x > ground_min_x and o1_max_x < ground_max_x) and
           (o1_min_z > ground_min_z and o1_max_z < ground_max_z)):
            table[16][0] = 'In'
    elif((o1_min_y < ground_min_y and o1_min_y > ground_max_y) and
           (o1_min_x < ground_min_x and o1_max_x > ground_max_x) and
           (o1_min_z < ground_min_z and o1_max_z > ground_max_z)):
            table[16][0] = 'Sa'        
    elif(((o1_min_y < ground_box.get_center()[2]  and ground_max_y > o1_max_y) or
        (o1_min_y > ground_box.get_center()[2]  and ground_max_y < o1_max_y)) and
           ((ground_min_x < o1_min_x and ground_max_x < o1_max_x) or
            (ground_min_x > o1_min_x and ground_max_x < o1_max_x) or
            (ground_min_x > o1_min_x and ground_max_x > o1_max_x) or
            (ground_min_x < o1_min_x and ground_max_x > o1_max_x))and
            ((ground_min_z < o1_min_z and ground_max_z < o1_max_z)or
            (ground_min_z > o1_min_z and ground_max_z < o1_max_z) or
            (ground_min_z > o1_min_z and ground_max_z > o1_max_z) or
            (ground_min_z < o1_min_z and ground_max_z > o1_max_z))):
        if table[6][0] == b'T':
            table[16][0] = 'VArT'
        else:
            table[16][0] = 'VAr'
    else:
        if table[6][0] == b'T':
                table[16][0] = 'HArT'
        else:
            table[16][0] = 'HAr'
        
    if(o2 == None or o3 == None):
        table[17][0] = 'U'
    elif(table[7][0] == b'A'):
        table[17][0] = 'A'
    elif((o2_min_y > o3_min_y and o2_min_y < o3_max_y) and
        #    (o2_min_x > o3_min_x and o2_max_x < o3_max_x) and
           (o2_min_z > o3_min_z and o2_max_z < o3_max_z)):
            table[17][0] = 'In'
    elif((o2_min_y < o3_min_y and o2_min_y > o3_max_y) and
           (o2_min_x < o3_min_x and o2_max_x > o3_max_x) and
           (o2_min_z < o3_min_z and o2_max_z > o3_max_z)):
            table[17][0] = 'Sa'        
    elif(((o2_min_y < o3_box.get_center()[2]  and o3_max_y > o2_max_y) or
        (o2_min_y > o3_box.get_center()[2]  and o3_max_y < o2_max_y)) and
           ((o3_min_x < o2_min_x and o3_max_x < o2_max_x) or
            (o3_min_x > o2_min_x and o3_max_x < o2_max_x) or
            (o3_min_x > o2_min_x and o3_max_x > o2_max_x) or
            (o3_min_x < o2_min_x and o3_max_x > o2_max_x))and
            ((o3_min_z < o2_min_z and o3_max_z < o2_max_z)or
            (o3_min_z > o2_min_z and o3_max_z < o2_max_z) or
            (o3_min_z > o2_min_z and o3_max_z > o2_max_z) or
            (o3_min_z < o2_min_z and o3_max_z > o2_max_z))):
        if table[7][0] == b'T':
            table[17][0] = 'VArT'
        else:
            table[17][0] = 'VAr'
    else:
        if table[7][0] == b'T':
                table[17][0] = 'HArT'
        else:
            table[17][0] = 'HAr'
    
    if(o2 == None):
        table[18][0] = 'U'
    elif(table[8][0] == b'A'):
        table[18][0] = 'A'
    elif((o2_min_y > ground_min_y and o2_min_y < ground_max_y) and
        #    (o2_min_x > ground_min_x and o2_max_x < ground_max_x) and
           (o2_min_z > ground_min_z and o2_max_z < ground_max_z)):
            table[18][0] = 'In'
    elif((o2_min_y < ground_min_y and o2_min_y > ground_max_y) and
           (o2_min_x < ground_min_x and o2_max_x > ground_max_x) and
           (o2_min_z < ground_min_z and o2_max_z > ground_max_z)):
            table[18][0] = 'Sa'        
    elif(((o2_min_y < ground_box.get_center()[2]  and ground_max_y > o2_max_y) or
        (o2_min_y > ground_box.get_center()[2]  and ground_max_y < o2_max_y)) and
           ((ground_min_x < o2_min_x and ground_max_x < o2_max_x) or
            (ground_min_x > o2_min_x and ground_max_x < o2_max_x) or
            (ground_min_x > o2_min_x and ground_max_x > o2_max_x) or
            (ground_min_x < o2_min_x and ground_max_x > o2_max_x))and
            ((ground_min_z < o2_min_z and ground_max_z < o2_max_z)or
            (ground_min_z > o2_min_z and ground_max_z < o2_max_z) or
            (ground_min_z > o2_min_z and ground_max_z > o2_max_z) or
            (ground_min_z < o2_min_z and ground_max_z > o2_max_z))):
        if table[8][0] == b'T':
            table[18][0] = 'VArT'
        else:
            table[18][0] = 'VAr'
    else:
        if table[8][0] == b'T':
                table[18][0] = 'HArT'
        else:
            table[18][0] = 'HAr'
            
    if(o3 == None):
        table[19][0] = 'U'
    elif(table[9][0] == b'A'):
        table[19][0] = 'A'
    elif((o3_min_y > ground_min_y and o3_min_y < ground_max_y) and
        #    (o3_min_x > ground_min_x and o3_max_x < ground_max_x) and
           (o3_min_z > ground_min_z and o3_max_z < ground_max_z)):
            table[19][0] = 'In'
    elif((o3_min_y < ground_min_y and o3_min_y > ground_max_y) and
           (o3_min_x < ground_min_x and o3_max_x > ground_max_x) and
           (o3_min_z < ground_min_z and o3_max_z > ground_max_z)):
            table[19][0] = 'Sa'        
    elif(((o3_min_y < ground_box.get_center()[2] and ground_max_y > o3_max_y) or
        (o3_min_y > ground_box.get_center()[2] and ground_max_y < o3_max_y)) and
           ((ground_min_x < o3_min_x and ground_max_x < o3_max_x) or
            (ground_min_x > o3_min_x and ground_max_x < o3_max_x) or
            (ground_min_x > o3_min_x and ground_max_x > o3_max_x) or
            (ground_min_x < o3_min_x and ground_max_x > o3_max_x))and
            ((ground_min_z < o3_min_z and ground_max_z < o3_max_z)or
            (ground_min_z > o3_min_z and ground_max_z < o3_max_z) or
            (ground_min_z > o3_min_z and ground_max_z > o3_max_z) or
            (ground_min_z < o3_min_z and ground_max_z > o3_max_z))):
        if table[9][0] == b'T':
            table[19][0] = 'VArT'
        else:
            table[19][0] = 'VAr'
        
    else:
        if table[9][0] == b'T':
                table[19][0] = 'HArT'
        else:
            table[19][0] = 'HAr'

def _fillSSR_4(hand, ground, table):
    '''
    Creates the SSR with pre-defined point clouds of hand and ground.
    
    Parameters:
        * hand: point cloud of the hand
        * ground: point cloud of the ground
        * table: chararray table   
    '''
#     SSR
#     H, 1
#     H, 2
#     H, 3
#     H, G
#     1, 2
#     1, 3
#     1, G
#     2, 3
#     2, G
#     3, G

    #get global objects 1, 2 and 3
    global o1, o2, o3
    #--------------------------------------------------
    #y-z replaced 
    # |y
    # |
    # |    - z
    # |   -
    # |  -
    # |-
    #  - - - - - - - - - - - - - -x
    #--------------------------------------------------
    #create AABB around object1 if it is defined and not absecent in the scene
    if(o1 != None and table[0][0] != b'A'):
        o1_box = o1.get_axis_aligned_bounding_box()
        points = np.asarray(o1_box.get_box_points())
        o1_max_x, o1_max_z, o1_max_y = np.max(points[:,0]), np.max(points[:,1]), np.max(points[:,2])
        o1_min_x, o1_min_z, o1_min_y = np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])
        
    #create AABB around object2 if it is defined and not absecent in the scene   
    if(o2 != None and table[1][0] != b'A'):
        o2_box = o2.get_axis_aligned_bounding_box()
        points = np.asarray(o2_box.get_box_points())
        o2_max_x, o2_max_z, o2_max_y = np.max(points[:,0]), np.max(points[:,1]), np.max(points[:,2])
        o2_min_x, o2_min_z, o2_min_y = np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])
    
    #create AABB around object2 if it is defined and not absecent in the scene
    if(o3 != None and table[2][0] != b'A'):
        o3_box = o3.get_axis_aligned_bounding_box()
        points = np.asarray(o3_box.get_box_points())
        o3_max_x, o3_max_z, o3_max_y = np.max(points[:,0]), np.max(points[:,1]), np.max(points[:,2])
        o3_min_x, o3_min_z, o3_min_y = np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])
        
    #create AABB around hand if it has a point cloud
    if(len(hand.points) > 0):
        hand_box = hand.get_axis_aligned_bounding_box()
        points = np.asarray(hand_box.get_box_points())
        hand_max_x, hand_max_z, hand_max_y = np.max(points[:,0]), np.max(points[:,1]), np.max(points[:,2])
        hand_min_x, hand_min_z, hand_min_y = np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])
    
    #create AABB around ground
    ground_box = ground.get_axis_aligned_bounding_box()
    points = np.asarray(ground_box.get_box_points())
    ground_max_x, ground_max_z, ground_max_y = np.max(points[:,0]), np.max(points[:,1]), np.max(points[:,2])
    ground_min_x, ground_min_z, ground_min_y = np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])
    

    
    if(o1 == None):
        table[10][0] = 'U'
    elif(table[0][0] == b'A'):
        table[10][0] = 'A'   
    elif(len(hand.points) > 0):
        if((hand_min_y > o1_min_y and hand_min_y < o1_max_y) and
        #    (hand_min_x > o1_min_x and hand_max_x < o1_max_x) and
           (hand_min_z > o1_min_z and hand_max_z < o1_max_z)):
            table[10][0] = 'In'  
            # print("Hand in o1:",
            # "(",hand_min_y > o1_min_y, "and", hand_min_y < o1_max_y,")", "and",
            # "(",hand_min_z > o1_min_z ,"and", hand_max_z < o1_max_z,"))") 

        elif((hand_min_y < o1_min_y and hand_min_y > o1_max_y) and
           (hand_min_x < o1_min_x and hand_max_x > o1_max_x) and
           (hand_min_z < o1_min_z and hand_max_z > o1_max_z)):
            table[10][0] = 'Sa'       
        elif(((hand.get_center()[2] >= o1_max_y) or
            (hand.get_center()[2] <= o1_min_y))and
             ((hand_max_x < o1_max_x) or
              (hand_min_x > o1_min_x) or
              (hand.get_center()[0] <= o1_max_x) or
              (hand.get_center()[0] >= o1_min_x))and
             ((hand_max_z < o1_max_z) or
              (hand_min_z > o1_min_z) or
              (hand.get_center()[1] <= o1_max_z) or
              (hand.get_center()[1] >= o1_min_z))):
            if table[0][0] == b'T':
                table[10][0] = 'VArT'
            else:
                table[10][0] = 'VAr'
        elif(((hand.get_center()[2] <= o1_max_y) or
              (hand.get_center()[2] >= o1_min_y)) and
             ((hand_max_x < o1_max_x) or
              (hand_min_x > o1_min_x) or
              (hand_min_x < o1_min_x) or 
              (hand_max_x > o1_max_x))and
             ((hand_max_z < o1_max_z) or
              (hand_min_z > o1_min_z) or
              (hand_max_z > o1_max_z) or
              (hand_min_z < o1_min_z))):
            if table[0][0] == b'T':
                table[10][0] = 'HArT'
            else:
                table[10][0] = 'HAr'
        else:
            print("Fehler: H,o1")
            # print("\n",
            # "((",hand_min_y >= o1_box.get_center()[2],")", "or","\n",
            # "(",hand_max_y <=o1_min_y,')', 'or',"\n",
            # '(',o1_min_y >= hand_max_y,")", "or","\n",
            # "(",o1_box.get_center()[2] <=hand_min_y,"))","and","\n",
            # "((",hand_max_x < o1_max_x,")", "or","\n",
            # "(",hand_min_x > o1_min_x,")", "or","\n",
            # "(",hand.get_center()[0] <= o1_max_x,")", "or","\n",
            # "(",hand.get_center()[0] >= o1_min_x,"))","and","\n",
            # "((",hand_max_z < o1_max_z,")", "or","\n",
            # "(",hand_min_z > o1_min_z,")", "or","\n",
            # "(",hand.get_center()[1] <= o1_max_z,")", "or","\n",
            # "(",hand.get_center()[1] >= o1_min_z,"))","\n",
            # "\n\n",
            # "((",hand_min_y > o1_min_y,")", "or","\n",
            # "(",hand_max_y < o1_max_y,")", "or","\n",
            # "(",hand.get_center()[2] <= o1_max_y,")", "or","\n",
            # "(",hand.get_center()[2] >= o1_min_y,"))", "and","\n",
            # "((",hand_max_x < o1_min_x,")", "or","\n",
            # "(",hand_min_x > o1_max_x,"))","and","\n",
            # "((",hand_max_z < o1_min_z,")", "or","\n",
            # "(",hand_min_z > o1_max_z,")))")
        
        
    if(o2 == None):
        table[11][0] = 'U'
    elif(table[1][0] == b'A'):
        table[11][0] = 'A'
    elif(len(hand.points) > 0):
        if((hand_min_y > o2_min_y and hand_min_y < o2_max_y) and
        #    (hand_min_x > o2_min_x and hand_max_x < o2_max_x) and
           (hand_min_z > o2_min_z and hand_max_z < o2_max_z)):
            table[11][0] = 'In'     
        elif((hand_min_y < o2_min_y and hand_min_y > o2_max_y) and
           (hand_min_x < o2_min_x and hand_max_x > o2_max_x) and
           (hand_min_z < o2_min_z and hand_max_z > o2_max_z)):
            table[11][0] = 'Sa'      
        elif(((hand.get_center()[2] >= o2_max_y) or
            (hand.get_center()[2] <= o2_min_y))and
             ((hand_max_x < o2_max_x) or
              (hand_min_x > o2_min_x) or
              (hand.get_center()[0] <= o2_max_x) or
              (hand.get_center()[0] >= o2_min_x))and
             ((hand_max_z < o2_max_z) or
              (hand_min_z > o2_min_z) or
              (hand.get_center()[1] <= o2_max_z) or
              (hand.get_center()[1] >= o2_min_z))):
            if table[1][0] == b'T':
                table[11][0] = 'VArT'
            else:
                table[11][0] = 'VAr'
        elif(((hand.get_center()[2] <= o2_max_y) or
              (hand.get_center()[2] >= o2_min_y)) and
             ((hand_max_x < o2_max_x) or
              (hand_min_x > o2_min_x) or
              (hand_max_x > o2_max_x) or
              (hand_min_x < o2_min_x))and
             ((hand_max_z < o2_max_z) or
              (hand_min_z > o2_min_z) or
              (hand_max_z > o2_max_z) or
              (hand_min_z < o2_min_z))):
                if table[1][0] == b'T':
                    table[11][0] = 'HArT'
                else:
                    table[11][0] = 'HAr'
        else:
            print("Fehler: H,o2")
            # print("\n",
            # "((",hand_min_y >= o2_box.get_center()[2],")", "or","\n",
            # "(",hand_max_y <=o2_min_y,')', 'or',"\n",
            # '(',o2_min_y >= hand_max_y,")", "or","\n",
            # "(",o2_box.get_center()[2] <=hand_min_y,"))","and","\n",
            # "((",hand_max_x < o2_max_x,")", "or","\n",
            # "(",hand_min_x > o2_min_x,")", "or","\n",
            # "(",hand.get_center()[0] <= o2_max_x,")", "or","\n",
            # "(",hand.get_center()[0] >= o2_min_x,"))","and","\n",
            # "((",hand_max_z < o2_max_z,")", "or","\n",
            # "(",hand_min_z > o2_min_z,")", "or","\n",
            # "(",hand.get_center()[1] <= o2_max_z,")", "or","\n",
            # "(",hand.get_center()[1] >= o2_min_z,"))","\n",
            # "\n\n",
            # "((",hand_min_y > o2_min_y,")", "or","\n",
            # "(",hand_max_y < o2_max_y,")", "or","\n",
            # "(",hand.get_center()[2] <= o2_max_y,")", "or","\n",
            # "(",hand.get_center()[2] >= o2_min_y,"))", "and","\n",
            # "((",hand_max_x < o2_min_x,")", "or","\n",
            # "(",hand_min_x > o2_max_x,"))","and","\n",
            # "((",hand_max_z < o2_min_z,")", "or","\n",
            # "(",hand_min_z > o2_max_z,")))")

    #print(table[11][0])

    if(o3 == None):
        table[12][0] = 'U'
    elif(table[2][0] == b'A'):
        table[12][0] = 'A'
    elif(len(hand.points) > 0):
        if((hand_min_y > o3_min_y and hand_min_y < o3_max_y) and
        #    (hand_min_x > o3_min_x and hand_max_x < o3_max_x) and
           (hand_min_z > o3_min_z and hand_max_z < o3_max_z)):
            table[12][0] = 'In'   
        if((hand_min_y < o3_min_y and hand_min_y > o3_max_y) and
           (hand_min_x < o3_min_x and hand_max_x > o3_max_x) and
           (hand_min_z < o3_min_z and hand_max_z > o3_max_z)):
            table[12][0] = 'Sa' 
        elif(((hand.get_center()[2] >= o3_max_y) or
            (hand.get_center()[2] <= o3_min_y))and
             ((hand_max_x < o3_max_x) or
              (hand_min_x > o3_min_x) or
              (hand.get_center()[0] <= o3_max_x) or
              (hand.get_center()[0] >= o3_min_x))and
             ((hand_max_z < o3_max_z) or
              (hand_min_z > o3_min_z) or
              (hand.get_center()[1] <= o3_max_z) or
              (hand.get_center()[1] >= o3_min_z))):
            if table[2][0] == b'T':
                table[12][0] = 'VArT'
            else:
                table[12][0] = 'VAr'
        elif(((hand.get_center()[2] <= o3_max_y) or
              (hand.get_center()[2] >= o3_min_y)) and
             ((hand_max_x < o3_max_x) or
              (hand_min_x > o3_min_x) or
              (hand_max_x > o3_max_x) or
              (hand_min_x < o3_min_x))and
             ((hand_max_z < o3_max_z) or
              (hand_min_z > o3_min_z) or
              (hand_max_z > o3_max_z) or
              (hand_min_z < o3_min_z))):
                if table[2][0] == b'T':
                    table[12][0] = 'HArT'
                else:
                    table[12][0] = 'HAr'
        else:
            print("Fehler: H,o3")
        
    if(len(hand.points) > 0): 
        # if((hand_min_y > ground_min_y and hand_min_y < ground_max_y) and
        # #    (hand_min_x > ground_min_x and hand_max_x < ground_max_x) and
        #    (hand_min_z > ground_min_z and hand_max_z < ground_max_z)):
        #     table[13][0] = 'In' 
        # if((hand_min_y < ground_min_y and hand_min_y > ground_max_y) and
        #    (hand_min_x < ground_min_x and hand_max_x > ground_max_x) and
        #    (hand_min_z < ground_min_z and hand_max_z > ground_max_z)):
        #     table[13][0] = 'Sa'    
        if(((hand.get_center()[2] >= ground_max_y) or
            (hand.get_center()[2] <= ground_min_y))and
             ((hand_max_x < ground_max_x) or
              (hand_min_x > ground_min_x) or
              (hand.get_center()[0] <= ground_max_x) or
              (hand.get_center()[0] >= ground_min_x))and
             ((hand_max_z < ground_max_z) or
              (hand_min_z > ground_min_z) or
              (hand.get_center()[1] <= ground_max_z) or
              (hand.get_center()[1] >= ground_min_z))):
            if table[3][0] == b'T':
                table[13][0] = 'VArT'
            else:
                table[13][0] = 'VAr'
        elif(((hand.get_center()[2] <= ground_max_y) or
              (hand.get_center()[2] >= ground_min_y)) and
             ((hand_max_x < ground_max_x) or
              (hand_min_x > ground_min_x) or
              (hand_max_x > ground_max_x) or
              (hand_min_x < ground_min_x))and
             ((hand_max_z < ground_max_z) or
              (hand_min_z > ground_min_z) or
              (hand_max_z > ground_max_z) or
              (hand_min_z < ground_min_z))):
                if table[3][0] == b'T':
                    table[13][0] = 'HArT'
                else:
                    table[13][0] = 'HAr'
        else:
            print("Fehler: H,G")
    else:
        table[13][0] = 'U'

    # if(o1 != None and o2 != None):
    #     print("\n","(",o1_min_y < o2_min_y, "and", o2_max_y > o1_max_y,")", "or","\n",
    #         "(",o1_min_y > o2_min_y, "and", o2_max_y < o1_max_y,"))", "and","\n",
    #         "((",o2_min_x < o1_min_x, "and", o2_max_x < o1_max_x,")", "or","\n",
    #         "(",o2_min_x > o1_min_x, "and", o2_max_x < o1_max_x,")" ,"or","\n",
    #         "(",o2_min_x > o1_min_x, "and", o2_max_x > o1_max_x,")", "or","\n",
    #         "(",o2_min_x < o1_min_x ,"and" ,o2_max_x > o1_max_x,"))","and","\n",
    #         "((",o2_min_z < o1_min_z, "and" ,o2_max_z < o1_max_z,")","or","\n",
    #         "(",o2_min_z > o1_min_z ,"and", o2_max_z < o1_max_z,")", "or","\n",
    #         "(",o2_min_z > o1_min_z, "and", o2_max_z > o1_max_z,")", "or","\n",
    #         "(",o2_min_z < o1_min_z,"and", o2_max_z > o1_max_z,")))")

    if(o1 == None or o2 == None):
        table[14][0] = 'U'
    elif(table[4][0] == b'A'):
        table[14][0] = 'A'
    elif((o1_min_y > o2_min_y and o1_min_y < o2_max_y) and
        #    (o1_min_x > o2_min_x and o1_max_x < o2_max_x) and
           (o1_min_z > o2_min_z and o1_max_z < o2_max_z)):
            table[14][0] = 'In'
    elif((o1_min_y < o2_min_y and o1_min_y > o2_max_y) and
       (o1_min_x < o2_min_x and o1_max_x > o2_max_x) and
       (o1_min_z < o2_min_z and o1_max_z > o2_max_z)):
        table[14][0] = 'Sa'       
    elif(((o1.get_center()[2] >= o2_max_y) or
            (o1.get_center()[2] <= o2_min_y)) and
             ((o1_max_x < o2_max_x) or
              (o1_min_x > o2_min_x) or
              (o1.get_center()[0] <= o2_max_x) or
              (o1.get_center()[0] >= o2_min_x))and
             ((o1_max_z < o2_max_z) or
              (o1_min_z > o2_min_z) or
              (o1.get_center()[1] <= o2_max_z) or
              (o1.get_center()[1] >= o2_min_z))):
            if table[4][0] == b'T':
                table[14][0] = 'VArT'
            else:
                table[14][0] = 'VAr'
    elif((  (o1.get_center()[2] <= o2_max_y) or
            (o1.get_center()[2] >= o2_min_y)) and
            ((o1_max_x < o2_max_x) or
            (o1_min_x > o2_min_x) or
            (o1_max_x > o2_max_x) or
            (o1_min_x < o2_min_x))and
            ((o1_max_z < o2_max_z) or
            (o1_min_z > o2_min_z) or 
            (o1_max_z > o2_max_z) or
            (o1_min_z < o2_min_z))):
            if table[4][0] == b'T':
                table[14][0] = 'HArT'
            else:
                table[14][0] = 'HAr'
    else:
        print("Fehler: o1,o2")
    
 
    if(o1 == None or o3 == None):
        table[15][0] = 'U'
    elif(table[5][0] == b'A'):
        table[15][0] = 'A'
    elif((o1_min_y > o3_min_y and o1_min_y < o3_max_y) and
        #    (o1_min_x > o3_min_x and o1_max_x < o3_max_x) and
           (o1_min_z > o3_min_z and o1_max_z < o3_max_z)):
            table[15][0] = 'In' 
    elif((o1_min_y < o3_min_y and o1_min_y > o3_max_y) and
           (o1_min_x < o3_min_x and o1_max_x > o3_max_x) and
           (o1_min_z < o3_min_z and o1_max_z > o3_max_z)):
            table[15][0] = 'Sa'       
    elif(((o1.get_center()[2] >= o3_max_y) or
            (o1.get_center()[2] <= o3_min_y))and
             ((o1_max_x < o3_max_x) or
              (o1_min_x > o3_min_x) or
              (o1.get_center()[0] <= o3_max_x) or
              (o1.get_center()[0] >= o3_min_x))and
             ((o1_max_z < o3_max_z) or
              (o1_min_z > o3_min_z) or
              (o1.get_center()[1] <= o3_max_z) or
              (o1.get_center()[1] >= o3_min_z))):
            if table[5][0] == b'T':
                table[15][0] = 'VArT'
            else:
                table[15][0] = 'VAr'
    elif(((o1.get_center()[2] <= o3_max_y) or
            (o1.get_center()[2] >= o3_min_y)) and
            ((o1_max_x < o3_max_x) or
            (o1_min_x > o3_min_x) or
            (o1_max_x > o3_max_x) or
            (o1_min_x < o3_min_x))and
            ((o1_max_z < o3_max_z) or
            (o1_min_z > o3_min_z) or
            (o1_max_z > o3_max_z) or
            (o1_min_z < o3_min_z))):
            if table[5][0] == b'T':
                table[15][0] = 'HArT'
            else:
                table[15][0] = 'HAr'
    else:
        print("Fehler: o1,o3")

    if(o1 == None):
        table[16][0] = 'U'
    elif(table[6][0] == b'A'):
        table[16][0] = 'A'
    # elif((o1_min_y > ground_min_y and o1_min_y < ground_max_y) and
    #     #    (o1_min_x > ground_min_x and o1_max_x < ground_max_x) and
    #        (o1_min_z > ground_min_z and o1_max_z < ground_max_z)):
    #         table[16][0] = 'In'
    # elif((o1_min_y < ground_min_y and o1_min_y > ground_max_y) and
    #        (o1_min_x < ground_min_x and o1_max_x > ground_max_x) and
    #        (o1_min_z < ground_min_z and o1_max_z > ground_max_z)):
    #         table[16][0] = 'Sa'        
    elif(((o1.get_center()[2] >= ground_max_y) or
            (o1.get_center()[2] <= ground_min_y))and
             ((o1_max_x < ground_max_x) or
              (o1_min_x > ground_min_x) or
              (o1.get_center()[0] <= ground_max_x) or
              (o1.get_center()[0] >= ground_min_x))and
             ((o1_max_z < ground_max_z) or
              (o1_min_z > ground_min_z) or
              (o1.get_center()[1] <= ground_max_z) or
              (o1.get_center()[1] >= ground_min_z))):
            if table[6][0] == b'T':
                table[16][0] = 'VArT'
            else:
                table[16][0] = 'VAr'
    elif((  (o1.get_center()[2] <= ground_max_y) or
            (o1.get_center()[2] >= ground_min_y)) and
            ((o1_max_x < ground_max_x) or
            (o1_min_x > ground_min_x) or
            (o1_max_x > ground_max_x) or
            (o1_min_x < ground_min_x))and
            ((o1_max_z < ground_max_z) or
            (o1_min_z > ground_min_z) or
            (o1_max_z > ground_max_z) or
            (o1_min_z < ground_min_z))):
            if table[6][0] == b'T':
                table[16][0] = 'HArT'
            else:
                table[16][0] = 'HAr'
    else:
        print("Fehler: o1,G")
        
    if(o2 == None or o3 == None):
        table[17][0] = 'U'
    elif(table[7][0] == b'A'):
        table[17][0] = 'A'
    elif((o2_min_y > o3_min_y and o2_min_y < o3_max_y) and
        #    (o2_min_x > o3_min_x and o2_max_x < o3_max_x) and
           (o2_min_z > o3_min_z and o2_max_z < o3_max_z)):
            table[17][0] = 'In'
    elif((o2_min_y < o3_min_y and o2_min_y > o3_max_y) and
           (o2_min_x < o3_min_x and o2_max_x > o3_max_x) and
           (o2_min_z < o3_min_z and o2_max_z > o3_max_z)):
            table[17][0] = 'Sa'        
    elif(((o2.get_center()[2] >= o3_max_y) or
            (o2.get_center()[2] <= o3_min_y))and
             ((o2_max_x < o3_max_x) or
              (o2_min_x > o3_min_x) or
              (o2.get_center()[0] <= o3_max_x) or
              (o2.get_center()[0] >= o3_min_x))and
             ((o2_max_z < o3_max_z) or
              (o2_min_z > o3_min_z) or
              (o2.get_center()[1] <= o3_max_z) or
              (o2.get_center()[1] >= o3_min_z))):
            if table[7][0] == b'T':
                table[17][0] = 'VArT'
            else:
                table[17][0] = 'VAr'
    elif((  (o2.get_center()[2] <= o3_max_y) or
            (o2.get_center()[2] >= o3_min_y)) and
            ((o2_max_x < o3_max_x) or
            (o2_min_x > o3_min_x) or
            (o2_max_x > o3_max_x) or
            (o2_min_x < o3_min_x))and
            ((o2_max_z < o3_max_z) or
            (o2_min_z > o3_min_z) or 
            (o2_max_z > o3_max_z) or
            (o2_min_z < o3_min_z))):
            if table[7][0] == b'T':
                table[17][0] = 'HArT'
            else:
                table[17][0] = 'HAr'
    else:
        print("Fehler: o2,o3")
    
    if(o2 == None):
        table[18][0] = 'U'
    elif(table[8][0] == b'A'):
        table[18][0] = 'A'
    # elif((o2_min_y > ground_min_y and o2_min_y < ground_max_y) and
    #     #    (o2_min_x > ground_min_x and o2_max_x < ground_max_x) and
    #        (o2_min_z > ground_min_z and o2_max_z < ground_max_z)):
    #         table[18][0] = 'In'
    # elif((o2_min_y < ground_min_y and o2_min_y > ground_max_y) and
    #        (o2_min_x < ground_min_x and o2_max_x > ground_max_x) and
    #        (o2_min_z < ground_min_z and o2_max_z > ground_max_z)):
    #         table[18][0] = 'Sa'        
    elif(((o2.get_center()[2] >= ground_max_y) or
            (o2.get_center()[2] <= ground_min_y))and
             ((o2_max_x < ground_max_x) or
              (o2_min_x > ground_min_x) or
              (o2.get_center()[0] <= ground_max_x) or
              (o2.get_center()[0] >= ground_min_x))and
             ((o2_max_z < ground_max_z) or
              (o2_min_z > ground_min_z) or
              (o2.get_center()[1] <= ground_max_z) or
              (o2.get_center()[1] >= ground_min_z))):
            if table[8][0] == b'T':
                table[18][0] = 'VArT'
            else:
                table[18][0] = 'VAr'
    elif((  (o2.get_center()[2] <= ground_max_y) or
            (o2.get_center()[2] >= ground_min_y)) and
            ((o2_max_x < ground_max_x) or
            (o2_min_x > ground_min_x) or
            (o2_max_x > ground_max_x) or
            (o2_min_x < ground_min_x))and
            ((o2_max_z < ground_max_z) or
            (o2_min_z > ground_min_z) or
            (o2_max_z > ground_max_z) or
            (o2_min_z < ground_min_z))):
            if table[8][0] == b'T':
                table[18][0] = 'HArT'
            else:
                table[18][0] = 'HAr'
    else:
        print("Fehler: o2,G")
            
    if(o3 == None):
        table[19][0] = 'U'
    elif(table[9][0] == b'A'):
        table[19][0] = 'A'
    # elif((o3_min_y > ground_min_y and o3_min_y < ground_max_y) and
    #     #    (o3_min_x > ground_min_x and o3_max_x < ground_max_x) and
    #        (o3_min_z > ground_min_z and o3_max_z < ground_max_z)):
    #         table[19][0] = 'In'
    # elif((o3_min_y < ground_min_y and o3_min_y > ground_max_y) and
    #        (o3_min_x < ground_min_x and o3_max_x > ground_max_x) and
    #        (o3_min_z < ground_min_z and o3_max_z > ground_max_z)):
    #         table[19][0] = 'Sa'        
    elif(((o3.get_center()[2] >= ground_max_y) or
            (o3.get_center()[2] <= ground_min_y))and
             ((o3_max_x < ground_max_x) or
              (o3_min_x > ground_min_x) or
              (o3.get_center()[0] <= ground_max_x) or
              (o3.get_center()[0] >= ground_min_x))and
             ((o3_max_z < ground_max_z) or
              (o3_min_z > ground_min_z) or
              (o3.get_center()[1] <= ground_max_z) or
              (o3.get_center()[1] >= ground_min_z))):
            if table[9][0] == b'T':
                table[19][0] = 'VArT'
            else:
                table[19][0] = 'VAr'
    elif((  (o3.get_center()[2] <= ground_max_y) or
            (o3.get_center()[2] >= ground_min_y)) and
            ((o3_max_x < ground_max_x) or
            (o3_min_x > ground_min_x) or
            (o3_max_x > ground_max_x) or
            (o3_min_x < ground_min_x))and
            ((o3_max_z < ground_max_z) or
            (o3_min_z > ground_min_z) or
            (o3_max_z > ground_max_z) or
            (o3_min_z < ground_min_z))):
            if table[9][0] == b'T':
                table[19][0] = 'HArT'
            else:
                table[19][0] = 'HAr'
    else:
        print("Fehler: o3,G")

def _fillDSR(hand, ground, previous_array, thresh, table):
    '''
    Creates the DSR with pre-defined point clouds of hand and ground. It also need the point cloud from the previous frame.
    
    Parameters:
        * hand: point cloud of the hand
        * ground: point cloud of the ground
        * previous_array: array that contains point clouds from previous frame of hand, ground, o1, o2 and o3
        * thresh: threshold distance of touching
        * table: chararray table   
    '''
    #     DSR
    #     H, 1
    #     H, 2
    #     H, 3
    #     H, G
    #     1, 2
    #     1, 3
    #     1, G
    #     2, 3
    #     2, G
    #     3, G 

    #get global objects 1, 2 and 3
    global o1,o2,o3
    
    #get varaibles from previous frame
    phand, pground, po1, po2, po3 = previous_array
    
    #creates AABB around object1 if o1 and po1 are defined and if TNR relation is not abscent
    if(o1 != None and po1 != None and table[0][0] != b'A'):
        o1_box = o1.get_axis_aligned_bounding_box()
        po1_box = po1.get_axis_aligned_bounding_box()
        
    #creates AABB around object2 if o2 and po2 are defined and if TNR relation is not abscent    
    if(o2 != None and po2 != None and table[1][0] != b'A'):
        o2_box = o2.get_axis_aligned_bounding_box()
        po2_box = po2.get_axis_aligned_bounding_box()
        
    #creates AABB around object1 if o3 and po3 are defined and if TNR relation is not abscent
    if(o3 != None and po3 != None and table[2][0] != b'A'):
        o3_box = o3.get_axis_aligned_bounding_box()
        po3_box = po3.get_axis_aligned_bounding_box()
        
    #creates AABB around hand if point cloud is in the frame
    if(len(hand.points) > 0):
        hand_box = hand.get_axis_aligned_bounding_box()
        phand_box = phand.get_axis_aligned_bounding_box()
    
    #creates AABB around ground
    ground_box = ground.get_axis_aligned_bounding_box()
    pground_box = pground.get_axis_aligned_bounding_box()
    
    multi = 1.5
    threshold = 0.1
    
    if(o1 == None or po1 == None):
        table[20][0] = 'U'
    elif(table[0][0] == b'A'):
        table[20][0] = 'A'
    elif(len(hand.points) > 0):

        if (table[0][0] == b'N' and abs(np.min(hand.compute_point_cloud_distance(o1)) - np.min(phand.compute_point_cloud_distance(po1))) < threshold):
            table[20][0] = 'S'
        elif ((table[0][0] == b'T' and (hand.get_center() != phand.get_center()).all() and (o1.get_center() != po1.get_center()).all()) 
              or (table[0][0] == b'T' and xor((hand.get_center() != phand.get_center()).all(), (o1.get_center() != po1.get_center()).all()))):
            table[20][0] = 'MT' 
        elif (table[0][0] == b'T' and not((hand.get_center() != phand.get_center()).all()) and not((o1.get_center() != po1.get_center())).all()):
            table[20][0] = 'HT' 
        elif (abs(np.min(hand.compute_point_cloud_distance(o1)) - np.min(phand.compute_point_cloud_distance(po1))) > threshold):
            table[20][0] = 'MA'
        elif (abs(np.min(hand.compute_point_cloud_distance(o1)) - np.min(phand.compute_point_cloud_distance(po1))) < threshold):
            table[20][0] = 'GC'
        else:
            table[20][0] = 'Q'
        
    
    if(o2 == None or po2 == None):
        table[21][0] = 'U'
    elif(table[1][0] == b'A'):
        table[21][0] = 'A'
    elif(len(hand.points) > 0):
        if (table[1][0] == b'N' and abs(np.min(hand.compute_point_cloud_distance(o2)) - np.min(phand.compute_point_cloud_distance(po2))) < threshold):
            table[21][0] = 'S'
        elif (table[1][0] == b'T' and (hand.get_center() != phand.get_center()).all() and (o2.get_center() != po2.get_center()).all()
              or (table[1][0] == b'T' and xor((hand.get_center() != phand.get_center()).all(), (o2.get_center() != po2.get_center()).all()))):
            table[21][0] = 'MT' 
        elif (table[1][0] == b'T' and not((hand.get_center() != phand.get_center()).all()) and not((o2.get_center() != po2.get_center()).all())):
            table[21][0] = 'HT' 
        elif (abs(np.min(hand.compute_point_cloud_distance(o2)) - np.min(phand.compute_point_cloud_distance(po2))) > threshold):
            table[21][0] = 'MA'
        elif (abs(np.min(hand.compute_point_cloud_distance(o2)) - np.min(phand.compute_point_cloud_distance(po2))) < threshold):
            table[21][0] = 'GC'
        else:
            table[21][0] = 'Q'
        
       
    if(o3 == None or po3 == None):
        table[22][0] = 'U'
    elif(table[2][0] == b'A'):
        table[22][0] = 'A'
    elif(len(hand.points) > 0):
        if (table[2][0] == b'N' and abs(np.min(hand.compute_point_cloud_distance(o3)) - np.min(phand.compute_point_cloud_distance(po3))) < threshold):
            table[22][0] = 'S'
        elif (table[2][0] == b'T' and (hand.get_center() != phand.get_center()).all() and (o3.get_center() != po3.get_center()).all()
              or (table[2][0] == b'T' and xor((hand.get_center() != phand.get_center()).all(), (o3.get_center() != po3.get_center()).all()))):
            table[22][0] = 'MT' 
        elif (table[2][0] == b'T' and not((hand.get_center() != phand.get_center()).all()) and not((o3.get_center() != po3.get_center()).all())):
            table[22][0] = 'HT' 
        elif (abs(np.min(hand.compute_point_cloud_distance(o3)) - np.min(phand.compute_point_cloud_distance(po3))) > threshold):
            table[22][0] = 'MA'
        elif (abs(np.min(hand.compute_point_cloud_distance(o3)) - np.min(phand.compute_point_cloud_distance(po3))) < threshold):
            table[22][0] = 'GC'
        else:
            table[22][0] = 'Q'
         

    if(len(hand.points) > 0):
        if (table[3][0] == b'N' and abs(np.min(hand.compute_point_cloud_distance(ground)) - np.min(phand.compute_point_cloud_distance(pground))) < threshold):
            table[23][0] = 'S'
        elif (table[3][0] == b'T' and (hand.get_center() != phand.get_center()).all() and (ground.get_center() != pground.get_center()).all()
              or (table[3][0] == b'T' and xor((hand.get_center() != phand.get_center()).all(), (ground.get_center() != pground.get_center()).all()))):
            table[23][0] = 'MT' 
        elif (table[3][0] == b'T' and not((hand.get_center() != phand.get_center()).all()) and not((ground.get_center() != pground.get_center()).all())):
            table[23][0] = 'HT' 
        elif (abs(np.min(hand.compute_point_cloud_distance(ground)) - np.min(phand.compute_point_cloud_distance(pground))) > threshold):
            table[23][0] = 'MA'
        elif (abs(np.min(hand.compute_point_cloud_distance(ground)) - np.min(phand.compute_point_cloud_distance(pground))) < threshold):
            table[23][0] = 'GC'
        else:
            table[23][0] = 'Q'
        
        
    if(o1 == None or o2 == None or po1 == None or po2 == None):
        table[24][0] = 'U'
    elif(table[4][0] == b'A'):
        table[24][0] = 'A'
    elif (table[4][0] == b'N' and abs(np.min(o1.compute_point_cloud_distance(o2)) - np.min(po1.compute_point_cloud_distance(po2))) < threshold):
        table[24][0] = 'S'
    elif (table[4][0] == b'T' and (o1.get_center() != po1.get_center()).all() and (o2.get_center() != po2.get_center()).all()
          or (table[4][0] == b'T' and xor((o1.get_center() != po1.get_center()).all(), (o2.get_center() != po2.get_center()).all()))):
        table[24][0] = 'MT' 
    elif (table[4][0] == b'T' and not((o1.get_center() != po1.get_center()).all()) and not((o2.get_center() != po2.get_center()).all())):
        table[24][0] = 'HT' 
    elif (abs(np.min(o1.compute_point_cloud_distance(o2)) - np.min(po1.compute_point_cloud_distance(po2))) > threshold):
        table[24][0] = 'MA'
    elif (abs(np.min(o1.compute_point_cloud_distance(o2)) - np.min(po1.compute_point_cloud_distance(po2))) < threshold):
        table[24][0] = 'GC'
    else:
        table[24][0] = 'Q'
    
    
    if(o1 == None or o3 == None or po1 == None or po3 == None):
        table[25][0] = 'U'
    elif(table[5][0] == b'A'):
        table[25][0] = 'A'
    elif (table[5][0] == b'N' and abs(np.min(o1.compute_point_cloud_distance(o3)) - np.min(po1.compute_point_cloud_distance(po3))) < threshold):
        table[25][0] = 'S'
    elif (table[5][0] == b'T' and (o1.get_center() != po1.get_center()).all() and (o3.get_center() != po3.get_center()).all()
          or (table[5][0] == b'T' and xor((o1.get_center() != po1.get_center()).all(), (o3.get_center() != po3.get_center()).all()))):
        table[25][0] = 'MT' 
    elif (table[5][0] == b'T' and not((o1.get_center() != po1.get_center()).all()) and not((o3.get_center() != po3.get_center()).all())):
        table[25][0] = 'HT'
    elif (abs(np.min(o1.compute_point_cloud_distance(o3)) - np.min(po1.compute_point_cloud_distance(po3))) > threshold):
        table[25][0] = 'MA'
    elif (abs(np.min(o1.compute_point_cloud_distance(o3)) - np.min(po1.compute_point_cloud_distance(po3))) < threshold):
        table[25][0] = 'GC'
    else:
        table[25][0] = 'Q'
    
        
    if(o1 == None or po1 == None):
        table[26][0] = 'U'
    elif(table[6][0] == b'A'):
        table[26][0] = 'A'
    elif (table[6][0] == b'N' and abs(np.min(o1.compute_point_cloud_distance(ground)) - np.min(po1.compute_point_cloud_distance(pground))) < threshold):
        table[26][0] = 'S'
    elif (table[6][0] == b'T' and (o1.get_center() != po1.get_center()).all() and (ground.get_center() != pground.get_center()).all()
          or (table[6][0] == b'T' and xor((o1.get_center() != po1.get_center()).all(), (ground.get_center() != pground.get_center()).all()))):
        table[26][0] = 'MT' 
    elif (table[6][0] == b'T' and not((o1.get_center() != po1.get_center()).all()) and not((ground.get_center() != pground.get_center()).all())):
        table[26][0] = 'HT'
    elif (abs(np.min(o1.compute_point_cloud_distance(ground)) - np.min(po1.compute_point_cloud_distance(pground))) > threshold):
        table[26][0] = 'MA'
    elif (abs(np.min(o1.compute_point_cloud_distance(ground)) - np.min(po1.compute_point_cloud_distance(pground))) < threshold):
        table[26][0] = 'GC'
    else:
        table[26][0] = 'Q'
    
      
    if(o2 == None or o3 == None or po2 == None or po3 == None):
        table[27][0] = 'U'
    elif(table[7][0] == b'A'):
        table[27][0] = 'A'
    elif (table[7][0] == b'N' and abs(np.min(o2.compute_point_cloud_distance(o3)) - np.min(po2.compute_point_cloud_distance(po3))) < threshold):
        table[27][0] = 'S'
    elif (table[7][0] == b'T' and (o2.get_center() != po2.get_center()).all() and (o3.get_center() != po3.get_center()).all()
          or (table[7][0] == b'T' and xor((o2.get_center() != po2.get_center()).all(), (o3.get_center() != po3.get_center()).all()))):
        table[27][0] = 'MT' 
    elif (table[7][0] == b'T' and not((o2.get_center() != po2.get_center()).all()) and not((o3.get_center() != po3.get_center()).all())):
        table[27][0] = 'HT' 
    elif (abs(np.min(o2.compute_point_cloud_distance(o3)) - np.min(po2.compute_point_cloud_distance(po3))) > threshold):
        table[27][0] = 'MA'
    elif (abs(np.min(o2.compute_point_cloud_distance(o3)) - np.min(po2.compute_point_cloud_distance(po3))) < threshold):
        table[27][0] = 'GC'
    else:
        table[27][0] = 'Q'
    
    # if o2 != None and po2 != None:
    #     print(len(o2.points), len(po2.points))
    
    if(o2 == None or po2 == None):
        table[28][0] = 'U'
    elif(table[8][0] == b'A'):
        table[28][0] = 'A'
    elif (table[8][0] == b'N' and abs(np.min(o2.compute_point_cloud_distance(ground)) - np.min(po2.compute_point_cloud_distance(pground))) < threshold):
        table[28][0] = 'S'
    elif (table[8][0] == b'T' and (o2.get_center() != po2.get_center()).all() and (ground.get_center() != pground.get_center()).all()
          or (table[8][0] == b'T' and xor((o2.get_center() != po2.get_center()).all(), (ground.get_center() != pground.get_center()).all()))):
        table[28][0] = 'MT' 
    elif (table[8][0] == b'T' and not((o2.get_center() != po2.get_center()).all()) and not((ground.get_center() != pground.get_center()).all())):
        table[28][0] = 'HT' 
    elif (abs(np.min(o2.compute_point_cloud_distance(ground)) - np.min(po2.compute_point_cloud_distance(pground))) > threshold):
        table[28][0] = 'MA'
    elif (abs(np.min(o2.compute_point_cloud_distance(ground)) - np.min(po2.compute_point_cloud_distance(pground))) < threshold):
        table[28][0] = 'GC'
    else:
        table[28][0] = 'Q'
        
    if(o3 == None or po3 == None):
        table[29][0] = 'U'
    elif(table[9][0] == b'A'):
        table[29][0] = 'A'
    elif (table[9][0] == b'N' and abs(np.min(o3.compute_point_cloud_distance(ground)) - np.min(po3.compute_point_cloud_distance(pground))) < threshold):
        table[29][0] = 'S'
    elif (table[9][0] == b'T' and (o3.get_center() != po3.get_center()).all() and (ground.get_center() != pground.get_center()).all()
          or (table[9][0] == b'T' and xor((o3.get_center() != po3.get_center()).all(), (ground.get_center() != pground.get_center()).all()))):
        table[29][0] = 'MT' 
    elif (table[9][0] == b'T' and not((o3.get_center() != po3.get_center()).all()) and not((ground.get_center() != pground.get_center()).all())):
        table[29][0] = 'HT' 
    elif (abs(np.min(o3.compute_point_cloud_distance(ground)) - np.min(po3.compute_point_cloud_distance(pground))) > threshold):
        table[29][0] = 'MA'
    elif (abs(np.min(o3.compute_point_cloud_distance(ground)) - np.min(po3.compute_point_cloud_distance(pground))) < threshold):
        table[29][0] = 'GC'
    else:
        table[29][0] = 'Q'


# def _fillDSR_new(hand, ground, previous_array, thresh, table):
#     '''
#     Creates the DSR with pre-defined point clouds of hand and ground. It also need the point cloud from the previous frame.
    
#     Parameters:
#         * hand: point cloud of the hand
#         * ground: point cloud of the ground
#         * previous_array: array that contains point clouds from previous frame of hand, ground, o1, o2 and o3
#         * thresh: threshold distance of touching
#         * table: chararray table   
#     '''
#     #     DSR
#     #     H, 1
#     #     H, 2
#     #     H, 3
#     #     H, G
#     #     1, 2
#     #     1, 3
#     #     1, G
#     #     2, 3
#     #     2, G
#     #     3, G 

#     #get global objects 1, 2 and 3
#     global o1,o2,o3
    
#     #get varaibles from previous frame
#     phand, pground, po1, po2, po3 = previous_array
    
#     #creates AABB around object1 if o1 and po1 are defined and if TNR relation is not abscent
#     if(o1 != None and po1 != None and table[0][0] != b'A'):
#         o1_box = o1.get_axis_aligned_bounding_box()
#         po1_box = po1.get_axis_aligned_bounding_box()
        
#     #creates AABB around object2 if o2 and po2 are defined and if TNR relation is not abscent    
#     if(o2 != None and po2 != None and table[1][0] != b'A'):
#         if len(o2.points) != 0:
#             o2_box = o2.get_axis_aligned_bounding_box()
#             po2_box = po2.get_axis_aligned_bounding_box()
        
#     #creates AABB around object1 if o3 and po3 are defined and if TNR relation is not abscent
#     if(o3 != None and po3 != None and table[2][0] != b'A'):
#         if len(o3.points) != 0:
#             o3_box = o3.get_axis_aligned_bounding_box()
#             po3_box = po3.get_axis_aligned_bounding_box()
        
#     #creates AABB around hand if point cloud is in the frame
#     if(len(hand.points) > 0):
#         hand_box = hand.get_axis_aligned_bounding_box()
#         phand_box = phand.get_axis_aligned_bounding_box()
    
#     #creates AABB around ground
#     ground_box = ground.get_axis_aligned_bounding_box()
#     pground_box = pground.get_axis_aligned_bounding_box()
        
    
#     #multi = 1.5
#     threshold = thresh
    
#     P1 = [table[0][0] == b'T', table[1][0] == b'T', table[2][0] == b'T', table[3][0] == b'T', table[4][0] == b'T', table[5][0] == b'T', table[6][0] == b'T', table[7][0] == b'T', table[8][0] == b'T', table[9][0] == b'T']
#     P2 = [table[0][0] == b'N', table[1][0] == b'N', table[2][0] == b'N', table[3][0] == b'N', table[4][0] == b'N', table[5][0] == b'N', table[6][0] == b'N', table[7][0] == b'N', table[8][0] == b'N', table[9][0] == b'N']
#     # if o1 != None and po1 != None and o2 != None and po2 != None and o3 != None and po3 != None:
#     #     if len(o1.points) != 0 and len(o2.points) != 0 and len(o3.points) != 0:
#     #         P3 = [all(ground_box.get_center() != pground_box.get_center()), 
#     #               all(hand_box.get_center() != phand_box.get_center()), 
#     #               all(o1_box.get_center() != po1_box.get_center()), 
#     #               all(o2_box.get_center() != po2_box.get_center()), 
#     #               all(o3_box.get_center() != po3_box.get_center())]
#     #     elif len(o1.points) != 0 and len(o2.points) != 0 :
#     #         P3 = [all(ground_box.get_center() != pground_box.get_center()), all(hand_box.get_center() != phand_box.get_center()), all(o1_box.get_center() != po1_box.get_center()), all(o2_box.get_center() != po2_box.get_center()), 0]
#     # elif o1 != None and po1 != None and o2 != None and po2 != None:
#     #     if len(o1.points) != 0 and len(o2.points) != 0:
#     #         P3 = [all(ground_box.get_center() != pground_box.get_center()), all(hand_box.get_center() != phand_box.get_center()), all(o1_box.get_center() != po1_box.get_center()), all(o2_box.get_center() != po2_box.get_center()), 0]
#     # elif o1 != None and po1 != None and o3 != None and po3 != None:
#     #     if len(o1.points) != 0 and len(o3.points) != 0:
#     #         P3 = [all(ground_box.get_center() != pground_box.get_center()), all(hand_box.get_center() != phand_box.get_center()), all(o1_box.get_center() != po1_box.get_center()), 0, all(o3_box.get_center() != po3_box.get_center())]
#     # elif o2 != None and po2 != None and o3 != None and po3 != None:
#     #     if len(o2.points) != 0 and  len(o3.points) != 0:
#     #         P3 = [all(ground_box.get_center() != pground_box.get_center()), all(hand_box.get_center() != phand_box.get_center()), 0, all(o2_box.get_center() != po2_box.get_center()), all(o3_box.get_center() != po3_box.get_center())]
#     # elif o1 != None and po1 != None:
#     #     if len(o1.points) != 0:
#     #         P3 = [all(ground_box.get_center() != pground_box.get_center()), all(hand_box.get_center() != phand_box.get_center()),all(o1_box.get_center() != po1_box.get_center()), 0, 0]
#     # elif o2 != None and po2 != None:
#     #     if len(o2.points) != 0:
#     #         P3 = [all(ground_box.get_center() != pground_box.get_center()), all(hand_box.get_center() != phand_box.get_center()), 0, all(o2_box.get_center() != po2_box.get_center()), 0]
#     # elif o3 != None and po3 != None:
#     #     if len(o3.points) != 0:
#     #         P3 = [all(ground_box.get_center() != pground_box.get_center()), all(hand_box.get_center() != phand_box.get_center()), 0, 0, all(o3_box.get_center() != po3_box.get_center())]
#     # else:
#     #     P3 = [all(ground_box.get_center() != pground_box.get_center()), all(hand_box.get_center() != phand_box.get_center()), 0, 0, 0]
 

#     # if o1 != None and po1 != None and o2 != None and po2 != None and o3 != None and po3 != None:
#     #     print("1")
#     #     if len(o1.points) != 0 and len(o2.points) != 0 and len(o3.points) != 0:
#     #         P5 = [(_distance(hand_box.get_center(), o1_box.get_center()) - _distance(phand_box.get_center(), po1_box.get_center())) < threshold,
#     #             (_distance(hand_box.get_center(), o2_box.get_center()) - _distance(phand_box.get_center(), po2_box.get_center())) < threshold,
#     #             (_distance(hand_box.get_center(),o3_box.get_center()) - _distance(phand_box.get_center(), po3_box.get_center())) < threshold,
#     #             (_distance(hand_box.get_center(), ground_box.get_center()) - _distance(phand_box.get_center(), pground_box.get_center())) < threshold,
#     #             (_distance(o1_box.get_center(), o2_box.get_center()) - _distance(po1_box.get_center(), po2_box.get_center())) < threshold,
#     #             (_distance(o1_box.get_center(), o3_box.get_center()) - _distance(po1_box.get_center(), po3_box.get_center())) < threshold,
#     #             (_distance(o1_box.get_center(), ground_box.get_center()) - _distance(po1_box.get_center(), pground_box.get_center())) < threshold,
#     #             (_distance(o2_box.get_center(), o3_box.get_center()) - _distance(po2_box.get_center(), po3_box.get_center())) < threshold,
#     #             (_distance(o2_box.get_center(), ground_box.get_center()) - _distance(po2_box.get_center(), pground_box.get_center())) < threshold,
#     #             (_distance(o3_box.get_center(), ground_box.get_center()) - _distance(po3_box.get_center(), pground_box.get_center())) < threshold]
#     #     elif len(o1.points) != 0 and len(o2.points) != 0 :
#     #         P5 = [(_distance(hand_box.get_center(), o1_box.get_center()) - _distance(phand_box.get_center(), po1_box.get_center())) < threshold,
#     #             (_distance(hand_box.get_center(), o2_box.get_center()) - _distance(phand_box.get_center(), po2_box.get_center())) < threshold,
#     #             0,
#     #             (_distance(hand_box.get_center(), ground_box.get_center()) - _distance(phand_box.get_center(), pground_box.get_center())) < threshold,
#     #             (_distance(o1_box.get_center(), o2_box.get_center()) - _distance(po1_box.get_center(), po2_box.get_center())) < threshold,
#     #             0,
#     #             (_distance(o1_box.get_center(), ground_box.get_center()) - _distance(po1_box.get_center(), pground_box.get_center())) < threshold,
#     #             0,
#     #             (_distance(o2_box.get_center(), ground_box.get_center()) - _distance(po2_box.get_center(), pground_box.get_center())) < threshold,
#     #             0]
#     # elif o1 != None and po1 != None and o2 != None and po2 != None:
#     #     print("2")
#     #     if len(o1.points) != 0 and len(o2.points) != 0:
#     #         P5 = [(_distance(hand_box.get_center(), o1_box.get_center()) - _distance(phand_box.get_center(), po1_box.get_center())) < threshold,
#     #         (_distance(hand_box.get_center(), o2_box.get_center()) - _distance(phand_box.get_center(), po2_box.get_center())) < threshold,
#     #         0,
#     #         (_distance(hand_box.get_center(), ground_box.get_center()) - _distance(phand_box.get_center(), pground_box.get_center())) < threshold,
#     #         (_distance(o1_box.get_center(), o2_box.get_center()) - _distance(po1_box.get_center(), po2_box.get_center())) < threshold,
#     #         0,
#     #         (_distance(o1_box.get_center(), ground_box.get_center()) - _distance(po1_box.get_center(), pground_box.get_center())) < threshold,
#     #         0,
#     #         (_distance(o2_box.get_center(), ground_box.get_center()) - _distance(po2_box.get_center(), pground_box.get_center())) < threshold,
#     #         0]
#     # elif o1 != None and po1 != None and o3 != None and po3 != None:
#     #     print("3")
#     #     if len(o1.points) != 0 and len(o3.points) != 0:
#     #         P5 = [(_distance(hand_box.get_center(), o1_box.get_center()) - _distance(phand_box.get_center(), po1_box.get_center())) < threshold,
#     #         0,
#     #         (_distance(hand_box.get_center(), o3_box.get_center()) - _distance(phand_box.get_center(), po3_box.get_center())) < threshold,
#     #         (_distance(hand_box.get_center(), ground_box.get_center()) - _distance(phand_box.get_center(), pground_box.get_center())) < threshold,
#     #         0,
#     #         (_distance(o1_box.get_center(), o3_box.get_center()) - _distance(po1_box.get_center(), po3_box.get_center())) < threshold,
#     #         (_distance(o1_box.get_center(), ground_box.get_center()) - _distance(po1_box.get_center(), pground_box.get_center())) < threshold,
#     #         0,
#     #         0,
#     #         (_distance(o3_box.get_center(), ground_box.get_center()) - _distance(po3_box.get_center(), pground_box.get_center())) < threshold]
#     # elif o2 != None and po2 != None and o3 != None and po3 != None:
#     #     print("4")
#     #     if len(o2.points) != 0 and len(o3.points) != 0:
#     #         P5 = [0,
#     #         (_distance(hand_box.get_center(), o2_box.get_center()) - _distance(phand_box.get_center(), po2_box.get_center())) < threshold,
#     #         (_distance(hand_box.get_center(), o3_box.get_center()) - _distance(phand_box.get_center(), po3_box.get_center())) < threshold,
#     #         (_distance(hand_box.get_center(), ground_box.get_center()) - _distance(phand_box.get_center(), pground_box.get_center())) < threshold,
#     #         0,
#     #         0,
#     #         0,
#     #         (_distance(o2_box.get_center(), o3_box.get_center()) - _distance(po2_box.get_center(), po3_box.get_center())) < threshold,
#     #         (_distance(o2_box.get_center(), ground_box.get_center()) - _distance(po2_box.get_center(), pground_box.get_center())) < threshold,
#     #         (_distance(o3_box.get_center(), ground_box.get_center()) - _distance(po3_box.get_center(), pground_box.get_center())) < threshold]
#     # elif o1 != None and po1 != None:
#     #     print("5")
#     #     if len(o1.points) != 0:
#     #         P5 = [(_distance(hand_box.get_center(), o1_box.get_center()) - _distance(phand_box.get_center(), po1_box.get_center())) < threshold,
#     #         0,
#     #         0,
#     #         (_distance(hand_box.get_center(), ground_box.get_center()) - _distance(phand_box.get_center(), pground_box.get_center())) < threshold,
#     #         0,
#     #         0,
#     #         (_distance(o1_box.get_center(), ground_box.get_center()) - _distance(po1_box.get_center(), pground_box.get_center())) < threshold,
#     #         0,
#     #         0,
#     #         0]
#     # elif o2 != None and po2 != None:
#     #     print("6")
#     #     if len(o2.points) != 0:
#     #         P5 = [0,
#     #         (_distance(hand_box.get_center(), o2_box.get_center()) - _distance(phand_box.get_center(), po2_box.get_center())) < threshold,
#     #         0,
#     #         (_distance(hand_box.get_center(), ground_box.get_center()) - _distance(phand_box.get_center(), pground_box.get_center())) < threshold,
#     #         0,
#     #         0,
#     #         0,
#     #         0,
#     #         (_distance(o2_box.get_center(), ground_box.get_center()) - _distance(po2_box.get_center(), pground_box.get_center())) < threshold,
#     #         0]
#     # elif o3 != None and po3 != None:
#     #     print("7")
#     #     if len(o2.points) != 0:
#     #         P5 = [0,
#     #         0,
#     #         (_distance(hand_box.get_center(), o3_box.get_center()) - _distance(phand_box.get_center(), po3_box.get_center())) < threshold,
#     #         (_distance(hand_box.get_center(), ground_box.get_center()) - _distance(phand_box.get_center(), pground_box.get_center())) < threshold,
#     #         0,
#     #         0,
#     #         0,
#     #         0,
#     #         0,
#     #         (_distance(o3_box.get_center(), ground_box.get_center()) - _distance(po3_box.get_center(), pground_box.get_center())) < threshold]
#     # else:
#     #     print("8")
#     #     P5 = [0,
#     #       0,
#     #       0,
#     #       (_distance(hand_box.get_center(), ground_box.get_center()) - _distance(phand_box.get_center(), pground_box.get_center())) < threshold,
#     #       0,
#     #       0,
#     #       0,
#     #       0,
#     #       0,
#     #       0]

#     # P5 = [(abs(hand_box.get_center() - o1_box.get_center()) - abs(phand_box.get_center() - po1_box.get_center())) < threshold,
#     #       (abs(hand_box.get_center() - o2_box.get_center()) - abs(phand_box.get_center() - po2_box.get_center())) < threshold,
#     #       (abs(hand_box.get_center() - o3_box.get_center()) - abs(phand_box.get_center() - po3_box.get_center())) < threshold,
#     #       (abs(hand_box.get_center() - ground_box.get_center()) - abs(phand_box.get_center() - pground_box.get_center())) < threshold,
#     #       (abs(o1_box.get_center() - o2_box.get_center()) - abs(po1_box.get_center() - po2_box.get_center())) < threshold,
#     #       (abs(o1_box.get_center() - o3_box.get_center()) - abs(po1_box.get_center() - po3_box.get_center())) < threshold,
#     #       (abs(o1_box.get_center() - ground_box.get_center()) - abs(po1_box.get_center() - pground_box.get_center())) < threshold,
#     #       (abs(o2_box.get_center() - o3_box.get_center()) - abs(po2_box.get_center() - po3_box.get_center())) < threshold,
#     #       (abs(o2_box.get_center() - ground_box.get_center()) - abs(po2_box.get_center() - pground_box.get_center())) < threshold,
#     #       (abs(o3_box.get_center() - ground_box.get_center()) - abs(po3_box.get_center() - pground_box.get_center())) < threshold]

#     center_distance = threshold/10
#     if(o1 == None or po1 == None):
#         table[20][0] = 'U'
#     elif(table[0][0] == b'A'):
#         table[20][0] = 'A'
#     elif(len(hand.points) > 0):
#         if (P2[0] and (_distance(hand_box.get_center(), o1_box.get_center()) - _distance(phand_box.get_center(), po1_box.get_center())) < threshold):
#             table[20][0] = 'S'
#         elif ((P1[0] and all(hand_box.get_center() != phand_box.get_center()) and all(o1_box.get_center() != po1_box.get_center())) or 
#               (P1[0] and (xor(all(hand_box.get_center() != phand_box.get_center()), all(o1_box.get_center() != po1_box.get_center()))))):
#             table[20][0] = 'MT' 
#         elif (P1[0] and not all(hand_box.get_center() != phand_box.get_center()) and not all(o1_box.get_center() != po1_box.get_center())):
#             table[20][0] = 'HT' 
#         elif ((not (_distance(hand_box.get_center(), o1_box.get_center()) - _distance(phand_box.get_center(), po1_box.get_center())) < threshold) or 
#              ((_distance(hand_box.get_center(), o1_box.get_center()) - _distance(phand_box.get_center(), po1_box.get_center())) < threshold)):
#             table[20][0] = 'MA'
#         #elif ((_distance(hand_box.get_center(), o1_box.get_center()) - _distance(phand_box.get_center(), po1_box.get_center())) < threshold):
#         #    table[20][0] = 'GC'
#         else:
#             table[20][0] = 'Q'
        
    
#     if(o2 == None or po2 == None):
#         table[21][0] = 'U'
#     elif(table[1][0] == b'A'):
#         table[21][0] = 'A'
#     elif(len(hand.points) > 0):
#         if (P2[1] and (_distance(hand_box.get_center(), o2_box.get_center()) - _distance(phand_box.get_center(), po2_box.get_center())) < threshold):
#             table[21][0] = 'S'
#         elif ((P1[1] and all(hand_box.get_center() != phand_box.get_center()) and all(o2_box.get_center() != po2_box.get_center())) or 
#               (P1[1] and (xor(all(hand_box.get_center() != phand_box.get_center()), all(o2_box.get_center() != po2_box.get_center()))))):
#             table[21][0] = 'MT' 
#         elif (P1[1] and not all(hand_box.get_center() != phand_box.get_center()) and not all(o2_box.get_center() != po2_box.get_center())):
#             table[21][0] = 'HT' 
#         elif ((not (_distance(hand_box.get_center(), o2_box.get_center()) - _distance(phand_box.get_center(), po2_box.get_center())) < threshold) or 
#              ((_distance(hand_box.get_center(), o2_box.get_center()) - _distance(phand_box.get_center(), po2_box.get_center())) < threshold)):
#             table[21][0] = 'MA'
#         #elif ((_distance(hand_box.get_center(), o2_box.get_center()) - _distance(phand_box.get_center(), po2_box.get_center())) < threshold):
#         #    table[21][0] = 'GC'
#         else:
#             table[21][0] = 'Q'
        
       
#     if(o3 == None or po3 == None):
#         table[22][0] = 'U'
#     elif(table[2][0] == b'A'):
#         table[22][0] = 'A'
#     elif(len(hand.points) > 0):
#         if (P2[2] and (_distance(hand_box.get_center(),o3_box.get_center()) - _distance(phand_box.get_center(), po3_box.get_center())) < threshold):
#             table[22][0] = 'S'
#         elif ((P1[2] and all(hand_box.get_center() != phand_box.get_center()) and all(o3_box.get_center() != po3_box.get_center())) or 
#               (P1[2] and (xor(all(hand_box.get_center() != phand_box.get_center()), all(o3_box.get_center() != po3_box.get_center()))))):
#             table[22][0] = 'MT' 
#         elif (P1[2] and not all(hand_box.get_center() != phand_box.get_center()) and not all(o3_box.get_center() != po3_box.get_center())):
#             table[22][0] = 'HT' 
#         elif ((not (_distance(hand_box.get_center(),o3_box.get_center()) - _distance(phand_box.get_center(), po3_box.get_center())) < threshold) or 
#              ((_distance(hand_box.get_center(),o3_box.get_center()) - _distance(phand_box.get_center(), po3_box.get_center())) < threshold)):
#             table[22][0] = 'MA'
#         #elif ((_distance(hand_box.get_center(),o3_box.get_center()) - _distance(phand_box.get_center(), po3_box.get_center())) < threshold):
#         #    table[22][0] = 'GC'
#         else:
#             table[22][0] = 'Q'
         
#     if(len(hand.points) > 0):
#         if (P2[3] and (_distance(hand_box.get_center(), ground_box.get_center()) - _distance(phand_box.get_center(), pground_box.get_center())) < threshold):
#             table[23][0] = 'S'
#         elif ((P1[3] and all(hand_box.get_center() != phand_box.get_center()) and all(ground_box.get_center() != pground_box.get_center())) or 
#               (P1[3] and (xor(all(hand_box.get_center() != phand_box.get_center()), all(ground_box.get_center() != pground_box.get_center()))))):
#             table[23][0] = 'MT' 
#         elif (P1[3] and not all(hand_box.get_center() != phand_box.get_center()) and not all(ground_box.get_center() != pground_box.get_center())):
#             table[23][0] = 'HT' 
#         elif ((not (_distance(hand_box.get_center(), ground_box.get_center()) - _distance(phand_box.get_center(), pground_box.get_center())) < threshold) or
#              ((_distance(hand_box.get_center(), ground_box.get_center()) - _distance(phand_box.get_center(), pground_box.get_center())) < threshold)):
#             table[23][0] = 'MA'
#         #elif ((_distance(hand_box.get_center(), ground_box.get_center()) - _distance(phand_box.get_center(), pground_box.get_center())) < threshold):
#         #    table[23][0] = 'GC'
#         else:
#             table[23][0] = 'Q'
        
        
#     if(o1 == None or o2 == None or po1 == None or po2 == None):
#         table[24][0] = 'U'
#     elif(table[4][0] == b'A'):
#         table[24][0] = 'A'
#     elif (P2[4] and (_distance(o1_box.get_center(), o2_box.get_center()) - _distance(po1_box.get_center(), po2_box.get_center())) < threshold):
#         table[24][0] = 'S'
#     elif ((P1[4] and all(o1_box.get_center() != po1_box.get_center()) and all(o2_box.get_center() != po2_box.get_center())) or 
#           (P1[4] and (xor(all(o1_box.get_center() != po1_box.get_center()), all(o2_box.get_center() != po2_box.get_center()))))):
#         table[24][0] = 'MT' 
#     elif (P1[4] and not all(o1_box.get_center() != po1_box.get_center()) and not all(o2_box.get_center() != po2_box.get_center())):
#         table[24][0] = 'HT' 
#     elif ((not (_distance(o1_box.get_center(), o2_box.get_center()) - _distance(po1_box.get_center(), po2_box.get_center())) < threshold) or 
#          ((_distance(o1_box.get_center(), o2_box.get_center()) - _distance(po1_box.get_center(), po2_box.get_center())) < threshold)):
#         table[24][0] = 'MA'
#     #elif ((_distance(o1_box.get_center(), o2_box.get_center()) - _distance(po1_box.get_center(), po2_box.get_center())) < threshold):
#     #    table[24][0] = 'GC'
#     else:
#         table[24][0] = 'Q'
    
    
#     if(o1 == None or o3 == None or po1 == None or po3 == None):
#         table[25][0] = 'U'
#     elif(table[5][0] == b'A'):
#         table[25][0] = 'A'
#     elif (P2[5] and (_distance(o1_box.get_center(), o3_box.get_center()) - _distance(po1_box.get_center(), po3_box.get_center())) < threshold):
#         table[25][0] = 'S'
#     elif ((P1[5] and all(o1_box.get_center() != po1_box.get_center()) and all(o3_box.get_center() != po3_box.get_center())) or 
#           (P1[5] and (xor(all(o1_box.get_center() != po1_box.get_center()), all(o3_box.get_center() != po3_box.get_center()))))):
#         table[25][0] = 'MT' 
#     elif (P1[5] and not all(o1_box.get_center() != po1_box.get_center()) and not all(o3_box.get_center() != po3_box.get_center())):
#         table[25][0] = 'HT' 
#     elif ((not (_distance(o1_box.get_center(), o3_box.get_center()) - _distance(po1_box.get_center(), po3_box.get_center())) < threshold) or 
#          ((_distance(o1_box.get_center(), o3_box.get_center()) - _distance(po1_box.get_center(), po3_box.get_center())) < threshold)):
#         table[25][0] = 'MA'
#     #elif ((_distance(o1_box.get_center(), o3_box.get_center()) - _distance(po1_box.get_center(), po3_box.get_center())) < threshold):
#     #    table[25][0] = 'GC'
#     else:
#         table[25][0] = 'Q'
    
        
#     if(o1 == None or po1 == None):
#         table[26][0] = 'U'
#     elif(table[6][0] == b'A'):
#         table[26][0] = 'A'
#     elif (P2[6] and (_distance(o1_box.get_center(), ground_box.get_center()) - _distance(po1_box.get_center(), pground_box.get_center())) < threshold):
#         table[26][0] = 'S'
#     elif ((P1[6] and all(o1_box.get_center() != po1_box.get_center()) and all(ground_box.get_center() != pground_box.get_center())) or 
#           (P1[6] and (xor(all(o1_box.get_center() != po1_box.get_center()), all(ground_box.get_center() != pground_box.get_center()))))):
#         table[26][0] = 'MT' 
#     elif (P1[6] and not all(o1_box.get_center() != po1_box.get_center()) and not all(ground_box.get_center() != pground_box.get_center())):
#         table[26][0] = 'HT' 
#     elif ((not (_distance(o1_box.get_center(), ground_box.get_center()) - _distance(po1_box.get_center(), pground_box.get_center())) < threshold) or
#          ((_distance(o1_box.get_center(), ground_box.get_center()) - _distance(po1_box.get_center(), pground_box.get_center())) < threshold)):
#         table[26][0] = 'MA'
#     #elif ((_distance(o1_box.get_center(), ground_box.get_center()) - _distance(po1_box.get_center(), pground_box.get_center())) < threshold):
#     #    table[26][0] = 'GC'
#     else:
#         table[26][0] = 'Q'
    
      
#     if(o2 == None or o3 == None or po2 == None or po3 == None):
#         table[27][0] = 'U'
#     elif(table[7][0] == b'A'):
#         table[27][0] = 'A'
#     elif (P2[7] and (_distance(o2_box.get_center(), o3_box.get_center()) - _distance(po2_box.get_center(), po3_box.get_center())) < threshold):
#         table[27][0] = 'S'
#     elif ((P1[7] and all(o2_box.get_center() != po2_box.get_center()) and all(o3_box.get_center() != po3_box.get_center())) or 
#           (P1[7] and (xor(all(o2_box.get_center() != po2_box.get_center()), all(o3_box.get_center() != po3_box.get_center()))))):
#         table[27][0] = 'MT' 
#     elif (P1[7] and not all(o2_box.get_center() != po2_box.get_center()) and not all(o3_box.get_center() != po3_box.get_center())):
#         table[27][0] = 'HT' 
#     elif ((not (_distance(o2_box.get_center(), o3_box.get_center()) - _distance(po2_box.get_center(), po3_box.get_center())) < threshold) or 
#          ((_distance(o2_box.get_center(), o3_box.get_center()) - _distance(po2_box.get_center(), po3_box.get_center())) < threshold)):
#         table[27][0] = 'MA'
#     #elif ((_distance(o2_box.get_center(), o3_box.get_center()) - _distance(po2_box.get_center(), po3_box.get_center())) < threshold):
#     #    table[27][0] = 'GC'
#     else:
#         table[27][0] = 'Q'
    
#     # if o2 != None and po2 != None:
#     #     print(len(o2.points), len(po2.points))
    
#     if(o2 == None or po2 == None):
#         table[28][0] = 'U'
#     elif(table[8][0] == b'A'):
#         table[28][0] = 'A'
#     elif (P2[8] and (_distance(o2_box.get_center(), ground_box.get_center()) - _distance(po2_box.get_center(), pground_box.get_center())) < threshold):
#         table[28][0] = 'S'
#     elif ((P1[8] and all(o2_box.get_center() != po2_box.get_center()) and all(ground_box.get_center() != pground_box.get_center())) or 
#           (P1[8] and (xor(all(o2_box.get_center() != po2_box.get_center()), all(ground_box.get_center() != pground_box.get_center()))))):
#         table[28][0] = 'MT' 
#     elif (P1[8] and not all(o2_box.get_center() != po2_box.get_center()) and not all(ground_box.get_center() != pground_box.get_center())):
#         table[28][0] = 'HT' 
#     elif ((not (_distance(o2_box.get_center(), ground_box.get_center()) - _distance(po2_box.get_center(), pground_box.get_center())) < threshold) or 
#          ((_distance(o2_box.get_center(), ground_box.get_center()) - _distance(po2_box.get_center(), pground_box.get_center())) < threshold)):
#         table[28][0] = 'MA'
#     #elif ((_distance(o2_box.get_center(), ground_box.get_center()) - _distance(po2_box.get_center(), pground_box.get_center())) < threshold):
#     #    table[28][0] = 'GC'
#     else:
#         table[28][0] = 'Q'
        
#     if(o3 == None or po3 == None):
#         table[29][0] = 'U'
#     elif(table[9][0] == b'A'):
#         table[29][0] = 'A'
#     elif (P2[9] and (_distance(o3_box.get_center(), ground_box.get_center()) - _distance(po3_box.get_center(), pground_box.get_center())) < threshold):
#         table[29][0] = 'S'
#     elif ((P1[9] and all(o3_box.get_center() != po3_box.get_center()) and all(ground_box.get_center() != pground_box.get_center())) or 
#           (P1[9] and (xor(all(o3_box.get_center() != po3_box.get_center()), all(ground_box.get_center() != pground_box.get_center()))))):
#         table[29][0] = 'MT' 
#     elif (P1[9] and not all(o3_box.get_center() != po3_box.get_center()) and not all(ground_box.get_center() != pground_box.get_center())):
#         table[29][0] = 'HT' 
#     elif ((not (_distance(o3_box.get_center(), ground_box.get_center()) - _distance(po3_box.get_center(), pground_box.get_center())) < threshold) or 
#          ((_distance(o3_box.get_center(), ground_box.get_center()) - _distance(po3_box.get_center(), pground_box.get_center())) < threshold)):
#         table[29][0] = 'MA'
#     #elif ((_distance(o3_box.get_center(), ground_box.get_center()) - _distance(po3_box.get_center(), pground_box.get_center())) < threshold):
#     #    table[29][0] = 'GC'
#     else:
#         table[29][0] = 'Q'

def _fillDSR_new(hand, ground, previous_array, thresh, table):
    '''
    Creates the DSR with pre-defined point clouds of hand and ground. It also need the point cloud from the previous frame.
    
    Parameters:
        * hand: point cloud of the hand
        * ground: point cloud of the ground
        * previous_array: array that contains point clouds from previous frame of hand, ground, o1, o2 and o3
        * thresh: threshold distance of touching
        * table: chararray table   
    '''
    #     DSR
    #     H, 1
    #     H, 2
    #     H, 3
    #     H, G
    #     1, 2
    #     1, 3
    #     1, G
    #     2, 3
    #     2, G
    #     3, G 

    #get global objects 1, 2 and 3
    global o1,o2,o3
    
    #get varaibles from previous frame
    phand, pground, po1, po2, po3 = previous_array
    
    #creates AABB around object1 if o1 and po1 are defined and if TNR relation is not abscent
    if(o1 != None and po1 != None and table[0][0] != b'A'):
        o1_box = o1.get_axis_aligned_bounding_box()
        po1_box = po1.get_axis_aligned_bounding_box()
        # o1_box_edges = o1_box.get_box_points()
        # po1_box_edges = po1_box.get_box_points()
        
    #creates AABB around object2 if o2 and po2 are defined and if TNR relation is not abscent    
    if(o2 != None and po2 != None and table[1][0] != b'A'):
        if len(o2.points) != 0:
            o2_box = o2.get_axis_aligned_bounding_box()
            po2_box = po2.get_axis_aligned_bounding_box()
            # o2_box_edges = o2_box.get_box_points()
            # po2_box_edges = po2_box.get_box_points()
        
    #creates AABB around object1 if o3 and po3 are defined and if TNR relation is not abscent
    if(o3 != None and po3 != None and table[2][0] != b'A'):
        if len(o3.points) != 0:
            o3_box = o3.get_axis_aligned_bounding_box()
            po3_box = po3.get_axis_aligned_bounding_box()
            # o3_box_edges = o3_box.get_box_points()
            # po3_box_edges = po3_box.get_box_points()
        
    #creates AABB around hand if point cloud is in the frame
    if(len(hand.points) > 0):
        hand_box = hand.get_axis_aligned_bounding_box()
        phand_box = phand.get_axis_aligned_bounding_box()
        # hand_box_edges = hand_box.get_box_points()
        # phand_box_edges = phand_box.get_box_points()
    
    #creates AABB around ground
    ground_box = ground.get_axis_aligned_bounding_box()
    pground_box = pground.get_axis_aligned_bounding_box()
    # ground_box_edges = ground_box.get_box_points()
    # pground_box_edges = pground_box.get_box_points()
          

    # o1_left_bot_front, o1_left_bot_back   = [o1_], []
    # o1_left_top_front, o1_left_top_back   = [], []
    # o1_right_bot_front, o1_right_bot_back = [], []
    # o1_right_top_front, o1_right_top_back = [], []
    # if(o1 != None and po1 != None and table[0][0] != b'A'):   
    #     print("\n Distance: ", _distance(o1_box.get_center(), po1_box.get_center()))
        # p = o3d.geometry.PointCloud()
        # p.points = o3d.utility.Vector3dVector([o1_box.get_center()])
        # q = o3d.geometry.PointCloud()
        # q.points = o3d.utility.Vector3dVector([po1_box.get_center()])
        # p.paint_uniform_color([1, 0, 0])
        # q.paint_uniform_color([0, 1, 0])
        # o3d.visualization.draw_geometries([o1_box, po1_box, p, q, ground])
    #multi = 1.5
    threshold = thresh
    #center_distance = threshold/10
    center_distance = threshold/20
    stable_dist = threshold/50
    #center_distance = 0.02

    P1 = [table[0][0] == b'T', table[1][0] == b'T', table[2][0] == b'T', table[3][0] == b'T', table[4][0] == b'T', table[5][0] == b'T', table[6][0] == b'T', table[7][0] == b'T', table[8][0] == b'T', table[9][0] == b'T']
    P2 = [table[0][0] == b'N', table[1][0] == b'N', table[2][0] == b'N', table[3][0] == b'N', table[4][0] == b'N', table[5][0] == b'N', table[6][0] == b'N', table[7][0] == b'N', table[8][0] == b'N', table[9][0] == b'N']

    if(o1 == None or po1 == None):
        table[20][0] = 'U'
    elif(table[0][0] == b'A'):
        table[20][0] = 'A'
    elif(len(hand.points) > 0):
        #print("\n", _distance(hand.get_center(), phand.get_center()), " > ", center_distance)
        if ((P1[0] and _distance(hand.get_center(), phand.get_center()) > center_distance and _distance(o1_box.get_center(), po1_box.get_center()) > center_distance) or 
              (P1[0] and (xor(_distance(hand.get_center(), phand.get_center()) > center_distance, _distance(o1_box.get_center(), po1_box.get_center()) > center_distance)))):
            table[20][0] = 'MT' 
        elif (P1[0] and not _distance(hand.get_center(), phand.get_center()) > center_distance and not _distance(o1_box.get_center(),po1_box.get_center()) > center_distance):
            table[20][0] = 'HT' 
        elif ((_distance(hand.get_center(), o1_box.get_center()) - _distance(phand.get_center(), po1_box.get_center())) > -stable_dist and (_distance(hand.get_center(), o1_box.get_center()) - _distance(phand.get_center(), po1_box.get_center())) < stable_dist):
            table[20][0] = 'S'
        elif ((_distance(hand.get_center(), o1_box.get_center()) > _distance(phand.get_center(), po1_box.get_center()))):
            table[20][0] = 'MA'
        elif ((_distance(hand.get_center(), o1_box.get_center()) < _distance(phand.get_center(), po1_box.get_center()))):
            table[20][0] = 'GC'
        else:
            table[20][0] = 'Q'

        
    
    if(o2 == None or po2 == None):
        table[21][0] = 'U'
    elif(table[1][0] == b'A'):
        table[21][0] = 'A'
    elif(len(hand.points) > 0):
        if ((P1[1] and _distance(hand.get_center() , phand.get_center() > center_distance) and _distance(o2_box.get_center() , po2_box.get_center()) > center_distance) or 
              (P1[1] and (xor(_distance(hand.get_center() , phand.get_center()) > center_distance, _distance(o2_box.get_center() , po2_box.get_center()) > center_distance)))):
            table[21][0] = 'MT' 
        elif (P1[1] and not _distance(hand.get_center() , phand.get_center()) > center_distance and not _distance(o2_box.get_center() , po2_box.get_center()) > center_distance):
            table[21][0] = 'HT'
        elif ((_distance(hand.get_center(), o2_box.get_center()) - _distance(phand.get_center(), po2_box.get_center())) > -stable_dist and (_distance(hand.get_center(), o2_box.get_center()) - _distance(phand.get_center(), po2_box.get_center())) < stable_dist):
            table[21][0] = 'S'
        elif ((_distance(hand.get_center(), o2_box.get_center()) > _distance(phand.get_center(), po2_box.get_center()))):
            table[21][0] = 'MA'
        elif ((_distance(hand.get_center(), o2_box.get_center()) < _distance(phand.get_center(), po2_box.get_center()))):
            table[21][0] = 'GC' 
        else:
            table[21][0] = 'Q'
        
       
    if(o3 == None or po3 == None):
        table[22][0] = 'U'
    elif(table[2][0] == b'A'):
        table[22][0] = 'A'
    elif(len(hand.points) > 0):
        if ((P1[2] and _distance(hand.get_center() , phand.get_center() > center_distance) and _distance(o3_box.get_center() , po3_box.get_center()) > center_distance) or 
              (P1[2] and (xor(_distance(hand.get_center() , phand.get_center()) > center_distance, _distance(o3_box.get_center() , po3_box.get_center()) > center_distance)))):
            table[22][0] = 'MT' 
        elif (P1[2] and not _distance(hand.get_center() , phand.get_center()) > center_distance and not _distance(o3_box.get_center() , po3_box.get_center()) > center_distance):
            table[22][0] = 'HT' 
        elif ((_distance(hand.get_center(),o3_box.get_center()) - _distance(phand.get_center(), po3_box.get_center())) > -stable_dist and (_distance(hand.get_center(),o3_box.get_center()) - _distance(phand.get_center(), po3_box.get_center())) < stable_dist):
            table[22][0] = 'S'
        elif ((_distance(hand.get_center(),o3_box.get_center()) > _distance(phand.get_center(), po3_box.get_center()))):
            table[22][0] = 'MA'
        elif ((_distance(hand.get_center(),o3_box.get_center()) < _distance(phand.get_center(), po3_box.get_center()))):
            table[22][0] = 'GC'
        else:
            table[22][0] = 'Q'
         
    if(len(hand.points) > 0):
        if ((P1[3] and _distance(hand.get_center() , phand.get_center()) > center_distance and _distance(ground_box.get_center() , pground_box.get_center()) > center_distance) or 
              (P1[3] and (xor(_distance(hand.get_center() , phand.get_center()) > center_distance, _distance(ground_box.get_center() , pground_box.get_center()) > center_distance)))):
            table[23][0] = 'MT' 
        elif (P1[3] and not _distance(hand.get_center() , phand.get_center()) > center_distance and not _distance(ground_box.get_center() , pground_box.get_center()) > center_distance):
            table[23][0] = 'HT' 
        elif ((abs(hand.get_center()[2] - ground_box.get_center()[2]) - abs(phand.get_center()[2] - pground_box.get_center()[2])) > -stable_dist and (abs(hand.get_center()[2] - ground_box.get_center()[2]) - abs(phand.get_center()[2] - pground_box.get_center()[2])) < stable_dist):
            table[23][0] = 'S'
        elif ((abs(hand.get_center()[2] - ground_box.get_center()[2]) > abs(phand.get_center()[2] - pground_box.get_center()[2]))):
            table[23][0] = 'MA'
        elif ((abs(hand.get_center()[2] - ground_box.get_center()[2]) < abs(phand.get_center()[2] - pground_box.get_center()[2]))):
            table[23][0] = 'GC'
        else:
            table[23][0] = 'Q'
        
    if(o1 == None or o2 == None or po1 == None or po2 == None):
        table[24][0] = 'U'
    elif(table[4][0] == b'A'):
        table[24][0] = 'A'
    elif ((P1[4] and _distance(o1_box.get_center() , po1_box.get_center()) > center_distance and _distance(o2_box.get_center() , po2_box.get_center()) > center_distance) or 
          (P1[4] and (xor(_distance(o1_box.get_center() , po1_box.get_center()) > center_distance, _distance(o2_box.get_center() , po2_box.get_center()) > center_distance)))):
        #print("\n!!!MT!!!")
        #print("\nMT: (",P1[4], "and", _distance(o1_box.get_center() , po1_box.get_center()) > center_distance, "and", _distance(o2_box.get_center() , po2_box.get_center()) > center_distance, ") or (",P1[5], "and", (xor(_distance(o1_box.get_center() , po1_box.get_center()) > center_distance, _distance(o2_box.get_center() , po2_box.get_center()) > center_distance)))
        #print("\nMA: (",not (_distance(o1_box.get_center(), o2_box.get_center()) - _distance(po1_box.get_center(), po2_box.get_center()) < threshold), "or", ((_distance(o1_box.get_center(), o2_box.get_center()) - _distance(po1_box.get_center(), po2_box.get_center())) < threshold))
        table[24][0] = 'MT' 
    elif (P1[4] and not _distance(o1_box.get_center() , po1_box.get_center()) > center_distance and not _distance(o2_box.get_center() , po2_box.get_center()) > center_distance):
        #print("\n!!!HT!!!")
        #print("\nMT: (",P1[4], "and", _distance(o1_box.get_center() , po1_box.get_center()) > center_distance, "and", _distance(o2_box.get_center() , po2_box.get_center()) > center_distance, ") or (",P1[5], "and", (xor(_distance(o1_box.get_center() , po1_box.get_center()) > center_distance, _distance(o2_box.get_center() , po2_box.get_center()) > center_distance)))
        #print("\nMA: (",not (_distance(o1_box.get_center(), o2_box.get_center()) - _distance(po1_box.get_center(), po2_box.get_center()) < threshold), "or", ((_distance(o1_box.get_center(), o2_box.get_center()) - _distance(po1_box.get_center(), po2_box.get_center())) < threshold))
        table[24][0] = 'HT' 
    elif ((_distance(o1_box.get_center(), o2_box.get_center()) - _distance(po1_box.get_center(), po2_box.get_center())) > -stable_dist and (_distance(o1_box.get_center(), o2_box.get_center()) - _distance(po1_box.get_center(), po2_box.get_center())) < stable_dist):
        table[24][0] = 'S'
    elif ((_distance(o1_box.get_center(), o2_box.get_center()) > _distance(po1_box.get_center(), po2_box.get_center()))):
        table[24][0] = 'MA'
    elif ((_distance(o1_box.get_center(), o2_box.get_center()) < _distance(po1_box.get_center(), po2_box.get_center()))):
        table[24][0] = 'GC'
    else:
        table[24][0] = 'Q'
    # if o1 != None and o2 != None and po1 != None and po2 != None:
        #print("\n", "center cloud: ", o1.get_center(),"center aabb: ", o1_box.get_center())
        # print("\n",(_distance(o1_box.get_center(), o2_box.get_center()) - _distance(po1_box.get_center(), po2_box.get_center())))
        # print((_distance(o1_box.get_center(), o2_box.get_center()) - _distance(po1_box.get_center(), po2_box.get_center())), center_distance)
        

    if(o1 == None or o3 == None or po1 == None or po3 == None):
        table[25][0] = 'U'
    elif(table[5][0] == b'A'):
        table[25][0] = 'A'
    elif ((P1[5] and _distance(o1_box.get_center() , po1_box.get_center()) > center_distance and _distance(o3_box.get_center() , po3_box.get_center()) > center_distance) or 
          (P1[5] and (xor(_distance(o1_box.get_center() , po1_box.get_center()) > center_distance, _distance(o3_box.get_center() , po3_box.get_center()) > center_distance)))):
        table[25][0] = 'MT' 
    elif (P1[5] and not _distance(o1_box.get_center() , po1_box.get_center()) > center_distance and not _distance(o3_box.get_center() , po3_box.get_center()) > center_distance):
        table[25][0] = 'HT' 
    elif ((_distance(o1_box.get_center(), o3_box.get_center()) - _distance(po1_box.get_center(), po3_box.get_center())) > -stable_dist and (_distance(o1_box.get_center(), o3_box.get_center()) - _distance(po1_box.get_center(), po3_box.get_center())) < stable_dist):
        table[25][0] = 'S'
    elif ((_distance(o1_box.get_center(), o3_box.get_center()) > _distance(po1_box.get_center(), po3_box.get_center()))):
        table[25][0] = 'MA'
    elif ((_distance(o1_box.get_center(), o3_box.get_center()) < _distance(po1_box.get_center(), po3_box.get_center()))):
        table[25][0] = 'GC'
    else:
        table[25][0] = 'Q'
    
        
    if(o1 == None or po1 == None):
        table[26][0] = 'U'
    elif(table[6][0] == b'A'):
        table[26][0] = 'A'
    elif ((P1[6] and _distance(o1_box.get_center() , po1_box.get_center()) > center_distance and _distance(ground_box.get_center() , pground_box.get_center()) > center_distance) or 
          (P1[6] and (xor(_distance(o1_box.get_center() , po1_box.get_center()) > center_distance, _distance(ground_box.get_center() , pground_box.get_center()) > center_distance)))):
        table[26][0] = 'MT' 
    elif (P1[6] and not _distance(o1_box.get_center() , po1_box.get_center()) > center_distance and not _distance(ground_box.get_center() , pground_box.get_center()) > center_distance):
        table[26][0] = 'HT' 
    elif ((abs(o1_box.get_center()[2] - ground_box.get_center()[2]) - abs(po1_box.get_center()[2] - pground_box.get_center()[2])) > -stable_dist and (abs(o1_box.get_center()[2] - ground_box.get_center()[2]) - abs(po1_box.get_center()[2] - pground_box.get_center()[2])) < stable_dist):
        table[26][0] = 'S'
    elif ((abs(o1_box.get_center()[2] - ground_box.get_center()[2]) > abs(po1_box.get_center()[2] - pground_box.get_center()[2]))):
        table[26][0] = 'MA'
    elif ((abs(o1_box.get_center()[2] - ground_box.get_center()[2]) < abs(po1_box.get_center()[2] - pground_box.get_center()[2]))):
        table[26][0] = 'GC'
        #print("\n",(_distance(o1.get_center(), ground_box.get_center()), "<", _distance(po1.get_center(), pground_box.get_center())))
        # print("\n",(_distance(o1_box.get_center(), ground_box.get_center()), "<", _distance(po1_box.get_center(), pground_box.get_center())))
    else:
        table[26][0] = 'Q'
    
      
    if(o2 == None or o3 == None or po2 == None or po3 == None):
        table[27][0] = 'U'
    elif(table[7][0] == b'A'):
        table[27][0] = 'A'
    elif ((P1[7] and _distance(o2_box.get_center() , po2_box.get_center()) > center_distance and _distance(o3_box.get_center() , po3_box.get_center()) > center_distance) or 
          (P1[7] and (xor(_distance(o2_box.get_center() , po2_box.get_center()) > center_distance, _distance(o3_box.get_center() , po3_box.get_center()) > center_distance)))):
        table[27][0] = 'MT' 
    elif (P1[7] and not _distance(o2_box.get_center() , po2_box.get_center()) > center_distance and not _distance(o3_box.get_center() , po3_box.get_center()) > center_distance):
        table[27][0] = 'HT' 
    elif ((_distance(o2_box.get_center(), o3_box.get_center()) - _distance(po2_box.get_center(), po3_box.get_center())) > -stable_dist and (_distance(o2_box.get_center(), o3_box.get_center()) - _distance(po2_box.get_center(), po3_box.get_center())) < stable_dist):
        table[27][0] = 'S'
    elif ((_distance(o2_box.get_center(), o3_box.get_center()) > _distance(po2_box.get_center(), po3_box.get_center()))):
        table[27][0] = 'MA'
    elif ((_distance(o2_box.get_center(), o3_box.get_center()) < _distance(po2_box.get_center(), po3_box.get_center()))):
        table[27][0] = 'GC'
    else:
        table[27][0] = 'Q'
    
    # if o2 , None and po2 , None:
    #     print(len(o2.points), len(po2.points))
    
    if(o2 == None or po2 == None):
        table[28][0] = 'U'
    elif(table[8][0] == b'A'):
        table[28][0] = 'A'
    elif ((P1[8] and _distance(o2_box.get_center() , po2_box.get_center()) > center_distance and _distance(ground_box.get_center() , pground_box.get_center()) > center_distance) or 
          (P1[8] and (xor(_distance(o2_box.get_center() , po2_box.get_center()) > center_distance, _distance(ground_box.get_center() , pground_box.get_center()) > center_distance)))):
        table[28][0] = 'MT' 
    elif (P1[8] and not _distance(o2_box.get_center() , po2_box.get_center()) > center_distance and not _distance(ground_box.get_center() , pground_box.get_center()) > center_distance):
        table[28][0] = 'HT'
    elif ((abs(o2_box.get_center()[2] - ground_box.get_center()[2]) - abs(po2_box.get_center()[2] - pground_box.get_center()[2])) > -stable_dist and (abs(o2_box.get_center()[2] - ground_box.get_center()[2]) - abs(po2_box.get_center()[2] - pground_box.get_center()[2])) < stable_dist):
        table[28][0] = 'S' 
    elif ((abs(o2_box.get_center()[2] - ground_box.get_center()[2]) > abs(po2_box.get_center()[2] - pground_box.get_center()[2]))):
        table[28][0] = 'MA'
    elif ((abs(o2_box.get_center()[2] - ground_box.get_center()[2]) < abs(po2_box.get_center()[2] - pground_box.get_center()[2]))):
        table[28][0] = 'GC'
    else:
        table[28][0] = 'Q'
    
    if(o3 == None or po3 == None):
        table[29][0] = 'U'
    elif(table[9][0] == b'A'):
        table[29][0] = 'A'
    elif ((P1[9] and _distance(o3_box.get_center() , po3_box.get_center()) > center_distance and _distance(ground_box.get_center() , pground_box.get_center()) > center_distance) or 
          (P1[9] and (xor(_distance(o3_box.get_center() , po3_box.get_center()) > center_distance, _distance(ground_box.get_center() , pground_box.get_center()) > center_distance)))):
        table[29][0] = 'MT' 
    elif (P1[9] and not _distance(o3_box.get_center() , po3_box.get_center()) > center_distance and not _distance(ground_box.get_center() , pground_box.get_center()) > center_distance):
        table[29][0] = 'HT' 
    elif ((abs(o3_box.get_center()[2] - ground_box.get_center()[2]) - abs(po3_box.get_center()[2] - pground_box.get_center()[2])) > -stable_dist and (abs(o3_box.get_center()[2] - ground_box.get_center()[2]) - abs(po3_box.get_center()[2] - pground_box.get_center()[2])) < stable_dist):
        table[29][0] = 'S'
    elif ((abs(o3_box.get_center()[2] - ground_box.get_center()[2]) > abs(po3_box.get_center()[2] - pground_box.get_center()[2]))):
        table[29][0] = 'MA'
    elif ((abs(o3_box.get_center()[2] - ground_box.get_center()[2]) < abs(po3_box.get_center()[2] - pground_box.get_center()[2]))):
        table[29][0] = 'GC'
    else:
        table[29][0] = 'Q'


def _region_filter(pcd):
    #get center of point cloud
    center = pcd.get_center()

    #transorfm point cloud to numpy
    pcd_array = np.asarray(pcd.points)

    #define dist dummy for calculation
    dist = sys.maxsize
    
    #find the nearest point in the point cloud from middle point
    for i in range(len(pcd_array)):
        dist_temp = _distance(pcd_array[i], center)
        if dist_temp < dist:
            dist = dist_temp
            first_point = pcd_array[i]

    #calculate mean distance between points
    mean_distance = 0
    for i in range(len(pcd_array)):
        mean_distance += _distance(pcd_array[i], center)
    mean_distance /= i
    new_cloud = copy.deepcopy(first_point)

    next_point = False
    for i in range(len(pcd_array)):
        if isinstance(next_point, bool):
            dist_temp = _distance(pcd_array[i], first_point)
        else:
            dist_temp = _distance(pcd_array[i], next_point)
        if dist_temp <= mean_distance:
            next_point = pcd_array[i]
            new_cloud = np.row_stack((new_cloud,next_point))
            i = 0

    new_cloudi = o3d.geometry.PointCloud()
    new_cloudi.points = o3d.utility.Vector3dVector(new_cloud)

    return new_cloudi

def _region_filter(pcd):
    #get center of point cloud
    center = pcd.get_center()

    #transorfm point cloud to numpy
    pcd_array = np.asarray(pcd.points)

    #define dist dummy for calculation
    dist = sys.maxsize
    
    #find the nearest point in the point cloud from middle point
    for i in range(len(pcd_array)):
        dist_temp = _distance(pcd_array[i], center)
        if dist_temp < dist:
            dist = dist_temp
            first_point = pcd_array[i]

    #calculate mean distance between points
    mean_distance = 0
    for i in range(len(pcd_array)):
        mean_distance += _distance(pcd_array[i], center)
    mean_distance /= i
    new_cloud = copy.deepcopy(first_point)

    next_point = False
    for i in range(len(pcd_array)):
        if isinstance(next_point, bool):
            dist_temp = _distance(pcd_array[i], first_point)
        else:
            dist_temp = _distance(pcd_array[i], next_point)
        if dist_temp <= mean_distance:
            next_point = pcd_array[i]
            new_cloud = np.row_stack((new_cloud,next_point))
            i = 0

    new_cloudi = o3d.geometry.PointCloud()
    new_cloudi.points = o3d.utility.Vector3dVector(new_cloud)

    return new_cloudi


def _process_rotation(pcd_file, label_file, ground_label, hand_label, 
                support_hand, rotation, frame, fps, ESEC_table, 
                relations, replace = False, old = [], new = [], ignored_labels = [], thresh = 0.1, debug = False, cython = False):
    '''
    Creates raw eSEC table from a point cloud with corresponding label file. 
    
    Parameters:
        * pcd_file: pcd file to process (.pcd)
        * label_file: label file corresponding to pcd file (.dat)
        * ground_label: label of the ground (int)
        * hand_label: label of the hand (int)
        * rotation: rotation of the scene, will be retured from function (start with 0)
        * frame: frame of manipulation, start with zero and count (int)
        * ESEC_table: empty chararray 10x1 for T/N; 20x1 for T/N, SSR; 30x1 for T/N, SSR, DSR
        * relations: relations to proceed in the computation 1:T/N; 2:T/N, SSR; 3:T/N, SSR, DSR
        * replace: True if labels should be replaces, False otherwise
        * old: old labels to raplace [int]
        * new: new labels that will replace old labels [int]
        * ignored_labels: labels that will be ignored in this manipulation [int]
        * threshold that defines distance for touching
        * cython: if true a self created filter will be used (experimental)
    
    Returns:
        * rotation: rotation of the scene
        * table: ESEC table
    '''
 
    
    #resize labels to point cloud size
    my_mat = np.zeros((640, 480))
    label = pd.read_csv(label_file,delim_whitespace=True, dtype =np.float64, header=None)
    label = label.to_numpy()

    #get global variables
    global hand_label_inarray, total_unique_labels
    
    if (frame == 0):
        #calculate the rotation with the first frame and define ground
        global count_ground, ground
        rotation = _rotateSceneNewNew(pcd_file, label_file, ground_label)
        ground = _getGroundiNew(pcd_file, label_file, ground_label, rotation)
        count_ground = 1
        #find unique labels and replace old with new if replace == True
        unique_labels = np.unique(label)
        total_unique_labels = unique_labels
        if replace == True:
            for values in old:
                total_unique_labels = np.delete(total_unique_labels, np.where(total_unique_labels == old))

        #remove ignored labels from total_unique_labels array
        for values in ignored_labels:
            if values in total_unique_labels:
                total_unique_labels = np.delete(total_unique_labels, np.where(total_unique_labels == values))

        #add hand label if not in total_unique_labels else append total_unique_labels by hand label
        if hand_label not in total_unique_labels:
            total_unique_labels = np.append(total_unique_labels, hand_label)
            hand_label_inarray = 0
        else:
            hand_label_inarray = np.where(total_unique_labels == hand_label)[0][0]
            
    #if hand is missing return roation and eSEC table      
    if hand_label not in np.unique(label):
        return rotation, ESEC_table
    
    #resize the label file to point cloud size
    label_resized = cv2.resize(label, my_mat.shape, interpolation = cv2.INTER_NEAREST)
    if replace == True:
        label_resized = _replace_labels(label_resized, old, new)
    label_resized = label_resized.flatten()
    
    #load pcd file with nan points to map lables     
    pcd = o3d.io.read_point_cloud(pcd_file, remove_nan_points=False)
    
    #cast cloud to numpy array and replace nan values with int value -100
    cloud = np.asarray(pcd.points)
    cloud = np.nan_to_num(cloud, nan=-100)

    #add labels to points from the cloud and sort
    pcd_array =  np.column_stack((cloud, label_resized))
    pcd_array_sorted = pcd_array[pcd_array[:, 3].argsort()]
    
    #search for -100 values and delete them
    result_1 = np.where(pcd_array_sorted == -100)
    pcd_array_sorted = np.delete(pcd_array_sorted, np.unique(result_1[0]), 0)

    #calculate the array index for the labels
    index = {}
    i = 0
    empty_index = []
    for value in total_unique_labels:
            #if no hand label is defined previous hand must be the last entry in total_unique_labels
            if(hand_label_inarray == 0):
                hand_label_inarray = np.where(total_unique_labels == np.max(total_unique_labels))[0][0]
                
            #find the index of the different objects in the scene
            if float(value) in pcd_array_sorted[:,3]:
                index[i] = np.where(pcd_array_sorted[:,3] == float(value))
            
            #if an value from total_unique_labels is not in the cloud then index is defined as -1
            else:
                index[i] = -1
                empty_index.append(i)
            i += 1

    #if hand is missing in scene return rotation and eSEC table
    if hand_label_inarray in empty_index:
        return rotation, ESEC_table
    
    #delete the labels from cloud array
    pcd_array_sorted = np.delete(pcd_array_sorted, 3,1)
    
    #rotate the point cloud with calculated rotation
    rotated_cloud = o3d.geometry.PointCloud()
    rotated_cloud.points = o3d.utility.Vector3dVector(pcd_array_sorted)
    rotated_cloud = rotated_cloud.rotate(R=rotation.T, center=rotated_cloud.get_center())
    
    #extract points from rotated cloud
    pcd_array_sorted = np.asarray(rotated_cloud.points)

    #create objects dict
    objects = {}

    j = 0
    #define single objects in the manipulation
    while(j < len(total_unique_labels)):
        #if an label has no point cloud assigned sat its value to -1
        if(j in empty_index):
            objects[j] = -1
            j += 1
            
        #otherwise assign the point cloud of this label to objects dict
        else:
            objects[j] = pcd_array_sorted[index[j]]
            j += 1
    
    #get global counts, lables, previous point cloud and counts
    global count1, count2, count3
    global o1_label, o2_label, o3_label, previous_array, internal_count
    #empty dict for point clouds of single objects, 
    pcd = {}
    filtered_pcd_voxel = {}
    j = 0
    for i in range(len(total_unique_labels)):
        #if label has a point cloud (i.e. objects[i] != -1) proceed
        if not isinstance(objects[i], int):
            #convert arrays to point clouds
            pcd[i] = o3d.geometry.PointCloud()
            pcd[i].points = o3d.utility.Vector3dVector(objects[i])
            if  i != ground_label and i != 0:
                if cython == True:
                    center = np.array(pcd[i].get_center())
                    filtered_pcd_voxel_array = filter_cython.region_filter_cython(center, objects[i], i == hand_label_inarray)
                    if isinstance(filtered_pcd_voxel_array, int):
                        filtered_pcd_voxel[i] = o3d.geometry.PointCloud()
                    else:
                        filtered_pcd_voxel[i] = o3d.geometry.PointCloud()
                        filtered_pcd_voxel[i].points = o3d.utility.Vector3dVector(filtered_pcd_voxel_array)
                else:
                    #filter objects with statistical filter except ground and label 0 (borders)
                    filtered_pcd_voxel[i], _ = pcd[i].remove_statistical_outlier(nb_neighbors=20, std_ratio=1)
                    if len(filtered_pcd_voxel[i].points)  == 0 and len(objects[i]) < 3:
                        filtered_pcd_voxel[i] = pcd[i]

                    else:
                        filtered_pcd_voxel[i] = pcd[i]
            else:
                filtered_pcd_voxel[i] = pcd[i]

        else:
            filtered_pcd_voxel[i] = o3d.geometry.PointCloud()
            
    #define hand variable as hand point cloud
    hand = filtered_pcd_voxel[hand_label_inarray]

    #if hand has no points return rotation and eSEC table
    if len(hand.points) == 0:
        #print('Hand is misssing 3')
        return rotation, ESEC_table
    
    #get global objects 1,2 and 3
    global o1, o2, o3
    
    #get possible combinations of objects, threshold and an empty char 
    list_ = np.arange(1, len(objects))
    combis_two = list(combinations(list_, 2))  

    #define object1, 2 and 3 in this loop
    for i in range(len(total_unique_labels)):
        if support_hand != None:
            #possible o1, o2, o3 candidates must not be hand_label, ground_label, support_hand_label, label 0
            #cloud has to be bigger than zero
            if(i != hand_label_inarray and i != np.where(total_unique_labels == ground_label)[0][0] and i != np.where(total_unique_labels == support_hand)[0][0] and i > 0 and len(filtered_pcd_voxel[i].points) > 0):
                #checkt if clouds have any points and if hand is in this frame
                if(len(filtered_pcd_voxel[i].points) > 0 and len(hand.points) > 0):
                    #if no o1 is defined, define it when distance to hand is smaller than thresh
                    if(count1 == 0 and np.min(hand.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh):
                            o1_label = i
                            count1 = 1
                            print('o1 found!, label: %d'%total_unique_labels[i])
                            print(label_file)
                            o1 = filtered_pcd_voxel[i]

                    #if o1 is defined and o2 not, define it when distance to hand or o1 is smaller than thresh
                    elif(o1 != None and count2 == 0 and i != o1_label):
                        if(np.min(hand.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh
                           or np.min(o1.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh):  
                                o2_label = i
                                count2 = 1
                                print('o2 found!, label: %d'%total_unique_labels[i])
                                o2 = filtered_pcd_voxel[i]

                    #if o1, o2 is defined and o3 not, define it when distance to hand, o1 or o2 is smaller than thresh
                    elif(o2 != None and count3 == 0 and i != o1_label and i != o2_label):
                        if(np.min(hand.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh
                           or np.min(o1.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh
                           or np.min(o2.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh):
                                o3_label = i
                                count3 = 1
                                print('o3 found!, label: %d'%total_unique_labels[i])
                                o3 = filtered_pcd_voxel[i]
        
        else:
            #possible o1, o2, o3 candidates must not be hand_label, ground_label, label 0
            #cloud has to be bigger than zero
            if(i != hand_label_inarray and i != np.where(total_unique_labels == ground_label)[0][0] and i > 0 and len(filtered_pcd_voxel[i].points) > 0):
                #checkt if clouds have any points and if hand is in this frame
                if(len(filtered_pcd_voxel[i].points) > 0 and len(hand.points) > 0):
                    #if no o1 is defined, define it when distance to hand is smaller than thresh
                    if(count1 == 0 and np.min(hand.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh):
                            o1_label = i
                            count1 = 1
                            print('o1 found!, label: %d'%total_unique_labels[i])
                            #print(label_file)
                            #print(np.min(hand.compute_point_cloud_distance(filtered_pcd_voxel[i])))
                            o1 = filtered_pcd_voxel[i]

                    #if o1 is defined and o2 not, define it when distance to hand or o1 is smaller than thresh
                    elif(o1 != None and count2 == 0 and i != o1_label):
                        if(np.min(hand.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh
                           or np.min(o1.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh):  
                                o2_label = i
                                count2 = 1
                                print('o2 found!, label: %d'%total_unique_labels[i])
                                o2 = filtered_pcd_voxel[i]

                    #if o1, o2 is defined and o3 not, define it when distance to hand, o1 or o2 is smaller than thresh
                    #o1 is in the frame
                    elif(o2 != None and count3 == 0 and i != o1_label and i != o2_label and len(o2.points) > 0 and len(o1.points) > 0 ):
                        if(np.min(hand.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh
                           or np.min(o1.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh
                           or np.min(o2.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh):
                                o3_label = i
                                count3 = 1
                                print('o3 found!, label: %d'%total_unique_labels[i])
                                o3 = filtered_pcd_voxel[i]

                    #if o1, o2 is defined and o3 not, define it when distance to hand, o1 or o2 is smaller than thresh
                    #o1 is not in the frame            
                    elif(o2 != None and count3 == 0 and i != o1_label and i != o2_label and len(o2.points) > 0 and len(o1.points) == 0):
                        if(np.min(hand.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh
                           or np.min(o2.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh):
                                o3_label = i
                                count3 = 1
                                print('o3 found!, label: %d'%total_unique_labels[i])
                                o3 = filtered_pcd_voxel[i]
                    
                    #if o1, o2 is defined and o3 not, define it when distance to hand, o1 or o2 is smaller than thresh
                    #o2 is not in the frame             
                    elif(o2 != None and count3 == 0 and i != o1_label and i != o2_label and len(o1.points) > 0 and len(o2.points) == 0):
                        if(np.min(hand.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh
                           or np.min(o1.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh):
                                o3_label = i
                                count3 = 1
                                print('o3 found!, label: %d'%total_unique_labels[i])
                                o3 = filtered_pcd_voxel[i]

    #if objects are recognized assign the point cloud to the varaibles o1, o2, o3 to track them during the manipulation
    if count1 == 1:
        o1 = filtered_pcd_voxel[o1_label]
    if count2 == 1:
        o2 = filtered_pcd_voxel[o2_label]
    if count3 == 1:
        o3 = filtered_pcd_voxel[o3_label]
    
    global count_esec
    #if hand is in the frame continue to proceed else return rotation and eSEC table
    if(len(hand.points) > 0):
        #relation == 1 means only TNR
        if (relations == 1):
            #empty array that will be filled in _fillTN_absent function
            add = np.chararray((10,1), itemsize=5)
            
            #find TNR and fill the table
            _fillTN_absent(hand, ground, thresh, add)
            compare_array = np.reshape(ESEC_table[:,count_esec], (-1, 1))
            
            #for the first frame the eSEC table is just this add array
            if frame == 0:
                ESEC_table = add
                
            #in case the add array is equal to the previous eSEC table row ignore it
            elif(np.array_equal(add, compare_array)):
                return rotation, ESEC_table
            
            #otherwise add column to eSEC table
            else:
                ESEC_table = np.column_stack((ESEC_table,add))
                #save image of manipulation in this frame
                plt.imsave("event_images/%s.png"%label_file[-21:-16], label)
                count_esec += 1
                
        #relation == 2 means TNR and SSR
        elif (relations == 2):
            #empty array that will be filled in _fillTN_absent and _fillSSR_2 function
            add = np.chararray((20,1), itemsize=5)
            
            #find TNR and fill the table
            _fillTN_absent(hand, ground, thresh, add)
            
            #find SSR and fill the table
            _fillSSR_4(hand, ground, add)
            compare_array = np.reshape(ESEC_table[:,count_esec], (-1, 1))
            
            #for the first frame the eSEC table is just this add array
            if frame == 0:
                ESEC_table = add
                
            #in case the add array is equal to the previous eSEC table row ignore it
            elif(np.array_equal(add, compare_array)):
                return rotation, ESEC_table
        
            #otherwise add column to eSEC table
            else:
                ESEC_table = np.column_stack((ESEC_table,add))
                #save image of manipulation in this frame
                plt.imsave("event_images/%s.png"%label_file[-21:-16], label)
                count_esec += 1
                
        #relation == 3 means TNR, SSR and DSR
        elif (relations == 3):
            #dont consider first appearance of manipulation
            #save first objects into previous_array
            if (len(hand.points) > 0 and internal_count == 0):
                previous_array = [hand, ground, o1, o2, o3]
                internal_count = 1
                return rotation, ESEC_table
            
            #internal_count is bigger zero after first appearance of manipulation
            elif(internal_count > 0):
                #empty array that will be filled in _fillTN_absent, _fillSSR_2 and _fillDSR function
                add = np.chararray((30,1), itemsize=5)
                
                #find T/N relations and fill the table
                _fillTN_absent(hand, ground, thresh, add)
                
                #find SSR relations and fill the table
                _fillSSR_4(hand, ground, add)
                
                #find DSR relations and fill the table
                _fillDSR(hand, ground, previous_array, thresh, add)
                
                #define the new previous array after calculation of TNR, SSR, DSR
                previous_array = [hand, ground, o1, o2, o3]
                compare_array = np.reshape(ESEC_table[:,count_esec], (-1, 1))
                
                #for the first frame the eSEC table is just this add array
                if internal_count == 1:
                    ESEC_table = add
                    internal_count = 2
                    
                #in case the add array is equal to the previous eSEC table row ignore it    
                elif(np.array_equal(add, compare_array)):
                    return rotation, ESEC_table
                
                #otherwise add column to eSEC table
                else:
                    ESEC_table = np.column_stack((ESEC_table,add))
                    #save image of manipulation in this frame
                    if debug == True:
                        if count_ground > 0:
    #                         mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #                          size=0.6, origin=[0,0,0])
                            #o3d.visualization.draw_geometries([ground, hand, mesh_frame])
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20,10))
                            ax1.set_title("x-y-view")
                            ax2.set_title("y-z-view")
                            #ax1.set_ylim(0.7,1.5)
                            ax2.set_ylim(0.7,1.5)
                            #ax1.set_xlim(-1,1)
                            ax2.set_xlim(-1,1)

                            #ax1.plot(np.array(mesh_frame)[:,0], np.array(mesh_frame)[:,1], ".g", label = 'hand')
                            ax1.plot(np.array(hand.points)[:,0], np.array(hand.points)[:,1], ".g", label = 'hand')
                            ax2.plot(np.array(hand.points)[:,1], np.array(hand.points)[:,2], ".g", label = 'hand')
        #                     ax1.plot(np.array(ground2.points)[:,0], np.array(ground2.points)[:,2], ".r", label = 'ground frame %d (unfiltered)'%frame)
        #                     ax2.plot(np.array(ground2.points)[:,1], np.array(ground2.points)[:,2], ".r", label = 'ground frame %d (unfiltered)'%frame)
                            ax1.plot(np.array(ground.points)[:,0], np.array(ground.points)[:,1], ".k", label = 'ground frame 0 (filtered)')
                            ax2.plot(np.array(ground.points)[:,1], np.array(ground.points)[:,2], ".k", label = 'ground frame 0 (filtered)')

                            if(count1 == 1):
        #                         hand.paint_uniform_color([1, 0, 0])
        #                         o1.paint_uniform_color([0, 1, 0])
        #                         ground.paint_uniform_color([0, 0, 0])
                                # o3d.visualization.draw_geometries([o1, hand])
                                ax1.plot(np.array(o1.points)[:,0], np.array(o1.points)[:,1], ".r", label = 'o1 label:%d'%total_unique_labels[o1_label])
                                ax2.plot(np.array(o1.points)[:,1], np.array(o1.points)[:,2], ".r", label = 'o1 label:%d'%total_unique_labels[o1_label])
                                if(count2 == 1):
                                    ax1.plot(np.array(o2.points)[:,0], np.array(o2.points)[:,1], ".b", label = 'o2 label:%d'%total_unique_labels[o2_label])
                                    ax2.plot(np.array(o2.points)[:,1], np.array(o2.points)[:,2], ".b", label = 'o2 label:%d'%total_unique_labels[o2_label])
                                    if count3 == 1:
                                        ax1.plot(np.array(o3.points)[:,0], np.array(o3.points)[:,1], ".y", label = 'o3 label:%d'%total_unique_labels[o3_label])
                                        ax2.plot(np.array(o3.points)[:,1], np.array(o3.points)[:,2], ".y", label = 'o3 label:%d'%total_unique_labels[o3_label])
                        ax1.legend(loc = 'upper right')
                        ax2.legend(loc = 'upper right')
                        #plt.axis('off')
                        #plt.savefig("debug/%d.png"%frame)
                        plt.savefig("debug_images/%d.png"%(count_esec+1), bbox_inches='tight')
                        plt.clf()  
                    plt.imsave("event_images/%s.png"%label_file[-21:-16], label)
                    count_esec += 1

    
    return rotation, ESEC_table

def _process(pcd_file, label_file, ground_label, hand_label, 
                support_hand, translation, roll, frame, fps, ESEC_table, 
                relations, replace = False, old = [], new = [], ignored_labels = [], cutted_labels = [], thresh = 0.1, debug = False, cython = False, savename = ""):
    '''
    Creates raw eSEC table from a point cloud with corresponding label file. 
    
    Parameters:
        * pcd_file: pcd file to process (.pcd)
        * label_file: label file corresponding to pcd file (.dat)
        * ground_label: label of the ground (int)
        * hand_label: label of the hand (int)
        * translation: translation of the scene, will be retured from function (start with 0)
        * frame: frame of manipulation, start with zero and count (int)
        * ESEC_table: empty chararray 10x1 for T/N; 20x1 for T/N, SSR; 30x1 for T/N, SSR, DSR
        * relations: relations to proceed in the computation 1:T/N; 2:T/N, SSR; 3:T/N, SSR, DSR
        * replace: True if labels should be replaces, False otherwise
        * old: old labels to raplace [int]
        * new: new labels that will replace old labels [int]
        * ignored_labels: labels that will be ignored in this manipulation [int]
        * cutted_labels: label of the cutted object [int]
        * threshold that defines distance for touching
        * cython: if true a self created filter will be used (experimental)
    
    Returns:
        * translation: translation of the scene
        * table: ESEC table
    '''
 
    
    #resize labels to point cloud size
    my_mat = np.zeros((640, 480))
    label = pd.read_csv(label_file,delim_whitespace=True, dtype =np.float64, header=None)
    label = label.to_numpy()

    #get global variables
    global hand_label_inarray, total_unique_labels
    
    if (frame == 0):
        #calculate the translation with the first frame and define ground
        global count_ground, ground
        #rotation = _rotateSceneNewNew(pcd_file, label_file, ground_label)
        ground = _getGroundiNewNew(pcd_file, label_file, ground_label)
        translation, roll = _getTranslation(ground)
        ground = ground.transform(translation)
        ground = ground.transform(roll)
        count_ground = 1
        #find unique labels and replace old with new if replace == True
        unique_labels = np.unique(label)
        total_unique_labels = unique_labels
        if replace == True:
            for values in old:
                total_unique_labels = np.delete(total_unique_labels, np.where(total_unique_labels == old))

        #remove ignored labels from total_unique_labels array
        for values in ignored_labels:
            if values in total_unique_labels:
                total_unique_labels = np.delete(total_unique_labels, np.where(total_unique_labels == values))

        #add hand label if not in total_unique_labels else append total_unique_labels by hand label
        if hand_label not in total_unique_labels:
            if cutted_labels != []:
                total_unique_labels = np.append(total_unique_labels, cutted_labels)
            total_unique_labels = np.append(total_unique_labels, hand_label)
            hand_label_inarray = 0
        else:
            if cutted_labels != []:
                total_unique_labels = np.append(total_unique_labels, cutted_labels)
            hand_label_inarray = np.where(total_unique_labels == hand_label)[0][0]

    #if hand is missing return roation and eSEC table      
    if hand_label not in np.unique(label):
        return translation, roll, ESEC_table
    
    #resize the label file to point cloud size
    label_resized = cv2.resize(label, my_mat.shape, interpolation = cv2.INTER_NEAREST)
    if replace == True:
        label_resized = _replace_labels(label_resized, old, new)
    label_resized = label_resized.flatten()

    #load pcd file with nan points to map lables     
    pcd = o3d.io.read_point_cloud(pcd_file, remove_nan_points=False)
    
    #cast cloud to numpy array and replace nan values with int value -100
    cloud = np.asarray(pcd.points)
    cloud = np.nan_to_num(cloud, nan=-100)

    #add labels to points from the cloud and sort
    pcd_array =  np.column_stack((cloud, label_resized))
    pcd_array_sorted = pcd_array[pcd_array[:, 3].argsort()]
    
    #search for -100 values and delete them
    result_1 = np.where(pcd_array_sorted == -100)
    pcd_array_sorted = np.delete(pcd_array_sorted, np.unique(result_1[0]), 0)

    #calculate the array index for the labels
    index = {}
    i = 0
    empty_index = []
    for value in total_unique_labels:
            #if no hand label is defined previous hand must be the last entry in total_unique_labels
            if(hand_label_inarray == 0):
                hand_label_inarray = np.where(total_unique_labels == hand_label)[0][0]
                
            #find the index of the different objects in the scene
            if float(value) in pcd_array_sorted[:,3]:
                index[i] = np.where(pcd_array_sorted[:,3] == float(value))
            
            #if an value from total_unique_labels is not in the cloud then index is defined as -1
            else:
                index[i] = -1
                empty_index.append(i)
            i += 1

    #if hand is missing in scene return translation and eSEC table
    if hand_label_inarray in empty_index:
        return translation, roll, ESEC_table
    
    #delete the labels from cloud array
    pcd_array_sorted = np.delete(pcd_array_sorted, 3,1)
    
    #rotate the point cloud with calculated translation
    rotated_cloud = o3d.geometry.PointCloud()
    rotated_cloud.points = o3d.utility.Vector3dVector(pcd_array_sorted)
    rotated_cloud = rotated_cloud.transform(translation)
    rotated_cloud = rotated_cloud.transform(roll)

    # o3d.visualization.draw_geometries([rotated_cloud])

    #extract points from rotated cloud
    pcd_array_sorted = np.asarray(rotated_cloud.points)

    #create objects dict
    objects = {}

    j = 0
    #define single objects in the manipulation
    while(j < len(total_unique_labels)):
        #if an label has no point cloud assigned sat its value to -1
        if(j in empty_index):
            objects[j] = -1
            j += 1
            
        #otherwise assign the point cloud of this label to objects dict
        else:
            objects[j] = pcd_array_sorted[index[j]]
            j += 1
    
    #get global counts, lables, previous point cloud and counts
    global count1, count2, count3
    global o1_label, o2_label, o3_label, previous_array, internal_count

    #empty dict for point clouds of single objects, 
    pcd = {}
    filtered_pcd_voxel = {}
    j = 0
    for i in range(len(total_unique_labels)):
        #if label has a point cloud (i.e. objects[i] != -1) proceed
        if not isinstance(objects[i], int):
            #convert arrays to point clouds
            pcd[i] = o3d.geometry.PointCloud()
            pcd[i].points = o3d.utility.Vector3dVector(objects[i])
            # if i == np.where(total_unique_labels == 3)[0][0]:
            #     o3d.io.write_point_cloud("../%s.pcd"%pcd_file.replace("/", "_"), pcd[i])
            #     print("saving")
            if  i != np.where(total_unique_labels == ground_label)[0][0] and i != 0 and len(objects[i]) > 3:
                if cython == True:
                    center = np.array(pcd[i].get_center())
                    hand_center = np.median(objects[i], axis = 0)
                    median_distance = np.median(pdist(objects[i]))
                    filtered_pcd_voxel_array = filter_cython.region_filter_cython(center, objects[i], i == hand_label_inarray, hand_center, hand_center, median_distance)

                    #alpha = 0.02
                    #tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd[i])
                    #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd[i], alpha, tetra_mesh, pt_map)
                    #mesh.compute_vertex_normals()
                    
                    #filtered_pcd_voxel[i] = mesh.sample_points_uniformly(number_of_points=10000)
                    if isinstance(filtered_pcd_voxel_array, int):
                        filtered_pcd_voxel[i] = o3d.geometry.PointCloud()

                    else:
                        filtered_pcd_voxel[i] = o3d.geometry.PointCloud()
                        filtered_pcd_voxel[i].points = o3d.utility.Vector3dVector(filtered_pcd_voxel_array)
                        
                else:
                    #filter objects with statistical filter except ground and label 0 (borders)
                    filtered_pcd_voxel[i], _ = pcd[i].remove_statistical_outlier(nb_neighbors=20, std_ratio=1)
                    if len(filtered_pcd_voxel[i].points)  == 0 and len(objects[i]) < 3:
                        filtered_pcd_voxel[i] = pcd[i]

                    else:
                        filtered_pcd_voxel[i] = pcd[i]
            else:
                filtered_pcd_voxel[i] = pcd[i]
        else:
            filtered_pcd_voxel[i] = o3d.geometry.PointCloud()
            pcd[i] = o3d.geometry.PointCloud()

            
    #define hand variable as hand point cloud
    #hand = pcd[hand_label_inarray]
    hand = filtered_pcd_voxel[hand_label_inarray]
    #print("\n",len(pcd), len(filtered_pcd_voxel))
    # if frame > 200:
    #     o3d.io.write_point_cloud("ownCloud/bowl_unfiltered_%d.pcd"%frame, pcd[3])
    #     o3d.io.write_point_cloud("ownCloud/bowl_filtered_%d.pcd"%frame, filtered_pcd_voxel[3])

    #o3d.visualization.draw_geometries([ground])
    #cutted = filtered_pcd_voxel[int(len(total_unique_labels)-2)]

    #if hand has no points return translation and eSEC table
    if len(hand.points) == 0:
        #print('Hand is misssing 3')
        return translation, roll, ESEC_table
    
    #get global objects 1,2 and 3
    global o1, o2, o3, first_o1, first_o2, first_o3, ao1, ao2, ao3
    
    #get possible combinations of objects, threshold and an empty char 
    list_ = np.arange(1, len(objects))
    combis_two = list(combinations(list_, 2))  

    #define object1, 2 and 3 in this loop
    for i in range(len(total_unique_labels)):
        if support_hand != None:
            #possible o1, o2, o3 candidates must not be hand_label, ground_label, support_hand_label, label 0
            #cloud has to be bigger than zero
            if(i != hand_label_inarray and i != np.where(total_unique_labels == ground_label)[0][0] and i != np.where(total_unique_labels == support_hand)[0][0] and i > 0 and len(filtered_pcd_voxel[i].points) > 0):
                #checkt if clouds have any points and if hand is in this frame
                if(len(filtered_pcd_voxel[i].points) > 0 and len(hand.points) > 0):
                    #if no o1 is defined, define it when distance to hand is smaller than thresh
                    if(count1 == 0 and np.min(hand.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh):
                            o1_label = i
                            count1 = 1
                            print('o1 found!, label: %d'%total_unique_labels[i])
                            #print(label_file)
                            o1 = filtered_pcd_voxel[i]
                            first_o1 = filtered_pcd_voxel[o1_label]

                    #if o1 is defined and o2 not, define it when distance to hand or o1 is smaller than thresh
                    elif(o1 != None and count2 == 0 and i != o1_label):
                        if(np.min(hand.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh
                           or np.min(o1.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh):  
                                o2_label = i
                                count2 = 1
                                print('o2 found!, label: %d'%total_unique_labels[i])
                                o2 = filtered_pcd_voxel[i]
                                first_o2 = filtered_pcd_voxel[o2_label]

                    #if o1, o2 is defined and o3 not, define it when distance to hand, o1 or o2 is smaller than thresh
                    elif(o2 != None and count3 == 0 and i != o1_label and i != o2_label):
                        if(np.min(hand.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh
                           or np.min(o1.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh
                           or np.min(o2.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh):
                                o3_label = i
                                count3 = 1
                                print('o3 found!, label: %d'%total_unique_labels[i])
                                o3 = filtered_pcd_voxel[i]
                                first_o3 = filtered_pcd_voxel[o3_label]
        
        else:
            #possible o1, o2, o3 candidates must not be hand_label, ground_label, label 0
            #cloud has to be bigger than zero
            if(i != hand_label_inarray and i != np.where(total_unique_labels == ground_label)[0][0] and i > 0 and len(filtered_pcd_voxel[i].points) > 0):
                #checkt if clouds have any points and if hand is in this frame
                if(len(filtered_pcd_voxel[i].points) > 0 and len(hand.points) > 0):
                    #if no o1 is defined, define it when distance to hand is smaller than thresh
                    if(count1 == 0 and np.min(hand.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh):
                            o1_label = i
                            count1 = 1
                            print('o1 found!, label: %d'%total_unique_labels[i])
                            #print(label_file)
                            #print(np.min(hand.compute_point_cloud_distance(filtered_pcd_voxel[i])))
                            o1 = filtered_pcd_voxel[i]
                            first_o1 = filtered_pcd_voxel[o1_label]

                    #if o1 is defined and o2 not, define it when distance to hand or o1 is smaller than thresh
                    elif(o1 != None and count2 == 0 and i != o1_label):
                        if(np.min(hand.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh
                           or np.min(o1.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh):  
                                o2_label = i
                                count2 = 1
                                print('o2 found!, label: %d'%total_unique_labels[i])
                                o2 = filtered_pcd_voxel[i]
                                first_o2 = filtered_pcd_voxel[o2_label]

                    #if o1, o2 is defined and o3 not, define it when distance to hand, o1 or o2 is smaller than thresh
                    #o1 is in the frame
                    elif(o2 != None and count3 == 0 and i != o1_label and i != o2_label and len(o2.points) > 0 and len(o1.points) > 0 ):
                        if(np.min(hand.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh
                           or np.min(o1.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh
                           or np.min(o2.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh):
                                o3_label = i
                                count3 = 1
                                print('o3 found!, label: %d'%total_unique_labels[i])
                                o3 = filtered_pcd_voxel[i]
                                first_o3 = filtered_pcd_voxel[o3_label]

                    #if o1, o2 is defined and o3 not, define it when distance to hand, o1 or o2 is smaller than thresh
                    #o1 is not in the frame            
                    elif(o2 != None and count3 == 0 and i != o1_label and i != o2_label and len(o2.points) > 0 and len(o1.points) == 0):
                        if(np.min(hand.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh
                           or np.min(o2.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh):
                                o3_label = i
                                count3 = 1
                                print('o3 found!, label: %d'%total_unique_labels[i])
                                o3 = filtered_pcd_voxel[i]
                                first_o3 = filtered_pcd_voxel[o3_label]
                    
                    #if o1, o2 is defined and o3 not, define it when distance to hand, o1 or o2 is smaller than thresh
                    #o2 is not in the frame             
                    elif(o2 != None and count3 == 0 and i != o1_label and i != o2_label and len(o1.points) > 0 and len(o2.points) == 0):
                        if(np.min(hand.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh
                           or np.min(o1.compute_point_cloud_distance(filtered_pcd_voxel[i])) < thresh):
                                o3_label = i
                                count3 = 1
                                print('o3 found!, label: %d'%total_unique_labels[i])
                                o3 = filtered_pcd_voxel[i]
                                first_o3 = filtered_pcd_voxel[o3_label]

    #if objects are recognized assign the point cloud to the varaibles o1, o2, o3 to track them during the manipulation
    # if count1 == 1:
    #     o1 = pcd[o1_label]
    # if count2 == 1:
    #     o2 = pcd[o2_label]
    # if count3 == 1:
    #     o3 = pcd[o3_label]

    # if count1 == 1:
    #     o1 = filtered_pcd_voxel[o1_label]
    # if count2 == 1:
    #     o2 = filtered_pcd_voxel[o2_label]
    # if count3 == 1:
    #     o3 = filtered_pcd_voxel[o3_label]

    thresh_registration = 0.02
    #thresh_registration = 0.05
    if o1 != None:
        #print("original vol: ",pcd[o1_label].get_axis_aligned_bounding_box().volume(), " new vol: ", o1.get_axis_aligned_bounding_box().volume())

        trans_init = np.asarray([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        if (len(pcd[o1_label].points)) != 0:
            if ao1 == 0:
                reg_p2p1 = o3d.pipelines.registration.registration_icp(
                        first_o1, pcd[o1_label], thresh_registration, trans_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint())
                o1 = first_o1.transform(reg_p2p1.transformation)
                ao1 = 2
            else:
                reg_p2p1 = o3d.pipelines.registration.registration_icp(
                        o1, pcd[o1_label], thresh_registration, trans_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint())
                o1 = o1.transform(reg_p2p1.transformation)
        else:
            o1 = o3d.geometry.PointCloud()

    if o2 != None:
        trans_init = np.asarray([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        if (len(pcd[o2_label].points)) != 0:
            if ao2 == 0:
                # first_o2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=30))
                # pcd[o2_label].estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=30))
                # loss = o3d.pipelines.registration.TukeyLoss(k=0.1)
                # p2l = o3d.pipelines.registration.TransformationEstimationPointToPoint(loss)
                reg_p2p2 = o3d.pipelines.registration.registration_icp(
                        first_o2, pcd[o2_label], thresh_registration, trans_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint())
                o2 = first_o2.transform(reg_p2p2.transformation)
                ao2 = 2
            else:
                # first_o2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=30))
                # pcd[o2_label].estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=30))
                # loss = o3d.pipelines.registration.TukeyLoss(k=0.1)
                # p2l = o3d.pipelines.registration.TransformationEstimationPointToPoint(loss)
                reg_p2p2 = o3d.pipelines.registration.registration_icp(
                        o2, pcd[o2_label], thresh_registration, trans_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint())
                o2 = o2.transform(reg_p2p2.transformation)
        else:
            o2 = o3d.geometry.PointCloud()

    if o3 != None:
        trans_init = np.asarray([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        if (len(pcd[o3_label].points)) != 0:
            if ao3 == 0:
                reg_p2p3 = o3d.pipelines.registration.registration_icp(
                        first_o3, pcd[o3_label], thresh_registration, trans_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint())
                o3 = first_o3.transform(reg_p2p3.transformation)
                ao3 = 2
            else:
                reg_p2p3 = o3d.pipelines.registration.registration_icp(
                        o3, pcd[o3_label], thresh_registration, trans_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint())
                o3 = o3.transform(reg_p2p3.transformation)
        else:
            o3 = o3d.geometry.PointCloud()

    
    global count_esec
    #if hand is in the frame continue to proceed else return translation and eSEC table
    if(len(hand.points) > 0):
        #relation == 1 means only TNR
        if (relations == 1):
            #empty array that will be filled in _fillTN_absent function
            add = np.chararray((10,1), itemsize=5)
            
            #find TNR and fill the table
            _fillTN_absent(hand, ground, thresh, add)
            compare_array = np.reshape(ESEC_table[:,count_esec], (-1, 1))
            
            #for the first frame the eSEC table is just this add array
            if frame == 0:
                ESEC_table = add
                
            #in case the add array is equal to the previous eSEC table row ignore it
            elif(np.array_equal(add, compare_array)):
                return translation, roll, ESEC_table
            
            #otherwise add column to eSEC table
            else:
                ESEC_table = np.column_stack((ESEC_table,add))
                #save image of manipulation in this frame
                plt.imsave("event_images/%s.png"%label_file[-21:-16], label)
                count_esec += 1
                
        #relation == 2 means TNR and SSR
        elif (relations == 2):
            #empty array that will be filled in _fillTN_absent and _fillSSR_2 function
            add = np.chararray((20,1), itemsize=5)
            
            #find TNR and fill the table
            _fillTN_absent(hand, ground, thresh, add)
            
            #find SSR and fill the table
            _fillSSR_4(hand, ground, add)
            compare_array = np.reshape(ESEC_table[:,count_esec], (-1, 1))
            
            #for the first frame the eSEC table is just this add array
            if frame == 0:
                ESEC_table = add
                
            #in case the add array is equal to the previous eSEC table row ignore it
            elif(np.array_equal(add, compare_array)):
                return translation, roll, ESEC_table
        
            #otherwise add column to eSEC table
            else:
                ESEC_table = np.column_stack((ESEC_table,add))
                #save image of manipulation in this frame
                plt.imsave("event_images/%s.png"%label_file[-21:-16], label)
                count_esec += 1
                
        #relation == 3 means TNR, SSR and DSR
        elif (relations == 3):
            #dont consider first appearance of manipulation
            #save first objects into previous_array
            if (len(hand.points) > 0 and internal_count == 0):
                previous_array = [hand, ground, copy.deepcopy(o1), copy.deepcopy(o2), copy.deepcopy(o3)]
                internal_count = 1
                return translation, roll, ESEC_table
            
            #internal_count is bigger zero after first appearance of manipulation
            elif(internal_count > 0):
                #empty array that will be filled in _fillTN_absent, _fillSSR_2 and _fillDSR function
                add = np.chararray((30,1), itemsize=5)
                
                #find T/N relations and fill the table
                _fillTN_absent(hand, ground, thresh, add)
                
                #find SSR relations and fill the table
                _fillSSR_4(hand, ground, add)
                
                #find DSR relations and fill the table
                _fillDSR_new(hand, ground, previous_array, thresh, add)

                # if previous_array[2] != None:
                #     o3d.io.write_point_cloud("before_spoon_%d.pcd"%frame,previous_array[2])
                #     o3d.io.write_point_cloud("after_spoon%d.pcd"%frame,o1)

                #define the new previous array after calculation of TNR, SSR, DSR
                previous_array = [hand, ground, copy.deepcopy(o1), copy.deepcopy(o2), copy.deepcopy(o3)]
                compare_array = np.reshape(ESEC_table[:,count_esec], (-1, 1))
                
                #for the first frame the eSEC table is just this add array
                if internal_count == 1:
                    ESEC_table = add
                    internal_count = 2
                    
                #in case the add array is equal to the previous eSEC table row ignore it    
                elif(np.array_equal(add, compare_array)):
                    return translation, roll, ESEC_table
                
                #otherwise add column to eSEC table
                else:
                    ESEC_table = np.column_stack((ESEC_table,add))
                    #save image of manipulation in this frame
                    if debug == True:
                        if count_ground > 0:
                            # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                            # size=0.6, origin=[0,0,0])
                            # o3d.visualization.draw_geometries([ground, hand, mesh_frame])
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20,10))
                            ax1.set_title("x-y-view")
                            ax2.set_title("y-z-view")
                            ax1.set_ylim(-0.8, 0.6)
                            ax2.set_ylim(-1.5,0.5)
                            #ax1.set_xlim(-1,1)
                            ax2.set_xlim(-1,0.75)
                            ax1.set_ylabel("y")
                            ax1.set_xlabel("x")
                            ax2.set_ylabel("z")
                            ax2.set_xlabel("y")

                            #ax1.plot(np.array(mesh_frame)[:,0], np.array(mesh_frame)[:,1], ".g", label = 'hand')
                            hand_box = hand.get_axis_aligned_bounding_box()
                            points_hand = np.asarray(hand_box.get_box_points())
                            hand_max_x, hand_max_y, hand_max_z = np.max(points_hand[:,0]), np.max(points_hand[:,1]), np.max(points_hand[:,2])
                            hand_min_x, hand_min_y, hand_min_z = np.min(points_hand[:,0]), np.min(points_hand[:,1]), np.min(points_hand[:,2])

                            ground_box = ground.get_axis_aligned_bounding_box()
                            points_ground = np.asarray(ground_box.get_box_points())
                            ground_max_x, ground_max_y, ground_max_z = np.max(points_ground[:,0]), np.max(points_ground[:,1]), np.max(points_ground[:,2])
                            ground_min_x, ground_min_y, ground_min_z = np.min(points_ground[:,0]), np.min(points_ground[:,1]), np.min(points_ground[:,2])

                            ax1.plot(np.array(hand.points)[:,0], np.array(hand.points)[:,1], ".g", label = 'hand')
                            
                            # ax1.plot(np.array(hand.get_center())[0], np.array(hand.get_center())[1], ".r", label = 'hand')
                            # ax1.plot(np.array(ground_box.get_center())[0], np.array(ground_box.get_center())[1], ".g", label = 'hand')
                            # # ax2.plot(np.array(hand.get_center())[1], np.array(hand.get_center())[2], ".r", label = 'hand')
                            # ax2.plot(np.array(ground_box.get_center())[1], np.array(ground_box.get_center())[2], ".g", label = 'hand')
                            
                            ax1.plot((hand_max_x, hand_max_x), (hand_min_y,hand_max_y), "-g")
                            ax1.plot((hand_min_x, hand_min_x), (hand_max_y,hand_min_y), "-g")

                            ax1.plot((hand_max_x, hand_min_x), (hand_max_y,hand_max_y), "-g")
                            ax1.plot((hand_min_x, hand_max_x), (hand_min_y,hand_min_y), "-g")

                            ax1.plot(np.array(ground.points)[:,0], np.array(ground.points)[:,1], ".k", label = 'ground frame 0 (filtered)')

                            ax1.plot((ground_max_x, ground_max_x), (ground_min_y,ground_max_y), "-k")
                            ax1.plot((ground_min_x, ground_min_x), (ground_max_y,ground_min_y), "-k")

                            ax1.plot((ground_max_x, ground_min_x), (ground_max_y,ground_max_y), "-k")
                            ax1.plot((ground_min_x, ground_max_x), (ground_min_y,ground_min_y), "-k")
                            
                            ax2.text(-0.9, 0.4, pcd_file)
                            #------------------------------------------------------------------
                            #print relations to plot
                            ax2.text(-0.9, 0.35, "TNR")
                            ax2.text(-0.9, 0.32, "H,o1  : %s"%add[0][0].decode("utf-8") )
                            ax2.text(-0.9, 0.28, "H,o2  : %s"%add[1][0].decode("utf-8") )
                            ax2.text(-0.9, 0.24, "H,o3  : %s"%add[2][0].decode("utf-8") )
                            ax2.text(-0.9, 0.20, "o1,o2 : %s"%add[4][0].decode("utf-8") )
                            ax2.text(-0.9, 0.16, "o1,o3 : %s"%add[5][0].decode("utf-8") )
                            ax2.text(-0.9, 0.12, "o1,G  : %s"%add[6][0].decode("utf-8") )
                            ax2.text(-0.9, 0.08, "o2,o3 : %s"%add[7][0].decode("utf-8") )
                            ax2.text(-0.9, 0.04, "o2,G  : %s"%add[8][0].decode("utf-8") )

                            ax2.text(-0.6, 0.35, "SSR")
                            ax2.text(-0.6, 0.32, "H,o1  : %s"%add[10][0].decode("utf-8") )
                            ax2.text(-0.6, 0.28, "H,o2  : %s"%add[11][0].decode("utf-8") )
                            ax2.text(-0.6, 0.24, "H,o3  : %s"%add[12][0].decode("utf-8") )
                            ax2.text(-0.6, 0.20, "o1,o2 : %s"%add[14][0].decode("utf-8") )
                            ax2.text(-0.6, 0.16, "o1,o3 : %s"%add[15][0].decode("utf-8") )
                            ax2.text(-0.6, 0.12, "o1,G  : %s"%add[16][0].decode("utf-8") )
                            ax2.text(-0.6, 0.08, "o2,o3 : %s"%add[17][0].decode("utf-8") )
                            ax2.text(-0.6, 0.04, "o2,G  : %s"%add[18][0].decode("utf-8") )

                            ax2.text(-0.3, 0.35, "DSR")
                            ax2.text(-0.3, 0.32, "H,o1  : %s"%add[20][0].decode("utf-8") )
                            ax2.text(-0.3, 0.28, "H,o2  : %s"%add[21][0].decode("utf-8") )
                            ax2.text(-0.3, 0.24, "H,o3  : %s"%add[22][0].decode("utf-8") )
                            ax2.text(-0.3, 0.20, "o1,o2 : %s"%add[24][0].decode("utf-8") )
                            ax2.text(-0.3, 0.16, "o1,o3 : %s"%add[25][0].decode("utf-8") )
                            ax2.text(-0.3, 0.12, "o1,G  : %s"%add[26][0].decode("utf-8") )
                            ax2.text(-0.3, 0.08, "o2,o3 : %s"%add[27][0].decode("utf-8") )
                            ax2.text(-0.3, 0.04, "o2,G  : %s"%add[28][0].decode("utf-8") )
                            #------------------------------------------------------------------

                            ax2.plot(np.array(hand.points)[:,1], np.array(hand.points)[:,2], ".g", label = 'hand')

                            ax2.plot((hand_max_y, hand_max_y), (hand_min_z,hand_max_z), "-g")
                            ax2.plot((hand_min_y, hand_min_y), (hand_max_z,hand_min_z), "-g")

                            ax2.plot((hand_max_y, hand_min_y), (hand_max_z,hand_max_z), "-g")
                            ax2.plot((hand_min_y, hand_max_y), (hand_min_z,hand_min_z), "-g")

                            ax2.plot(np.array(ground.points)[:,1], np.array(ground.points)[:,2], ".k", label = 'ground frame 0 (filtered)')

                            ax2.plot((ground_max_y, ground_max_y), (ground_min_z,ground_max_z), "-k")
                            ax2.plot((ground_min_y, ground_min_y), (ground_max_z,ground_min_z), "-k")

                            ax2.plot((ground_max_y, ground_min_y), (ground_max_z,ground_max_z), "-k")
                            ax2.plot((ground_min_y, ground_max_y), (ground_min_z,ground_min_z), "-k")

                            if(count1 == 1):
                                # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                            #  size=0.6, origin=[0,0,0])
        #                         hand.paint_uniform_color([1, 0, 0])
        #                         o1.paint_uniform_color([0, 1, 0])
        #                         ground.paint_uniform_color([0, 0, 0])
                                # o3d.visualization.draw_geometries([o1, hand,ground, mesh_frame])
                                # o1 = pcd[o1_label]
                                o1_box = o1.get_axis_aligned_bounding_box()
                                points_o1 = np.asarray(o1_box.get_box_points())
                                o1_max_x, o1_max_y, o1_max_z = np.max(points_o1[:,0]), np.max(points_o1[:,1]), np.max(points_o1[:,2])
                                o1_min_x, o1_min_y, o1_min_z = np.min(points_o1[:,0]), np.min(points_o1[:,1]), np.min(points_o1[:,2])
                           
                                ax1.plot(np.array(o1.points)[:,0], np.array(o1.points)[:,1], ".r", label = 'o1 label:%d'%total_unique_labels[o1_label])

                                # ax1.plot(np.array(o1.points)[:,0], np.array(o1.points)[:,1], ".g", label = 'hand')
                                # ax1.plot(np.array(o1.get_center())[0], np.array(o1.get_center())[1], ".r", label = 'hand')
                                # ax1.plot(np.array(o1_box.get_center())[0], np.array(o1_box.get_center())[1], ".g", label = 'hand')
                                # ax2.plot(np.array(o1.get_center())[1], np.array(o1.get_center())[2], ".r", label = 'hand')
                                # ax2.plot(np.array(o1_box.get_center())[1], np.array(o1_box.get_center())[2], ".g", label = 'hand')
                                # ax1.plot(np.array(ground_box.get_center())[0], np.array(ground_box.get_center())[1], ".g", label = 'hand')
                                # ax2.plot(np.array(hand.get_center())[1], np.array(hand.get_center())[2], ".r", label = 'hand')
                                # ax2.plot(np.array(ground_box.get_center())[1], np.array(ground_box.get_center())[2], ".g", label = 'hand')
    
                                ax1.plot((o1_max_x, o1_max_x), (o1_min_y,o1_max_y), "-r")
                                ax1.plot((o1_min_x, o1_min_x), (o1_max_y,o1_min_y), "-r")

                                ax1.plot((o1_max_x, o1_min_x), (o1_max_y,o1_max_y), "-r")
                                ax1.plot((o1_min_x, o1_max_x), (o1_min_y,o1_min_y), "-r")
                          
                                ax2.plot(np.array(o1.points)[:,1], np.array(o1.points)[:,2], ".r", label = 'o1 label:%d'%total_unique_labels[o1_label])
                                
                                ax2.plot((o1_max_y, o1_max_y), (o1_min_z,o1_max_z), "-r")
                                ax2.plot((o1_min_y, o1_min_y), (o1_max_z,o1_min_z), "-r")

                                ax2.plot((o1_max_y, o1_min_y), (o1_max_z,o1_max_z), "-r")
                                ax2.plot((o1_min_y, o1_max_y), (o1_min_z,o1_min_z), "-r")
                                
                                if(count2 == 1):
                                   
                                    o2_box = o2.get_axis_aligned_bounding_box()
                                    points_o2 = np.asarray(o2_box.get_box_points())
                                    o2_max_x, o2_max_y, o2_max_z = np.max(points_o2[:,0]), np.max(points_o2[:,1]), np.max(points_o2[:,2])
                                    o2_min_x, o2_min_y, o2_min_z = np.min(points_o2[:,0]), np.min(points_o2[:,1]), np.min(points_o2[:,2])
                                    
                                    ax1.plot(np.array(o2.points)[:,0], np.array(o2.points)[:,1], ".b", label = 'o2 label:%d'%total_unique_labels[o2_label])
                                    
                                    ax1.plot((o2_max_x, o2_max_x), (o2_min_y,o2_max_y), "-b")
                                    ax1.plot((o2_min_x, o2_min_x), (o2_max_y,o2_min_y), "-b")

                                    ax1.plot((o2_max_x, o2_min_x), (o2_max_y,o2_max_y), "-b")
                                    ax1.plot((o2_min_x, o2_max_x), (o2_min_y,o2_min_y), "-b")


                                    #ax2.scatter(o2_min_y, o2_max_x, "y")
                                    ax2.plot(np.array(o2.points)[:,1], np.array(o2.points)[:,2], ".b", label = 'o2 label:%d'%total_unique_labels[o2_label])

                                    ax2.plot((o2_max_y, o2_max_y), (o2_min_z,o2_max_z), "-b")
                                    ax2.plot((o2_min_y, o2_min_y), (o2_max_z,o2_min_z), "-b")

                                    ax2.plot((o2_max_y, o2_min_y), (o2_max_z,o2_max_z), "-b")
                                    ax2.plot((o2_min_y, o2_max_y), (o2_min_z,o2_min_z), "-b")
                                    
                                    if count3 == 1:
                                        
                                        o3_box = o3.get_axis_aligned_bounding_box()
                                        points_o3 = np.asarray(o3_box.get_box_points())
                                        o3_max_x, o3_max_y, o3_max_z = np.max(points_o3[:,0]), np.max(points_o3[:,1]), np.max(points_o3[:,2])
                                        o3_min_x, o3_min_y, o3_min_z = np.min(points_o3[:,0]), np.min(points_o3[:,1]), np.min(points_o3[:,2])
                                        ax1.plot(np.array(o3.points)[:,0], np.array(o3.points)[:,1], ".y", label = 'o3 label:%d'%total_unique_labels[o3_label])
                                        
                                        ax1.plot((o3_max_x, o3_max_x), (o3_min_y,o3_max_y), "-y")
                                        ax1.plot((o3_min_x, o3_min_x), (o3_max_y,o3_min_y), "-y")

                                        ax1.plot((o3_max_x, o3_min_x), (o3_max_y,o3_max_y), "-y")
                                        ax1.plot((o3_min_x, o3_max_x), (o3_min_y,o3_min_y), "-y")
                                        
                                        ax2.plot(np.array(o3.points)[:,1], np.array(o3.points)[:,2], ".y", label = 'o3 label:%d'%total_unique_labels[o3_label])

                                        ax2.plot((o3_max_y, o3_max_y), (o3_min_z,o3_max_z), "-y")
                                        ax2.plot((o3_min_y, o3_min_y), (o3_max_z,o3_min_z), "-y")

                                        ax2.plot((o3_max_y, o3_min_y), (o3_max_z,o3_max_z), "-y")
                                        ax2.plot((o3_min_y, o3_max_y), (o3_min_z,o3_min_z), "-y")
                        
                        ax1.legend(loc = 'upper right')
                        ax2.legend(loc = 'upper right')
                        #plt.axis('off')
                        #plt.savefig("debug/%d.png"%frame)
                        plt.savefig("debug_images/"+savename+"/%d.png"%(count_esec+1), bbox_inches='tight')
                        plt.clf()  
                        plt.imsave("event_images/"+savename+"/%s.png"%label_file[-21:-16], label)
                    count_esec += 1

    
    return translation, roll, ESEC_table

def analyse_maniac_manipulation(pcl_path, label_path, ground_label, hand_label, support_hand, relations,
                                replace, old, new, ignored_labels, thresh, cutted_labels = [],  debug = False, cython = False, savename = ""):
    '''
    Analyses a complete manipulation from the MANIAC dataset. Therefore, it needs the path
    of the folder that contains all the .pcd files (pcl_path) and the label files(label_path). 
    The other parameters are listed below.
    This functions returns and saves the calculated e2SEC matrix.
    
    Parameters:
        * pcl_path: path to pcl files (string)
        * label_path: path to label files (string)
        * ground_label: label of the ground (int)
        * hand_label: label of the hand (int)
        * support_hand: label of the support hand (int)
        * relations: relations to proceed in the computation 1:T/N; 2:T/N, SSR; 3:T/N, SSR, DSR (int)
        * replace: True if labels should be replaces, False otherwise (bool)
        * old: old labels to raplace [int]
        * new: new labels that will replace old labels [int]
        * ignored_labels: labels that will be ignored in this manipulation [int]
        * thresh: threshold that defines distance for touching (float)
        * cython: if true a self created filter will be used (experimental)
        * savename: name of the saved e2SEC file
    
    Returns:
        e2SEC matrix in the current folder as "e2sec_matrix.npy"
    '''

    #create folder for event images in case it does not exist yet
    if debug == True:
        if not os.path.exists("debug_images/"+savename+"/"):
            os.makedirs("debug_images/"+savename)
        else:
            shutil.rmtree("debug_images/"+savename+"/")
            os.makedirs("debug_images/"+savename+"/")
        if not os.path.exists("event_images/"+savename+"/"):
            os.makedirs("event_images/"+savename)
        else:
            shutil.rmtree("event_images/"+savename+"/")
            os.makedirs("event_images/"+savename+"/")

    ##define global variables
    global o1, o2, o3, count1, count2, count3, o1_label, o2_label, o3_label, previous_array, internal_count, total_unique_labels, hand_label_inarray, count_esec, ground, count_ground, absent_o1, absent_o2, absent_o3
    #o1,o2,o3 are the three main objects of the manipulation
    o1 = None
    o2 = None
    o3 = None
    #count variables esnure o1, o2, o3 are only once defined
    #count1 is assigned to object1 and so on
    count1, count2, count3 = 0, 0, 0

    #define variables for labels of o1, o2 and o3
    o1_label = 0
    o2_label = 0
    o3_label = 0

    #define previous frame for DSR
    previous_array = None

    #define variables for the internal of the algorithm
    internal_count = 0
    total_unique_labels = 0
    hand_label_inarray = -1
    count_esec = 0
    ground = 0
    count_ground = 0
    absent_o1, absent_o2, absent_03 = False, False, False
    
    if relations == 1:
        table = np.chararray((10,1), itemsize=5)
    if relations == 2:
        table = np.chararray((20,1), itemsize=5)
    if relations == 3:
        table = np.chararray((30,1), itemsize=5)
    
    #define first column of tables as "-"
    table[:] = '-'
    translation = 0
    roll = 0
    i = 0

    #define fps
    fps = 10
    frames = int(30/fps)
    relations = relations
   
    for file in progressbar.progressbar(sorted(os.listdir(pcl_path))):
        if(i%frames == 0):
            #print(file)
            translation, roll, table = _process(pcl_path+file[0:-7]+"_pc.pcd",
                               label_path+file[0:-7]+"_left-labels.dat",
                               ground_label = ground_label ,hand_label = hand_label, support_hand = support_hand, translation = translation, roll = roll, frame = i, fps=fps,
                                ESEC_table = table, relations = relations,
                                replace = replace, old = old, new = new, 
                                ignored_labels = ignored_labels,
                                thresh = thresh, cutted_labels = cutted_labels, debug = debug, cython = cython, savename = savename)
        i+=1
        
    e2sec, esec = esec_to_e2sec(table,relations)
    np.save("e2sec_%s.npy"%savename.replace("/", "_"),e2sec)
    plt.close('all')