cimport numpy as np
import numpy as np
import random
from libc.math cimport sqrt

cpdef double distance_cython(double[:] p1, double[:] p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

cpdef double[:,::1] region_filter_cython(double[:] center, double[:,::1] pcd, int hand_cloud):

    #define dist dummy for calculation
    cdef float dist = 50000
    cdef int length = pcd.shape[0]
    cdef int i
    cdef double[:] first_point 
    cdef float dist_temp 

    #find the nearest point in the point cloud from middle point
    for i in range(length):
        dist_temp = distance_cython(pcd[i], center)
        if dist_temp < dist:
            dist = dist_temp
            first_point = pcd[i]
    
    #calculate mean distance between points
    cdef float mean_distance = 0
    for i in range(length-1):
        mean_distance += distance_cython(pcd[i], pcd[i-1])

    if i != 0.0:
        mean_distance /= i
    else:
        return np.zeros((1, 3))

    cdef double[:] next_point
    cdef int test = -1

    dicti = {}
    dicti[0] = first_point
    cdef int count = 1
    indices = []
    cdef int abort = 0
    cdef int temp = 0
    cdef int length_deleted = 0
    if hand_cloud:
        mean_distance = 10 * mean_distance
    else:
        mean_distance = 2 * mean_distance
    while (abort < 30):
        indices = []
        temp = len(pcd)
        length_deleted = pcd.shape[0]
        for i in range(length_deleted):
            if test == -1:
                if distance_cython(pcd[i], first_point) <= mean_distance:
                    dicti[count] = pcd[i]
                    indices.append(i)
                    count += 1

            else: 
                if distance_cython(pcd[i], next_point) <= mean_distance:
                    dicti[count] = pcd[i]
                    indices.append(i)
                    count += 1
        test = 1        
        next_point = random.choice(list(dicti.values()))
        pcd = np.delete(pcd, indices, axis = 0)
        if temp == len(pcd):
            abort += 1
        if temp != len(pcd):
            abort = 0   

    length_2 = len(dicti)
    new_cloud = np.zeros((length_2, 3))
    
    #cdef int index = 0
    for key, value in dicti.items():
        new_cloud[key] = np.array(value)
 

    return new_cloud
