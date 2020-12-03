cimport numpy as np
import numpy as np
from libc.math cimport sqrt

cpdef double distance_cython(double[:] p1, double[:] p2):
	#calculate distance between points in 3d
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

cpdef region_filter_cython(double[:] center, double[:,::1] pcd):
	
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
    for i in range(length):
        if (distance_cython(pcd[i], first_point)) < 1.0:
            mean_distance += distance_cython(pcd[i], first_point)
#        mean_distance += distance_cython(pcd[i], center)
    if i == 0:
        return -1
    mean_distance /= i

    cdef double[:] next_point
    cdef int test = -1

    dicti = {}
    dicti[0] = first_point
    count = 1

    for i in range(length):
        if test == -1:
            dist_temp = distance_cython(pcd[i], first_point)
        else:
            dist_temp = distance_cython(pcd[i], next_point)
        if dist_temp <= mean_distance:
            test = 1
            next_point = pcd[i]
            dicti[count] = next_point
            count += 1
            i = 0
    length_2 = len(dicti)
    new_cloud = np.zeros((length_2, 3))
    
    cdef int index = 0
    for key, value in dicti.items():
        new_cloud[key] = np.array(value)
        #index += 1

    return new_cloud
