#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2020-2021

@author: Tobias Strübing
"""
import tabula
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import copy
from itertools import combinations
import os
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from scipy.spatial import distance
import seaborn as sn
from random import randrange
from sklearn.model_selection import ShuffleSplit
import itertools

def readPDF(PDFpath):
	'''
	Read PDF in a specific format and outputs a dict of the eSEC matrices. 
	There has to be one eSEC matrix per page with the relations on the left side and without titles in the PDF file.
	
	Parameters:
		* PDFpath: path of the PDF file (PDF)
		
	Returns:
		python dictionary with the eSEC matrices
	'''
	
	pdf_path = PDFpath
	#read PDF
	df = tabula.read_pdf(pdf_path, pages='all', pandas_options={'header': None})
	liste_array = {}
	
	#transform PDF to numpy
	for i in range(len(df)):
		liste_array[i] = df[i].loc[:, 1:].to_numpy()
		
	##uppercase every char
	for j in range(len(liste_array)):
		for i in range(liste_array[j].shape[0]):
			for k in range(liste_array[j].shape[1]):
				liste_array[j][i][k] = liste_array[j][i][k].upper()
				
	return liste_array

def plotRowRanking(esec_dict, savefig = False, e2sec = False):
	"""
	Plots the ranking of importance from the different rows of all eSEC manipulations in the eSEC dictionary.
	This dictionary can be obtained from function readPDF().
	
	Parameters:
		* esec_dict: dictionary with eSEC matrices (dict)
		* savefig: parameter if figure needs to be saved (bool)
		* e2sec: is true if the input dict consists of e2SEC matrices (bool)
    """

	matrix = []
	k = 0
	#in this loop the matrix array is filled with dissimilarity values from two compared manipulations
	for j in range(1, len(esec_dict)+1):
		k += 1
		for i in range(k, len(esec_dict)+1):
			if(j != i):
				man_1 = j
				man_2 = i
				matrix = np.append(matrix,_dissimilarity_array_new(man_1, man_2, esec_dict))
	
	#reshape the array so that every row corresponds to one comparison between two different manipulations
	matrix = np.reshape(matrix, (int((len(esec_dict)*(len(esec_dict)-1))/2), esec_dict[0].shape[0]))
	place = np.argsort(matrix, axis = 1)

	#rank the different values 
	matrix_ranked = np.zeros_like(matrix)
	for j in range(matrix.shape[0]):
		rank = 1
		for i in range(len(matrix[j])):
			if(i>0):
				if(matrix[j][place[j][i]] != matrix[j][place[j][i - 1]]):
					rank += 1
					matrix_ranked[j][place[j][i]] = rank
				else:
					matrix_ranked[j][place[j][i]] = rank
			else:
				matrix_ranked[j][place[j][i]] = rank

	#calculate mean and median from the ranks
	matrix_ranked_mean = np.mean(matrix_ranked, axis = 0)
	matrix_ranked_median = np.median(matrix_ranked, axis = 0)

	#plot the ranks of the rows
	if e2sec == False:
		x = ['H, 1', 'H, 2', 'H, 3', \
		 'H, G', "1, 2", '1,3 ',\
		 '1, G', '2, 3', '2, G', '3, G']
		fig, axs = plt.subplots(1,3, figsize = (30,5))
		axs[0].bar(x, matrix_ranked_mean[0:10], color='r')
		axs[0].set_title("SEC", fontsize=32)
		axs[0].set_xlabel("Row",  fontsize=25)
		axs[0].tick_params(axis='x', labelsize=15)
		axs[0].tick_params(axis='y', labelsize=15)
		axs[0].set_ylabel("Rank (mean)",  fontsize=25)

		axs[1].bar(x, matrix_ranked_mean[10:20], color = 'b')
		axs[1].set_title("SSR", fontsize=32)
		axs[1].set_xlabel("Row",  fontsize=25)
		axs[1].tick_params(axis='x', labelsize=15)
		axs[1].tick_params(axis='y', labelsize=15)

		axs[2].bar(x, matrix_ranked_mean[20:30], color = 'g')
		axs[2].set_title("DSR", fontsize=32)
		axs[2].set_xlabel("Row",  fontsize=25)
		axs[2].tick_params(axis='x', labelsize=15)
		axs[2].tick_params(axis='y', labelsize=15)
		if(savefig == True):
			plt.savefig("ranked_rows.png", bbox_inches='tight')
		plt.show()
	
	if e2sec == True:
		x = ['H, 1', 'H, 2', 'H, 3', \
		  "1, 2", '1,3 ',\
		 '1, G', '2, 3', '2, G']
		fig, axs = plt.subplots(1,3, figsize = (30,5))
		axs[0].bar(x, matrix_ranked_mean[0:8], color='r')
		axs[0].set_title("SEC", fontsize=32)
		axs[0].set_xlabel("Row",  fontsize=25)
		axs[0].tick_params(axis='x', labelsize=15)
		axs[0].tick_params(axis='y', labelsize=15)
		axs[0].set_ylabel("Rank (mean)",  fontsize=25)

		axs[1].bar(x, matrix_ranked_mean[8:16], color = 'b')
		axs[1].set_title("SSR", fontsize=32)
		axs[1].set_xlabel("Row",  fontsize=25)
		axs[1].tick_params(axis='x', labelsize=15)
		axs[1].tick_params(axis='y', labelsize=15)

		axs[2].bar(x, matrix_ranked_mean[16:24], color = 'g')
		axs[2].set_title("DSR", fontsize=32)
		axs[2].set_xlabel("Row",  fontsize=25)
		axs[2].tick_params(axis='x', labelsize=15)
		axs[2].tick_params(axis='y', labelsize=15)
		if(savefig == True):
			plt.savefig("ranked_rows.png", bbox_inches='tight')
		plt.show()
	
def removeRows(esec_dict, row):
	'''
	Removes the rows assigned in the parameter from all the eSEC matrices.
	
	Parameters:
		* esec_dict: dictionary with the eSEC matrices (dict)
		* row: indicates which rows will be deleted (int, tuple)
	
	Returns:
		dictionary with removed rows.
	'''
	#copy the input disc
	new_dict = copy.deepcopy(esec_dict)

	#dependent of the numbers in parameter "row" remove rows
	if (isinstance(row, int) and not(isinstance(row, tuple))):
		for i in range(len(new_dict)):
			new_dict[i] = _delete_rows(row, new_dict[i])
	elif (len(row) == 2):
		for i in range(len(new_dict)):
			new_dict[i] = _delete_rows_two(row, new_dict[i])
	elif (len(row) == 3):
		for i in range(len(new_dict)):
			new_dict[i] = _delete_rows_three(row, new_dict[i])
	elif (len(row) == 4):
		for i in range(len(new_dict)):
			new_dict[i] = _delete_rows_four(row, new_dict[i])
	elif (len(row) == 5):
		for i in range(len(new_dict)):
			new_dict[i] = _delete_rows_five(row, new_dict[i])
			
	return new_dict
	
def removeCobinationRowsSave(esec_dict, rows = [3, 2, 5, 7, 9]):
	'''
	Removes the rows from the input in every combination and saves it in the folder "array". 
	First it removes single rows and then combinations of tuple, triple, quadruple and quintuple rows. 
	Rows can have a maximum of 5 values.
	If rows == None the Dissimilarity matrix is calculated without removing rows and output is saved as d_shaped.npy.
	example:	rows = [1,2,3] 
				first step: remove single rows 1, 2, 3

				second step: remove combinations of two 1/2, 1/3, 2/3 

				third step: remove combinations of three 1/2/3
	
	Parameters:
		* esec_dict: dictionary with the eSEC matrices (dict)
		* rows: rows that will be removed (array) 
	
	Returns:
		folder structure with .npy files or one .npy file if rows == None
	'''

	if rows == None:
		liste_array = copy.deepcopy(esec_dict)

		#save the dissimilarity matrix as "d_shaped.npy"
		np.save("d_shaped", _calc_D_shaped(_make_triple(liste_array)))
	
	if rows != None:
		#create folder structure if not existent
		if not(os.path.exists("arrays/single")):
			os.makedirs("arrays/single/")
			os.mkdir("arrays/nothing_removed/")
			os.mkdir("arrays/tuple/")
			os.mkdir("arrays/triple/")
			os.mkdir("arrays/quadruple/")
			os.mkdir("arrays/quintuple/")

		liste_array = copy.deepcopy(esec_dict)
		np.save("arrays/nothing_removed/nothing_removed.npy", _calc_D_shaped(_make_triple(liste_array)))
		#remove all combinations of single rows and saves the array as .npy files in array/single
		for value in rows:
			array = {}
			for i in range(len(liste_array)):
				array[i] = _delete_rows(value, liste_array[i])
			np.save("arrays/single/matrix_removed_%d"%value, _calc_D_shaped(_make_triple(array)))
		
		combis_two = list(combinations(rows, 2))
		#remove all combinations of two rows and saves the array as .npy files in array/tuple
		for values in combis_two:
			array = {}
			for i in range(len(liste_array)):
				array[i] = _delete_rows_two(values, liste_array[i])
			np.save("arrays/tuple/matrix_removed_%d,%d"%(values[0], values[1]), _calc_D_shaped(_make_triple(array)))
		
		#remove all combinations of three rows and saves the array as .npy files in array/triple
		combis_three = list(combinations(rows, 3))
		for values in combis_three:
			array = {}
			for i in range(len(liste_array)):
				array[i] = _delete_rows_three(values, liste_array[i])
			np.save("arrays/triple/matrix_removed_%d,%d,%d"%(values[0], values[1], values[2]), _calc_D_shaped(_make_triple(array)))
		
		#remove all combinations of four rows and saves the array as .npy files in array/quadruple
		combis_four = list(combinations(rows, 4))
		for values in combis_four:
			array = {}
			for i in range(len(liste_array)):
				array[i] = _delete_rows_four(values, liste_array[i])
			np.save("arrays/quadruple/matrix_removed_%d,%d,%d,%d"%(values[0], values[1], values[2], values[3]), _calc_D_shaped(_make_triple(array)))
		
		#remove all combinations of five rows and saves the array as .npy files in array/quintuple
		combis_five = list(combinations(rows, 5))
		for values in combis_five:
			array = {}
			for i in range(len(liste_array)):
				array[i] = _delete_rows_five(values, liste_array[i])
			np.save("arrays/quintuple/matrix_removed_%d,%d,%d,%d,%d"%(values[0], values[1], values[2], values[3], values[4]), _calc_D_shaped(_make_triple(array)))

def checkSimilarRows(esec_dict, combination, rows = [3, 2, 5, 7, 9]):
	'''
	This function checks if manipulations are the same in case of removed rows. The input parameter combination defines how many combinations are considered
	(e.g. 1 for single combination, 3 for triple combinations). 
	
	Parameters:
		* esec_dict: dictionary with the eSEC matrices (dict)
		* combination: which type of combination (int)
		* rows: rows that will be removed (from 0 for row 1 to 9 for row 10) (array) 
	'''
	temp = 0
	value = []
	if (combination > len(rows)):
		print("combination parameter must be <= len(rows)")
		return 
	#for combination == 1 only single rows are removed and if two manipulations are same it prints out
	#the rows where manipulations are same otherwise it prints 'No other equalities'
	liste_array = copy.deepcopy(esec_dict)
	if combination == 1:
		for values in rows:
			for i in range(len(liste_array)):
				for j in range(len(liste_array)):
					if(i != j):
						if(_compare_manipulations(_delete_rows(values, liste_array[i]), _delete_rows(values, liste_array[j]))):
							print("equalities bettween manipulation %d and %d"%(i+1, j+1))
							if values not in value:
								value.append(values)
							temp += 1
		if temp == 0:
			print('No manipulations are same for %d combinations'%combination)
		else:
			print("Same manipulations due to removing of row(s)", value)
	
	#for combination == 2 only two rows are removed and if two manipulations are same it prints out
	#the combination of rows where manipulations are same otherwise it prints 'No other equalities'
	if combination == 2:
		combis_two = list(combinations(rows, 2))
		for values in combis_two:
			for i in range(len(liste_array)):
				for j in range(len(liste_array)):
					if(i != j):
						if(_compare_manipulations(_delete_rows_two(values, liste_array[i]), _delete_rows_two(values, liste_array[j]))):
							print("equalities bettween manipulation %d and %d"%(i+1, j+1))
							if values not in value:
								value.append(values)
							temp += 1
		if temp == 0:
			print('No manipulations are same for %d combinations'%combination)
		else:
			print("Same manipulations due to removing of row(s)", value)

	#for combination == 3 only three rows are removed and if two manipulations are same it prints out
	#the combination of rows where manipulations are same otherwise it prints 'No other equalities'
	if combination == 3:
		combis_three = list(combinations(rows, 3))
		##remove 3 permutations
		for values in combis_three:
			for i in range(len(liste_array)):
				for j in range(len(liste_array)):
					if(i != j):
						if(_compare_manipulations(_delete_rows_three(values, liste_array[i]), _delete_rows_three(values, liste_array[j]))):
							print("equalities bettween manipulation %d and %d"%(i+1, j+1))
							if values not in value:
								value.append(values)
							temp += 1
		if temp == 0:
			print('No manipulations are same for %d combinations'%combination)
		else:
			print("Same manipulations due to removing of row(s)", value)
	
	#for combination == 2 only four rows are removed and if two manipulations are same it prints out
	#the combination of rows where manipulations are same otherwise it prints 'No other equalities'
	if combination == 4:
		combis_four = list(combinations(rows, 4))
		##remove 4 permutations
		for values in combis_four:
			for i in range(len(liste_array)):
				for j in range(len(liste_array)):
					if(i != j):
						if(_compare_manipulations(_delete_rows_four(values, liste_array[i]), _delete_rows_four(values, liste_array[j]))):
							print("equalities bettween manipulation %d and %d"%(i+1, j+1))
							if values not in value:
								value.append(values)
							temp += 1
		if temp == 0:
			print('No manipulations are same for %d combinations'%combination)
		else:
			print("Same manipulations due to removing of row(s)", value)

	#for combination == 2 only five rows are removed and if two manipulations are same it prints out
	#the combination of rows where manipulations are same otherwise it prints 'No other equalities'			
	if combination == 5:
		combis_five = list(combinations(rows, 5))
		##remove 5 permutations
		for values in combis_five:
			for i in range(len(liste_array)):
				for j in range(len(liste_array)):
					if(i != j):
						if(_compare_manipulations(_delete_rows_five(values, liste_array[i]), _delete_rows_five(values, liste_array[j]))):
							print("equalities bettween manipulation %d and %d"%(i+1, j+1))
							if values not in value:
								value.append(values)
							temp += 1
		if temp == 0:
			print('No manipulations are same for %d combinations'%combination)
		else:
			print("Same manipulations due to removing of row(s)", value)
		
def plotDendrogram(rows, labels, threshold = 0.4, save = False):
	'''
	| Plots the dendrogram for specific rows for the .npy arrays.
	| Warning: needs the output from removeCobinationRowsSave()

	Parameters:
		* rows: indicates which rows will be considered (int, tuple) 
		* label: name of the manipulations in the right order [string array]
		* threshold: color threshold for the dendrogram (float)
		* save: parameter if figure need to be saved (bool)
	'''

	if (isinstance(rows, int) and not(isinstance(rows, tuple))):
		D_shaped = np.load("arrays/single/matrix_removed_%d.npy"%rows)
		indexes = "%d"%rows
	elif (len(rows) == 2):
		D_shaped = np.load("arrays/tuple/matrix_removed_%d,%d.npy"%(rows[0], rows[1]))
		indexes = "%d,%d"%(rows[0], rows[1])
	elif (len(rows) == 3):
		D_shaped = np.load("arrays/triple/matrix_removed_%d,%d,%d.npy"%(rows[0], rows[1], rows[2]))
		indexes = "%d,%d,%d"%(rows[0], rows[1], rows[2])
	elif (len(rows) == 4):
		D_shaped = np.load("arrays/quadruple/matrix_removed_%d,%d,%d,%d.npy"%(rows[0], rows[1], rows[2], rows[3]))
		indexes = "%d,%d,%d,%d"%(rows[0], rows[1], rows[2], rows[3])
	elif (len(rows) == 5):
		D_shaped = np.load("arrays/quintuple/matrix_removed_%d,%d,%d,%d,%d.npy"%(rows[0], rows[1], rows[2], rows[3], rows[4]))
		indexes = "%d,%d,%d,%d,%d"%(rows[0], rows[1], rows[2], rows[3], rows[4])

	fig = plt.figure(figsize=(30,16))

	#reshape the D_shaped matrix in squareform
	dissimilarity = distance.squareform(D_shaped)

	#define the linkage, color threshold and produce dendrogram
	Z = hierarchy.linkage(dissimilarity, method = 'complete', optimal_ordering = True)
	ax = fig.add_subplot(1, 1, 1)
	dn = hierarchy.dendrogram(Z,orientation='right', labels = labels, ax = ax, count_sort = 'ascending', color_threshold = threshold)

	#define plot parameters				
	ax.tick_params(axis='x', which='major', labelsize=27)
	ax.tick_params(axis='y', which='major', labelsize=27)
	ax.set_xlabel('Dissimilarity', fontsize=40)
	plt.tight_layout()
	if save == True:
		plt.savefig("dendrogram_rows_%s.png"%indexes, bbox_inches = 'tight')

def plotDendrogramFromMatrix(D_shaped, labels, figsize = (30,16), threshold = 0.4, save = False, lablesize = 27, fontsize = 40, name_of_plot = ""):
	'''
	| Plots the dendrogram for specific rows for the .npy arrays.
	| Warning: needs the output from removeCobinationRowsSave()

	Parameters:
		* D_shaped: dissimilarity matrix of all manipulations from function removeCobinationRowsSave() (numpy array)
		* labels: name of the manipulations in the right order [string array]
		* figsize: size of plot in case labels are squeezed (int,int)
		* threshold: color threshold for the dendrogram (float)
		* save: parameter if figure needs to be saved (bool)
		* lablesize: fontsize of x- and y-labels (int)
		* fontsize: fontsize of x label (int)
		* name_of_plot: name of the plot that will be saved (string)
	'''
	
	fig = plt.figure(figsize=figsize, facecolor='white')

	#reshape the D_shaped matrix in squareform
	dissimilarity = distance.squareform(D_shaped)

	#define the linkage, color threshold and produce dendrogram
	Z = hierarchy.linkage(dissimilarity, method = 'complete', optimal_ordering = True)
	ax = fig.add_subplot(1, 1, 1)
	dn = hierarchy.dendrogram(Z,orientation='right', labels = labels, ax = ax,
							count_sort = 'ascending', color_threshold = threshold)

	#define plot parameters				
	ax.tick_params(axis='x', which='major', labelsize=lablesize)
	ax.tick_params(axis='y', which='major', labelsize=lablesize)
	ax.set_xlabel('Dissimilarity', fontsize=fontsize)
	plt.tight_layout()
	if save == True:
		plt.savefig(name_of_plot, bbox_inches = 'tight')


def plotDissi(rows, label, save = False):
	'''
	| Plots the dissimilarity matrix for specific rows for the .npy arrays.
	| Warning: needs the output from removeCobinationRowsSave()

	Parameters:
		* rows: indicates which rows will be considered (int, tuple) (None if nothing is removed)
		* label: name of the manipulations in the right order [string array]
		* save: parameter if figure needs to be saved (bool)
	'''
	#load the dissimilarity matrix
	if (isinstance(rows, bool)):
		D_shaped = np.load("arrays/nothing_removed/nothing_removed.npy")
		indexes = "nothing_removed"
	if (isinstance(rows, int) and not(isinstance(rows, tuple))):
		D_shaped = np.load("arrays/single/matrix_removed_%d.npy"%rows)
		indexes = "%d"%rows
	elif (len(rows) == 2):
		D_shaped = np.load("arrays/tuple/matrix_removed_%d,%d.npy"%(rows[0], rows[1]))
		indexes = "%d,%d"%(rows[0], rows[1])
	elif (len(rows) == 3):
		D_shaped = np.load("arrays/triple/matrix_removed_%d,%d,%d.npy"%(rows[0], rows[1], rows[2]))
		indexes = "%d,%d,%d"%(rows[0], rows[1], rows[2])
	elif (len(rows) == 4):
		D_shaped = np.load("arrays/quadruple/matrix_removed_%d,%d,%d,%d.npy"%(rows[0], rows[1], rows[2], rows[3]))
		indexes = "%d,%d,%d,%d"%(rows[0], rows[1], rows[2], rows[3])
	elif (len(rows) == 5):
		D_shaped = np.load("arrays/quintuple/matrix_removed_%d,%d,%d,%d,%d.npy"%(rows[0], rows[1], rows[2], rows[3], rows[4]))
		indexes = "%d,%d,%d,%d,%d"%(rows[0], rows[1], rows[2], rows[3], rows[4])
	#round the numbers in D_shaped for better visualization 
	D_shaped_rounded = np.around(D_shaped, decimals=2)

	#create pandas dataframe with rounded D_shaped and input labels
	df_cm = pd.DataFrame(D_shaped_rounded, index = label, columns = label)
	fig, ax = plt.subplots(figsize = (35,26))

	#define plot parameters
	sn.set(font_scale=3)
	ax.xaxis.tick_top() # x axis on top
	ax.xaxis.set_label_position('top')
	ax.tick_params(labelsize=29)
	ax.tick_params(labelsize=29)
	sn.heatmap(df_cm, annot=True, annot_kws={'size':18})#, linewidths=.5)
	plt.tight_layout()
	plt.show()
	if save == True:
		plt.savefig("dissimilarity_rows_%s.eps"%indexes, format = "eps", dpi = 350, bbox_inches= 'tight')

def plotDissiFromMatrix(D_shaped, label, save = False, name_of_plot = ""):
	'''
	| Plots the dissimilarity matrix for specific rows for the .npy arrays.
	| Warning: needs the output from removeCobinationRowsSave()

	Parameters:
		* D_shaped: dissimilarity matrix of all manipulations from function removeCobinationRowsSave() (numpy array)
		* labels: name of the manipulations in the right order [string array]
		* save: parameter if figure needs to be saved (bool)
		* name_of_plot: name of the plot that will be saved (string)
	'''

	#round the numbers in D_shaped for better visualization 
	D_shaped_rounded = np.around(D_shaped, decimals=2)

	#create pandas dataframe with rounded D_shaped and input labels
	df_cm = pd.DataFrame(D_shaped_rounded, index = label, columns = label)
	fig, ax = plt.subplots(figsize = (35,26))

	#define plot parameters
	sn.set(font_scale=3)
	ax.xaxis.tick_top() # x axis on top
	ax.xaxis.set_label_position('top')
	ax.tick_params(labelsize=29)
	ax.tick_params(labelsize=29)
	sn.heatmap(df_cm, annot=True, annot_kws={'size':18})#, linewidths=.5)
	plt.tight_layout()
	plt.show()
	if save == True:
		plt.savefig("dissimilarity_%s.eps"%name_of_plot, format = "eps", dpi = 350, bbox_inches= 'tight')

def plotDendroDissimi(rows,  label, threshold = 0.4, save = False): 
	'''
	| Plots the dendrogram and the dissimilarity matrix for specific rows for the .npy arrays.
	| e.g. rows=[3,4,5] will plot the dendrogram and dissimilarity matrix with removed rows 3,4 and 5
	| Warning: needs the folder structure from removeCobinationRowsSave()
	
	Parameters:
		* rows: indicates which rows will be considered (int, tuple) 
		* label: name of the manipulations in the right order [string array]
		* threshold: color threshold for the dendrogram (float)
		* save: parameter if figure needs to be saved (bool)
	'''

	#load the dissimilarity matrix
	if (isinstance(rows, int) and not(isinstance(rows, tuple))):
		D_shaped = np.load("arrays/single/matrix_removed_%d.npy"%rows)
		indexes = "%d"%rows
	elif (len(rows) == 2):
		D_shaped = np.load("arrays/tuple/matrix_removed_%d,%d.npy"%(rows[0], rows[1]))
		indexes = "%d,%d"%(rows[0], rows[1])
	elif (len(rows) == 3):
		D_shaped = np.load("arrays/triple/matrix_removed_%d,%d,%d.npy"%(rows[0], rows[1], rows[2]))
		indexes = "%d,%d,%d"%(rows[0], rows[1], rows[2])
	elif (len(rows) == 4):
		D_shaped = np.load("arrays/quadruple/matrix_removed_%d,%d,%d,%d.npy"%(rows[0], rows[1], rows[2], rows[3]))
		indexes = "%d,%d,%d,%d"%(rows[0], rows[1], rows[2], rows[3])
	elif (len(rows) == 5):
		D_shaped = np.load("arrays/quintuple/matrix_removed_%d,%d,%d,%d,%d.npy"%(rows[0], rows[1], rows[2], rows[3], rows[4]))
		indexes = "%d,%d,%d,%d,%d"%(rows[0], rows[1], rows[2], rows[3], rows[4])
	
	#label = ["Hit/Flick", "Poke", "Bore, Rub, Rotate", "Lay", "Push/ Pull", "Stir", "Knead", "Lever", "Cut", "Draw", "Scoop", "Take down", "Push down", "Break", "Uncover(Pick & Place)", "Uncover(Push)", "Put on top", "Put inside", "Push on top", "Put over", "Push over", "Push from x to y", "Push together", "Push apart", "Take & invert", "Shake", "Rotate align", "Pour to ground(v1)", "Pour to ground(v2)", "Pour to cup(v1)", "Pour to cup(v2)", "Pick & place", "Chop", "Scratch", "Squash"]
	
	#transforms the dissimilarity matrix into squareform
	dissimilarity = distance.squareform(D_shaped)

	#define the linkage, color threshold and produce dendrogram
	Z = hierarchy.linkage(dissimilarity, 'complete')

	#round the numbers in D_shaped for better visualization 
	D_shaped_rounded = np.around(D_shaped, decimals=2)

	#create pandas dataframe with rounded D_shaped and input labels
	df_cm = pd.DataFrame(D_shaped_rounded, index = label,
					  columns = label)
	
	#define plot parameters
	f = plt.figure(figsize=(40,31))
	ax = f.add_subplot(221)
	hierarchy.dendrogram(Z, orientation='right', labels = label, ax = ax, count_sort = 'ascending', color_threshold = threshold)#, color_threshold = threshold)
	ax.tick_params(axis='x', which='major', labelsize=25)
	ax.tick_params(axis='y', which='major', labelsize=25)
	ax.set_xlabel('Dissimilarity', fontsize=35)
	
	ax2 = f.add_subplot(223)
	ax2.xaxis.tick_top() # x axis on top
	ax2.xaxis.set_label_position('top')
	ax2.tick_params(labelsize=15)
	ax2.tick_params(labelsize=15)
	sn.heatmap(df_cm, annot=True, linewidths=.5, annot_kws={'size':12})
	plt.tight_layout()
	if save == True:
		plt.savefig("sissimilarity_dendrogram_rows_%s.png"%indexes, bbox_inches = 'tight', pad_inches = 0)
	plt.show()

def _plotDendroDissimiFromMatrix(D_shaped,  label, threshold = 0.4, save = False, name_of_plot = ""): 
	'''
	| Plots the dendrogram and the dissimilarity matrix for specific rows for the .npy arrays.
	e.g. rows=[3,4,5] will plot the dendrogram and dissimilarity matrix with removed rows 3,4 and 5
	| Warning: needs the folder structure from removeCobinationRowsSave()
	
	Parameters:
		* rows: indicates which rows will be considered (int, tuple) 
		* label: name of the manipulations in the right order [string array]
		* threshold: color threshold for the dendrogram (float)
		* save: parameter if figure needs to be saved (bool)
	'''
	#label = ["Hit/Flick", "Poke", "Bore, Rub, Rotate", "Lay", "Push/ Pull", "Stir", "Knead", "Lever", "Cut", "Draw", "Scoop", "Take down", "Push down", "Break", "Uncover(Pick & Place)", "Uncover(Push)", "Put on top", "Put inside", "Push on top", "Put over", "Push over", "Push from x to y", "Push together", "Push apart", "Take & invert", "Shake", "Rotate align", "Pour to ground(v1)", "Pour to ground(v2)", "Pour to cup(v1)", "Pour to cup(v2)", "Pick & place", "Chop", "Scratch", "Squash"]
	
	#transforms the dissimilarity matrix into squareform
	dissimilarity = distance.squareform(D_shaped)

	#define the linkage, color threshold and produce dendrogram
	Z = hierarchy.linkage(dissimilarity, 'complete')

	#round the numbers in D_shaped for better visualization 
	D_shaped_rounded = np.around(D_shaped, decimals=2)

	#create pandas dataframe with rounded D_shaped and input labels
	df_cm = pd.DataFrame(D_shaped_rounded, index = label,
					  columns = label)
	
	#define plot parameters
	f = plt.figure(figsize=(40,31))
	ax = f.add_subplot(221)
	hierarchy.dendrogram(Z, orientation='right', labels = label, ax = ax, count_sort = 'ascending', color_threshold = threshold)#, color_threshold = threshold)
	ax.tick_params(axis='x', which='major', labelsize=25)
	ax.tick_params(axis='y', which='major', labelsize=25)
	ax.set_xlabel('Dissimilarity', fontsize=35)
	
	ax2 = f.add_subplot(223)
	ax2.xaxis.tick_top() # x axis on top
	ax2.xaxis.set_label_position('top')
	ax2.tick_params(labelsize=15)
	ax2.tick_params(labelsize=15)
	sn.heatmap(df_cm, annot=True, linewidths=.5, annot_kws={'size':12})
	plt.tight_layout()
	if save == True:
		plt.savefig("%s"%name_of_plot, bbox_inches = 'tight', pad_inches = 0)
	#plt.show()

def plotAllMatrices(path, label):
	'''
	| Plots the dendrogram and dissimilarity matrix for all combinations of removed rows in the "array" folder structure.
	| Warning: Needs folder structure from function removeCobinationRowsSave()

	Parameters:
		* path: path to the "array" folder from function removeCobinationRowsSave()
		* label: name of the manipulations in the right order [string array]
        
    Returns:
        plots of dendrogram and dissimilarity matrices
    '''
	if not(os.path.exists("plots/single")):
			os.makedirs("plots/single/")
			os.mkdir("plots/nothing_removed/")
			os.mkdir("plots/tuple/")
			os.mkdir("plots/triple/")
			os.mkdir("plots/quadruple/")
			os.mkdir("plots/quintuple/")

	##plots for non removed rows
	path = "arrays/nothing_removed/"
	for filename in os.listdir(path):
		D_shaped = np.load(path+filename)
		_plotDendroDissimiFromMatrix(D_shaped, label, save=True, name_of_plot = "plots/nothing_removed/"+os.path.splitext(filename)[0]+".png")

	##plots for single removed rows
	path = "arrays/single/"
	for filename in os.listdir(path):
		D_shaped = np.load(path+filename)
		_plotDendroDissimiFromMatrix(D_shaped, label, save=True, name_of_plot = "plots/single/"+os.path.splitext(filename)[0]+".png")

	##plots for tuple removed rows
	path = "arrays/tuple/"
	for filename in os.listdir(path):
		D_shaped = np.load(path+filename)
		_plotDendroDissimiFromMatrix(D_shaped, label, save=True, name_of_plot ="plots/tuple/"+os.path.splitext(filename)[0]+".png")

	##plots for tripe removed rows
	path = "arrays/triple/"
	for filename in os.listdir(path):
		D_shaped = np.load(path+filename)
		_plotDendroDissimiFromMatrix(D_shaped, label, save=True, name_of_plot ="plots/triple/"+os.path.splitext(filename)[0]+".png")

	##plots for quadruple removed rows
	path = "arrays/quadruple/"
	for filename in os.listdir(path):
		D_shaped = np.load(path+filename)
		_plotDendroDissimiFromMatrix(D_shaped, label, save=True, name_of_plot ="plots/quadruple/"+os.path.splitext(filename)[0]+".png")

	##plots for quintuple removed rows
	path = "arrays/quintuple/"
	for filename in os.listdir(path):
		D_shaped = np.load(path+filename)
		_plotDendroDissimiFromMatrix(D_shaped, label, save=True, name_of_plot ="plots/quintuple/"+os.path.splitext(filename)[0]+".png")

def esec_to_e2sec(pdf_path):
    '''
    Takes an pdf file with eSEC matrices as input and returns these matrices as e2SEC matrices dict.
    
    Parameters:
        * pdf_path: path of the pdf file which contains eSEC matrices (string)
        
    Returns:
        dict of e2SEC matrices (dict)
    '''
    
    #read PDF file and save in liste_array dict
    df = tabula.read_pdf(pdf_path, pages='all', pandas_options={'header': None})
    liste_array = {}
    for i in range(len(df)):
        liste_array[i] = df[i].loc[:, 1:].to_numpy()
    #----------------------------------------------------------------  
    
    #uppercase every entry
    for j in range(len(liste_array)):
        for i in range(liste_array[j].shape[0]):
            for k in range(liste_array[j].shape[1]):
                liste_array[j][i][k] = liste_array[j][i][k].upper()
    #----------------------------------------------------------------  
    
    #copy esec matrix 
    e2sec_array = copy.deepcopy(liste_array)            
    for j in range(len(e2sec_array)):
        
    #----------------------------------------------------------------
        #find index of semantics to merge
        replace_Ab_1, replace_Ab_2 = np.where(e2sec_array[j] == "AB") #VAr
        replace_Be_1, replace_Be_2 = np.where(e2sec_array[j] == "BE") #VAr

        replace_To_1, replace_To_2 = np.where(e2sec_array[j] == "TO") #TVAr
        replace_Bo_1, replace_Bo_2 = np.where(e2sec_array[j] == "BO") #TVAr

        replace_Ar_1, replace_Ar_2 = np.where(e2sec_array[j] == "AR") #HAr
        replace_ArT_1, replace_ArT_2 = np.where(e2sec_array[j] == "ART") #THAr
        
        replace_MT_1, replace_MT_2 = np.where(e2sec_array[j] == "MT")
        replace_FMT_1, replace_FMT_2 = np.where(e2sec_array[j] == "FMT")
    #----------------------------------------------------------------
        
        #replace old semantics with merged ones
        for i in range(len(replace_Ab_1)):
            e2sec_array[j][replace_Ab_1[i]][replace_Ab_2[i]] = 'VAR'
        for i in range(len(replace_Be_1)):
            e2sec_array[j][replace_Be_1[i]][replace_Be_2[i]] = 'VAR'

        for i in range(len(replace_To_1)):
            e2sec_array[j][replace_To_1[i]][replace_To_2[i]] = 'VART'
        for i in range(len(replace_Bo_1)):
            e2sec_array[j][replace_Bo_1[i]][replace_Bo_2[i]] = 'VART'

        for i in range(len(replace_Ar_1)):
            e2sec_array[j][replace_Ar_1[i]][replace_Ar_2[i]] = 'HAR'
        for i in range(len(replace_ArT_1)):
            e2sec_array[j][replace_ArT_1[i]][replace_ArT_2[i]] = 'HART'

        for i in range(len(replace_MT_1)):
            e2sec_array[j][replace_MT_1[i]][replace_MT_2[i]] = 'MT'
        for i in range(len(replace_FMT_1)):
            e2sec_array[j][replace_FMT_1[i]][replace_FMT_2[i]] = 'MT'
    #----------------------------------------------------------------  
    #remove rows 4 and 10 in T/N, SSR, DSR
    for i in range(len(e2sec_array)):
        e2sec_array[i] = np.delete(e2sec_array[i], [3,9,13,19,23,29], 0)
    #----------------------------------------------------------------  
    
    #find columns that are same due to e2SEC and remove them
    k = {}
    for i in range(len(e2sec_array)):
        for j in range(e2sec_array[i].shape[1]-1):
            if(np.array_equal(e2sec_array[i][:,j], e2sec_array[i][:,j+1])):
                _add_element(k, i, j+1)

    for i in range(len(e2sec_array)):
        e2sec_array[i] = np.delete(e2sec_array[i], k[i], 1)
    #----------------------------------------------------------------  
    
    #return e2sec and esec dict
    return e2sec_array, liste_array

def getDissimilarityMatrix(eSEC_matrices):
	'''
    Takes a dict of eSEC matrices as input to calculate the dissimilarity matrix from paper [1].
    
    Parameters:
        * eSEC_matrices: dict contains eSEC matrices
        
    Returns:
        dissimilarity matrix
    '''
	return _calc_D_shaped(_make_triple(eSEC_matrices))

def _compare_manipulations(manipulation_1, manipulation_2):
    '''
    Compares lengths of two eSEc matrices. If the length is not same, the last column of the shorter manipulation is repeated until size is same.
    
    Parameters:
        * manipulation_1: first eSEC matrix
		* manipulation_2: second eSEC matrix
    '''
    if (manipulation_1.shape[1] > manipulation_2.shape[1]):
        new_array = copy.copy(manipulation_2)

        for i in range(manipulation_2.shape[1], manipulation_1.shape[1]):
            new_array = np.column_stack((new_array, new_array[:, i-1]))
        return(np.array_equal(new_array, manipulation_1))

    elif (manipulation_1.shape[1] < manipulation_2.shape[1]):

        new_array = copy.copy(manipulation_1)
        for i in range(manipulation_1.shape[1], manipulation_2.shape[1]):
            new_array = np.column_stack((new_array, new_array[:, i-1]))
        return(np.array_equal(new_array, manipulation_2))

    else:
        return(np.array_equal(manipulation_1, manipulation_2))
		
def _calc_D_shaped(table_triples):
	'''
    Calculates dissimilarity matrix.
    
    Parameters:
        * table_triples: resulting triple matrix from _make_triple()
    '''
	D = np.zeros(len(table_triples)*len(table_triples))
	index = 0
	for i in range(len(table_triples)):
		for j in range(len(table_triples)):
			man_1, man_2 = _compare_triple_matrics(table_triples[i], table_triples[j])
			L1 = np.array([(man_1[j][:,0] == man_2[j][:,0]) for j in range(len(man_1))])
			L1 = np.invert(L1.T) * 1
			L2 = np.array([(man_1[j][:,1] == man_2[j][:,1]) for j in range(len(man_1))])
			L2 = np.invert(L2.T) * 1 
			L3 = np.array([(man_1[j][:,2] == man_2[j][:,2]) for j in range(len(man_1))])
			L3 = np.invert(L3.T) * 1 
			d = (np.sqrt(L1+L2+L3))/(np.sqrt(3))
			D[index] = (1/(L1.shape[1] * 10)) * np.sum(d)
			index += 1
	D_shaped = np.reshape(D , (len(table_triples),len(table_triples)))
	return(D_shaped)

def similarity_manipulations(manipulation_1, manipulation_2):
	'''
    Calculates the similarity between two e2SEC matrices in percent.
    
    Parameters:
        * manipulation_1: first e2SEC manipulation
		* manipulation_2: second e2SEC manipulation
    '''

	man_1, man_2 = _compare_triple_matrics(_make_triple_one_manipualtion(manipulation_1), _make_triple_one_manipualtion(manipulation_2))
	L1 = np.array([(man_1[j][:,0] == man_2[j][:,0]) for j in range(len(man_1))])
	L1 = np.invert(L1.T) * 1
	L2 = np.array([(man_1[j][:,1] == man_2[j][:,1]) for j in range(len(man_1))])
	L2 = np.invert(L2.T) * 1 
	L3 = np.array([(man_1[j][:,2] == man_2[j][:,2]) for j in range(len(man_1))])
	L3 = np.invert(L3.T) * 1 
	d = (np.sqrt(L1+L2+L3))/(np.sqrt(3))
	D = (1/(L1.shape[1] * 8)) * np.sum(d)

	sim = (1 - D)*100

	return(sim)

def _make_triple(liste_array):
	'''
    Creates triples for 10-row eSEC matrix.
    
    Parameters:
		* liste_array: eSEC matrices
    '''
	
	table_triples = {}
	frames_triples = {}
	opal = int(liste_array[0].shape[0]/3)
	triple_frame = np.chararray((opal,3), itemsize=7)
	for k in range(len(liste_array)):
		for j in range(liste_array[k][1].size):
			for i in range(opal):
				triple_frame[i][0] = liste_array[k][i][j]
				triple_frame[i][1] = liste_array[k][i+opal][j]
				triple_frame[i][2] = liste_array[k][i+2*opal][j]
			frames_triples[j] = copy.copy(triple_frame)
			#print("new:",frames_triples)
		table_triples[k] = copy.copy(frames_triples)
		##empty dict to get rid of zeug
		frames_triples.clear()
	return table_triples

def _make_triple_one_manipualtion(liste_array):
	'''
    Creates triples for 10-row e2SEC matrix.
    
    Parameters:
		* liste_array: e2SEC matrix
    '''

	table_triples = {}
	frames_triples = {}
	opal = int(liste_array.shape[0]/3)
	triple_frame = np.chararray((opal,3), itemsize=7)
	for j in range(liste_array[1].size):
		for i in range(opal):
			triple_frame[i][0] = liste_array[i][j]
			triple_frame[i][1] = liste_array[i+opal][j]
			triple_frame[i][2] = liste_array[i+2*opal][j]
		frames_triples[j] = copy.copy(triple_frame)
	table_triples = copy.copy(frames_triples)
	##empty dict to get rid of zeug
	frames_triples.clear()

	return table_triples

def _compare_triple_matrics(manipulation_1, manipulation_2):
	'''
    Compares triple matrices (TNR, SSR, DSR) from two manipulations.
    
    Parameters:
        * manipulation_1: triples of first manipulation
		* manipulation_2: triples of second manipulation

    '''
	
	#if the length of the triple matrices don't match repeat last
	#column of the longer manipulation
	if (len(manipulation_1) > len(manipulation_2)):
		new_array = copy.copy(manipulation_2)
		length = len(manipulation_1) - len(manipulation_2)

		for i in range(length):
			new_array[len(manipulation_2) + i] = manipulation_2[len(manipulation_2) - 1]
		return(copy.copy(manipulation_1), copy.copy(new_array))

	elif (len(manipulation_1) < len(manipulation_2)):
		new_array = copy.copy(manipulation_1)
		length = len(manipulation_2) - len(manipulation_1)
		
		for i in range(length):
			new_array[len(manipulation_1) + i] = manipulation_1[len(manipulation_1) - 1]
		return(copy.copy(new_array), copy.copy(manipulation_2))

	else:
		return(copy.copy(manipulation_1), copy.copy(manipulation_2))	

def _delete_rows(relation, manipulation):
	'''
    Deletes one row in eSEC table.
    
    Parameters:
        * relation: row to delete (int)
		* manipulation: eSEC matrix
    '''

	return np.delete(manipulation, [relation,relation+10,relation+20], 0)

def _delete_rows_two(combination, manipulation):
	'''
    Deletes two rows in eSEC table.
    
    Parameters:
        * combination: two rows to delete [int]
		* manipulation: eSEC matrix
    '''

	relation_1 = combination[0]
	relation_2 = combination[1]
	return np.delete(manipulation, [relation_1,relation_1+10,relation_1+20, relation_2,relation_2+10,relation_2+20], 0)

def _delete_rows_three(combination, manipulation):
	'''
    Deletes three rows in eSEC table.
    
    Parameters:
        * combination: three rows to delete [int]
		* manipulation: eSEC matrix
    '''

	relation_1 = combination[0]
	relation_2 = combination[1]
	relation_3 = combination[2]
	return np.delete(manipulation, [relation_1,relation_1+10,relation_1+20, 
									relation_2,relation_2+10,relation_2+20,
									relation_3,relation_3+10,relation_3+20], 0)

def _delete_rows_four(combination, manipulation):
	'''
    Deletes four rows in eSEC table.
    
    Parameters:
        * combination: four rows to delete [int]
		* manipulation: eSEC matrix
    '''

	relation_1 = combination[0]
	relation_2 = combination[1]
	relation_3 = combination[2]
	relation_4 = combination[3]
	return np.delete(manipulation, [relation_1,relation_1+10,relation_1+20, 
									relation_2,relation_2+10,relation_2+20,
									relation_3,relation_3+10,relation_3+20,
									relation_4,relation_4+10,relation_4+20], 0)
def _delete_rows_five(combination, manipulation):
	'''
    Deletes five rows in eSEC table.
    
    Parameters:
        * combination: five rows to delete [int]
		* manipulation: eSEC matrix
    '''

	relation_1 = combination[0]
	relation_2 = combination[1]
	relation_3 = combination[2]
	relation_4 = combination[3]
	relation_5 = combination[4]
	return np.delete(manipulation, [relation_1,relation_1+10,relation_1+20, 
									relation_2,relation_2+10,relation_2+20,
									relation_3,relation_3+10,relation_3+20,
									relation_4,relation_4+10,relation_4+20,
									relation_5,relation_5+10,relation_5+20], 0)

		
def _dissimilarity_array_new(manipulation_1, manipulation_2, liste_array):
	"""
	Returns how many entries are different if two manipulations are compared. Furthermore it makes the number of columns same in two manipulations. 
	If manipulation_1 has less columns than manipulation_2 the last column of manipulation_2 is repeated until its same. This happens
	vice versa if manipulation_2 has more columns than manipulation_1
	
	Parameters:
		* manipulation_1: first manipulation to check (int)
		* manipulation_2: second manipulation to check (int)
		* liste_array: eSEC matrices
    """
	#a and b are the number of manipulation - 1 (because array starts at zero but manipulations at one)
	a = manipulation_1 - 1
	b = manipulation_2 - 1
	array = []
	if (liste_array[a].shape[1] > liste_array[b].shape[1]):
		##repeat last column of longer esec
		new_array = copy.copy(liste_array[b])
		for i in range(liste_array[b].shape[1], liste_array[a].shape[1]):
			new_array = np.column_stack((new_array, new_array[:, i-1]))

		##compare entries and returns how many of them are different
		for i in range(liste_array[0].shape[0]):
			c = liste_array[a][i, 0:new_array.shape[1]] == new_array[i]
			array = np.append(array, liste_array[a].shape[1] - np.sum(c))
		array /= np.sum(array)
		return(array)
	
	elif (liste_array[a].shape[1] < liste_array[b].shape[1]):
		##repeat last column of longer esec
		new_array = copy.copy(liste_array[a])
		for i in range(liste_array[a].shape[1], liste_array[b].shape[1]):
			new_array = np.column_stack((new_array, new_array[:, i-1]))
			
		##compare entries and returns how many of them are different 
		for i in range(liste_array[0].shape[0]):
			c = new_array[i] == liste_array[b][i, 0:new_array.shape[1]]
			array = np.append(array, (liste_array[b].shape[1] - np.sum(c)))
		array /= np.sum(array)
		return(array)

	else:
		##compare entries and returns how many of them are different
		for i in range(liste_array[0].shape[0]):
			c = liste_array[a][i] == liste_array[b][i]
			array = np.append(array, (liste_array[a].shape[1] - np.sum(c)))
		array /= np.sum(array)
		return(array)
		

#*************************************************
# Author: dkarchmer
# Availability: https://stackoverflow.com/questions/33272588/appending-elements-to-an-empty-dictionary-of-lists-in-python
#*************************************************
def _add_element(dict, key, value):
	"""
	Adds an element to dict

	Parameters:
		* dict: input dict
		* key: index to add
		* value: value to add
	"""
	if key not in dict:
		dict[key] = []
	dict[key].append(value)


#*************************************************
# Availability: https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html
#*************************************************
def _plot_confusion_matrix(cm, classes,
							normalize=False,
							title='Confusion matrix',
							cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""

	matplotlib.rcParams.update(matplotlib.rcParamsDefault)
	fig, ax = plt.subplots(figsize = (11,11))
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation = 90)
	plt.yticks(tick_marks, classes)
	plt.tick_params(labelsize=20)
	plt.tick_params(labelsize=20)
	plt.title('e²SEC', fontsize=30, fontweight='bold')

	ax.set_ylabel('Predicted class', fontsize=25, fontweight='bold')
	ax.set_xlabel('Target class', fontsize=25, fontweight='bold')

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, str(cm[i, j])+"%",
					horizontalalignment="center",
					color="white" if cm[i, j] > thresh else "black", size = "xx-large",family = "sans-serif")
	plt.savefig("Confusion_matrix.png", bbox_inches = 'tight', dpi = 300)

def _classify_monte_carlo(input_groups, manipulation, dissi_matrix):
	"""
	Calculates the confusion matrix for all 120 e2SEC matrices from the MANIAC dataset. 

	Parameters:
		* input_groups: train manipulations from _split()
		* manipulation: test manipulations from _split()
		* dissi_matrix: empty zeros 8x8 array
	"""
	#assign test manipulations to train manipulations
	for k in range(8):
		for y in range(5):
			sim = 0
			temp_i = 0
			for i in range(8):
				for j in range(10):
					#calculate similarity between train groups (input_groups[i][j]) and test manipulation (manipulation[k][y])
					sim_temp = similarity_manipulations(input_groups[i][j], manipulation[k][y])
					#save the biggest similarity from test manipulation (manipulation[k][y]) to a train manipulation (input_groups[i][j])
					if sim_temp > sim:
						sim = sim_temp
						temp_i = i
			#add a one in the confusion matrix where train class was predicted
			dissi_matrix[temp_i][k] += 1

	#return confusion matrix
	#have to divide it to max value(5) and multiply it by 100 to get percentage values from 0%-100%
	return (dissi_matrix/5)*100

def _split(rand, chopping, cutting, hiding, pushing, put_on_top, stirring, take_down, uncover):
	#do a train/test split using ShuffleSplit for each manipulation
	indices = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])   
	mc = ShuffleSplit(n_splits=8, test_size = 0.3 ,random_state = rand) 
	mc.get_n_splits(indices)    
	traini = []
	testi = []
	for train, test in mc.split(indices):
			traini.append(indices[train])
			testi.append(indices[test])

	#define train groups and add them in one array 
	chop_train = []
	cut_train = []
	hiding_train = []
	pushing_train = []
	put_on_top_train = []
	stirring_train = []
	take_down_train = []
	uncover_train = []
	groups_train = []
	for i in range(10):
		chop_train.append(chopping[traini[0][i]])
		cut_train.append(cutting[traini[1][i]])
		hiding_train.append(hiding[traini[2][i]])
		pushing_train.append(pushing[traini[3][i]])
		put_on_top_train.append(put_on_top[traini[4][i]])
		stirring_train.append(stirring[traini[5][i]])
		take_down_train.append(take_down[traini[6][i]])
		uncover_train.append(uncover[traini[7][i]])
	groups_train = [chop_train, cut_train, hiding_train, pushing_train, put_on_top_train, stirring_train, take_down_train, uncover_train]

	#define test groups and add them in one array 
	chop_test = []
	cut_test = []
	hiding_test = []
	pushing_test = []
	put_on_top_test = []
	stirring_test = []
	take_down_test = []
	uncover_test = []
	groups_test = []
	for i in range(5):
		chop_test.append(chopping[testi[0][i]])
		cut_test.append(cutting[testi[1][i]])
		hiding_test.append(hiding[testi[2][i]])
		pushing_test.append(pushing[testi[3][i]])
		put_on_top_test.append(put_on_top[testi[4][i]])
		stirring_test.append(stirring[testi[5][i]])
		take_down_test.append(take_down[testi[6][i]])
		uncover_test.append(uncover[testi[7][i]])
	groups_test = [chop_test, cut_test, hiding_test, pushing_test, put_on_top_test, stirring_test, take_down_test, uncover_test]
	
	#return final train and test arrays
	return groups_train, groups_test

def classification_e2sec(location_e2sec_matrices):
	"""
	Plots the confusion matrix, the classification accuracy for each manipulation and average classification accuracy using cross validation from 120 e2SEC matrices using the MANIAC dataset.

	Parameters:
		* location_e2sec_matrices: folder of all 120 e2SEC matrices saved as .npy file
	"""

	#define location of e2SEC matrices folder 
	e2sec_matrices_without_filter = {}
	loc = location_e2sec_matrices
	i = 0

	#load all .npy files in e2sec_matrices_without_filter array
	for file in sorted(os.listdir(loc)):
		if file[-3:] == "npy":
			e2sec_matrices_without_filter[i] = np.load(loc+file)
		i += 1
	i = 0
	labels = []

	#import labels from .npy files 
	for file in sorted(os.listdir(loc)):
		if file[-3:] == "npy":
			#print(file)
			if i < 15:
				labels.append(file[15:-4])
			if i >= 15 and i < 30:
				labels.append(file[14:-4])
			if i >= 30 and i < 45:
				labels.append(file[13:-4])
			if i >= 45 and i < 60:
				labels.append(file[14:-4])
			if i >= 60 and i < 75:
				labels.append(file[17:-4])
			if i >= 75 and i < 90:
				labels.append(file[15:-4])
			if i >= 90 and i < 105:
				labels.append(file[16:-4])
			if i >= 105:
				labels.append(file[14:-4])
		i += 1
	
	#decode to UTF-8
	for i in range(len(e2sec_matrices_without_filter)):
		e2sec_matrices_without_filter[i] = np.char.decode(e2sec_matrices_without_filter[i].astype(np.bytes_), 'UTF-8')

	#create array for each manipulation and assign e2SEC matrices
	chopping = []
	cutting = []
	hiding = []
	pushing = []
	put_on_top = []
	stirring = []
	take_down = []
	uncover = []
	for i in range(120):
		if i < 15:
			chopping.append(e2sec_matrices_without_filter[i])
		if i >= 15 and i < 30:
			cutting.append(e2sec_matrices_without_filter[i])
		if i >= 30 and i < 45:
			hiding.append(e2sec_matrices_without_filter[i])
		if i >= 45 and i < 60:
			pushing.append(e2sec_matrices_without_filter[i])
		if i >= 60 and i < 75:
			put_on_top.append(e2sec_matrices_without_filter[i])
		if i >= 75 and i < 90:
			stirring.append(e2sec_matrices_without_filter[i])
		if i >= 90 and i < 105:
			take_down.append(e2sec_matrices_without_filter[i])
		if i >= 105:
			uncover.append(e2sec_matrices_without_filter[i])
		i += 1
	groups = [chopping, cutting, hiding, pushing, put_on_top, stirring, take_down, uncover]


	dissi_matrix_new = np.zeros((8,8))
	all_matrices = []
	final_matrices = np.zeros((8,8))
	#apply 20 iterations of cross validation to receive a confusion matrix
	for i in range(20):
		dissi_matrix = np.zeros((8,8))
		groups_train, groups_test = _split(randrange(100), chopping, cutting, hiding, pushing, put_on_top, stirring, take_down, uncover)
		dissi_matrix_new = _classify_monte_carlo(groups_train, groups_test, dissi_matrix)
		all_matrices.append(dissi_matrix_new)
		final_matrices += dissi_matrix_new

	#calculate errors and 95% confidence interval
	test = np.asarray(all_matrices)
	error = np.std(test, axis = 0)
	confi_interval_95 = (error.diagonal()*1.960)/np.sqrt(20)

	#define all manipulations
	manipulations = ['Chopping', 'Cutting', 'Hiding', 'Pushing', 'Put on top', 'Stirring', 'Take down', 'Uncover']

	#divide matrices by 20 because 20 matrices have added in the iteration
	final_matrices = final_matrices/20

	#sort the x and y columns the confusion matrices to compare them to these from  paper 
	#"Recognition and prediction of manipulation actions using Enriched Semantic Event Chains"
	new_sorted_dissi = copy.deepcopy(final_matrices)
	new_sorted_labels = copy.deepcopy(manipulations)
	new_sorted_dissi[[0, 1, 2, 3, 4, 5, 6, 7], :] = new_sorted_dissi[[4, 6, 3, 1, 0, 5, 2, 7], :]
	new_sorted_dissi[:,[0, 1, 2, 3, 4, 5, 6, 7]] = new_sorted_dissi[:,[4, 6, 3, 1, 0, 5, 2, 7]]
	values = [4, 6, 3, 1, 0, 5, 2, 7]
	k = 0
	old_array = copy.deepcopy(new_sorted_labels)
	for i in values:
		new_sorted_labels[k] = old_array[i]
		k += 1
	k = 0
	confi_interval_95_sorted_new = copy.deepcopy(confi_interval_95)
	old_array_confi = copy.deepcopy(confi_interval_95)
	for i in values:
		confi_interval_95_sorted_new[k] = old_array_confi[i]
		k += 1
	
	#plot confusion matrix and save it as png
	_plot_confusion_matrix(new_sorted_dissi, new_sorted_labels, cmap = "binary")

	#load measured values with "WebPlotDigitizer" from SEC and eSEC values from paper 
	#"Recognition and prediction of manipulation actions using Enriched Semantic Event Chains"
	values_sec_cleaned = [79,70,94,53,54,93,92,91]
	errors_sec_cleaned = [84.294094414429-79.14442233373222, 74.3055063268706-70.22128364218003, 98.07834597525962-93.88313897848509, 57.68005637669001-53.174093306080316, 58.98967125928101-54.17295211483616, 97.05729030408699-92.92867389456285, 96.4135812939999-91.95201194822378, 95.43691934766085-90.99754686430154]
	values_esec_cleaned = [89,95,100,75,77,98,100,99]
	errors_esec_cleaned = [93.51683642425462-89.05328527126964, 94.88805622299134-90.97906072480025, 0, 80.75553403722337-75.01505113778182, 76.98321670330462-71.1880625381541, 98.08632526696599-95.18874818439073, 0, 99.01573678401843-97.10224248420457]
	values_sec = [76.88634192932187, 67.9083094555874, 92.93218720152818, 50.907354345749766, 53.19961795606497, 92.93218720152817, 90.06685768863419, 89.11174785100286]
	errors_sec = [82.04393505253105-76.88634192932187, 73.35243553008596-67.90830945558739, 97.13467048710602-93.02769818529131, 55.491881566380115-51.00286532951293, 58.261700095511-53.19961795606499, 97.13467048710602-92.93218720152818, 94.46036294173828-90.06685768863419, 93.60076408787012-89.01623686723974]
	values_esec = [83.9541547277937, 88.15663801337153, 97.134670487106, 67.90830945558739, 65.99808978032475, 94.84240687679082, 93.8872970391595, 95.79751671442216]
	errors_esec = [88.53868194842408-83.9541547277937, 93.31423113658072-88.0611270296084, 100.38204393505252-97.134670487106, 72.49283667621779-67.9083094555874, 70.20057306590257-65.99808978032475, 98.75835721107927-94.84240687679082, 98.08978032473735-93.98280802292265, 99.71346704871061-95.79751671442216]
	
	#load diagonal values from confusion matrix
	values_e2sec = new_sorted_dissi.diagonal()

	#define labels for bar plot
	labels = ["Put on top", "Take down", "Pushing", "Cutting", "Chopping", "Stirring", "Hiding", "Uncover"]
	labels_numbers =  np.asarray([0,1,2,3,4,5,6,7])
	
	#print average classification accuracy of the classification
	print("Average classification accuracy e2SEC:",np.mean(values_e2sec),"+-", np.std(values_e2sec))

	#define the details for the bar plot, plot and save figure as png
	fig, ax = plt.subplots(1,1, figsize = (21,8))
	width = 0.13

	ax.bar(labels_numbers-2*width, values_e2sec, yerr = confi_interval_95_sorted_new, capsize = 5, width=width,color='g', label = "e$^2$SEC", edgecolor='k')
	ax.bar(labels_numbers-width, values_esec_cleaned, yerr = errors_esec_cleaned, width=width,  capsize = 5, color='m', label = "eSEC de-noised", edgecolor='k')
	ax.bar(labels_numbers, values_esec, yerr = errors_esec, width=width,  capsize = 5, color='r', label = "eSEC", edgecolor='k')
	ax.bar(labels_numbers+width, values_sec_cleaned, yerr = errors_sec_cleaned ,width=width,  capsize = 5, color='c',label = "SEC de-noised", edgecolor='k')
	ax.bar(labels_numbers+2*width, values_sec, yerr = errors_sec ,width=width,  capsize = 5, color='b',label = "SEC", edgecolor='k')

	plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize = 20)
	ax.set_xticks(labels_numbers)
	ax.set_xticklabels(labels,  fontsize=18)
	ax.set_yticklabels(np.arange(0, 110, step=20), fontsize=25)
	ax.set_ylim(0,110)

	ax.set_ylabel("Classification accuracy [%]",  fontsize=35)
	plt.tight_layout()
	plt.savefig("classification_accuracy.png", dpi = 350)

def cluster_e2sec(location_e2sec_matrices):
    """
    Plots and saves the dendrogram of the clustering result from 120 e2SEC matrices using the MANIAC dataset.

    Parameters:
    	* location_e2sec_matrices: folder of all 120 e2SEC matrices saved as .npy file
    """
	#load the e2SEC matrices in a dict
    loc = location_e2sec_matrices
    e2sec_matrices_without_filter = {}
    i = 0
    for file in sorted(os.listdir(loc)):
        if file[-3:] == "npy":
            e2sec_matrices_without_filter[i] = np.load(loc+file)
        i += 1

	#assign the lables to the e2SEC matrices
    i = 0
    labels = []
    for file in sorted(os.listdir(loc)):
        if file[-3:] == "npy":
            if i < 15:
                labels.append(file[15:-4])
            if i >= 15 and i < 30:
                labels.append(file[14:-4])
            if i >= 30 and i < 45:
                labels.append(file[13:-4])
            if i >= 45 and i < 60:
                labels.append(file[14:-4])
            if i >= 60 and i < 75:
                labels.append(file[17:-4])
            if i >= 75 and i < 90:
                labels.append(file[15:-4])
            if i >= 90 and i < 105:
                labels.append(file[16:-4])
            if i >= 105:
                labels.append(file[14:-4])
        i += 1

	#decode to UTF-8
    for i in range(len(e2sec_matrices_without_filter)):
        e2sec_matrices_without_filter[i] = np.char.decode(e2sec_matrices_without_filter[i].astype(np.bytes_), 'UTF-8')
    
    #find max column length
    column_length = []
    for i in range(len(e2sec_matrices_without_filter)):
        column_length.append(e2sec_matrices_without_filter[i].shape[1])
    max_column = np.max(column_length)

    #repeat lact column of every e2SEC matrix until all have same length
    i = 0
    while i < len(e2sec_matrices_without_filter):
        if e2sec_matrices_without_filter[i].shape[1] < max_column:
            add = np.reshape(e2sec_matrices_without_filter[i][:, e2sec_matrices_without_filter[i].shape[1]-1], (24, 1))
            e2sec_matrices_without_filter[i] = np.append(e2sec_matrices_without_filter[i], add, 1)
        if e2sec_matrices_without_filter[i].shape[1] == max_column:
            i += 1
	#remove arrays folder if present
    os.system("rm -r arrays")
	#calculate dissimilarity and save output in arrays folder
    removeCobinationRowsSave(e2sec_matrices_without_filter, rows = [])
	#load dissimilarity 
    D_shaped = np.load("arrays/nothing_removed/nothing_removed.npy")
	#plot dendrogram
    plotDendrogramFromMatrix(D_shaped, labels = labels, figsize = (35,50), threshold = 0.31, save = True, lablesize = 38, fontsize = 45, name_of_plot = loc[:-1])
    #remove arrays folder
    os.system("rm -r arrays")
