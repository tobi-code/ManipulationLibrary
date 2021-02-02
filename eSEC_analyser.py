#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2020

@author: Tobias Strübing
"""
import tabula
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from itertools import combinations
import os
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from scipy.spatial import distance
import seaborn as sn


def readPDF(PDFpath):
	'''
	Read PDF in a specific foramt and outputs a dict of the eSEC matrices. 
	In the PDF file there has to be one eSEC matrix per page with the relations 
	on the left side and without titles.
	
	Parameters:
		* PDFpath: path of the PDF file (PDF)
		
	Returns:
		python dictionary with the ESEC matrices
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
	This dictionary can be recived by the function readPDF().
	
	Parameters:
		* esec_dict: dictionary with ESEC matrices (dict)
		* savefig: paramter if figure need to be saved (bool)
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
	Removes the rows assigned in the parameter from all the ESEC tables.
	
	Parameters:
		* esec_dict: dictionary with the ESEC tables (dict)
		* row: indicated which rows will be deleted (int, tuple)
	
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
	First removes single rows and then combinations of tuple, triple, quadruple and quintuple rows. 
	Rows can have a maximum of 5 values.
	If rows == None the Dissimilarity matrix is calculated without removing rows and output is saved as d_shaped.npy.
	example:    rows = [1,2,3] 
				first step: remove single rows 1, 2, 3
				second step: remove combinations of two 1/2, 1/3, 2/3 
				third step: remove combinations of three 1/2/3
	
	Parameters:
		* esec_dict: dictionary with the ESEC tables (dict)
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
	This functions checks if manipulations are the same in case of removes rows. The input parameter combinations defines how much combinations are considered
	(e.g. 1 for single combinations, 3 for triple combinations). 
	
	Parameters:
		* esec_dict: dictionary with the ESEC tables (dict)
		* combination: which type of combination (int)
		* rows: rows that will be removed (array) (from 0 for row 1 to 9 for row 10)
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
			print("Same manipulations due to removement of row(s)", value)
	
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
			print("Same manipulations due to removement of row(s)", value)

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
			print("Same manipulations due to removement of row(s)", value)
	
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
			print("Same manipulations due to removement of row(s)", value)

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
			print("Same manipulations due to removement of row(s)", value)
		
def plotDendrogram(rows, labels, threshold = 0.4, save = False):
	'''
	| Plots the dendrogram for specific rows for the .npy arrays.
	| Warning: needs the output from removeCobinationRowsSave()

	Parameters:
		* rows: indicates which rows will be considered (int, tuple) 
		* label: name of the manipulations in the right order [string array]
		* threshold: color threshold for the dendrogram (float)
		* save: paramter if figure need to be saved (bool)
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

def plotDendrogramFromMatrix(D_shaped, labels, figsize = (30,16), threshold = 0.4, save = False, name_of_plot = ""):
	'''
	| Plots the dendrogram for specific rows for the .npy arrays.
	| Warning: needs the output from removeCobinationRowsSave()

	Parameters:
		* D_shaped: dissimilarity matrix of all manipulations from function removeCobinationRowsSave() (numpy array)
		* labels: name of the manipulations in the right order [string array]
		* figsize: size of plot in case labels are squeezed
		* threshold: color threshold for the dendrogram (float)
		* save: paramter if figure need to be saved (bool)
		* name_of_plot: name of the plot that will be saved (string)
	'''
	
	fig = plt.figure(figsize=figsize, facecolor='white')

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
		plt.savefig("dendrogram_%s.png"%name_of_plot, bbox_inches = 'tight')


def plotDissi(rows, label, save = False):
	'''
	| Plots the dissimilarity matrix for specific rows for the .npy arrays.
	| Warning: needs the output from removeCobinationRowsSave()

	Parameters:
		* rows: indicates which rows will be considered (int, tuple) 
		* label: name of the manipulations in the right order [string array]
		* save: paramter if figure need to be saved (bool)
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
		* save: paramter if figure need to be saved (bool)
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
		* save: paramter if figure need to be saved (bool)
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
		* save: paramter if figure need to be saved (bool)
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
    Takes an pdf file with eSEC matrices as input and return these matrices as e2SEC matrices dict.
    
    Parameters:
        * pdf_path: path of the pdf file which contains eSEC matrices
        
    Returns:
        dict of e2SEC matrices
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
        #find index of sematics to merge
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
    Takes a dict of eSEC matrices as input to calculate the dissimilarity matrix from paper.
    
    Parameters:
        * eSEC_matrices: dict contains eSEC matrices
        
    Returns:
        dissimilarity matrix
    '''
	return _calc_D_shaped(_make_triple(eSEC_matrices))

def _compare_manipulations(manipulation_1, manipulation_2):
    '''
    Compares manipulations
    
    Parameters:
        * manipulation_1: first eSEC manipulation
		* manipulation_2: second eSEC manipulation

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
	D = np.zeros(len(table_triples)*len(table_triples))
	#k = -1
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

def _make_triple(liste_array):
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

def _compare_triple_matrics(manipulation_1, manipulation_2):
	'''
    Compares triple matrices (TNR, SSR, DSR) from two manipulations
    
    Parameters:
        * manipulation_1: triples of first manipulation
		* manipulation_2: triples of second manipulation

    '''
	
	#if the length of the triple matrices dont match repeat last
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
	return np.delete(manipulation, [relation,relation+10,relation+20], 0)

def _delete_rows_two(combination, manipulation):
	relation_1 = combination[0]
	relation_2 = combination[1]
	return np.delete(manipulation, [relation_1,relation_1+10,relation_1+20, relation_2,relation_2+10,relation_2+20], 0)

def _delete_rows_three(combination, manipulation):
	relation_1 = combination[0]
	relation_2 = combination[1]
	relation_3 = combination[2]
	return np.delete(manipulation, [relation_1,relation_1+10,relation_1+20, 
									relation_2,relation_2+10,relation_2+20,
									relation_3,relation_3+10,relation_3+20], 0)

def _delete_rows_four(combination, manipulation):
	relation_1 = combination[0]
	relation_2 = combination[1]
	relation_3 = combination[2]
	relation_4 = combination[3]
	return np.delete(manipulation, [relation_1,relation_1+10,relation_1+20, 
									relation_2,relation_2+10,relation_2+20,
									relation_3,relation_3+10,relation_3+20,
									relation_4,relation_4+10,relation_4+20], 0)
def _delete_rows_five(combination, manipulation):
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
		
#https://stackoverflow.com/questions/33272588/appending-elements-to-an-empty-dictionary-of-lists-in-python
def _add_element(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)
