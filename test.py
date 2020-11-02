import eSEC_analyser as ell
import os

#print(os.listdir("."))
dicti = ell.readPDF("/home/sazu/ownCloud/ManipulationLibrary/esec_marti.pdf")
#ell.plotRowRanking(dicti)
ell.removeCobinationRowsSave(dicti, rows = [2])
#ell.checkSimilarRows(dicti, 5)
