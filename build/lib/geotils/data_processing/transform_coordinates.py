import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import re
import numpy as np

def transform_coordinates(B = 0.34527, D = -0.232645, A = 0.0000026948989 , C = 99507.421, F = 3115009.014, file=None):
	
	E = A
	excelFile=pd.read_excel(file)
	#columns = excelFile.columns.tolist()

	# Loop through indices and rows in the dataframe using iterrows
	f = open("BoundingBox.txt","w")
	for index, row in excelFile.iterrows():
		# Loop through columns
		cell = str(row[2])
		name = cell.split("/")[1]
		name = name.split("_")[0]
		# If we find it, print it out
		if name == "20170828bC0970430w280600n" and int(row[0]) == 8035:
				mx = float(row[9])
				my = float(row[8])
				
				a = np.array([[A,B], [D,E]])
				b = np.array([mx - C, my - F])
				Sol1 = np.linalg.solve(a, b)
				Sol1 = [int(round(i)) for i in Sol1]
				
				mx = float(row[7])
				my = float(row[10])
				a = np.array([[A,B], [D,E]])
				b = np.array([mx - C, my - F])
				Sol2 = np.linalg.solve(a, b)
				Sol2 = [int(round(i)) for i in Sol2]
				
				if row[1] == "none":
					f.write("0 ")
				else:
					f.write("1 ")
				f.write(str(Sol1[0]))
				f.write(" ")
				f.write(str(Sol1[1]))
				f.write(" ")
				f.write(str(Sol2[0]))
				f.write(" ")
				f.write(str(Sol2[1]))
				f.write("\n")
				break
	exit()