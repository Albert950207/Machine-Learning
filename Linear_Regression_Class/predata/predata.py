import pandas as pd
import numpy as np

class preprocessor:


    def __init__ (self, path):
        self.path = path
        self.data = np.array(pd.read_csv(path, header=None, sep='\s+'))
        print(self.data)

    def Transpose (self):
        self.data = self.data.T

    def tofloat(self,row):
        error_element_index = []
        for element_index in range(0,len(row)) :
            try:
                row[element_index] = float(row[element_index])
            except:
                error_element_index.append(element_index)
                row[element_index] = 0.0
        for i in error_element_index:
            row[i] = row.mean()
        return row

    def clear(self, index, key):

        for row in self.data:
            row = self.tofloat(row)

        if key == 'colum':
            self.data = np.delete(self.data, index, 0)

        elif key == 'row':
            self.data = np.delete(self.data, index, 1)


    def Normalization(self):

        for row in self.data:
            for element_index in range(0,len(row)) :
                row[element_index] = (row[element_index]- np.min(row))/(np.max(row)-np.min(row))
