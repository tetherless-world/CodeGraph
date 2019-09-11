# import csv
# import numpy as np

# # Leemos el fichero por defecto para hacer envios
# file = csv.reader(open('sampleSubmission.csv'))

# # Los datos leidos del csv lo convertimos en una lista
# data_file = list(file)

# # A continuación, lo convertimos en un numpy array
# data_file_array = np.array(data_file)

# # Utilizando el "advanced slicing" de los vectores numpy
# # le decimos que ponga las prediciones a partir de la segunda fila con "1:"
# # para no modificar los header del archivo de envío y lo ponemos en la segunda
# # columna poniendo "1"

# data_file_array[1:, 1] = predictions

# #Una vez modificado los datos, lo guardamos en un csv utilizando el siguiente código
# with open('output.csv', 'w') as f:
#   csv.writer(f).writerows(data_file_array)