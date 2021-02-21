import streamlit as st
import pandas as pd 
import joblib
from sklearn.datasets import load_iris
from PIL import Image
iris= load_iris()


# Store features matrix in X
X= iris.data
#Store target vector in y
y= iris.target
# Finalizing KNN Classifier after evaluation and choosing best 
# parameter
#Importing KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
# instantiate the model with the best known parameters
knn = KNeighborsClassifier(n_neighbors=12)
# train the model with X and y (not X_train and y_train)
knn_clf=knn.fit(X, y)


st.title("Iris flower species Classification App")
#Loading images
setosa= Image.open('setosa.png')
versicolor= Image.open('versicolor.png')
virginica = Image.open('virginica.png')

st.sidebar.title("Features")

#Intializing
parameter_list=['Sepal length (cm)','Sepal Width (cm)','Petal length (cm)','Petal Width (cm)']
parameter_input_values=[]
parameter_default_values=['5.2','3.2','4.2','1.2']

values=[]

#Display
for parameter, parameter_df in zip(parameter_list, parameter_default_values):
	
	values= st.sidebar.slider(label=parameter, key=parameter,value=float(parameter_df), min_value=0.0, max_value=8.0, step=0.1)
	parameter_input_values.append(values)
	
input_variables=pd.DataFrame([parameter_input_values],columns=parameter_list,dtype=float)
st.write('\n\n')

if st.button("Click Here to Classify"):
	prediction = knn_clf.predict(input_variables)
	st.image(setosa) if prediction == 0 else st.image(versicolor)  if prediction == 1 else st.image(virginica) 
































