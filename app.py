import streamlit as st
st.title("Machine Learning- iris")
sepal_length = st.slider('Sepal Length', 0.1, 7.9, 2.0)
sepal_width = st.slider('Sepal width', 0.1, 7.9, 2.0)
petal_length = st.slider('petal Length', 0.1, 7.9, 2.0)
petal_width = st.slider('petal width', 0.1, 7.9, 2.0)



from sklearn.datasets import load_iris
iris = load_iris()
x= iris.data
y= iris.target

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x,y)
y = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
y = iris.target_names[y[0]]
