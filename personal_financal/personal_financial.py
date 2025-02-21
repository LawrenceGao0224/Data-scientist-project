import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# App title
st.title('💰 Personal financial prediction')


data = pd.read_excel("financial.xlsx", sheet_name=None)
df = pd.ExcelFile("financial.xlsx")
sheet_names = df.sheet_names
all_sum = []
for sheet in sheet_names:
    all_sum.append(data[sheet].iloc[32,-1])
year = []
month = []
for i in range(len(sheet_names)):
    year.append(int(sheet_names[i].split('.')[0]))
    month.append(int(sheet_names[i].split('.')[-1]))
dict = {"year": year, "month": month, "cost": all_sum}
clean_data = pd.DataFrame(dict)


# Show data preview
if st.checkbox('Show raw data'):
    st.write(clean_data)
display = pd.DataFrame({"Y": all_sum}, index=sheet_names)
st.line_chart(display, x_label ="Date", y_label="Cost")

#x = sheet_names  # X-axis points
#y = all_sum  # Y-axis points

#plt.plot(x, y)  # Plot the chart
#st.pyplot(plt.gcf())  # display


# Preprocessing
numerical_features = ['month']
categorical_features = ['year']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
X = clean_data[['year', 'month']]
y = clean_data['cost']
model.fit(X, y)

# User input form
st.sidebar.header('Enter year/month for prediction')

year = st.sidebar.slider('Year', 2025, 2035, 2025)
month = st.sidebar.slider('Month', 1, 12, 3)


# Prediction
if st.sidebar.button('Estimate Price'):
    input_data = pd.DataFrame([[year, month]],
                             columns=['year', 'month'])
    
    prediction = model.predict(input_data)
    formatted_price = "${:,.2f}".format(prediction[0])
    
    st.success(f'**Estimated Property Value:** {formatted_price}')
    st.balloons()

# Model info
st.markdown("---")
st.subheader("Model Information")
st.write("Algorithm: Linear Regression")
st.write("Features Used:")
st.write("- Numerical Features: Month")
st.write("- Categorical Features: Year")