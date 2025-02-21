import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# App title
st.title('🏠 Real Estate Price Predictor')

# Function to load data
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    else:
        # Sample data
        data = pd.DataFrame({
            'price': [450000, 600000, 375000, 900000, 550000],
            'bedrooms': [3, 4, 2, 5, 3],
            'bathrooms': [2, 3, 1, 4, 2],
            'lot_size': [0.25, 0.5, 0.3, 1.0, 0.4],
            'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
            'state': ['NY', 'CA', 'IL', 'TX', 'AZ']
        })
    return data

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
data = load_data(uploaded_file)
data = data.dropna(axis=0, how='any')

# Show data preview
if st.checkbox('Show raw data'):
    st.write(data)

# Check required columns
required_columns = ['price', 'bedrooms', 'bathrooms', 'lot_size', 'city', 'state']
if not all(col in data.columns for col in required_columns):
    st.error(f"Dataset must contain these columns: {', '.join(required_columns)}")
    st.stop()

# Preprocessing
numerical_features = ['bedrooms', 'bathrooms', 'lot_size']
categorical_features = ['city', 'state']

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
X = data[['bedrooms', 'bathrooms', 'lot_size', 'city', 'state']]
y = data['price']
model.fit(X, y)

# User input form
st.sidebar.header('Enter Property Details')

bedrooms = st.sidebar.slider('Bedrooms', 1, 8, 3)
bathrooms = st.sidebar.slider('Bathrooms', 1, 6, 2)
lot_size = st.sidebar.slider('Lot Size (acres)', 0.1, 10.0, 0.5)
city = st.sidebar.selectbox('City', data['city'].unique())
state = st.sidebar.selectbox('State', data['state'].unique())

# Prediction
if st.sidebar.button('Estimate Price'):
    input_data = pd.DataFrame([[bedrooms, bathrooms, lot_size, city, state]],
                             columns=['bedrooms', 'bathrooms', 'lot_size', 'city', 'state'])
    
    prediction = model.predict(input_data)
    formatted_price = "${:,.2f}".format(prediction[0])
    
    st.success(f'**Estimated Property Value:** {formatted_price}')
    st.balloons()

# Model info
st.markdown("---")
st.subheader("Model Information")
st.write("Algorithm: Linear Regression")
st.write("Features Used:")
st.write("- Numerical Features: Bedrooms, Bathrooms, Lot Size")
st.write("- Categorical Features: City, State")