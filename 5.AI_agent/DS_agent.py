import streamlit as st
import pandas as pd
import os
from pathlib import Path

def create_upload_directory():
    """Create a directory for uploaded files if it doesn't exist"""
    upload_dir = Path("uploaded_files")
    upload_dir.mkdir(exist_ok=True)
    return upload_dir

def save_uploaded_file(uploaded_file, upload_dir):
    """Save the uploaded file to the specified directory"""
    file_path = upload_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def main():
    st.title("CSV File Analyzer")
    
    # Create upload directory
    upload_dir = create_upload_directory()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        # Save the uploaded file
        save_uploaded_file(uploaded_file, upload_dir)
        st.success(f"File {uploaded_file.name} successfully uploaded!")
    
    # Get list of uploaded files
    csv_files = list(upload_dir.glob("*.csv"))
    
    if csv_files:
        st.write("### Select files to analyze")
        # Create multiselect for uploaded files
        selected_files = st.multiselect(
            "Choose CSV files",
            options=csv_files,
            format_func=lambda x: x.name
        )
        
        if selected_files and st.button("Go"):
            # Create tabs for each selected file
            tabs = st.tabs([file.name for file in selected_files])
            
            # Display data in tabs
            for tab, file in zip(tabs, selected_files):
                with tab:
                    try:
                        df = pd.read_csv(file)
                        st.write(f"First few rows of {file.name}:")
                        st.dataframe(df.head())
                        st.write(f"Shape: {df.shape}")
                    except Exception as e:
                        st.error(f"Error reading {file.name}: {str(e)}")

if __name__ == "__main__":
    main()
