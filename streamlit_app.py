import streamlit as st
import pandas as pd
from getdiaryentries import get_diary_from_image
import os


def main():
    st.set_page_config(layout="wide")
    st.title('MigrAIne',)
    
    # File upload widget
    uploaded_file = st.file_uploader('Choose an image file', type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display the uploaded image
        #st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        impath = 'tmp/'+uploaded_file.name
        with open(impath, 'wb') as file:
            file.write(uploaded_file.getvalue())
        
        # Process the image
        result_df = get_diary_from_image(impath)
        
        # Display the resulting DataFrame as a table
        st.table(result_df)

        os.remove(impath)

if __name__ == '__main__':
    main()