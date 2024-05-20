import streamlit as st
import pandas as pd
import numpy as np
from getdiaryentries import get_diary_from_image
import os


def click_change_button(lab, r, c):
    st.session_state.diary_df.iloc[r, c] = lab
    st.rerun()

@st.experimental_dialog("Diary stats", width='large')
def show_diary_stats():
    H_count = 0
    M_count = 0
    df = st.session_state.diary_df
    for index, row in df.iterrows():
        for column in df.columns:
            if row[column] == "H":
                H_count = H_count + 1
            if row[column] == "M":
                M_count = M_count + 1
            
    st.text(f"Headache days: {H_count}")
    st.text(f"Migraine days: {M_count}")
@st.experimental_dialog("Diary image", width='large')
def show_diary_image():
    st.image(st.session_state.imfile, caption='Uploaded Image', use_column_width=True)

@st.experimental_dialog("Change diary entry")
def click_date_button(r, c, k):
    cols=st.columns(4)
    with cols[0]:
        if st.button(" ", key="modal_btn_blank", use_container_width=True):
            click_change_button(" ", r , c)

    with cols[1]:
        if st.button("M", key="modal_btn_M", use_container_width=True):
            click_change_button("M", r , c)

    with cols[2]:
        if st.button("H", key="modal_btn_H", use_container_width=True):
            click_change_button("H", r , c)


def main():
    st.set_page_config(layout="wide")
    st.title('MigrAIne',)

    if 'diary_df' not in st.session_state:
        # File upload widget
        uploaded_file = st.file_uploader('Choose an image file', type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Display the uploaded image
            #st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            
            st.session_state.imfile = uploaded_file
            impath = 'tmp/'+uploaded_file.name
            with open(impath, 'wb') as file:
                file.write(uploaded_file.getvalue())
            
            # Process the image
            diary_df = get_diary_from_image(impath)
            st.session_state.diary_df = diary_df

            os.remove(impath)
            st.rerun()

        #
    if 'diary_df' in st.session_state:
        cols = st.columns(31)
        c = 0
        for col in cols:
            with col:
                st.text(str(c+1))
                for r in range(0, 6):
                    k = "btn_" + str(r) + "_"+str(c)
                    lab = st.session_state.diary_df.iloc[r, c]
                    if st.button(lab, key=k, use_container_width=True):
                        click_date_button(r, c, k)
            c = c + 1

        cols = st.columns(6)
        with cols[2]:
            if st.button("show image"):          
                show_diary_image()
   
        with cols[3]:
            if st.button("calculate diary stats"): 
                show_diary_stats()
        
if __name__ == '__main__':
    main()