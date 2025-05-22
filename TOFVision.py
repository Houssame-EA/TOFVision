"""


Université de Montréal


Département de Chimie


Groupe Prof. Kevin J. Wilkinson
Boiphysicochimie de l'environnement


Aut: H-E Ahabchane, Amanda Wu

Date : 01/01/2024
Update :22/05/2025
"""


import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
import numpy as np
import itertools
import \
    plotly.graph_objects as go
import re
import os
import time
from io import StringIO
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, MiniBatchKMeans, MeanShift, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances
from kneed import KneeLocator
from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import NearestNeighbors
from statsmodels.nonparametric.smoothers_lowess import lowess
import plotly.subplots as sp
from io import BytesIO
import warnings
from collections import defaultdict
from mendeleev import element
from scipy.stats import poisson, norm, expon, binom, lognorm, gamma, weibull_min
from scipy.interpolate import CubicSpline
from scipy import stats

warnings.filterwarnings('ignore')



st.set_page_config(page_title='TOFVision', page_icon=':atom_symbol:', layout='wide')



logo_url = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBwgHBgkIBwgKCgkLDRYPDQwMDRsUFRAWIB0iIiAdHx8kKDQsJCYxJx8fLT0tMTU3Ojo6Iys/RD84QzQ5OjcBCgoKDQwNGg8PGjclHyU3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3N//AABEIAFwAXAMBEQACEQEDEQH/xAAcAAEAAgIDAQAAAAAAAAAAAAAABQcEBgIDCAH/xAA6EAABAwMCBAQDBAgHAAAAAAABAgMEAAURBiESEzFBByJRYRQykRVxgaEWM0JSYqKxwSMkQ3OS4fD/xAAbAQEAAgMBAQAAAAAAAAAAAAAAAQQDBQYCB//EACsRAAICAgEDAwIGAwAAAAAAAAABAgMEETEFEiETQVEyYQYUI3GRoSKBwf/aAAwDAQACEQMRAD8AvCvJIoBQCgIyDqC0T7hJt8S4R3JsZZQ9H4sLSR18p3I9xtUgk6gCgFAKAUAoBQCgFAQeo9X2LTK2G71PEdb4UptPLUskDqcJBxUgkLTdIV4gNTrZIRIjOpCkrT/cdQfY70B5q8VmHIXiLeS0VNrLqHm1pOCCptJyD9+akgs6/eLH6PM2lgWpU1cq2MS+aZHAPOCMfKe6agksPT1z+2bFb7oGuV8ZGbe5fFxcHEkHGe+M0BIUAqAKAUAoBQCgKQ8StBaq1Drl+VBiB2C8ltLTy5ACGgEAHIJyNwThIOc+uakgn9Fw7d4VW+Y3qi7xW350lJbDalKCkBIAIRjiG5Vk4x0oCv8AxrUxJ1excYTqHo023tOtuoOQsZUMj6CpBFavWJFg0dJG5NsXHJ/2nVJ/vQk3lnxJlaR0hpSNHtzMsP2/iUXHSgp4FcGBgH0oQWzpm6G96ft91UyGTLYS8Wwri4cjOM96hkknUAUAoBQCgFAKkHnTx1gmLrsyMHhlxG3AfcZSR/KPrQg1yY45eLFYI8Vtx+VCRIYdCUnyoLgWjJ6dFKH4VkhXOf0oxW31VebJJGXLtl0l6etNv+BKXIC5B41OpwpLikqAAz2IP1rOsK/4Kb6tiJ/UYl/TNNrs8aRCfb+z47jK1kApPE6pYwQT2I64rFOiyH1RZYqzKLvokj0X4dFP6B2DhUCBAayQc78IzWEtE+l5paihDqFKHVIUCRQHZQCoBxUkk5CsUByoCG1bqOHpayPXOdlSUEJbaT8zqz0SP/bDJqSCp7Nr5nVE/kagvFyty3VcLMeIv4eON9k8xJ4yfdRA9hWp6nZn1xcsZLS/ky1qD+ol9UeHbV3baeYuM1yRGB5Tc+Qt9tQzkpJUeIA47H8K0eJ+JLYWJZEU1/BksxlKLUHpmvwVDlKZ+H+GdYWWnY+McpY6j/vuDX1LCyKsilWVcM+e9Qx7aL3G17fyZFWyicVqShClrOEpGVE9hXmTSXk9VxbklHkhoi2UJJmTHYMOceNuAh5bbTmP2lgHBUc5x9c1x+ZlStsl6K8HZU+rXUq9ttcs6r+m1Wb4QM2xlLzq8JWyOWtABGSCMHO+29VMd22Nvu4MlLsnt93BZOh9TzYt1Ysd5krlMSciFKdOXErAzy1n9rIBIV12wc7Vax7/AFVp8lim71OeSyasFgVAFAVJ432+5Xy76bs9vTxB7nueY4SCngHET7An61hyMivGqdtnCJjFyekV3rDw/n6YgtTHJLUyMpQQ4ptBSW1HoMEnIPr+Va/p/Wqc2bhFaa+T3ZU4rZtnh5cNTGY1bFSFSm2eAy/iBlMRoZw2CNy6fQ54QPvxq+s04SrdjWm+Nct/P7GWqU96OzVj8aFrG5LcdbabVGjqWVHGXPMPxPCE/lXRfg+3twW7H434Od/EFErrIKtbfk7IVq1BcI5kwbDKVHxlKpC0sKcH8KVHP1xXRS6jBPSWzWw6FbKO5SSZCXhxT1muTfKdakMoUh5hxPC42oDdJH3fWstlqux5OBXpxZYubCNvycdTWkXi1tri+Z1pPEyAdlpI6f0ri6LfSsal7nQVWenNqRENRlzk2z7YdUy9CWtLoV1CUgLBV+A6+lWHJQ7uz3M7ko93Z52bZHcVMudkEZDiHXLmwW+NPCSAoKUcdflCutYsWLVpjx4tWF51szYCoAoCA1dZn7pFjyLa4hu5wXedFU58ijjCm1fwqTke2x7VhyaIZFTqnwyYtxezVLhfrVPhvWq8v/YlxUjdqaEpW0rstBV5VYO4UD9K4t9MzcG9TjHvivj3Rb9SE48mozhYLO7CRpy5zprafLNh26Y4nnerpcSeEL9id+m1b/plWZmTk8qja9m1x9v2KmRfTRHbnosTRVv0bKJuVhaakTB+tdlKU7KbPormEqSa3agoLtS1r2PKl3LuT2Yd+iarOtG5cR2U1p0Ox/ikplNgqwCStIUPK2NuMZyrG1SDVL/cIt61ZcLlbyFwi03GS4PlfUji4lj1Hm4c9+H0rb9Orfa5PhnNdcvi5xhHlEHypUBsR20vuwc+XkKAdaH7u/VP3bjpWuzekS73ZSt/YnG6hVYkrPEv6ZDXDmyL3bm4bUyQXnUtJbebKFrOdkcSsBXfqdsneqcce2EGpx0biiUZxai1/osvwo+zrldH586U19tx+Y0i2k+aIgK4VK/iUcDKhsBt60ppVUdIsVVKtaRatZjKKgCgFAVn4rQ+VebRcVISWHm3IbhIzhWy0fXCxV7Aklbp+5qur1ylj90eUawAEjCQAPQVvVFLg46UpPlmHcGYKQJkvhZW38shKyhafuUCDWC+ulrutLuHflxl2UNt/BiLcRLb/wAdm+T2AflkOPOI/wCK1b/StR+b6dCWmzofyXW7a960Z8KXGkpKI6sFvZTZSUKR7FJ3Fbii6q2O62c3l4uRjy1dFpmTWcqGRp6EbnrWxspGREcXNcP7qUp4R/MpP51q+pSWoxOh6DW+6c/bggNd2ib4da5YvNoymK+6X4x34Qf9RpXtufwPtWpOlL305eomobLFukBWWZCM8J6oV3SfcHaoJJKoAoBQEXqaysags0i3SVFAcAKHEjdtYOUrHuCAalNp+DzKKktMpz/MRJr1sujYZuMf9Y32cT2cR6pP5dDuK6HGyY3R+5xfUMCWNNtfT7ETIeTy5N0eWjLb3wsEOfKhWeFTmPXOfuCa5rqmTO7J9FcLk7XoODXi4X5iS/ykZouAbt/IsMeRLUk8AfCMo4s+ZXErHEep271pXVuzuuaX2Ok9bVfbQm/ucpcF1+Hz22XmpsZJUy88tJW53KVYJ2Pp27dKy4uXLGuUovx/RXzcCGZjuuyPnXPufftBj4FmXuUvJSW0JGVLUrolI7k9MV3rvjGv1JHyeOHZK90xXlMs7w8049Z4Ts65ICbnPwp1GQeQ2Plbz7ZJPuTWgutds+5naYuPHHqVcST1npuNqqwSLXJIQpQ4mXsZLTg+VQ/ofUE1iLJCeF2jbjo6DNYuFxbkiS6FoZZB4GyBgkE9zt27CgN3qAKAUAoCD1Tpa2anipauLakut5LEllXC6yfVKv7HapTae0eZRUlpooG/6Ju0O+S7OZyXhGVzWC+Snmtr34wBkdcg+4+6q998KX3SXPuW8fGlfHsi/C9jOtsW+MsN25dxRHdaRltAYSpK2wQNl+u+OnfvWttnjyfqKO0za015MV6blpr7f9MxSFwmHpcgy/ikeVKFqS6HVHZKUHh7nbAwa8x/UkoQS0zJL9KDsm3tfPn+Cy9CaBh6ejRJM5Spl0baCQ47jhj7bpbT0HpxdT69q30pyaSb4OZjXCLbiuTda8GQUAoBQCgFAKAUBCam0xA1Ey0JXMZksEmPLYPC60T1we4PcHINRKKktSXg9RnKD3FmoOaB1ChzDN2tjyRsl16KtKwPcJVg/hiqT6dU+GzYLqlyXlLZNad0Izbpzdyu0xVynNbs5QEMsH1Qjfze5JPpirNVEKVqCKd+RZe9zZuFZTCKAUAoBQCgFAKAUAoBUgUAqAKAUAoBQH//2Q=="



second_logo_url = 'https://upload.wikimedia.org/wikipedia/en/thumb/4/4b/Universite_de_Montreal_logo.svg/2560px-Universite_de_Montreal_logo.svg.png'

st.markdown(f"""
    <div style="display:flex; justify-content: space-between; align-items: center;">
        <a href="https://kevinjwilkinson.openum.ca">
            <img src="{logo_url}" alt="First Logo" style="height:100px; display: block; margin-top: 60px;">
        </a>
        <a href="https://www.umontreal.ca/en/">
            <img src="{second_logo_url}" alt="Second Logo" style="height:100px;">
        </a>
    </div>
    """, unsafe_allow_html=True)

st.title(""":atom_symbol: TOFVision""")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
st.markdown(
    """
    <style>
        .bottom-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f4f4f4;
            padding: 10px;
            text-align: center;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            z-index: 1000;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="bottom-bar">
                      Kevin J. Wilkinson Laboratory Environmental Biophysicochemistry - Department of Chemistry - Université de Montréal
    </div>
    """,
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["Single File Analysis", "Multi Files Analysis", "Isotopic Ratio Analysis"])





with tab1:

    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = None
    
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
        
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
        
    if 'element_selection_confirmed' not in st.session_state:
        st.session_state.element_selection_confirmed = False

    data_type = st.radio(
        "Select data structure type:",
        ["NuQuant", "SPCal", "IsoTrack"],  
        key='data_type_selection',
        on_change=lambda: setattr(st.session_state, 'data_processed', False)
    )

    combine_files = st.checkbox('Combine multiple files?', key='combine_files')


    if combine_files:
        fl = st.file_uploader(
            ':file_folder: Upload files',
            type=['csv'],
            accept_multiple_files=True,
            key='multiple_files',
            on_change=lambda: setattr(st.session_state, 'uploaded_files', st.session_state.multiple_files) or setattr(st.session_state, 'data_processed', False)
        )
    else:
        fl = st.file_uploader(
            ':file_folder: Upload a file',
            type=['csv'],
            key='single_file',
            on_change=lambda: setattr(st.session_state, 'uploaded_files', [st.session_state.single_file] if st.session_state.single_file else None) or setattr(st.session_state, 'data_processed', False)
        )
        
    if fl:
        with st.sidebar:
            st.title("Analysis Options")

    def create_periodic_table_selector(data_type, data_obj):
        """
        Creates a periodic table UI for element selection without causing reloads when selecting elements.
        Only processes data when Apply button is clicked.
        """
        VALID_ELEMENTS = [
            "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", 
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", 
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
            "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", 
            "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", 
            "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", 
            "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", 
            "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", 
            "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", 
            "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
            "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
            "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
        ]
   
        available_elements = []
        

        if data_type == "SPCal":
            for col in data_obj.columns:
                if not isinstance(col, str):
                    continue
                element = ''.join(c for c in col if c.isalpha())
                if element and element in VALID_ELEMENTS:
                    available_elements.append(element)
                    
        elif data_type == "NuQuant":
            for col in data_obj.columns:
                if not isinstance(col, str):
                    continue
                element = col.strip()
                if element in VALID_ELEMENTS:
                    available_elements.append(element)
                    
        elif data_type == "IsoTrack":
            for col in data_obj['mass'].columns:
                if not isinstance(col, str):
                    continue
                element = ''.join(c for c in col if c.isalpha())
                if element and element in VALID_ELEMENTS:
                    available_elements.append(element)
        else:
            st.error(f"Unknown data type: {data_type}")
            return []
        
        available_elements = sorted(list(set(available_elements)))
     
        session_key = f"selected_elements_{data_type}"
        temp_selection_key = f"temp_selected_elements_{data_type}"
        
   
        if session_key not in st.session_state:
            st.session_state[session_key] = available_elements.copy()
            
        if temp_selection_key not in st.session_state:
            st.session_state[temp_selection_key] = st.session_state[session_key].copy()
    
        element_to_column = {}
        full_columns_key = f"selected_columns_{data_type}"
        temp_full_columns_key = f"temp_selected_columns_{data_type}"
        
        if data_type == "SPCal":
            for col in data_obj.columns:
                if not isinstance(col, str):
                    continue
                element = ''.join(c for c in col if c.isalpha())
                if element and element in VALID_ELEMENTS:
                    if element not in element_to_column:
                        element_to_column[element] = []
                    element_to_column[element].append(col)
            

            if full_columns_key not in st.session_state:
                all_columns = []
                for elem in st.session_state[session_key]:
                    if elem in element_to_column:
                        all_columns.extend(element_to_column[elem])
                st.session_state[full_columns_key] = all_columns
                
            if temp_full_columns_key not in st.session_state:
                all_columns = []
                for elem in st.session_state[temp_selection_key]:
                    if elem in element_to_column:
                        all_columns.extend(element_to_column[elem])
                st.session_state[temp_full_columns_key] = all_columns
        
     
        periodic_table = [
            ["H", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "He"],
            ["Li", "Be", "", "", "", "", "", "", "", "", "", "", "B", "C", "N", "O", "F", "Ne"],
            ["Na", "Mg", "", "", "", "", "", "", "", "", "", "", "Al", "Si", "P", "S", "Cl", "Ar"],
            ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"],
            ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"],
            ["Cs", "Ba", " ", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"],
            ["Fr", "Ra", " ", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"],
            ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
            ["", "", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", ""],
            ["", "", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", ""]
        ]
        
        st.write("## Element Filter")
        st.write("Click elements to include/exclude from analysis. Green = included, Red = excluded.")
        
  
        need_ui_update = False
        
     
        for row_idx, row_elements in enumerate(periodic_table):
            cols = st.columns(len(row_elements))
            for col_idx, element in enumerate(row_elements):
                with cols[col_idx]:
                    if element == "":
                        st.write("")
                    elif element in available_elements:
                      
                        is_selected = element in st.session_state[temp_selection_key]
                        button_color = "primary" if is_selected else "secondary"  
                        
               
                        button_key = f"pt_{element}_{data_type}_{row_idx}_{col_idx}"
                        clicked = st.button(
                            element, 
                            key=button_key,
                            use_container_width=True,
                            type=button_color
                        )
                        
                      
                        if clicked:
                        
                            if is_selected:
                                st.session_state[temp_selection_key].remove(element)
                                
                             
                                if data_type == "SPCal" and temp_full_columns_key in st.session_state:
                                    for col in element_to_column.get(element, []):
                                        if col in st.session_state[temp_full_columns_key]:
                                            st.session_state[temp_full_columns_key].remove(col)
                            else:
                                st.session_state[temp_selection_key].append(element)
                                
                               
                                if data_type == "SPCal" and temp_full_columns_key in st.session_state:
                                    for col in element_to_column.get(element, []):
                                        if col not in st.session_state[temp_full_columns_key]:
                                            st.session_state[temp_full_columns_key].append(col)
                            
                         
                            need_ui_update = True
                    else:
                      
                        st.button(
                            element, 
                            key=f"pt_{element}_{data_type}_{row_idx}_{col_idx}", 
                            disabled=True, 
                            use_container_width=True,
                            type="tertiary"  
                        )
        
   
        st.write("### Selected Elements")
        selected_text = ", ".join(sorted(st.session_state[temp_selection_key])) if st.session_state[temp_selection_key] else "None"
        st.write(f"**{len(st.session_state[temp_selection_key])} elements selected:** {selected_text}")
        
     
        if data_type == "SPCal" and temp_full_columns_key in st.session_state:
            isotope_text = ", ".join(sorted(st.session_state[temp_full_columns_key])) if st.session_state[temp_full_columns_key] else "None"
            st.write(f"**Isotopes included:** {isotope_text}")
    
  
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select All", key=f"select_all_{data_type}", use_container_width=True):
                st.session_state[temp_selection_key] = available_elements.copy()
             
                if data_type == "SPCal" and temp_full_columns_key in st.session_state:
                    all_columns = []
                    for elem in available_elements:
                        if elem in element_to_column:
                            all_columns.extend(element_to_column[elem])
                    st.session_state[temp_full_columns_key] = all_columns
                    
                need_ui_update = True
        with col2:
            if st.button("Clear All", key=f"clear_all_{data_type}", use_container_width=True):
                st.session_state[temp_selection_key] = []
                
                if data_type == "SPCal" and temp_full_columns_key in st.session_state:
                    st.session_state[temp_full_columns_key] = []
                    
                need_ui_update = True
        
       
        if st.button("Apply Element Selection", key=f"apply_selection_{data_type}", type="primary", use_container_width=True):
      
            st.session_state[session_key] = st.session_state[temp_selection_key].copy()
 
            if data_type == "SPCal" and full_columns_key in st.session_state and temp_full_columns_key in st.session_state:
                st.session_state[full_columns_key] = st.session_state[temp_full_columns_key].copy()
            
           
            st.session_state.element_selection_confirmed = True
            
         
            st.rerun()
        
      
        if need_ui_update:
            st.rerun()
            
        
        if data_type == "SPCal" and full_columns_key in st.session_state:
            return st.session_state[full_columns_key]
        else:
            return st.session_state[session_key]


    def preprocess_csv_file(fl):
        lines = fl.getvalue().decode('utf-8').splitlines()
        max_fields = max([line.count(',') for line in lines]) + 1
        cleaned_lines = []

        for line in lines:
            fields = line.split(',')
            if len(fields) < max_fields:
                fields.extend([''] * (max_fields - len(fields)))
            cleaned_lines.append(','.join(fields))

        cleaned_file_content = "\n".join(cleaned_lines)
        return cleaned_file_content


    def find_last_value_after_keyword(data, keyword, column_index=1):
        keyword_found = False
        count = 0
        for value in data.iloc[:, column_index]:
            if keyword_found:
                if pd.notna(value):
                    count += 1
                else:
                    break
            elif keyword.lower() in str(value).lower():
                keyword_found = True
        return count


    def extract_numeric_value_from_string(s):
        match = re.search(r"(\d+(\.\d+)?)", s)
        return float(match.group(1)) if match else None


    def find_value_at_keyword(data, keyword, column_index=1):
        for value in data.iloc[:, column_index]:
            if keyword.lower() in str(value).lower():
                return extract_numeric_value_from_string(str(value))
        return None


    def calculate_particles_per_ml(event_number, q_plasma, acquisition_time, dilution_factor):
        try:
            value = (float(event_number) * dilution_factor) / ((float(q_plasma) / 1000) * (float(acquisition_time)))
            return f"{value:.2e}"
        except ValueError:
            return None


    def find_start_index(df, keyword, column_index=0):
        for i, value in enumerate(df.iloc[:, column_index]):
            if keyword.lower() in str(value).lower():
                return i
        return None
    
    def process_and_display_spcal(df):
        try:
            time_row_mask = df.iloc[:, 0] == "Time"
            if not time_row_mask.any():
                st.error("No 'Time' column found in the data. Please check the file format.")
                return None, None

            time_row = time_row_mask.idxmax()
            
            headers = df.iloc[time_row:time_row + 3]
            data = df.iloc[time_row + 2:].reset_index(drop=True) 
            
            counts_data = pd.DataFrame()
            fg_data = pd.DataFrame()
         
            VALID_ELEMENTS = [
                "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", 
                "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", 
                "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
                "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", 
                "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", 
                "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", 
                "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", 
                "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", 
                "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", 
                "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
                "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
                "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
            ]
            
            for col in range(1, len(headers.columns), 2):  
                raw_element = str(headers.iloc[0, col])
                
                if pd.notna(raw_element):
                    element_symbol = ''.join(c for c in raw_element if c.isalpha())
                    
                    if not element_symbol or element_symbol not in VALID_ELEMENTS:
                        continue
                    
                    counts_column = pd.to_numeric(data.iloc[:, col], errors='coerce')
                
                    if element_symbol in counts_data.columns:
                        counts_data[element_symbol] += counts_column.fillna(0)
                    else:
                        counts_data[element_symbol] = counts_column.fillna(0)
              
                    if col + 1 < len(headers.columns):
                        fg_column = pd.to_numeric(data.iloc[:, col + 1], errors='coerce')
           
                        if element_symbol in fg_data.columns:
                            fg_data[element_symbol] += fg_column.fillna(0)
                        else:
                            fg_data[element_symbol] = fg_column.fillna(0)
    
            particle_counts = {}
            for col in counts_data.columns:
                particle_count = (counts_data[col] > 0).sum()
                particle_counts[col] = particle_count
            
            return counts_data, fg_data
                
        except Exception as e:
            st.error(f"Error processing SPCal data: {str(e)}")
            return None, None

    def process_data(df, keyword='event number'):
        start_index = find_start_index(df, keyword)
        if start_index is not None:
            new_header = df.iloc[start_index]
            data = df.iloc[start_index + 1:].reset_index(drop=True)
            data.columns = [str(col) for col in new_header]

            mass_data_cols = [col for col in data.columns if
                            'mass' in col and 'total' not in col and not col.endswith('mass %')]
            mole_data_cols = [col for col in data.columns if
                            'mole' in col and 'total' not in col and not col.endswith('mole %')]
            count_cols = [col for col in data.columns if col.endswith('counts')]

            mass_data = data[mass_data_cols].rename(columns=lambda x: re.sub(r'\d+', '', x.split(' ')[0]).strip())
            mole_data = data[mole_data_cols].rename(columns=lambda x: re.sub(r'\d+', '', x.split(' ')[0]).strip())
            counts_data = data[count_cols].rename(columns=lambda x: re.sub(r'\d+', '', x.split(' ')[0]).strip())

            mass_data = mass_data.apply(pd.to_numeric, errors='coerce').fillna(0)
            counts_data = counts_data.apply(pd.to_numeric, errors='coerce').fillna(0)
            mole_data = mole_data.apply(pd.to_numeric, errors='coerce').fillna(0)

            mass_data = mass_data.loc[~(mass_data == 0).all(axis=1)]
            mole_data = mole_data.loc[~(mole_data == 0).all(axis=1)]
            counts_data = counts_data.loc[~(counts_data == 0).all(axis=1)]

            mass_percent_data = mass_data.div(mass_data.sum(axis=1), axis=0) * 100
            mass_percent_data = mass_percent_data.rename(columns=lambda x: re.sub(r'\d+', '', x.split(' ')[0]).strip())

            mole_percent_data = mole_data.div(mole_data.sum(axis=1), axis=0) * 100
            mole_percent_data = mole_percent_data.rename(columns=lambda x: re.sub(r'\d+', '', x.split(' ')[0]).strip())

            return mass_data, mass_percent_data, mole_data, mole_percent_data, counts_data
        else:
            return None, None, None, None, None

    def clean_data(df, keyword='event number'):
        start_index = find_start_index(df, keyword)

        if start_index is not None:
            new_header = df.iloc[start_index]
            data = df.iloc[start_index + 1:].reset_index(drop=True)
            data.columns = [str(col) for col in new_header]

            def make_unique(column_names):
                counts = {}
                for i, col in enumerate(column_names):
                    if col in counts:
                        counts[col] += 1
                        column_names[i] = f"{col}_{counts[col]}"
                    else:
                        counts[col] = 0
                return column_names

            data.columns = make_unique(data.columns.tolist())

            elements = set(col.split(' ')[0] for col in data.columns if 'fwhm' in col)

            for element in elements:
                count_col = f'{element} counts'
                mole_col = f'{element} moles [fmol]'
                mass_col = f'{element} mass [fg]'
                fwhm_col = f'{element} fwhm'
                mole_per = f'{element} mole %'
                mass_per = f'{element} mass %'

                if all(col in data.columns for col in [count_col, mole_col, mass_col, fwhm_col, mole_per, mass_per]):
                    data.loc[data[fwhm_col].isna(), [count_col, mole_col, mass_col, mass_per, mole_per]] = 0

            cleaned_df = df.copy()
            cleaned_df.iloc[start_index + 1:, :] = data.values

            return cleaned_df
        else:
            st.error("Header row with 'fwhm' not found.")
            return None
        
    def process_isotrack(df):
        """Process IsoTrack format data with specific structure for counts, mass, moles, and diameter"""
        try:
            transport_rate = None
            for idx, row in df.iterrows():
                if 'Transport Rate:' in str(row.iloc[0]):
                    try:
                        match = re.search(r'(\d+\.?\d*)', str(row.iloc[0]))
                        if match:
                            transport_rate = float(match.group(1))
                    except (ValueError, AttributeError):
                        continue
                    break
      
            start_idx = None
            for idx, row in df.iterrows():
                if 'Particle ID' in str(row.iloc[0]):
                    start_idx = idx
                    break
                        
            if start_idx is None:
                st.error("Could not find particle data section")
                return None
                        
            data = df.iloc[start_idx:].copy()
            if len(data) == 0:
                st.error("No data found after Particle ID")
                return None
                
            data.columns = [str(x).strip() for x in data.iloc[0]]
            data = data.iloc[1:].reset_index(drop=True)
            
            counts_cols = [col for col in data.columns if 'counts' in col]
            fg_cols = [col for col in data.columns if '(fg)' in col and 'Total' not in col and not col.endswith('Mass%')]
            fmol_cols = [col for col in data.columns if '(fmol)' in col and 'Total' not in col and not col.endswith('Mole%')]
            nm_cols = [col for col in data.columns if '(nm)' in col]
            
            counts_data = data[counts_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            mass_data = data[fg_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            mole_data = data[fmol_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            diameter_data = data[nm_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            def clean_element_name(col):
                col = col.split('(')[0].strip()
                return ''.join(c for c in col if not c.isdigit()).strip()
            
            counts_data.columns = [clean_element_name(col) for col in counts_data.columns]
            mass_data.columns = [clean_element_name(col) for col in mass_data.columns]
            mole_data.columns = [clean_element_name(col) for col in mole_data.columns]
            diameter_data.columns = [clean_element_name(col) for col in diameter_data.columns]

            mole_percent_data = mole_data.div(mole_data.sum(axis=1), axis=0) * 100
            
            return {
                'counts': counts_data,
                'mass': mass_data,
                'mole': mole_data,
                'mole_percent': mole_percent_data,
                'diameter': diameter_data,
                'transport_rate': transport_rate,
                'particle_count': len(data)
            }
                
        except Exception as e:
            st.error(f'Error processing IsoTrack data: {str(e)}')
            return None

    def process_uploaded_files(files, data_type):
        if isinstance(files, list):
            if not files:
                return None

            try:
                if 'csv' in files[0].name:
                    cleaned_content = preprocess_csv_file(files[0])
                    base_df = pd.read_csv(StringIO(cleaned_content))
                else:
                    st.error('File format not supported')
                    return None

                keyword = "Particle ID" if data_type == "IsoTrack" else "Time" if data_type == "SPCal" else "event number"
                start_index = find_start_index(base_df, keyword)
                
                if start_index is None:
                    st.error(f"Could not find {keyword} in the first file")
                    return None

                def find_end_of_data(df, start_idx):
                    data_section = df.iloc[start_idx + 1:]
                    for idx, row in data_section.iterrows():
                        if row.isna().all() or row.astype(str).str.strip().eq('').all():
                            return idx
                    return len(df)

                transport_rate_index = None
                if data_type == "IsoTrack":
                    for idx, row in base_df.iterrows():
                        if 'Transport Rate:' in str(row.iloc[0]):
                            transport_rate_index = idx
                            break
                    calibration = base_df.iloc[:transport_rate_index + 1] if transport_rate_index is not None else pd.DataFrame()
                
                header = base_df.iloc[:start_index + 1]
                
                end_index = find_end_of_data(base_df, start_index)
                data_frames = [base_df.iloc[start_index + 1:end_index]]

                for additional_file in files[1:]:
                    try:
                        if 'csv' in additional_file.name:
                            content = preprocess_csv_file(additional_file)
                            df = pd.read_csv(StringIO(content))
                            add_start_index = find_start_index(df, keyword)
                            
                            if add_start_index is not None:
                                end_index = find_end_of_data(df, add_start_index)
                                data_frames.append(df.iloc[add_start_index + 1:end_index])
                            else:
                                st.warning(f"Skipping file {additional_file.name}: Could not find {keyword}")
                    except Exception as e:
                        st.error(f"Error processing {additional_file.name}: {str(e)}")
                        continue

                combined_data = pd.concat(data_frames, ignore_index=True)
                
                if data_type == "IsoTrack" and not calibration.empty:
                    final_df = pd.concat([calibration, header, combined_data], ignore_index=True)
                else:
                    final_df = pd.concat([header, combined_data], ignore_index=True)
                
                st.success(f"Successfully combined {len(files)} files")
                csv = final_df.to_csv(index=False)
        
                st.download_button(
                    label="Download Combined File",
                    data=csv,
                    file_name="combined_data.csv",
                    mime="text/csv"
                )
                return final_df
            
            except Exception as e:
                st.error(f'Error processing combined files: {str(e)}')
                return None
                
        else: 
            try:
                if 'csv' in files.name:
                    cleaned_content = preprocess_csv_file(files)
                    df = pd.read_csv(StringIO(cleaned_content))
                    
                    keyword = "Particle ID" if data_type == "IsoTrack" else "Time" if data_type == "SPCal" else "event number"
                    start_index = find_start_index(df, keyword)
                    
                    if start_index is not None:
                        def find_end_of_data(df, start_idx):
                            data_section = df.iloc[start_idx + 1:]
                            for idx, row in data_section.iterrows():
                                if row.isna().all() or row.astype(str).str.strip().eq('').all():
                                    return idx
                            return len(df)
                        
                        end_index = find_end_of_data(df, start_index)
                        return pd.concat([df.iloc[:start_index + 1], df.iloc[start_index + 1:end_index]], ignore_index=True)
                    
                    return df
                else:
                    st.error('File format not supported')
                    return None
            except Exception as e:
                st.error(f'Error processing file: {str(e)}')
                return None

   
    if fl is not None and not st.session_state.data_processed:
        if combine_files:
            st.session_state.processed_df = process_uploaded_files(fl, data_type)
            if st.session_state.processed_df is not None:
                st.write('Processing combined files...')
                st.session_state.data_processed = True
        else:
            st.session_state.processed_df = process_uploaded_files(fl, data_type)
            if st.session_state.processed_df is not None:
                st.write('Uploaded File: ', fl.name)
                st.session_state.data_processed = True

    if st.session_state.processed_df is not None:
        df = st.session_state.processed_df
        try:
            
            if data_type == "SPCal":
             
                if ('original_fg_data' not in st.session_state or 
                    'original_counts_data' not in st.session_state):
                    
                    counts_data, fg_data = process_and_display_spcal(df)
                    if counts_data is not None and fg_data is not None:
                        st.write("### Data Summary")
                        st.write(f"Number of particles: {len(fg_data)}")
                        st.write(f"Number of elements: {len(fg_data.columns)}")
                        
                   
                        st.session_state.original_fg_data = fg_data.copy()
                        st.session_state.original_counts_data = counts_data.copy()
                        
               
                        if 'filtered_fg_data' not in st.session_state:
                            st.session_state.filtered_fg_data = fg_data.copy()
                        if 'filtered_counts_data' not in st.session_state:
                            st.session_state.filtered_counts_data = counts_data.copy()
                else:
               
                    fg_data = st.session_state.original_fg_data
                    counts_data = st.session_state.original_counts_data
                    
                    st.write("### Data Summary")
                    st.write(f"Number of particles: {len(fg_data)}")
                    st.write(f"Number of elements: {len(fg_data.columns)}")
                
               
                selected_elements = create_periodic_table_selector("SPCal", fg_data)
                
        
                if st.session_state.element_selection_confirmed:
                    st.session_state.filtered_fg_data = fg_data[selected_elements]
                    st.session_state.filtered_counts_data = counts_data[selected_elements]
                    st.session_state.element_selection_confirmed = False
                    st.rerun()  

      
            elif data_type == "IsoTrack":
               
                if 'original_results' not in st.session_state:
                    results = process_isotrack(df)
                    if results:
                        st.write(f"Transport Rate: {results['transport_rate']} µL/s")
                        st.write(f"Total Particles: {results['particle_count']}")
                        
                      
                        st.session_state.original_results = results.copy()
                
                        if 'filtered_results' not in st.session_state:
                            st.session_state.filtered_results = results.copy()
                else:
                    results = st.session_state.original_results
                    st.write(f"Transport Rate: {results['transport_rate']} µL/s")
                    st.write(f"Total Particles: {results['particle_count']}")
                
               
                selected_elements = create_periodic_table_selector("IsoTrack", results)
         
                if st.session_state.element_selection_confirmed:
                    filtered_results = results.copy()
                    filtered_results['mass'] = results['mass'][selected_elements]
                    filtered_results['mole'] = results['mole'][selected_elements]
                    filtered_results['counts'] = results['counts'][selected_elements]
                    filtered_results['mole_percent'] = results['mole_percent'][selected_elements]
                    if 'diameter' in results:
                        filtered_results['diameter'] = results['diameter'][selected_elements] if len(selected_elements) > 0 else pd.DataFrame()
                    
                    st.session_state.filtered_results = filtered_results
                    st.session_state.element_selection_confirmed = False
                    st.rerun()  
               
                dilution_factor = st.number_input('Enter Dilution Factor:', format="%f", value=1.0, key='isotrack_dilution')
                acquisition_time = st.number_input('Enter Total Acquisition Time (in seconds):', format="%f", value=60.0, key='isotrack_acquisition')
                
                if results['transport_rate'] is not None:
                    particles_per_ml = calculate_particles_per_ml(
                        results['particle_count'],
                        results['transport_rate'],
                        acquisition_time,
                        dilution_factor
                    )
                    if particles_per_ml is not None:
                        st.write(f'Particles per ml: {particles_per_ml} Particles/mL')
                        st.session_state.particles_per_ml = particles_per_ml
                                
      
            elif data_type == "NuQuant":
               
            
                if 'apply_fwhm_cleaning' not in st.session_state:
                    st.session_state.apply_fwhm_cleaning = False
                
               
                st.sidebar.title('Data Cleaning')
                apply_fwhm_cleaning = st.sidebar.checkbox(
                    "Clean data based on FWHM", 
                    value=st.session_state.apply_fwhm_cleaning,
                    key='fwhm_cleaning_checkbox'
                )
                
            
                if apply_fwhm_cleaning != st.session_state.apply_fwhm_cleaning:
                 
                    for key in ['original_mass_data', 'original_mass_percent_data', 
                            'original_mole_data', 'original_mole_percent_data', 
                            'original_counts_data', 'filtered_mass_data', 
                            'filtered_mass_percent_data', 'filtered_mole_data', 
                            'filtered_mole_percent_data', 'filtered_counts_data']:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    st.session_state.data_processed = False
                    st.session_state.apply_fwhm_cleaning = apply_fwhm_cleaning
                
                if ('original_mass_data' not in st.session_state):
                    if st.session_state.apply_fwhm_cleaning:
                        cleaned_df = clean_data(df)
                        if cleaned_df is not None:
                            st.success("Data cleaned based on FWHM values")
                            mass_data, mass_percent_data, mole_data, mole_percent_data, counts_data = process_data(cleaned_df)
                        else:
                            st.error("FWHM cleaning failed. Using original data instead.")
                            mass_data, mass_percent_data, mole_data, mole_percent_data, counts_data = process_data(df)
                    else:
                        mass_data, mass_percent_data, mole_data, mole_percent_data, counts_data = process_data(df)
                    
                    st.session_state.original_mass_data = mass_data.copy()
                    st.session_state.original_mass_percent_data = mass_percent_data.copy()
                    st.session_state.original_mole_data = mole_data.copy()
                    st.session_state.original_mole_percent_data = mole_percent_data.copy()
                    st.session_state.original_counts_data = counts_data.copy()
                
                    if 'filtered_mass_data' not in st.session_state:
                        st.session_state.filtered_mass_data = mass_data.copy()
                    if 'filtered_mass_percent_data' not in st.session_state:
                        st.session_state.filtered_mass_percent_data = mass_percent_data.copy()
                    if 'filtered_mole_data' not in st.session_state:
                        st.session_state.filtered_mole_data = mole_data.copy()
                    if 'filtered_mole_percent_data' not in st.session_state:
                        st.session_state.filtered_mole_percent_data = mole_percent_data.copy()
                    if 'filtered_counts_data' not in st.session_state:
                        st.session_state.filtered_counts_data = counts_data.copy()
                else:
                    mass_data = st.session_state.original_mass_data
                    mass_percent_data = st.session_state.original_mass_percent_data
                    mole_data = st.session_state.original_mole_data
                    mole_percent_data = st.session_state.original_mole_percent_data
                    counts_data = st.session_state.original_counts_data
                            
               
                selected_elements = create_periodic_table_selector("NuQuant", mass_data)
                
                if st.session_state.element_selection_confirmed:
                    st.session_state.filtered_mass_data = mass_data[selected_elements]
                    st.session_state.filtered_mass_percent_data = mass_percent_data[selected_elements]
                    st.session_state.filtered_mole_data = mole_data[selected_elements]
                    st.session_state.filtered_mole_percent_data = mole_percent_data[selected_elements]
                    st.session_state.filtered_counts_data = counts_data[selected_elements]
                    
                    st.session_state.element_selection_confirmed = False
                    st.rerun()  
              
                dilution_factor = st.number_input('Enter Dilution Factor:', format="%f", value=1.0, key='single_dilution')
                acquisition_time = st.number_input('Enter Total Acquisition Time (in seconds):', format="%f", value=60.0, key='single_acquisition')

                event_number_cell = find_last_value_after_keyword(df, 'event number', column_index=0)
                en = event_number_cell
                if event_number_cell is not None:
                    st.write(f'Total Particles Count: {event_number_cell} Particles')
                else:
                    st.write('Event number not found or no valid count available.')

                transport_rate_cell = find_value_at_keyword(df, 'calibrated transport rate', column_index=1)
                if transport_rate_cell is not None:
                    transport_rate = extract_numeric_value_from_string(str(transport_rate_cell))
                    if transport_rate is not None:
                        st.write(f'Calibrated Transport Rate: {transport_rate} µL/s')
                    else:
                        st.write('Invalid transport rate value.')
                
                if event_number_cell is not None and transport_rate_cell is not None:
                    particles_per_ml = calculate_particles_per_ml(event_number_cell, transport_rate_cell,
                                                               acquisition_time, dilution_factor)
                    ppm = particles_per_ml
                    if particles_per_ml is not None:
                        st.write(f'Particles per ml: {particles_per_ml} Particles/mL')
                        st.session_state.particles_per_ml = particles_per_ml

        except Exception as e:
            st.error(f'An error occurred: {str(e)}')
            st.write("Error details:", type(e).__name__)
                    


    
    def plot_histogram_for_elements_spcal(fg_data, elements, all_color, single_color, multiple_color, bin_size, x_max, title):
        """Plot histograms for selected elements from SPCal data"""
        try:
            if not elements:
                st.warning("Please select elements to plot.")
                return

            if len(elements) > 3:
                st.warning("Please select up to 3 elements.")
                return

        
            data_to_plot = fg_data[elements].copy()
            data_to_plot = data_to_plot.loc[~(data_to_plot == 0).all(axis=1)]
            
            if data_to_plot.empty:
                st.error(f"No particles found containing the selected elements: {', '.join(elements)}")
                return

            if len(elements) == 1:
            
                element = elements[0]
                fig = go.Figure()
                element_data = data_to_plot[element]
                element_data = element_data[element_data > 0] 

                if not element_data.empty:
                    fig.add_trace(go.Histogram(
                        x=element_data,
                        name='Detections',
                        marker_color=single_color,
                        xbins=dict(start=0, end=x_max, size=bin_size),
                        marker_line_color='black',
                        marker_line_width=1
                    ))

                fig.update_layout(
                    title=f"{title}: {element}",
                    xaxis_title="Mass (fg)",
                    yaxis_title="Frequency",
                    xaxis=dict(
                        range=[0, x_max], 
                        title_font=dict(size=40, color='black'), 
                        tickfont=dict(size=40, color='black'), 
                        linecolor='black', 
                        linewidth=1
                    ),
                    yaxis=dict(
                        title_font=dict(size=40, color='black'), 
                        tickfont=dict(size=40, color='black'), 
                        linecolor='black', 
                        linewidth=1
                    ),
                    barmode='overlay',
                    legend_title_font=dict(size=24, color='black'),
                    legend_font=dict(size=24, color='black')
                )

                fig.add_annotation(
                    x=0.5, y=0.95,
                    xref="paper", yref="paper",
                    text=f"Total particles: {len(element_data)}",
                    showarrow=False,
                    font=dict(size=14, color="black"),
                    align="center",
                )

            else:
                total_mass = data_to_plot.sum(axis=1)
                
                fig = go.Figure()

                fig.add_trace(go.Histogram(
                    x=total_mass,
                    name='Combined Mass',
                    marker_color=all_color,
                    xbins=dict(start=0, end=x_max, size=bin_size),
                    marker_line_color='black',
                    marker_line_width=1
                ))

                combination_key = ','.join(elements)
                fig.update_layout(
                    title=f"Mass Distribution: {combination_key}",
                    xaxis_title="Mass (fg)",
                    yaxis_title="Frequency",
                    xaxis=dict(
                        range=[0, x_max], 
                        title_font=dict(size=20, color='black'), 
                        tickfont=dict(size=20, color='black'), 
                        linecolor='black', 
                        linewidth=1
                    ),
                    yaxis=dict(
                        title_font=dict(size=20, color='black'), 
                        tickfont=dict(size=20, color='black'), 
                        linecolor='black', 
                        linewidth=1
                    ),
                    barmode='overlay',
                    legend_title_font=dict(size=16, color='black'),
                    legend_font=dict(size=16, color='black')
                )

                fig.add_annotation(
                    x=0.5, y=0.95,
                    xref="paper", yref="paper",
                    text=f"Total particles: {len(total_mass)}",
                    showarrow=False,
                    font=dict(size=14, color="black"),
                    align="center",
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )

            st.plotly_chart(fig)
            max_length = max([len(data_to_plot[elem].dropna()) for elem in elements])
            all_values = [list(data_to_plot[elem].dropna()) + [''] * (max_length - len(data_to_plot[elem].dropna())) for elem in elements]

            summary_data = [
                ['File name: ' + title, '', '', ''],
                ['Mass (fg) - ' + elem for elem in elements]
            ]

            for row in zip(*all_values):
                summary_data.append(list(row))

            summary_df = pd.DataFrame(summary_data)
            csv = summary_df.to_csv(index=False, header=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='histogram_element_data.csv',
                mime='text/csv'
            )

        except Exception as e:
            st.error(f"Error plotting histogram: {str(e)}")
            
            
    def plot_histogram_for_elements(mass_data, elements, all_color, single_color, multiple_color, bin_size, x_max, title, related_mass_data, only_selected_elements):
        mass_data = mass_data.apply(pd.to_numeric, errors='coerce')
        mass_data = mass_data.dropna(subset=elements)
        
        for elem in elements:
            mass_data[elem] = mass_data[elem][mass_data[elem] > 0]

        selected_elements_mask = np.all([mass_data[elem] > 0 for elem in elements], axis=0)
        
        if only_selected_elements and len(elements) > 1:
        
            other_elements = [col for col in mass_data.columns if col not in elements]
            other_elements_mask = np.all([mass_data[elem] == 0 for elem in other_elements], axis=0)
            filtered_data = mass_data[selected_elements_mask & other_elements_mask]
        else:
            filtered_data = mass_data[selected_elements_mask]

        if filtered_data.empty:
            st.error(f"No particles found containing {'only ' if only_selected_elements else ''}the selected elements: {', '.join(elements)}")
            return

        if len(elements) == 1:
            element = elements[0]
            fig = go.Figure()

            single_data = filtered_data.loc[filtered_data.apply(lambda row: (row > 0).sum(), axis=1) == 1]
            if not single_data.empty:
                fig.add_trace(go.Histogram(
                    x=single_data[element],
                    name='Single',
                    marker_color=single_color,
                    xbins=dict(start=0, end=x_max, size=bin_size),
                    marker_line_color='black',
                    marker_line_width=1
                ))

            multiple_data = filtered_data.loc[filtered_data.apply(lambda row: (row > 0).sum(), axis=1) > 1]
            if not multiple_data.empty:
                fig.add_trace(go.Histogram(
                    x=multiple_data[element],
                    name='Multiple',
                    marker_color=multiple_color,
                    xbins=dict(start=0, end=x_max, size=bin_size),
                    marker_line_color='black',
                    marker_line_width=1
                ))

            fig.update_layout(
                title=f"{title}: {element}",
                xaxis_title="Mass (fg)",
                yaxis_title="Frequency",
                xaxis=dict(range=[0, x_max], title_font=dict(size=40, color='black'), tickfont=dict(size=40, color='black'), linecolor='black', linewidth=1),
                yaxis=dict(title_font=dict(size=40, color='black'), tickfont=dict(size=40, color='black'), linecolor='black', linewidth=1),
                barmode='overlay',
                legend_title_font=dict(size=24, color='black'),
                legend_font=dict(size=24, color='black')
            )
            fig.add_annotation(
                x=0.5, y=0.95,
                xref="paper", yref="paper",
                text=f"Total NPs: {len(filtered_data)}, Single: {len(single_data)}, Multiple: {len(multiple_data)}",
                showarrow=False,
                font=dict(size=14, color="black"),
                align="center",
            )

            fig.update_traces(opacity=0.7)

        else:
            total_mass = filtered_data[elements].sum(axis=1)

            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=total_mass,
                name='Combined Mass',
                marker_color=all_color,
                xbins=dict(start=0, end=x_max, size=bin_size),
                marker_line_color='black',
                marker_line_width=1
            ))

            combination_key = ','.join(elements)
            fig.update_layout(
                title=f"Mass Distribution: {combination_key}{'(Only)' if only_selected_elements else ''}",
                xaxis_title="Mass (fg)",
                yaxis_title="Frequency",
                xaxis=dict(range=[0, x_max], title_font=dict(size=20, color='black'), tickfont=dict(size=20, color='black'), linecolor='black', linewidth=1),
                yaxis=dict(title_font=dict(size=20, color='black'), tickfont=dict(size=20, color='black'), linecolor='black', linewidth=1),
                barmode='overlay',
                legend_title_font=dict(size=16, color='black'),
                legend_font=dict(size=16, color='black')
            )

            fig.add_annotation(
                x=0.5, y=0.95,
                xref="paper", yref="paper",
                text=f"Total NPs: {len(total_mass)}",
                showarrow=False,
                font=dict(size=14, color="black"),
                align="center",
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )

            max_frequency = max(np.histogram(total_mass, bins=np.arange(0, x_max, bin_size))[0])
            fig.update_yaxes(range=[0, max_frequency * 1.1]) 

        st.plotly_chart(fig)
        max_length = max([len(filtered_data[elem]) for elem in elements])
        all_values = [list(filtered_data[elem]) + [''] * (max_length - len(filtered_data[elem])) for elem in elements]

        summary_data = [['File name: ' + title, '', '', ''],
                        ['Mass (fg) - ' + elem for elem in elements]]

        for row in zip(*all_values):
            summary_data.append(list(row))

        summary_df = pd.DataFrame(summary_data)
        csv = summary_df.to_csv(index=False, header=False)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='histogram_element_data.csv',
            mime='text/csv'
        )
        
        



    def get_combinations_and_related_data(selected_data, mass_data, mass_percent_data, mole_data, mole_percent_data):
        start_time = time.time()
        combination_data = {}
        selected_data = selected_data.apply(pd.to_numeric, errors='coerce')

        for index, row in selected_data.iterrows():
            elements = row[row > 0].index.tolist()
            combination_key = ', '.join(sorted(elements))
            combination_data.setdefault(combination_key, []).append(index)

        related_data = {
            'mass_data': {},
            'mass_percent_data': {},
            'mole_data': {},
            'mole_percent_data': {}
        }
        combinations = {}

        for combination_key, indices in combination_data.items():
            indices = pd.Index(indices)

            related_data['mass_data'][combination_key] = mass_data.loc[indices]
            related_data['mass_percent_data'][combination_key] = mass_percent_data.loc[indices]
            related_data['mole_data'][combination_key] = mole_data.loc[indices]
            related_data['mole_percent_data'][combination_key] = mole_percent_data.loc[indices]

            if indices.size > 0:
         
                filtered_mass_data = mass_data.loc[indices]
                mass_sums = filtered_mass_data.sum()
                
            
                filtered_mole_data = mole_data.loc[indices]
                mole_sums = filtered_mole_data.sum()
                
                counts = indices.size
                mass_average = mass_sums / counts
                mass_squared_diffs = ((filtered_mass_data - mass_average) ** 2).sum()

                combinations[combination_key] = {
                    'sums': mass_sums, 
                    'moles': mole_sums,  
                    'counts': counts,
                    'average': mass_average,
                    'squared_diffs': mass_squared_diffs,
                    'sd': np.sqrt(mass_squared_diffs / counts)
                }

        sd_data = {key: value['sd'] for key, value in combinations.items()}
        sd_df = pd.DataFrame(sd_data).transpose()

        elapsed_time = time.time() - start_time
        st.write(f"Time taken: {elapsed_time} seconds")

        return combinations, related_data['mole_percent_data'], related_data['mass_data'], related_data[
            'mass_percent_data'], related_data['mole_data'], sd_df






    def prepare_heatmap_data(data_combinations, combinations, start, end):
        heatmap_df = pd.DataFrame()
        combo_counts = {combo: info['counts'] for combo, info in combinations.items()}

        for combo, df in data_combinations.items():
            df = df.apply(pd.to_numeric, errors='coerce')

            df = df.fillna(0)  
            avg_percents = df.mean().to_frame().T
            avg_percents = avg_percents.div(avg_percents.sum(axis=1), axis=0)

            if data_type == "Mole %":
                avg_percents = avg_percents.div(avg_percents.sum(axis=1), axis=0)

            combo_with_count = f"{combo} ({combo_counts[combo]})"
            avg_percents.index = [combo_with_count]
            heatmap_df = pd.concat([heatmap_df, avg_percents])

        heatmap_df['Counts'] = heatmap_df.index.map(lambda x: combo_counts[x.split(' (')[0]])
        heatmap_df = heatmap_df.sort_values(by='Counts', ascending=False)

        heatmap_df = heatmap_df.iloc[start - 1:end]
        heatmap_df.drop(columns=['Counts'], inplace=True)

        return heatmap_df
    
    

    def text_color_based_on_background(avg_value, min_val, max_val):
        norm_value = (avg_value - min_val) / (max_val - min_val)
        return "black" if norm_value < 0.5 else "white"


    def text_color_based_on_background_2(avg_value, min_val, max_val):
        avg_value *= 100
        norm_value = (avg_value - min_val) / (max_val - min_val)
        return "black" if norm_value < 0.5 else "white"




    def plot_heatmap(heatmap_df, sd_df, selected_colorscale='ylgnbu', display_numbers=True, font_size=14):
        st.sidebar.header("Element Selection")
        all_elements = sorted(heatmap_df.columns.tolist())
        selected_elements = st.sidebar.multiselect(
            "Select elements to display (leave empty to show all):",
            options=all_elements,
            default=[]
        )
        

        st.sidebar.header("Download Options")
        download_format = st.sidebar.selectbox(
            "Select download format:",
            options=["PNG", "SVG", "PDF"],
            index=0
        )
        
        if selected_elements:
            filtered_indices = []
            for idx in heatmap_df.index:
                combination = idx.split(' (')[0]
                elements = combination.split(', ')
                if any(elem in elements for elem in selected_elements):
                    filtered_indices.append(idx)
            
            heatmap_df = heatmap_df.loc[filtered_indices]
            sd_df = sd_df.loc[[idx.split(' (')[0] for idx in filtered_indices]]

            if heatmap_df.empty:
                st.warning("No combinations found with the selected elements. Please adjust your selection.")
                return None

        elements_with_data = [col for col in heatmap_df.columns if heatmap_df[col].any()]
        heatmap_df = heatmap_df[elements_with_data]
        sd_df = sd_df[elements_with_data]

        elements = heatmap_df.columns.tolist()
        combinations_with_counts = heatmap_df.index.tolist()
        total_count = sum(int(comb.split('(')[-1].replace(')', '').strip()) for comb in combinations_with_counts)

   
        if data_type == "SPCal":
            colorbar_title = 'Mass (fg)'
            title_prefix = 'Mass'
        else:
            colorbar_title = ''
            title_prefix = ''
            if data_type == "Mole %":
                colorbar_title = 'Mole %'
                title_prefix = 'Molar Percentage'
            elif data_type == "Mass":
                colorbar_title = 'Mass (fg)'
                title_prefix = 'Mass'
            elif data_type == "Mole":
                colorbar_title = 'Moles (fmol)'
                title_prefix = 'Moles'

        z_values = heatmap_df.values
        
        if data_type == "Mole %":
            z_values = z_values * 100

        combinations_with_counts = np.array(combinations_with_counts).tolist()

        min_val = np.nanmin(z_values)
        max_val = np.nanmax(z_values)

        z_values = np.where(np.isnan(z_values), min_val - 1, z_values)
        z_values = np.where(z_values == 0, np.nan, z_values)

  
        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=elements,
            y=combinations_with_counts,
            colorscale=selected_colorscale,
            showscale=True,
            hoverongaps=False,
            colorbar=dict(
                title=colorbar_title,
                titlefont=dict(size=40, family='Times New Roman', color='black', weight = 'bold'),
                tickfont=dict(size=40, family='Times New Roman',color='black', weight = 'bold'),
                ticks='outside',
                ticklen=5,
                tickwidth=2,
                tickcolor='black',
                thickness=30
            )
        ))

        if display_numbers:
            for y, comb_with_count in enumerate(combinations_with_counts):
                comb = comb_with_count.split(' (')[0]
                for x, elem in enumerate(elements):
                    avg_value = heatmap_df.loc[comb_with_count, elem]
                    if pd.notna(avg_value) and avg_value != 0:
                        sd_value = sd_df.loc[comb, elem] if elem in sd_df.columns and comb in sd_df.index else np.nan
                        
                        if data_type == "Mole %":
                            if avg_value == 100:
                                sd_value = 0
                            color = text_color_based_on_background_2(avg_value, min_val, max_val)
                            annotation_text = f"{avg_value * 100:.1f}\n± {sd_value:.1f}" if not np.isnan(sd_value) else f"{avg_value * 100:.1f}"
                        else:  
                            color = text_color_based_on_background(avg_value, min_val, max_val)
                            annotation_text = f"{avg_value:.1f}\n± {sd_value:.1f}" if not np.isnan(sd_value) else f"{avg_value:.1f}"
                        
                        fig.add_annotation(
                            x=x, y=y, 
                            text=annotation_text, 
                            showarrow=False,
                            font=dict(size=font_size,family='Times New Roman',  color=color)
                        )

        selection_info = f" (Filtered by: {', '.join(selected_elements)})" if selected_elements else ""
        

        fig.update_layout(






            xaxis=dict(
                title='Éléments',
                titlefont=dict(size=40, family='Times New Roman', color='black', weight='bold'),
                tickfont=dict(size=40, family='Times New Roman', color='black', weight='bold'),
                tickangle=0,
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                title='Particule (Fréquence)',
                titlefont=dict(size=40, family='Times New Roman', color='black', weight='bold'),
                tickfont=dict(size=40, family='Times New Roman', color='black', weight='bold'),
                autorange='reversed',
                showgrid=False,
                zeroline=False
            ),
            height=max(1200, 60 * len(combinations_with_counts)),
            width=3175,
            paper_bgcolor='rgba(0,0,0,0)',  
            plot_bgcolor='rgba(0,0,0,0)', 
            margin=dict(
                l=500,  # Increased left margin
                r=100,
                t=100,
                b=100
            ),
            showlegend=False)

    
        fig.update_traces(
            xgap=0,  
            ygap=0   
        )


        if st.sidebar.button(f"Download as {download_format}"):
            base_filename = "heatmap"
            extension = download_format.lower()
            

            counter = 0
            filename = f"{base_filename}.{extension}"
            
            while os.path.exists(filename):
                counter += 1
                filename = f"{base_filename}_{counter}.{extension}"
                
            if download_format == "PNG":
                fig.write_image("heatmap.png", scale=4)  # Higher scale for better resolution
            st.sidebar.success(f"Downloaded as heatmap.{download_format.lower()}")

        return fig

    def display_aggregated_data(aggregated_data, data_type):
        st.header(f"All {data_type.replace('_', ' ').title()}")
        st.dataframe(aggregated_data)




    def aggregate_combination_data(data_dict):
        aggregated_df = pd.concat(data_dict.values(), keys=data_dict.keys())
        aggregated_df.reset_index(level=0, inplace=True)
        aggregated_df.rename(columns={'level_0': 'Combination'}, inplace=True)
        return aggregated_df




    
   
    def plot_combination_distribution_by_counts(combinations, elements_to_analyze, elements_to_exclude, percentage_threshold):
        st.sidebar.subheader("Percentage Calculation Options")
        percentage_basis = st.sidebar.selectbox(
            "Calculate percentages based on:",
            ["Count", "Mass (fg)", "Mole (fmol)"],
            key="percentage_basis"
        )
        

        only_selected_elements_mass = st.sidebar.checkbox(
            "Consider only mass of selected elements", 
            value=True,
            key="only_selected_elements_mass"
        )


        show_breakdown_on_slices = st.sidebar.checkbox(
            "Show element breakdown on pie slices", 
            value=True,
            key="show_breakdown_on_slices"
        )

        use_total_file = st.sidebar.checkbox(
            "Calculate percentages based on total sample", 
            value=False,
            key="use_total_file"
        )


        if percentage_basis == "Count":
            all_total = sum(info['counts'] for info in combinations.values())
            value_key = 'counts'
        elif percentage_basis == "Mass (fg)":
            if only_selected_elements_mass:
                all_total = sum(sum(info['sums'].filter(items=elements_to_analyze)) for info in combinations.values())
            else:
                all_total = sum(sum(info['sums']) for info in combinations.values())
            value_key = 'sums'
        elif percentage_basis == "Mole (fmol)":
            if only_selected_elements_mass:
                all_total = sum(sum(info.get('moles', info['sums']).filter(items=elements_to_analyze)) for info in combinations.values())
            else:
                all_total = sum(sum(info.get('moles', info['sums'])) for info in combinations.values())
            value_key = 'moles' if 'moles' in next(iter(combinations.values()), {}) else 'sums'
        

        element_filtered_combinations = {}
        for combo, info in combinations.items():
            if any(elem in combo.split(', ') for elem in elements_to_analyze) and \
            not any(elem in combo.split(', ') for elem in elements_to_exclude):
                element_filtered_combinations[combo] = info
        
        if use_total_file:
            total_value = all_total
        else:
            if percentage_basis == "Count":
                filtered_total = sum(info['counts'] for info in element_filtered_combinations.values())
            elif percentage_basis == "Mass (fg)":
                if only_selected_elements_mass:
                    filtered_total = sum(sum(info['sums'].filter(items=elements_to_analyze)) for info in element_filtered_combinations.values())
                else:
                    filtered_total = sum(sum(info['sums']) for info in element_filtered_combinations.values())
            elif percentage_basis == "Mole (fmol)":
                if only_selected_elements_mass:
                    filtered_total = sum(sum(info.get('moles', info['sums']).filter(items=elements_to_analyze)) for info in element_filtered_combinations.values())
                else:
                    filtered_total = sum(sum(info.get('moles', info['sums'])) for info in element_filtered_combinations.values())
            total_value = filtered_total
        

        filtered_combinations = {}
        other_combinations = {}
        other_counts = 0
        other_value = 0
        

        sorted_by_value = sorted(
            [(combo, info) for combo, info in element_filtered_combinations.items()],
            key=lambda item: (
                item[1]['counts'] if percentage_basis == "Count" 
                else sum(item[1]['sums'].filter(items=elements_to_analyze)) if percentage_basis == "Mass (fg)" and only_selected_elements_mass
                else sum(item[1]['sums']) if percentage_basis == "Mass (fg)"
                else sum(item[1].get('moles', item[1]['sums']).filter(items=elements_to_analyze)) if percentage_basis == "Mole (fmol)" and only_selected_elements_mass
                else sum(item[1].get('moles', item[1]['sums']))
            ),
            reverse=True
        )
        

        for combo, info in sorted_by_value:
            if percentage_basis == "Count":
                value = info['counts']
            elif percentage_basis == "Mass (fg)":
                if only_selected_elements_mass:
                    value = sum(info['sums'].filter(items=elements_to_analyze))
                else:
                    value = sum(info['sums'])
            elif percentage_basis == "Mole (fmol)":
                if only_selected_elements_mass:
                    value = sum(info.get('moles', info['sums']).filter(items=elements_to_analyze))
                else:
                    value = sum(info.get('moles', info['sums']))
                    
            percentage = (value / total_value) * 100
            
            if percentage >= percentage_threshold:
                filtered_combinations[combo] = info
            else:
                other_combinations[combo] = info
                other_counts += info['counts']
                other_value += value
        
        if not filtered_combinations and not other_combinations:
            st.write(f"No valid combinations found after filtering out {' or '.join(elements_to_exclude)}.")
            return

        total_counts = sum(info['counts'] for info in filtered_combinations.values()) + other_counts
        
        default_colors = [
            '#FF6347', '#FFD700', '#FFA500', '#20B2AA', '#00BFFF',
            '#F0E68C', '#E0FFFF', '#AFEEEE', '#DDA0DD', '#FFE4E1',
            '#FAEBD7', '#D3D3D3', '#90EE90', '#FFB6C1', '#FFA07A',
            '#C71585', '#DB7093', '#FF1493', '#FF69B4', '#FFA07A'
        ]

        st.sidebar.title("Customize Combination Colors")

        labels = []
        values = []
        texts = []
        hover_texts = []
        colors = []
        element_breakdowns = []  # Store element-specific breakdowns

        if other_counts > 0:
            percentage = other_value / total_value * 100
            labels.append(f"Others ({other_counts}) : ({percentage:.2f}%)")
            

            if show_breakdown_on_slices:
                texts.append(f"Others ({other_counts})<br>({percentage:.2f}%)")
            else:
                texts.append(f"Others ({other_counts})<br>({percentage:.2f}%)")
                
            hover_texts.append(f"Others: {other_counts}<br>({percentage:.2f}%)")
            colors.append('#abcdef')
            values.append(other_value)
            element_breakdowns.append({}) # No breakdown for Others

        sorted_combinations = sorted(
            [(k, v) for k, v in filtered_combinations.items()],
            key=lambda item: (
                item[1]['counts'] if percentage_basis == "Count" 
                else sum(item[1]['sums'].filter(items=elements_to_analyze)) if percentage_basis == "Mass (fg)" and only_selected_elements_mass
                else sum(item[1]['sums']) if percentage_basis == "Mass (fg)"
                else sum(item[1].get('moles', item[1]['sums']).filter(items=elements_to_analyze)) if percentage_basis == "Mole (fmol)" and only_selected_elements_mass
                else sum(item[1].get('moles', item[1]['sums']))
            ),
            reverse=True
        )

        for i, (combo, info) in enumerate(sorted_combinations):

            if percentage_basis == "Count":
                value = info['counts']
                element_breakdown = {}  # No element breakdown for count
            elif percentage_basis == "Mass (fg)":

                element_breakdown = {}
                if only_selected_elements_mass:
                    for elem in elements_to_analyze:
                        if elem in info['sums']:
                            element_breakdown[elem] = float(info['sums'][elem])
                    value = sum(element_breakdown.values())
                else:
                    value = sum(info['sums'])
                    for elem in elements_to_analyze:
                        if elem in info['sums']:
                            element_breakdown[elem] = float(info['sums'][elem])
            elif percentage_basis == "Mole (fmol)":
                element_breakdown = {}
                if only_selected_elements_mass:
                    for elem in elements_to_analyze:
                        if elem in info.get('moles', info['sums']):
                            element_breakdown[elem] = float(info.get('moles', info['sums'])[elem])
                    value = sum(element_breakdown.values())
                else:
                    value = sum(info.get('moles', info['sums']))
                    for elem in elements_to_analyze:
                        if elem in info.get('moles', info['sums']):
                            element_breakdown[elem] = float(info.get('moles', info['sums'])[elem])
                    
            percentage = value / total_value * 100
            
            color = st.sidebar.color_picker(
                f"Pick color for {combo}", 
                default_colors[i % len(default_colors)],
                key=f"color_picker_{combo}"
            )


            breakdown_text = ""
            if percentage_basis in ["Mass (fg)", "Mole (fmol)"] and element_breakdown:

                hover_breakdown = "<br>Element breakdown:<br>"
                for elem, elem_val in element_breakdown.items():
                    elem_percentage = (elem_val / value) * 100
                    hover_breakdown += f"- {elem}: {elem_val:.2f} ({elem_percentage:.1f}%)<br>"
                

                if show_breakdown_on_slices:
                    breakdown_text = "<br>"
                    for elem, elem_val in element_breakdown.items():
                        if elem_val > 0:  # Only show non-zero elements
                            elem_percentage = (elem_val / value) * 100
                            breakdown_text += f"{elem}: ({elem_percentage:.1f}%)<br>"
            

            element_breakdowns.append(element_breakdown)
            

            label_text = f"{combo} ({info['counts']}) : ({percentage:.2f}%)"
            labels.append(label_text)
            values.append(value)
            

            if show_breakdown_on_slices and percentage_basis in ["Mass (fg)", "Mole (fmol)"]:
                combo_elements = combo.split(', ')
                

                all_elements_selected = all(elem in elements_to_analyze for elem in combo_elements)
                

                if all_elements_selected and len(combo_elements) > 1:
                    breakdown_text = "<br>"
                    for elem, elem_val in element_breakdown.items():
                        if elem_val > 0:  # Only show non-zero elements
                            elem_percentage = (elem_val / value) * 100
                            breakdown_text += f"{elem}: {elem_percentage:.1f}%<br>"
                    
                    texts.append(f"{combo} ({info['counts']})<br>({percentage:.2f}%){breakdown_text}")
                else:

                    texts.append(f"{combo} ({info['counts']})<br>({percentage:.2f}%)")
            else:
                texts.append(f"{combo} ({info['counts']})<br>({percentage:.2f}%)")
                    

            hover_texts.append(f"{combo}: {info['counts']}<br>({percentage:.2f}%){hover_breakdown if percentage_basis in ['Mass (fg)', 'Mole (fmol)'] else ''}")
            colors.append(color)


        rotation = 90
        if other_counts > 0:
            total_angle = 360
            others_percentage = values[0] / sum(values)
            others_angle = others_percentage * total_angle
            rotation = 180 - (others_angle / 2)

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            text=texts,
            hoverinfo="text",  # Use custom hover text
            hovertext=hover_texts,
            textinfo="text",  # Show custom text on slices
            textfont=dict(size=40, color='black', weight='bold'),
            marker=dict(colors=colors, line=dict(color='#000000', width=1)),
            pull=[0.1 if 'Others' in label else 0 for label in labels],
            direction='clockwise',
            rotation=rotation,
            sort=False,
            showlegend=False
        )])


        percentage_scope = "total file" if use_total_file else "selected combinations"
        mass_description = "selected elements only" if only_selected_elements_mass and percentage_basis in ["Mass (fg)", "Mole (fmol)"] else "all elements"
        title = f"Distribution of particles containing {' or '.join(elements_to_analyze)}"
        if elements_to_exclude:
            title += f" excluding {' and '.join(elements_to_exclude)}"
        if percentage_basis in ["Mass (fg)", "Mole (fmol)"]:
            title += f" ({mass_description})"

        fig.update_layout(
            title=title,
            title_font_size=20,
            title_x=0.4,
            title_y=0.95,
            title_xanchor='center',
            title_yanchor='top',
            width=2000,
            height=1000,
            margin=dict(
                l=200,
                r=200,
                t=100,
                b=100,
                pad=4
            ),
            annotations=[
                dict(
                    x=0.95,
                    y=1,
                    xref="paper",
                    yref="paper",
                    text=f"Total number of particles: {total_counts}",
                    showarrow=False,
                    font=dict(size=28, color="black"),
                )
            ]
        )

        st.sidebar.title("Download Options")
        st.plotly_chart(fig, use_container_width=True)

        button_key = f"download_png_button_{','.join(elements_to_analyze)}_{percentage_basis}"
        if st.sidebar.button("Download as PNG", key=button_key):
            fig.write_image("element_distribution.png", scale=4)
            st.sidebar.success("Downloaded as element_distribution.png")


        percentage_scope_label = "total_file" if use_total_file else "selected_combinations"
        

        header_row = ['Combination', 'Count', f'Percentage ({percentage_basis})']
        if percentage_basis in ["Mass (fg)", "Mole (fmol)"]:
            for elem in elements_to_analyze:
                header_row.append(f"{elem} {percentage_basis}")
                header_row.append(f"{elem} %")
        
        summary_data = [header_row]
        

        for i, (combo, info) in enumerate(sorted_combinations):
            if percentage_basis == "Count":
                value = info['counts']
                row = [combo, info['counts'], f"{value / total_value * 100:.2f}%"]
            elif percentage_basis in ["Mass (fg)", "Mole (fmol)"]:
                element_breakdown = element_breakdowns[i+1 if other_counts > 0 else i]  # Adjust index for "Others"
                value = sum(element_breakdown.values()) if only_selected_elements_mass else (
                    sum(info['sums']) if percentage_basis == "Mass (fg)" else sum(info.get('moles', info['sums']))
                )
                

                row = [combo, info['counts'], f"{value / total_value * 100:.2f}%"]
                

                for elem in elements_to_analyze:
                    elem_value = element_breakdown.get(elem, 0)
                    elem_percentage = (elem_value / value * 100) if value > 0 else 0
                    row.append(f"{elem_value:.4f}")
                    row.append(f"{elem_percentage:.2f}%")
                    
            summary_data.append(row)
        

        if other_counts > 0:
            if percentage_basis == "Count":
                others_row = ['Others', other_counts, f"{other_value / total_value * 100:.2f}%"]
            else:
                others_row = ['Others', other_counts, f"{other_value / total_value * 100:.2f}%"]

                for elem in elements_to_analyze:
                    others_row.append("N/A")
                    others_row.append("N/A")
            
            summary_data.append(others_row)

        summary_df = pd.DataFrame(summary_data)
        csv = summary_df.to_csv(index=False, header=False)
        

        download_key = f"download_csv_button_{','.join(elements_to_analyze)}_{percentage_basis}"
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=f'combination_distribution_{percentage_basis.lower()}_{percentage_scope_label}.csv',
            mime='text/csv',
            key=download_key
        ) 
        


    def visualize_pie_chart_single_and_multiple(combinations, particles_per_ml_value):
        count_threshold_Multiple = st.sidebar.number_input('Multiple Elements Threshold', min_value=0, value=30, step=5)
        count_threshold_Single = st.sidebar.number_input('Single Elements Threshold', min_value=0, value=30, step=5)

        sorted_combinations = sorted(combinations.items(), key=lambda item: item[1]['counts'], reverse=True)
        total_counts_all = sum(details['counts'] for _, details in sorted_combinations)

        filtered_combinations_Multiple = [(combination, details) for combination, details in sorted_combinations if
                                        details['counts'] > count_threshold_Multiple]
        filtered_combinations_Single = [(combination, details) for combination, details in sorted_combinations if
                                        details['counts'] > count_threshold_Single]

        single_element_combinations = [(combination, details) for combination, details in filtered_combinations_Single if
                                    len(combination.split(', ')) == 1]
        multiple_element_combinations = [(combination, details) for combination, details in filtered_combinations_Multiple
                                        if len(combination.split(', ')) > 1]

        col1, col2 = st.columns(2)  

        summary_data_single = [['Combination', 'Count', 'Percentage']]
        if single_element_combinations:
            fig_single = plot_pie_chart(single_element_combinations, "Single Elements", total_counts_all,
                                        particles_per_ml_value)
            with col1: 
                st.plotly_chart(fig_single, use_container_width=True)

            for combination, details in single_element_combinations:
                percentage = details['counts'] / total_counts_all * 100
                summary_data_single.append([combination, details['counts'], f"{percentage:.1f}%"])

        summary_data_multiple = [['Combination', 'Count', 'Percentage']]
        if multiple_element_combinations:
            fig_multiple = plot_pie_chart(multiple_element_combinations, "Multiple Elements", total_counts_all,
                                        particles_per_ml_value)
            with col2:  
                st.plotly_chart(fig_multiple, use_container_width=True)

            for combination, details in multiple_element_combinations:
                percentage = details['counts'] / total_counts_all * 100
                summary_data_multiple.append([combination, details['counts'], f"{percentage:.1f}%"])


        if summary_data_single:
            summary_df_single = pd.DataFrame(summary_data_single)
            csv_single = summary_df_single.to_csv(index=False, header=False)
            st.download_button(
                label="Download Single Element Data as CSV",
                data=csv_single,
                file_name='single_element_combinations_data.csv',
                mime='text/csv'
            )

        if summary_data_multiple:
            summary_df_multiple = pd.DataFrame(summary_data_multiple)
            csv_multiple = summary_df_multiple.to_csv(index=False, header=False)
            st.download_button(
                label="Download Multiple Element Data as CSV",
                data=csv_multiple,
                file_name='multiple_element_combinations_data.csv',
                mime='text/csv'
            )




    def plot_pie_chart(combinations, title_suffix, total_counts, particles_per_ml_value):
        labels = [combination for combination, _ in combinations]
        values = [details['counts'] for _, details in combinations]

        fig = px.pie(values=values, names=labels, title=f"{title_suffix}",
                    color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_traces(textinfo='percent+label')
        return fig


    md = None




    def create_color_map(_elements, base_colors):
        color_map = {}
        for i, element in enumerate(_elements):
            default_color = base_colors[i % len(base_colors)]  
            color = st.sidebar.color_picker(f"Color for {element}", value=default_color, key=f"color_{element}")
            color_map[element] = color
        color_map['Others'] = st.sidebar.color_picker(f"Color for Others", '#777777')  
        return color_map



    def summarize_data(data, threshold):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")

        total_data = data.sum()

        percentages = total_data * 100 / total_data.sum()

        main = percentages[percentages > threshold]
        others = percentages[percentages <= threshold]

        if others.sum() > 0:
            main['Others'] = others.sum()

        return main, others

    def get_element_counts(data, others_elements):
        element_counts = (data > 0).sum(axis=0)

        others_counts = element_counts[others_elements.index].sum()

        return element_counts, others_counts
    
    def visualize_mass_and_mole_percentages_pie_charts(mass_data, mole_data, color_map, threshold):
        mass_percent, mass_others = summarize_data(mass_data, threshold)
        mole_percent, mole_others = summarize_data(mole_data, threshold)
        element_counts, others_counts = get_element_counts(mass_data, mass_others)

        if 'Others' in mass_percent.index:
            element_counts['Others'] = others_counts
        if 'Others' in mole_percent.index:
            element_counts['Others'] = others_counts

    

        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'pie'},{'type': 'pie'}]], horizontal_spacing=0.001)

        colors_mass = [color_map.get(index, '#CCCCCC') for index in mass_percent.index]
        colors_mole = [color_map.get(index, '#CCCCCC') for index in mole_percent.index]

        fig.add_trace(go.Pie(labels=mass_percent.index, values=mass_percent.values,
                            marker=dict(colors=colors_mass, line=dict(color='#000000', width=1)),
                            textinfo='label+percent',
                            texttemplate='%{label}(%{customdata:d}): %{value:.1f}%',
                            customdata=[element_counts.get(index, 0) for index in mass_percent.index],
                            hoverinfo='label+percent+value',
                            hovertext=[f"{label}: {element_counts.get(label, 0):d} counts" for label in
                                        mass_percent.index],
                            textfont=dict(size=25, color='black' , weight = 'bold'),
                            title=dict(text='', font=dict(size=18, color='black')),
                            direction='clockwise',  # clockwise 
                            rotation=180  # the first slice at the bottom
                            ), row=1, col=1)

        fig.add_trace(go.Pie(labels=mole_percent.index, values=mole_percent.values,
                            marker=dict(colors=colors_mole, line=dict(color='#000000', width=1)),
                            textinfo='label+percent',
                            texttemplate='%{label}(%{customdata:d}): %{value:.1f}%',
                            customdata=[element_counts.get(index, 0) for index in mole_percent.index],
                            hoverinfo='label+percent+value',
                            hovertext=[f"{label}: {element_counts.get(label, 0):d} counts" for label in
                                        mole_percent.index],
                            textfont=dict(size=25, color='black' , weight = 'bold'),
                            title=dict(text='', font=dict(size=18, color='black')),
                            direction='clockwise', 
                            rotation=180  # first slice at the bottom
                            ), row=1, col=2)

        fig.update_layout(
            title_text='Mass and Mole Percentages', 
            title_x=0.4, 
            height=600, 
            width=2500,
            legend=dict(font=dict(size=16, color='black')),
            margin=dict(b=100)  # Increase bottom margin to add space
        )

        st.plotly_chart(fig)

     
        summary_data_mass = [['Element', 'Mass Percentage', 'Count']]
        for index, value in mass_percent.items():
            summary_data_mass.append([index, f"{value:.2f}%", element_counts.get(index, 0)])

        summary_data_counts = [['Element', 'Count']]
        for index, value in element_counts.items():
            summary_data_counts.append([index, value])

        summary_data_mole = [['Element', 'Mole Percentage', 'Count']]
        for index, value in mole_percent.items():
            summary_data_mole.append([index, f"{value:.2f}%", element_counts.get(index, 0)])

        if summary_data_mass:
            summary_df_mass = pd.DataFrame(summary_data_mass)
            csv_mass = summary_df_mass.to_csv(index=False, header=False)
            st.download_button(
                label="Download Mass Percentages Data as CSV",
                data=csv_mass,
                file_name='mass_percentages_data.csv',
                mime='text/csv'
            )

        if summary_data_counts:
            summary_df_counts = pd.DataFrame(summary_data_counts)
            csv_counts = summary_df_counts.to_csv(index=False, header=False)
            st.download_button(
                label="Download Element Counts Data as CSV",
                data=csv_counts,
                file_name='element_counts_data.csv',
                mime='text/csv'
            )

        if summary_data_mole:
            summary_df_mole = pd.DataFrame(summary_data_mole)
            csv_mole = summary_df_mole.to_csv(index=False, header=False)
            st.download_button(
                label="Download Mole Percentages Data as CSV",
                data=csv_mole,
                file_name='mole_percentages_data.csv',
                mime='text/csv'
            )

    def find_value_in_column(data, keyword, column_index):
        for value in data.iloc[:, column_index]:
            if keyword.lower() in str(value).lower():
                return value
        return None




    
    
    def get_combinations_and_related_data_spcal(fg_data):
        """Process SPCal data for heatmap analysis"""
        try:
            combination_data = {}
            mass_data = fg_data.apply(pd.to_numeric, errors='coerce')
            
            for index, row in mass_data.iterrows():
                elements = row[row > 0].index.tolist()
                if elements:  
                    combination_key = ', '.join(sorted(elements))
                    combination_data.setdefault(combination_key, []).append(index)

            combinations = {}
            mass_data_combinations = {}

            for combination_key, indices in combination_data.items():
                indices = pd.Index(indices)
                if indices.size > 0:
                    filtered_data = mass_data.loc[indices]
                    
      
                    sums = filtered_data.sum()
                    counts = indices.size
                    average = sums / counts
                    squared_diffs = ((filtered_data - average) ** 2).sum()
                    
                    combinations[combination_key] = {
                        'sums': sums,
                        'counts': counts,
                        'average': average,
                        'squared_diffs': squared_diffs,
                        'sd': np.sqrt(squared_diffs / counts)
                    }
                    
                    mass_data_combinations[combination_key] = filtered_data
            sd_data = {key: value['sd'] for key, value in combinations.items()}
            sd_df = pd.DataFrame(sd_data).transpose()

            return combinations, mass_data_combinations, sd_df

        except Exception as e:
            st.error(f"error processing combinations: {str(e)}")
            return None, None, None

    def prepare_heatmap_data_spcal(mass_data_combinations, combinations, start, end):
        """prepare heatmap data for SPCal data"""
        try:
            heatmap_df = pd.DataFrame()
            combo_counts = {combo: info['counts'] for combo, info in combinations.items()}

            for combo, df in mass_data_combinations.items():
                avg_mass = df.mean().to_frame().T
                combo_with_count = f"{combo} ({combo_counts[combo]})"
                avg_mass.index = [combo_with_count]
                heatmap_df = pd.concat([heatmap_df, avg_mass])

            heatmap_df['Counts'] = heatmap_df.index.map(lambda x: combo_counts[x.split(' (')[0]])
            heatmap_df = heatmap_df.sort_values(by='Counts', ascending=False)
            
            heatmap_df = heatmap_df.iloc[start - 1:end]
            heatmap_df.drop(columns=['Counts'], inplace=True)

            return heatmap_df

        except Exception as e:
            st.error(f"Error preparing heatmap data: {str(e)}")
            return None
        
    def visualize_mass_percentages_pie_chart_spcal(fg_data, color_map, threshold):
        """Create mass percentage pie chart for SPCal data"""
        try:
            mass_percent, mass_others = summarize_data(fg_data, threshold)
            element_counts = (fg_data > 0).sum(axis=0)
            if 'Others' in mass_percent.index:
                others_counts = element_counts[mass_others.index].sum()
                element_counts['Others'] = others_counts

         
            fig = go.Figure()
            colors_mass = [color_map.get(index, '#CCCCCC') for index in mass_percent.index]
            fig.add_trace(go.Pie(
                labels=mass_percent.index,
                values=mass_percent.values,
                marker=dict(
                    colors=colors_mass,
                    line=dict(color='#000000', width=1)
                ),
                textinfo='label+percent',
                texttemplate='%{label}(%{customdata:d}): %{value:.1f}%',
                customdata=[element_counts.get(index, 0) for index in mass_percent.index],
                hoverinfo='label+percent+value',
                hovertext=[f"{label}: {element_counts.get(label, 0):d} counts" for label in mass_percent.index],
                textfont=dict(size=25, color='black'),
                title=dict(text='Mass Distribution', font=dict(size=18, color='black'))
            ))

            fig.update_layout(
                title_text='Mass Percentages',
                title_x=0.4,
                height=600,
                width=1200,
                legend=dict(font=dict(size=16, color='black'))
            )

            st.plotly_chart(fig)
            summary_data = [['Element', 'Mass Percentage', 'Count']]
            for index, value in mass_percent.items():
                summary_data.append([index, f"{value:.2f}%", element_counts.get(index, 0)])

            summary_df = pd.DataFrame(summary_data)
            csv = summary_df.to_csv(index=False, header=False)
            st.download_button(
                label="Download Mass Distribution Data as CSV",
                data=csv,
                file_name='spcal_mass_distribution.csv',
                mime='text/csv'
            )

        except Exception as e:
            st.error(f"Error creating mass distribution visualization: {str(e)}")
            

    def apply_clustering(data, method, n_clusters):
        if data is None or data.empty:
            return None

        clustering_models = {
            'Hierarchical': AgglomerativeClustering(n_clusters=n_clusters),
            'Spectral': SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', assign_labels='kmeans', n_jobs=-1),
            'Gaussian': GaussianMixture(n_components=n_clusters),
            'K-Means': KMeans(n_clusters=n_clusters),
            'Mini-Batch K-Means': MiniBatchKMeans(n_clusters=n_clusters),
            'Mean Shift': MeanShift()
        }

        cluster_model = clustering_models.get(method)
        if not cluster_model:
            st.error(f"Unsupported clustering method: {method}")
            return None

        labels = cluster_model.fit_predict(data)
        return labels

    def evaluate_clusters(data, method, max_clusters=70):
        n_clusters_range = range(2, max_clusters + 1)
        scores = {
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': [],
            'inertia': [],
            'bic': [],
            'aic': []
        }

        for n_clusters in n_clusters_range:
            if method == 'K-Means':
                model = KMeans(n_clusters=n_clusters, random_state=42)
            elif method == 'Hierarchical':
                model = AgglomerativeClustering(n_clusters=n_clusters)
            elif method == 'Spectral':
                model = SpectralClustering(n_clusters=n_clusters, random_state=42, n_jobs=-1)
            elif method == 'Gaussian':
                model = GaussianMixture(n_components=n_clusters, random_state=42)
            elif method == 'Mini-Batch K-Means':
                model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
            else:
                break

            labels = model.fit_predict(data)

            scores['silhouette'].append(silhouette_score(data, labels))
            scores['calinski_harabasz'].append(calinski_harabasz_score(data, labels))
            scores['davies_bouldin'].append(davies_bouldin_score(data, labels))

            if hasattr(model, 'inertia_'):
                scores['inertia'].append(model.inertia_)
            elif method == 'Hierarchical':
                cluster_centers = np.array([data[labels == i].mean(axis=0) for i in range(n_clusters)])
                distances = pairwise_distances(data, cluster_centers[labels])
                scores['inertia'].append(np.sum(np.min(distances, axis=1)**2))
            if isinstance(model, GaussianMixture):
                scores['bic'].append(model.bic(data))
                scores['aic'].append(model.aic(data))


        optimal_clusters = {}
        if scores['silhouette']:
            optimal_clusters['Silhouette'] = np.argmax(scores['silhouette']) + 2
        if scores['calinski_harabasz']:
            optimal_clusters['Calinski-Harabasz'] = np.argmax(scores['calinski_harabasz']) + 2
        if scores['davies_bouldin']:
            optimal_clusters['Davies-Bouldin'] = np.argmin(scores['davies_bouldin']) + 2
        if scores['inertia']:
            kl = KneeLocator(n_clusters_range, scores['inertia'], curve="convex", direction="decreasing")
            optimal_clusters['Elbow'] = kl.elbow
        if scores['bic']:
            optimal_clusters['BIC'] = np.argmin(scores['bic']) + 2
        if scores['aic']:
            optimal_clusters['AIC'] = np.argmin(scores['aic']) + 2


        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            'Silhouette Score (Higher is Better)',
            'Calinski-Harabasz Score (Higher is Better)', 
            'Davies-Bouldin Score (Lower is Better)',
            'Dendrogram / Additional Metric'
        ))
        
        def add_trace_with_optimal(row, col, metric_name, y_values, optimal_value):
            fig.add_trace(go.Scatter(x=list(n_clusters_range), y=y_values, mode='lines+markers', name=metric_name), row=row, col=col)
            if optimal_value:
                fig.add_trace(go.Scatter(x=[optimal_value, optimal_value], y=[min(y_values), max(y_values)], 
                                        mode='lines', name=f'Optimal {metric_name}', line=dict(color='red', dash='dash')), row=row, col=col)

        add_trace_with_optimal(1, 1, 'Silhouette', scores['silhouette'], optimal_clusters.get('Silhouette'))
        add_trace_with_optimal(1, 2, 'Calinski-Harabasz', scores['calinski_harabasz'], optimal_clusters.get('Calinski-Harabasz'))
        add_trace_with_optimal(2, 1, 'Davies-Bouldin', scores['davies_bouldin'], optimal_clusters.get('Davies-Bouldin'))

        if method == 'Hierarchical' and len(data) <= 5000:
            try:
              
                linkage_matrix = linkage(data, method='ward')
                
                dendrogram_data = dendrogram(linkage_matrix, no_plot=True)
            
                x_coords = []
                y_coords = []
                for i in range(len(dendrogram_data['icoord'])):
                    x_coords.extend(dendrogram_data['icoord'][i])
                    x_coords.append(None)
                    y_coords.extend(dendrogram_data['dcoord'][i])
                    y_coords.append(None)
                
            
                fig.add_trace(
                    go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='lines',
                        line=dict(color='black'),
                        name='Dendrogram',
                        showlegend=False
                    ),
                    row=2,
                    col=2
                )
                
                fig.update_xaxes(title_text="Sample Index", title_font=dict(size=25, color='black'), tickfont=dict(size=20, color='black'), row=2, col=2)
                fig.update_yaxes(title_text="Distance", title_font=dict(size=25, color='black'), tickfont=dict(size=20, color='black'), row=2, col=2)
                
            except Exception as e:
                st.warning(f"Could not create dendrogram: {str(e)}")
                if scores['inertia']:
                    add_trace_with_optimal(2, 2, 'Inertia', scores['inertia'], optimal_clusters.get('Elbow'))
        else:
            if scores['inertia']:
                add_trace_with_optimal(2, 2, 'Inertia', scores['inertia'], optimal_clusters.get('Elbow'))
            elif scores['bic']:
                add_trace_with_optimal(2, 2, 'BIC', scores['bic'], optimal_clusters.get('BIC'))
            elif scores['aic']:
                add_trace_with_optimal(2, 2, 'AIC', scores['aic'], optimal_clusters.get('AIC'))

        fig.update_layout(
            height=800, 
            width=1000, 
            title_text=f"Cluster Evaluation Metrics for {method}",
            font=dict(size=25, color='black'),
            showlegend=True
        )

        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(title_text="Number of Clusters", row=i, col=j, title_font=dict(size=25, color='black'), tickfont=dict(size=20, color='black'))
                fig.update_yaxes(title_text="Score", row=i, col=j, title_font=dict(size=25, color='black'), tickfont=dict(size=20, color='black'))

        if method == 'Hierarchical' and len(data) <= 1000:
            fig.update_xaxes(title_text="Sample Index", row=2, col=2)
            fig.update_yaxes(title_text="Distance", row=2, col=2)

        return fig, optimal_clusters
    
    
    def evaluate_mean_shift(data, max_bandwidth=50, min_bandwidth=0.5, num_bandwidths=20):
        """
        Evaluate Mean Shift clustering with different bandwidth values.
        
        Parameters:
        -----------
        data : pandas DataFrame
            The data to cluster
        max_bandwidth : float
            Maximum bandwidth value to test
        min_bandwidth : float
            Minimum bandwidth value to test
        num_bandwidths : int
            Number of bandwidth values to test
        
        Returns:
        --------
        fig : plotly Figure
            A figure showing the evaluation metrics
        optimal_bandwidths : dict
            Dictionary containing optimal bandwidth values for each metric
        """
        from scipy.signal import argrelextrema
        

        bandwidth_values = np.logspace(
            np.log10(min_bandwidth), 
            np.log10(max_bandwidth), 
            num_bandwidths
        )
        

        scores = {
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': [],
            'num_clusters': []
        }
        

        for bandwidth in bandwidth_values:
            model = MeanShift(bandwidth=bandwidth, n_jobs=-1)
            labels = model.fit_predict(data)
            

            n_clusters = len(np.unique(labels))
            scores['num_clusters'].append(n_clusters)
            

            if n_clusters <= 1:
                scores['silhouette'].append(np.nan)
                scores['calinski_harabasz'].append(np.nan)
                scores['davies_bouldin'].append(np.nan)
                continue
            

            scores['silhouette'].append(silhouette_score(data, labels))
            scores['calinski_harabasz'].append(calinski_harabasz_score(data, labels))
            scores['davies_bouldin'].append(davies_bouldin_score(data, labels))
        

        optimal_bandwidths = {}
        

        valid_indices = [i for i, s in enumerate(scores['silhouette']) if not np.isnan(s)]
        if valid_indices:
            optimal_idx = valid_indices[np.argmax([scores['silhouette'][i] for i in valid_indices])]
            optimal_bandwidths['silhouette'] = bandwidth_values[optimal_idx]
        else:
            optimal_bandwidths['silhouette'] = None
        

        valid_indices = [i for i, s in enumerate(scores['calinski_harabasz']) if not np.isnan(s)]
        if valid_indices:
            optimal_idx = valid_indices[np.argmax([scores['calinski_harabasz'][i] for i in valid_indices])]
            optimal_bandwidths['calinski_harabasz'] = bandwidth_values[optimal_idx]
        else:
            optimal_bandwidths['calinski_harabasz'] = None
        

        valid_indices = [i for i, s in enumerate(scores['davies_bouldin']) if not np.isnan(s)]
        if valid_indices:
            optimal_idx = valid_indices[np.argmin([scores['davies_bouldin'][i] for i in valid_indices])]
            optimal_bandwidths['davies_bouldin'] = bandwidth_values[optimal_idx]
        else:
            optimal_bandwidths['davies_bouldin'] = None
        

        num_clusters = np.array(scores['num_clusters'])
        if len(num_clusters) > 3:

            diffs = np.diff(num_clusters)
            


            if len(diffs) > 2:
                second_diffs = np.diff(diffs)
                elbow_idx = np.argmax(np.abs(second_diffs)) + 1
                optimal_bandwidths['num_clusters'] = bandwidth_values[elbow_idx]
            else:
                optimal_bandwidths['num_clusters'] = None
        else:
            optimal_bandwidths['num_clusters'] = None
        

        fig = make_subplots(
            rows=2, cols=2, 
            subplot_titles=(
                'Silhouette Score (Higher is Better)',
                'Calinski-Harabasz Score (Higher is Better)', 
                'Davies-Bouldin Score (Lower is Better)',
                'Number of Clusters'
            )
        )
        

        fig.add_trace(
            go.Scatter(
                x=bandwidth_values, 
                y=scores['silhouette'], 
                mode='lines+markers',
                name='Silhouette'
            ), 
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=bandwidth_values, 
                y=scores['calinski_harabasz'], 
                mode='lines+markers',
                name='Calinski-Harabasz'
            ), 
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=bandwidth_values, 
                y=scores['davies_bouldin'], 
                mode='lines+markers',
                name='Davies-Bouldin'
            ), 
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=bandwidth_values, 
                y=scores['num_clusters'], 
                mode='lines+markers',
                name='Number of Clusters'
            ), 
            row=2, col=2
        )
        

        if optimal_bandwidths['silhouette']:
            fig.add_vline(
                x=optimal_bandwidths['silhouette'], 
                line_dash="dash", 
                line_color="red",
                row=1, col=1
            )
        
        if optimal_bandwidths['calinski_harabasz']:
            fig.add_vline(
                x=optimal_bandwidths['calinski_harabasz'], 
                line_dash="dash", 
                line_color="red",
                row=1, col=2
            )
        
        if optimal_bandwidths['davies_bouldin']:
            fig.add_vline(
                x=optimal_bandwidths['davies_bouldin'], 
                line_dash="dash", 
                line_color="red",
                row=2, col=1
            )
        
        if optimal_bandwidths['num_clusters']:
            fig.add_vline(
                x=optimal_bandwidths['num_clusters'], 
                line_dash="dash", 
                line_color="red",
                row=2, col=2
            )
        

        fig.update_layout(
            height=800, 
            width=1000, 
            title_text="Mean Shift Bandwidth Evaluation",
            font=dict(size=25, color='black'),
            showlegend=True
        )
        

        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(
                    title_text="Bandwidth (log scale)", 
                    type="log",
                    row=i, 
                    col=j, 
                    title_font=dict(size=25, color='black'), 
                    tickfont=dict(size=20, color='black')
                )
                fig.update_yaxes(
                    title_text="Score", 
                    row=i, 
                    col=j, 
                    title_font=dict(size=25, color='black'), 
                    tickfont=dict(size=20, color='black')
                )
        
        return fig, optimal_bandwidths


    def plot_heatmap_cluster(data, labels, title, colorbar_title, use_log=False):
        if labels is None:
            st.error("No labels available for plotting.")
            return

        data_with_labels = data.copy()
        data_with_labels['Cluster'] = labels

        cluster_summary = data_with_labels.groupby('Cluster').mean()
        cluster_counts = data_with_labels.groupby('Cluster').size()

        if cluster_summary.empty:
            st.error("Failed to compute cluster summary.")
            return

        cluster_labels = []
        for i in cluster_summary.index:
            top_elements = cluster_summary.loc[i][cluster_summary.loc[i] > 1].sort_values(ascending=False).head(3).index.tolist()
            top_elements = ', '.join(top_elements)
            count = cluster_counts[i]
            label = f"{top_elements} ({count})"
            cluster_labels.append(label)

        sorted_indices = cluster_counts.sort_values(ascending=True).index
        sorted_labels = [cluster_labels[i] for i in sorted_indices]
        sorted_summary = cluster_summary.loc[sorted_indices]

        z_values = sorted_summary.values
        z_values = np.where(z_values == 0, np.nan, z_values)
        
        if use_log and 'Mass (fg)' in colorbar_title:
            z_values = np.sqrt(z_values)
            colorbar_title = 'sqrt(Mass (fg))'

        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=sorted_summary.columns,
            y=sorted_labels,
            colorscale='ylGnBu',
            colorbar=dict(
                    title=colorbar_title,
                    titlefont=dict(size=30, color='black'),
                    tickfont=dict(size=30, color='black'),
                    ticks='outside',
                    ticklen=5,
                    tickwidth=2,
                    tickcolor='black'
                )
        ))

        fig.update_layout(






            xaxis=dict(
                title='Elements',
                titlefont=dict(size=40, family='Times New Roman', color='black', weight='bold'),
                tickfont=dict(size=40, family='Times New Roman', color='black', weight='bold'),
                tickangle=0,
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                title='Particle (Frequency)',
                titlefont=dict(size=40, family='Times New Roman', color='black', weight='bold'),
                tickfont=dict(size=40, family='Times New Roman', color='black', weight='bold'),
                showgrid=False,
                zeroline=False
            ),
            height=max(1200, 60 * len(sorted_labels)),
            width=3175,
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(
                l=600,  # Increased left margin
                r=100,
                t=100,
                b=100
            ),
            showlegend=False)

    
        fig.update_traces(
            xgap=0,  
            ygap=0   
        )
        st.plotly_chart(fig, use_container_width=True)
        

        st.sidebar.header("Download Options")
        

        img_bytes = fig.to_image(format="png", scale=4)
        

        st.sidebar.download_button(
            label="Download Cluster Heatmap",
            data=img_bytes,
            file_name="heatmap_cluster.png",
            mime="image/png",
            key="download_cluster_heatmap"
        )


        
    def visualize_isotrack_distributions(results, color_map, threshold):
        """Create mass and mole percentage pie charts for IsoTrack data"""
        try:

            mass_data = results['mass']
            mole_data = results['mole']
            

            mass_sums = mass_data.sum()
            mass_percent = mass_sums * 100 / mass_sums.sum()
            

            mole_sums = mole_data.sum()
            mole_percent = mole_sums * 100 / mole_sums.sum()
            

            mass_counts = (mass_data > 0).sum()
            

            fig = make_subplots(rows=1, cols=2, 
                            specs=[[{'type': 'pie'}, {'type': 'pie'}]],
                            subplot_titles=('Mass Distribution', 'Mole Distribution'))
            

            mass_main = mass_percent[mass_percent > threshold]
            mass_others = mass_percent[mass_percent <= threshold]
            if not mass_others.empty:
                mass_main['Others'] = mass_others.sum()
                mass_counts['Others'] = mass_counts[mass_others.index].sum()
                
            mole_main = mole_percent[mole_percent > threshold]
            mole_others = mole_percent[mole_percent <= threshold]
            if not mole_others.empty:
                mole_main['Others'] = mole_others.sum()
                

            colors_mass = [color_map.get(index, '#CCCCCC') for index in mass_main.index]
            fig.add_trace(
                go.Pie(
                    labels=mass_main.index,
                    values=mass_main.values,
                    marker=dict(colors=colors_mass, line=dict(color='#000000', width=1)),
                    textinfo='label+percent',
                    texttemplate='%{label}(%{customdata:d}): %{value:.1f}%',
                    customdata=[mass_counts.get(index, 0) for index in mass_main.index],
                    hoverinfo='label+percent+value',
                    hovertext=[f"{label}: {mass_counts.get(label, 0):d} counts" for label in mass_main.index],
                    textfont=dict(size=25, color='black')
                ),
                row=1, col=1
            )
            

            colors_mole = [color_map.get(index, '#CCCCCC') for index in mole_main.index]
            fig.add_trace(
                go.Pie(
                    labels=mole_main.index,
                    values=mole_main.values,
                    marker=dict(colors=colors_mole, line=dict(color='#000000', width=1)),
                    textinfo='label+percent',
                    texttemplate='%{label}(%{customdata:d}): %{value:.1f}%',
                    customdata=[mass_counts.get(index, 0) for index in mole_main.index],
                    hoverinfo='label+percent+value',
                    hovertext=[f"{label}: {mass_counts.get(label, 0):d} counts" for label in mole_main.index],
                    textfont=dict(size=25, color='black')
                ),
                row=1, col=2
            )
            

            fig.update_layout(
                height=600,
                width=2500,
                title_text='Mass and Mole Distribution',
                title_x=0.4,
                showlegend=False,
                font=dict(size=16, color='black')
            )
            
            st.plotly_chart(fig)
            

            summary_data = []
            summary_data.append(['Element', 'Mass %', 'Mole %', 'Counts'])
            all_elements = sorted(set(mass_main.index) | set(mole_main.index))
            
            for element in all_elements:
                summary_data.append([
                    element,
                    f"{mass_main.get(element, 0):.2f}%",
                    f"{mole_main.get(element, 0):.2f}%",
                    mass_counts.get(element, 0)
                ])
                
            summary_df = pd.DataFrame(summary_data)
            csv = summary_df.to_csv(index=False, header=False)
            st.download_button(
                label="Download Distribution Data as CSV",
                data=csv,
                file_name='isotrack_distribution_data.csv',
                mime='text/csv'
            )
            
        except Exception as e:
            st.error(f"Error creating distribution visualization: {str(e)}")
            
    def get_combinations_and_related_data_isotrack(selected_data, mass_data, mole_data, mole_percent_data):
        """Get combinations and related data for IsoTrack format"""
        start_time = time.time()
        combination_data = {}
        selected_data = selected_data.apply(pd.to_numeric, errors='coerce')


        for index, row in selected_data.iterrows():
            elements = row[row > 0].index.tolist()
            combination_key = ', '.join(sorted(elements))
            combination_data.setdefault(combination_key, []).append(index)

        related_data = {
            'mass_data': {},
            'mole_data': {},
            'mole_percent_data': {}
        }
        combinations = {}

        for combination_key, indices in combination_data.items():
            indices = pd.Index(indices)


            related_data['mass_data'][combination_key] = mass_data.loc[indices]
            related_data['mole_data'][combination_key] = mole_data.loc[indices]
            related_data['mole_percent_data'][combination_key] = mole_percent_data.loc[indices]

            if indices.size > 0:
                filtered_data = selected_data.loc[indices]
                sums = filtered_data.sum()
                counts = indices.size
                average = sums / counts
                squared_diffs = ((filtered_data - average) ** 2).sum()

                combinations[combination_key] = {
                    'sums': sums,
                    'counts': counts,
                    'average': average,
                    'squared_diffs': squared_diffs,
                    'sd': np.sqrt(squared_diffs / counts)
                }

        sd_data = {key: value['sd'] for key, value in combinations.items()}
        sd_df = pd.DataFrame(sd_data).transpose()

        elapsed_time = time.time() - start_time
        st.write(f"Time taken: {elapsed_time} seconds")

        return combinations, related_data['mole_percent_data'], related_data['mass_data'], related_data['mole_data'], sd_df


    def prepare_heatmap_data_isotrack(data_combinations, combinations, start, end):
        """Prepare heatmap data for IsoTrack format"""
        heatmap_df = pd.DataFrame()
        combo_counts = {combo: info['counts'] for combo, info in combinations.items()}

        for combo, df in data_combinations.items():
            df = df.apply(pd.to_numeric, errors='coerce')
            avg_percents = df.mean().to_frame().T

            if data_type == "Mole %":
                avg_percents = avg_percents.div(avg_percents.sum(axis=1), axis=0)

            combo_with_count = f"{combo} ({combo_counts[combo]})"
            avg_percents.index = [combo_with_count]
            heatmap_df = pd.concat([heatmap_df, avg_percents])

        heatmap_df['Counts'] = heatmap_df.index.map(lambda x: combo_counts[x.split(' (')[0]])
        heatmap_df = heatmap_df.sort_values(by='Counts', ascending=False)

        heatmap_df = heatmap_df.iloc[start - 1:end]
        heatmap_df.drop(columns=['Counts'], inplace=True)

        return heatmap_df
    
    def plot_histogram_for_elements_isotrack(mass_data, elements, all_color, single_color, multiple_color, bin_size, x_max, title):
        """Create mass distribution histogram for IsoTrack data"""
        mass_data = mass_data.apply(pd.to_numeric, errors='coerce')
        mass_data = mass_data.dropna(subset=elements)
        
        for elem in elements:
            mass_data[elem] = mass_data[elem][mass_data[elem] > 0]

        selected_elements_mask = np.all([mass_data[elem] > 0 for elem in elements], axis=0)
        filtered_data = mass_data[selected_elements_mask]

        if filtered_data.empty:
            st.error(f"No particles found containing the selected elements: {', '.join(elements)}")
            return

        if len(elements) == 1:
            element = elements[0]
            fig = go.Figure()

            single_data = filtered_data.loc[filtered_data.apply(lambda row: (row > 0).sum(), axis=1) == 1]
            if not single_data.empty:
                fig.add_trace(go.Histogram(
                    x=single_data[element],
                    name='Single',
                    marker_color=single_color,
                    xbins=dict(start=0, end=x_max, size=bin_size),
                    marker_line_color='black',
                    marker_line_width=1
                ))

            multiple_data = filtered_data.loc[filtered_data.apply(lambda row: (row > 0).sum(), axis=1) > 1]
            if not multiple_data.empty:
                fig.add_trace(go.Histogram(
                    x=multiple_data[element],
                    name='Multiple',
                    marker_color=multiple_color,
                    xbins=dict(start=0, end=x_max, size=bin_size),
                    marker_line_color='black',
                    marker_line_width=1
                ))

            fig.update_layout(
                title=f"{title}: {element}",
                xaxis_title="Mass (fg)",
                yaxis_title="Frequency",
                xaxis=dict(range=[0, x_max], title_font=dict(size=40, color='black'), tickfont=dict(size=40, color='black'), linecolor='black', linewidth=1),
                yaxis=dict(title_font=dict(size=40, color='black'), tickfont=dict(size=40, color='black'), linecolor='black', linewidth=1),
                barmode='overlay',
                legend_title_font=dict(size=24, color='black'),
                legend_font=dict(size=24, color='black')
            )
            
            fig.add_annotation(
                x=0.5, y=0.95,
                xref="paper", yref="paper",
                text=f"Total NPs: {len(filtered_data)}, Single: {len(single_data)}, Multiple: {len(multiple_data)}",
                showarrow=False,
                font=dict(size=14, color="black"),
                align="center",
            )

            fig.update_traces(opacity=0.7)

        else:
            total_mass = filtered_data[elements].sum(axis=1)
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=total_mass,
                name='Combined Mass',
                marker_color=all_color,
                xbins=dict(start=0, end=x_max, size=bin_size),
                marker_line_color='black',
                marker_line_width=1
            ))

            combination_key = ','.join(elements)
            fig.update_layout(
                title=f"Mass Distribution: {combination_key}",
                xaxis_title="Mass (fg)",
                yaxis_title="Frequency",
                xaxis=dict(range=[0, x_max], title_font=dict(size=20, color='black'), tickfont=dict(size=20, color='black'), linecolor='black', linewidth=1),
                yaxis=dict(title_font=dict(size=20, color='black'), tickfont=dict(size=20, color='black'), linecolor='black', linewidth=1),
                barmode='overlay',
                legend_title_font=dict(size=16, color='black'),
                legend_font=dict(size=16, color='black')
            )

            fig.add_annotation(
                x=0.5, y=0.95,
                xref="paper", yref="paper",
                text=f"Total NPs: {len(total_mass)}",
                showarrow=False,
                font=dict(size=14, color="black"),
                align="center",
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )

            max_frequency = max(np.histogram(total_mass, bins=np.arange(0, x_max, bin_size))[0])
            fig.update_yaxes(range=[0, max_frequency * 1.1])

        st.plotly_chart(fig)
        
        total_mass_all_particles = filtered_data[elements].sum().sum()
        st.write(f"**Total Mass of All Particles:** {total_mass_all_particles:.2f} fg")


        max_length = max([len(filtered_data[elem]) for elem in elements])
        all_values = [list(filtered_data[elem]) + [''] * (max_length - len(filtered_data[elem])) for elem in elements]

        summary_data = [['File name: ' + title, '', '', ''],
                        ['Mass (fg) - ' + elem for elem in elements]]

        for row in zip(*all_values):
            summary_data.append(list(row))

        summary_df = pd.DataFrame(summary_data)
        csv = summary_df.to_csv(index=False, header=False)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='histogram_element_data.csv',
            mime='text/csv'
        )

   
    def create_parallel_sets(results, particles_per_ml=None, show_concentration=False):
        """
        Create parallel sets diagram with option to show counts or particles/mL and colored connections
        """
        mass_data = results['mass']
        

        element_total_counts = {}
        element_singles = {}
        two_elem_combos = defaultdict(int)
        three_elem_combos = defaultdict(int)
        

        element_threshold = st.sidebar.number_input("Minimum single element count threshold", value=200, min_value=0)
        pair_threshold = st.sidebar.number_input("Minimum pair count threshold", value=100, min_value=0)
        triple_threshold = st.sidebar.number_input("Minimum triple count threshold", value=50, min_value=0)
        

        total_particles = 0
        for index, row in mass_data.iterrows():
            present_elements = row[row > 0].index.tolist()
            total_particles += 1
            

            for elem in present_elements:
                element_total_counts[elem] = element_total_counts.get(elem, 0) + 1
            

            if len(present_elements) == 1:
                elem = present_elements[0]
                element_singles[elem] = element_singles.get(elem, 0) + 1
            elif len(present_elements) == 2:
                combo = '-'.join(sorted(present_elements))
                two_elem_combos[combo] += 1
            elif len(present_elements) == 3:
                combo = '-'.join(sorted(present_elements))
                three_elem_combos[combo] += 1


        def count_to_pml(count):
            if show_concentration and particles_per_ml:
                ratio = count / total_particles
                return ratio * float(particles_per_ml.split()[0])
            return count


        filtered_elements = {elem: count for elem, count in element_total_counts.items() 
                            if count >= element_threshold}
        

        others_count = sum(count for elem, count in element_total_counts.items() 
                        if count < element_threshold)
        
        if not filtered_elements and others_count == 0:
            st.warning("No elements found.")
            return None


        if others_count > 0:
            filtered_elements["Others"] = others_count


        nodes = []
        links = []
        node_indices = {}
        current_idx = 0
        

        sorted_elements = sorted(filtered_elements.items(), key=lambda x: -x[1])


        default_colors = [
                        '#FF6347', '#FFD700', '#FFA500', '#20B2AA', '#00BFFF',
                        '#F0E68C', '#E0FFFF', '#AFEEEE', '#DDA0DD', '#FFE4E1',
                        '#FAEBD7', '#D3D3D3', '#90EE90', '#FFB6C1', '#FFA07A',
                        '#C71585', '#DB7093', '#FF1493', '#FF69B4', '#FFA07A',
                        '#FFB6C1', '#87CEEB', '#98FB98', '#FFFFE0', '#FFDAB9',
                        '#E6E6FA', '#FFF0F5', '#B0E0E6', '#FFC0CB', '#F5DEB3',
                    ]


        st.sidebar.title("Element Colors")
        element_colors = {}


        for i, (elem, _) in enumerate(sorted_elements):
            if elem != "Others":
                default_color = default_colors[i % len(default_colors)]
                element_colors[elem] = st.sidebar.color_picker(
                    f"Color for {elem}",
                    default_color,
                    key=f"color_{elem}"
                )


        element_colors["Others"] = st.sidebar.color_picker(
            "Color for Others",
            "#808080",  # Default gray
            key="color_others"
        )
        total_nodes = len(sorted_elements)

        def format_scientific(value):
            """Format number in scientific notation with smaller exponent"""
            sci = f"{value:.2e}"
            base, exponent = sci.split('e+')
            return f"{base}×10<sup>{int(exponent)}</sup>"


        for i, (elem, total) in enumerate(sorted_elements):
            y_pos = 1 - (i / (total_nodes - 1)) if total_nodes > 1 else 0.5
            value = count_to_pml(total)
            display_value = format_scientific(value) if show_concentration else str(total)
            nodes.append({
                "name": f"{elem}",
                "level": 0,
                "color": element_colors[elem],
                "y": y_pos
            })
            node_indices[elem] = current_idx
            current_idx += 1


        singles = []
        doubles = []
        triples = []


        others_singles = 0
        for elem, count in element_singles.items():
            if elem in filtered_elements and elem != "Others":
                value = count_to_pml(count)
                display_value = format_scientific(value) if show_concentration else str(count)
                singles.append((count, f"{elem} Single: {display_value}", {elem}, elem))
            else:
                others_singles += count
        
        if others_singles > 0:
            value = count_to_pml(others_singles)
            display_value = format_scientific(value) if show_concentration else str(others_singles)
            singles.append((others_singles, f"Others Singles: {display_value}", {"Others"}, "Others"))
        singles.sort(key=lambda x: -x[0])


        others_doubles = 0
        for combo, count in two_elem_combos.items():
            elements = set(combo.split('-'))
            if count >= pair_threshold and any(elem in filtered_elements and elem != "Others" for elem in elements):
                value = count_to_pml(count)
                display_value = format_scientific(value) if show_concentration else str(count)
                doubles.append((count, f"{combo}: {display_value}", elements, None))
            else:
                others_doubles += count

        if others_doubles > 0:
            value = count_to_pml(others_doubles)
            display_value = format_scientific(value) if show_concentration else str(others_doubles)
            doubles.append((others_doubles, f"Other Pairs: {display_value}", {"Others"}, None))
        doubles.sort(key=lambda x: -x[0])
        

        others_triples = 0
        for combo, count in three_elem_combos.items():
            elements = set(combo.split('-'))
            if count >= triple_threshold and any(elem in filtered_elements and elem != "Others" for elem in elements):
                value = count_to_pml(count)
                display_value = format_scientific(value) if show_concentration else str(count)
                triples.append((count, f"{combo}: {display_value}", elements, None))
            else:
                others_triples += count

        if others_triples > 0:
            value = count_to_pml(others_triples)
            display_value = format_scientific(value) if show_concentration else str(others_triples)
            triples.append((others_triples, f"Other Triples: {display_value}", {"Others"}, None))
        triples.sort(key=lambda x: -x[0])
                

        all_combinations = singles + doubles + triples
        

        total_combinations = len(all_combinations)
        link_colors = []  # Store colors for each link
        
        for i, (count, label, elements, single_elem) in enumerate(all_combinations):
            y_pos = 1 - (i / (total_combinations - 1)) if total_combinations > 1 else 0.5
            nodes.append({
                "name": label,
                "level": 1,
                "color": element_colors[single_elem] if single_elem else "lightgray",
                "y": y_pos
            })
            for elem in elements:
                if elem in node_indices:
                    value = count_to_pml(count)
                    links.append({
                        "source": node_indices[elem],
                        "target": current_idx,
                        "value": value
                    })
                    if len(elements) == 1:
                        link_colors.append(element_colors[elem])  # Use element color for single element connections
                    else:
                        link_colors.append(f"rgba{tuple(int(element_colors[elem].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.7,)}")  # Semi-transparent version of element color
            current_idx += 1


        fig = go.Figure(data=[go.Sankey(
            node = {
                "pad": 60,
                "thickness": 20,
                "line": dict(color="black", width=1.5),
                "label": [node["name"] for node in nodes],
                "x": [node["level"] for node in nodes],
                "y": [node["y"] for node in nodes],
                "color": [node["color"] for node in nodes]
            },
            link = {
                "source": [link["source"] for link in links],
                "target": [link["target"] for link in links],
                "value": [link["value"] for link in links],
                "color": link_colors
            }
        )])

        title_text = "Element Flow Distribution"
        title_text += " (P/mL)" if show_concentration else " (Counts)"

        fig.update_layout(
            title_text=title_text,
            title_font_size=20,
            font=dict(size=35, weight= 'bold', color = 'black'),
            height=900,
            width=1400,
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(
                l=50,  # Left margin
            )
        )
        st.sidebar.title("Download Options")
        if st.sidebar.button("Download as PNG"):
            
            fig.write_image("element_flow_distribution.png", scale=4) 
            st.sidebar.success("Downloaded as element_flow_distribution.png")
                
        return fig
                                                                                                    
                                                                                        

    if 'df' in globals():
        if data_type == "SPCal":
            counts_data, fg_data = process_and_display_spcal(df)
            if counts_data is not None and fg_data is not None:
                
                st.sidebar.title('Pie chart for mass')
                if st.sidebar.checkbox("Show Mass Distribution", value=False):
                    default_colors = [
                        '#FF6347', '#FFD700', '#FFA500', '#20B2AA', '#00BFFF',
                        '#F0E68C', '#E0FFFF', '#AFEEEE', '#DDA0DD', '#FFE4E1',
                        '#FAEBD7', '#D3D3D3', '#90EE90', '#FFB6C1', '#FFA07A',
                        '#C71585', '#DB7093', '#FF1493', '#FF69B4', '#FFA07A',
                        '#FFB6C1', '#87CEEB', '#98FB98', '#FFFFE0', '#FFDAB9',
                        '#E6E6FA', '#FFF0F5', '#B0E0E6', '#FFC0CB', '#F5DEB3',
                    ]
                    
                    color_map = create_color_map(st.session_state.filtered_fg_data.columns, default_colors)
                    threshold = st.sidebar.number_input('Threshold for Others (%)', format="%f")
                    visualize_mass_percentages_pie_chart_spcal(st.session_state.filtered_fg_data, color_map, threshold)
                    
                st.sidebar.title('Mass Distribution')
                perform_mass_distribution_analysis = st.sidebar.checkbox('Perform Mass distribution Analysis?')

                if perform_mass_distribution_analysis:
                    elements_to_plot = st.sidebar.multiselect(
                        'Select up to 3 elements to view the histogram:', 
                        st.session_state.filtered_fg_data.columns,
                        max_selections=3
                    )
                    
                    bin_size = st.sidebar.slider(
                        'Select bin size:', 
                        min_value=0.001, 
                        max_value=10.0, 
                        value=0.01, 
                        step=0.01
                    )
                    
                    x_max = st.sidebar.slider(
                        'Select max value for x-axis (Mass (fg)):', 
                        min_value=0, 
                        max_value=1000, 
                        value=10,
                        step=1
                    )

                    all_color = st.sidebar.color_picker('Pick a color for All data', '#00f900')
                    single_color = st.sidebar.color_picker('Pick a color for Single detections', '#0000ff')
                    multiple_color = st.sidebar.color_picker('Pick a color for Multiple detections', '#fff000')

                    plot_histogram_for_elements_spcal(
                        st.session_state.filtered_fg_data,
                        elements_to_plot, 
                        all_color, 
                        single_color, 
                        multiple_color, 
                        bin_size, 
                        x_max, 
                        "Mass Distribution"
                    )
                    

                combinations, mass_data_combinations, sd_df = get_combinations_and_related_data_spcal(st.session_state.filtered_fg_data)
            
                if combinations is not None:
                    st.sidebar.title("Single and Multiple Element Analysis")
                    show_s_m = st.sidebar.checkbox("Single and Multiple Element Analysis?")
                    if show_s_m:
                        visualize_pie_chart_single_and_multiple(combinations, None)

                    st.sidebar.title('Element Distribution')
                    perform_element_distribution = st.sidebar.checkbox("Perform Element Distribution")
                    if perform_element_distribution:
                        elements_to_analyze = st.sidebar.multiselect(
                            'Select elements to analyze:',
                            st.session_state.filtered_fg_data.columns.tolist()
                        )
                        elements_to_exclude = st.sidebar.multiselect(
                            'Select elements to exclude from combinations:',
                            st.session_state.filtered_fg_data.columns.tolist(),
                            default=[]
                        )
                        count_threshold = st.sidebar.number_input(
                            'Set a count threshold for display:', 
                            min_value=0, 
                            value=10, 
                            step=1
                        )
                        if elements_to_analyze:
                            plot_combination_distribution_by_counts(
                                combinations,
                                elements_to_analyze,
                                elements_to_exclude,
                                count_threshold
                            )

                st.sidebar.title('Heatmap')
                perform_heatmap = st.sidebar.checkbox("Perform Mass Heatmap Analysis?")

                if perform_heatmap:
                    display_numbers = st.sidebar.checkbox("Display Numbers on Heatmap", value=True)
                    font_size = st.sidebar.slider("Font Size for Numbers on Heatmap", min_value=5, max_value=30, value=14)


                    combinations, mass_data_combinations, sd_df = get_combinations_and_related_data_spcal(st.session_state.filtered_fg_data)

                    colorscale_options = [
                        'aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance', 'blackbody', 'bluered', 'blues', 'blugrn',
                        'bluyl', 'brbg', 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl', 'darkmint', 'deep',
                        'delta', 'dense', 'earth', 'edge', 'electric', 'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens',
                        'greys', 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet', 'magenta', 'magma', 'matter',
                        'mint', 'mrybm', 'mygbm', 'oranges', 'orrd', 'oryel', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg',
                        'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 'puor', 'purd', 'purp', 'purples', 'purpor',
                        'rainbow', 'rdbu', 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar', 'spectral', 'speed',
                        'sunset', 'sunsetdark', 'teal', 'tealgrn', 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
                        'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd'
                    ]

                    st.sidebar.title('Heatmap Parameters')
                    selected_colorscale = st.sidebar.selectbox('Select a colorscale:', colorscale_options, index=89)

                    if combinations:
                        if len(combinations) <= 1:
                            st.warning("At least two combinations are needed to create a heatmap. Please select more elements.")
                        else:
                            max_idx = max(1, len(combinations) - 1)  # Ensure max_value is at least 1
                            start = st.sidebar.number_input('Start from combination:', min_value=1, max_value=max_idx, value=1)
                            end = st.sidebar.number_input('End at combination:', min_value=2, max_value=len(combinations), value=min(2, len(combinations)))
                            end = max(end, start + 1)

                            try:
                                heatmap_df = prepare_heatmap_data_spcal(mass_data_combinations, combinations, start, end)
                                if heatmap_df is not None:
                                    plot = plot_heatmap(heatmap_df, sd_df, selected_colorscale=selected_colorscale, 
                                                    display_numbers=display_numbers, font_size=font_size)
                                    st.plotly_chart(plot)
                            except Exception as e:
                                st.error(f"Error preparing heatmap: {str(e)}")
                            
                st.sidebar.title('Clustering Analysis')
                perform_clustering = st.sidebar.checkbox('Perform Clustering Analysis?')
                if perform_clustering:

                    clustering_method = st.sidebar.selectbox(
                        'Select clustering method:',
                        ['K-Means', 'Hierarchical', 'Spectral', 'Gaussian', 'Mini-Batch K-Means', 'Mean Shift']
                    )
            
                    st.sidebar.title('Analysis Options')
                    perform_clustering_number = st.sidebar.checkbox('Evaluate optimal number of clusters')
                    perform_heatmap = st.sidebar.checkbox('Generate clustering heatmap')
                    
                    if perform_clustering_number:
                        if clustering_method != 'Mean Shift':
                            max_clusters = st.sidebar.slider('Maximum number of clusters to evaluate:', 
                                                        min_value=5, max_value=100, value=20)
                            fig, optimal_clusters = evaluate_clusters(st.session_state.filtered_fg_data, clustering_method, max_clusters)
                            st.plotly_chart(fig)
                            st.write("Suggested number of clusters based on different metrics:")
                            for metric, n_clusters in optimal_clusters.items():
                                st.write(f"- {metric}: {n_clusters}")
                    
                    if perform_heatmap:
                        num_clusters = st.sidebar.slider('Select number of clusters:', 
                                                    min_value=2, max_value=100, value=5)
                        
                        labels = apply_clustering(st.session_state.filtered_fg_data, clustering_method, num_clusters)
                        if labels is not None:
                            plot_heatmap_cluster(st.session_state.filtered_fg_data, labels, 
                                            f"{clustering_method} Clustering Results",
                                            "Mass (fg)",
                                            use_log=True)
                                
       
        
        elif data_type == "NuQuant":
            mass_data, mass_percent_data, mole_data, mole_percent_data, counts_data = process_data(df)
            

            elements = st.session_state.filtered_mass_data.columns

            default_colors = [
                '#FF6347', '#FFD700', '#FFA500', '#20B2AA', '#00BFFF',
                '#F0E68C', '#E0FFFF', '#AFEEEE', '#DDA0DD', '#FFE4E1',
                '#FAEBD7', '#D3D3D3', '#90EE90', '#FFB6C1', '#FFA07A',
                '#C71585', '#DB7093', '#FF1493', '#FF69B4', '#FFA07A',
                '#FFB6C1', '#87CEEB', '#98FB98', '#FFFFE0', '#FFDAB9',
                '#E6E6FA', '#FFF0F5', '#B0E0E6', '#FFC0CB', '#F5DEB3',
            ]

            st.sidebar.title('Elemental Distribution')
            if st.sidebar.checkbox("Show Mass and Mole Distribution", value=False):
                color_map = create_color_map(elements, default_colors)
                threshold = st.sidebar.number_input('Threshold for Others (%)', format="%f")
                visualize_mass_and_mole_percentages_pie_charts(st.session_state.filtered_mass_data, st.session_state.filtered_mole_data, color_map, threshold)


            if 'perform_combination_test' not in st.session_state:
                st.session_state['perform_combination_test'] = False
            if 'combinations' not in st.session_state:
                st.session_state['combinations'] = None
            if 'mass_data_combinations' not in st.session_state:
                st.session_state['mass_data_combinations'] = None
            if 'mole_data_combinations' not in st.session_state:
                st.session_state['mole_data_combinations'] = None
            if 'mole_percent_data_combinations' not in st.session_state:
                st.session_state['mole_percent_data_combinations'] = None
            if 'sd_df' not in st.session_state:
                st.session_state['sd_df'] = None
            if 'data_loaded' not in st.session_state:
                st.session_state['data_loaded'] = False
            if 'heatmap_df' not in st.session_state:
                st.session_state['heatmap_df'] = None
            if 'heatmap_params' not in st.session_state:
                st.session_state['heatmap_params'] = None
            if 'heatmap_plot' not in st.session_state:
                st.session_state['heatmap_plot'] = None

        
            st.sidebar.title('Heatmap')
            perform_combination_test = st.sidebar.checkbox("Perform Heatmap Test?",
                                                        value=st.session_state.get('perform_combination_test', False))
            st.session_state['perform_combination_test'] = perform_combination_test

            if perform_combination_test:
                data_type = st.sidebar.selectbox("Select data type for heatmap:", ["Mass", "Mole", "Mole %"])
                

                selected_data = {
                    "Mass": st.session_state.filtered_mass_data,
                    "Mole": st.session_state.filtered_mole_data,
                    "Mole %": st.session_state.filtered_mole_percent_data
                }[data_type]

                display_numbers = st.sidebar.checkbox("Display Numbers on Heatmap", value=True)
                font_size = st.sidebar.slider("Font Size for Numbers on Heatmap", min_value=5, max_value=30, value=14)

                combinations, mole_percent_data_combinations, mass_data_combinations, mass_percent_data_combination, mole_data_combinations, sd_df = get_combinations_and_related_data(
                    selected_data, st.session_state.filtered_mass_data, st.session_state.filtered_mass_percent_data, 
                    st.session_state.filtered_mole_data, st.session_state.filtered_mole_percent_data)
                
                st.session_state['combinations'] = combinations
                st.session_state['mass_data_combinations'] = mass_data_combinations
                st.session_state['mole_data_combinations'] = mole_data_combinations
                st.session_state['mole_percent_data_combinations'] = mole_percent_data_combinations
                st.session_state['sd_df'] = sd_df
                st.session_state['data_loaded'] = True

                combinations = st.session_state['combinations']
                sd_df = st.session_state['sd_df']

                colorscale_options = [
                    'aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance', 'blackbody', 'bluered', 'blues', 'blugrn',
                    'bluyl', 'brbg', 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl', 'darkmint', 'deep',
                    'delta', 'dense', 'earth', 'edge', 'electric', 'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens',
                    'greys', 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet', 'magenta', 'magma', 'matter',
                    'mint', 'mrybm', 'mygbm', 'oranges', 'orrd', 'oryel', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg',
                    'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 'puor', 'purd', 'purp', 'purples', 'purpor',
                    'rainbow', 'rdbu', 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar', 'spectral', 'speed',
                    'sunset', 'sunsetdark', 'teal', 'tealgrn', 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
                    'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd'
                ]
                st.sidebar.title('Heatmap Parameters')

                selected_colorscale = st.sidebar.selectbox('Select a colorscale:', colorscale_options, index=89)

                if combinations:
                    if len(combinations) <= 1:
                        st.warning("At least two combinations are needed to create a heatmap. Please select more elements.")
                    else:
                        max_idx = max(1, len(combinations) - 1)  # Ensure max_value is at least 1
                        start = st.sidebar.number_input('Start from combination:', min_value=1, max_value=max_idx, value=1)
                        end = st.sidebar.number_input('End at combination:', min_value=2, max_value=len(combinations), value=min(2, len(combinations)))
                        end = max(end, start + 1)

                        current_params = (selected_colorscale, start, end)
                        heatmap_df = None

                        try:
                            if data_type == "Mass":
                                heatmap_df = prepare_heatmap_data(st.session_state['mass_data_combinations'], combinations, start, end)
                            elif data_type == "Mole":
                                heatmap_df = prepare_heatmap_data(st.session_state['mole_data_combinations'], combinations, start, end)
                            elif data_type == "Mole %":
                                heatmap_df = prepare_heatmap_data(st.session_state['mole_percent_data_combinations'], combinations,
                                                                start, end)

                            st.session_state['heatmap_df'] = heatmap_df
                            st.session_state['heatmap_params'] = current_params

                            plot = plot_heatmap(st.session_state['heatmap_df'], st.session_state['sd_df'],
                                                selected_colorscale=selected_colorscale, display_numbers=display_numbers, font_size=font_size)
                            st.session_state['heatmap_plot'] = plot

                        except Exception as e:
                            st.write(f"Error preparing heatmap data: {e}")  # debug statement

                        if 'heatmap_plot' in st.session_state:
                            st.plotly_chart(st.session_state['heatmap_plot'])
                            
                        heatmap_df = st.session_state['heatmap_df']

            st.sidebar.title('Mass Distribution')

            perform_mass_distribution_analysis = st.sidebar.checkbox('Perform Mass distribution Analysis?')

            if perform_mass_distribution_analysis:
                elements_to_plot = st.sidebar.multiselect('Select up to 3 elements to view the histogram:', 
                                                    st.session_state.filtered_mass_data.columns,
                                                    max_selections=3)
                only_selected_elements = st.checkbox("Show only particles with exactly the selected elements")

                bin_size = st.sidebar.slider('Select bin size:', min_value=0.001, max_value=10.0, value=0.01, step=0.01)
                x_max = st.sidebar.slider('Select max value for x-axis (Mass (fg)):', min_value=0, max_value=1000, value=10,
                                        step=1)

                all_color = st.sidebar.color_picker('Pick a color for All data', '#00f900')  # Default color is light green
                single_color = st.sidebar.color_picker('Pick a color for Single detections', '#0000ff')  # Default blue
                multiple_color = st.sidebar.color_picker('Pick a color for Multiple detections', '#fff000')  # Default red

                plot_histogram_for_elements(st.session_state.filtered_mass_data, elements_to_plot, all_color, single_color, multiple_color, bin_size,
                                            x_max, "Mass Distribution", st.session_state['mass_data_combinations'], only_selected_elements)

            st.sidebar.title("Single and Multiple Element Analysis")
            show_s_m = st.sidebar.checkbox("Single and Multiple Element Analysis?")
            if show_s_m:
                visualize_pie_chart_single_and_multiple(st.session_state['combinations'], ppm)

            st.sidebar.title('Element Distribution')
            perform_element_distribution = st.sidebar.checkbox("Perform Element Distribution")
            if perform_element_distribution:
                if 'heatmap_df' in st.session_state and st.session_state['heatmap_df'] is not None:
                    elements_to_analyze = st.sidebar.multiselect('Select elements to analyze:',
                                                                st.session_state['heatmap_df'].columns.tolist())
                    elements_to_exclude = st.sidebar.multiselect('Select elements to exclude from combinations:',
                                                                st.session_state['heatmap_df'].columns.tolist(), default=[])
                else:
                    elements_to_analyze = st.sidebar.multiselect('Select elements to analyze:',
                                                                st.session_state.filtered_mass_data.columns.tolist())
                    elements_to_exclude = st.sidebar.multiselect('Select elements to exclude from combinations:',
                                                                st.session_state.filtered_mass_data.columns.tolist(), default=[])
                    
                percentage_threshold = st.sidebar.slider(
                    'Minimum percentage threshold for display (%):', 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=0.5, 
                    step=0.1
                )
                if elements_to_analyze:
                    plot_combination_distribution_by_counts(st.session_state['combinations'], elements_to_analyze, elements_to_exclude, percentage_threshold)

            st.sidebar.title('Raw Data after combination')

            Raw_data_after_combination = st.sidebar.checkbox('Export Raw Data?')
            if Raw_data_after_combination:

                _, mole_percent_data_after_combination, mass_data_after_combination, mass_percent_data_after_combination, mole_data_after_combination, sd_df = get_combinations_and_related_data(
                    st.session_state.filtered_mole_percent_data, st.session_state.filtered_mass_data, 
                    st.session_state.filtered_mass_percent_data, st.session_state.filtered_mole_data, 
                    st.session_state.filtered_mole_percent_data)

                data_type_options = ['Mass Data', 'Mole Data', 'Mass Percent Data', 'Mole Percent Data']
                selected_data_type = st.selectbox('Select data type to aggregate and display:', options=data_type_options)

                if selected_data_type == 'Mass Data':
                    aggregated_mass_data = aggregate_combination_data(mass_data_after_combination)
                    display_aggregated_data(aggregated_mass_data, 'Mass Data')
                elif selected_data_type == 'Mole Data':
                    aggregated_mole_data = aggregate_combination_data(mole_data_after_combination)
                    display_aggregated_data(aggregated_mole_data, 'Mole Data')
                elif selected_data_type == 'Mass Percent Data':
                    aggregated_mass_percent_data = aggregate_combination_data(mass_percent_data_after_combination)
                    display_aggregated_data(aggregated_mass_percent_data, 'Mass Percent Data')
                elif selected_data_type == 'Mole Percent Data':
                    aggregated_mole_percent_data = aggregate_combination_data(mole_percent_data_after_combination)
                    display_aggregated_data(aggregated_mole_percent_data, 'Mole Percent Data')
            
            st.sidebar.title('Clustering Analysis')
            perform_clustering = st.sidebar.checkbox('Perform Clustering Analysis?')
            if perform_clustering:

                clustering_method = st.sidebar.selectbox(
                    'Select clustering method:',
                    ['K-Means', 'Hierarchical', 'Spectral', 'Gaussian', 'Mini-Batch K-Means', 'Mean Shift']
                )
                
                st.sidebar.title('Analysis Options')
                perform_clustering_number = st.sidebar.checkbox('Evaluate optimal number of clusters/parameters')
                perform_heatmap = st.sidebar.checkbox('Generate clustering heatmap')
                
                if perform_clustering_number:
                    if clustering_method == 'Mean Shift':
                    
                        min_bandwidth = st.sidebar.number_input('Minimum bandwidth:', min_value=0.1, value=0.5, step=0.1)
                        max_bandwidth = st.sidebar.number_input('Maximum bandwidth:', min_value=1.0, value=50.0, step=1.0)
                        num_bandwidths = st.sidebar.slider('Number of bandwidths to test:', min_value=5, max_value=50, value=20)
                       
                        fig, optimal_bandwidths = evaluate_mean_shift(
                            st.session_state.filtered_mole_percent_data, 
                            max_bandwidth=max_bandwidth,
                            min_bandwidth=min_bandwidth,
                            num_bandwidths=num_bandwidths
                        )
                        st.plotly_chart(fig)
                        
                        st.write("### Optimal bandwidth values from different metrics:")
                        if optimal_bandwidths['silhouette']:
                            st.write(f"- Silhouette Score (higher is better): {optimal_bandwidths['silhouette']:.2f}")
                        if optimal_bandwidths['calinski_harabasz']:
                            st.write(f"- Calinski-Harabasz Index (higher is better): {optimal_bandwidths['calinski_harabasz']:.2f}")
                        if optimal_bandwidths['davies_bouldin']:
                            st.write(f"- Davies-Bouldin Index (lower is better): {optimal_bandwidths['davies_bouldin']:.2f}")
                        if optimal_bandwidths['num_clusters']:
                            st.write(f"- Number of Clusters (elbow method): {optimal_bandwidths['num_clusters']:.2f}")


                            st.session_state.optimal_bandwidths = optimal_bandwidths
                        else:
                            st.warning("Could not determine optimal bandwidth. Try adjusting the bandwidth range.")
                    else:

                        max_clusters = st.sidebar.slider('Maximum number of clusters to evaluate:', 
                                                min_value=5, max_value=100, value=20)
                        fig, optimal_clusters = evaluate_clusters(st.session_state.filtered_mole_percent_data, clustering_method, max_clusters)
                        st.plotly_chart(fig)
                        st.write("Suggested number of clusters based on different metrics:")
                        for metric, n_clusters in optimal_clusters.items():
                            st.write(f"- {metric}: {n_clusters}")
                
                if perform_heatmap:
                    if clustering_method == 'Mean Shift':

                        if hasattr(st.session_state, 'optimal_bandwidth'):
                            default_bandwidth = st.session_state.optimal_bandwidth
                        else:
                            default_bandwidth = 5.0
                            
                        bandwidth = st.sidebar.number_input(
                            'Select bandwidth:', 
                            min_value=0.1, 
                            max_value=100.0, 
                            value=default_bandwidth,
                            step=0.1
                        )
                        

                        from sklearn.cluster import MeanShift
                        model = MeanShift(bandwidth=bandwidth, n_jobs=-1)
                        labels = model.fit_predict(st.session_state.filtered_mole_percent_data)
                        

                        n_clusters = len(np.unique(labels))
                        st.write(f"Number of clusters found: {n_clusters}")
                    else:

                        num_clusters = st.sidebar.slider('Select number of clusters:', 
                                                    min_value=2, max_value=100, value=5)
                        labels = apply_clustering(st.session_state.filtered_mole_percent_data, clustering_method, num_clusters)
                    
                    if labels is not None:
                        plot_heatmap_cluster(st.session_state.filtered_mole_percent_data, labels, 
                                        f"{clustering_method} Clustering Results",
                                        "Mole %")
                    
            
        elif data_type == "IsoTrack":
            results = process_isotrack(df)
            if results:
                

                if 'filtered_results' not in st.session_state:
                    st.session_state.filtered_results = results.copy()
                
                st.sidebar.title('Element Distribution')
                if st.sidebar.checkbox("Show Mass and Mole Distribution", value=False):

                    default_colors = [
                        '#FF6347', '#FFD700', '#FFA500', '#20B2AA', '#00BFFF',
                        '#F0E68C', '#E0FFFF', '#AFEEEE', '#DDA0DD', '#FFE4E1',
                        '#FAEBD7', '#D3D3D3', '#90EE90', '#FFB6C1', '#FFA07A',
                        '#C71585', '#DB7093', '#FF1493', '#FF69B4', '#FFA07A'
                    ]
                    all_elements = sorted(set(st.session_state.filtered_results['mass'].columns))
                    color_map = create_color_map(all_elements, default_colors)
                    threshold = st.sidebar.number_input('Threshold for Others (%)', format="%f")
                    visualize_isotrack_distributions(st.session_state.filtered_results, color_map, threshold)
                    
                st.sidebar.title('Mass Distribution')
                perform_mass_distribution_analysis = st.sidebar.checkbox('Perform Mass distribution Analysis?')

                if perform_mass_distribution_analysis:
                    elements_to_plot = st.sidebar.multiselect(
                        'Select up to 3 elements to view the histogram:', 
                        st.session_state.filtered_results['mass'].columns,
                        max_selections=3
                    )
                    
                    bin_size = st.sidebar.slider(
                        'Select bin size:', 
                        min_value=0.001, 
                        max_value=10.0, 
                        value=0.01, 
                        step=0.01
                    )
                    
                    x_max = st.sidebar.slider(
                        'Select max value for x-axis (Mass (fg)):', 
                        min_value=0, 
                        max_value=1000, 
                        value=10,
                        step=1
                    )

                    all_color = st.sidebar.color_picker('Pick a color for All data', '#00f900')
                    single_color = st.sidebar.color_picker('Pick a color for Single detections', '#0000ff')
                    multiple_color = st.sidebar.color_picker('Pick a color for Multiple detections', '#fff000')

                    if elements_to_plot:
                        plot_histogram_for_elements_isotrack(
                            st.session_state.filtered_results['mass'], 
                            elements_to_plot, 
                            all_color, 
                            single_color, 
                            multiple_color, 
                            bin_size, 
                            x_max, 
                            "Mass Distribution"
                        )

                st.sidebar.title('Heatmap')
                perform_heatmap = st.sidebar.checkbox("Perform Heatmap Analysis?")

                if perform_heatmap:
                    data_type = st.sidebar.selectbox("Select data type for heatmap:", ["Mass", "Mole", "Mole %"])
                    

                    selected_data = {
                        "Mass": st.session_state.filtered_results['mass'],
                        "Mole": st.session_state.filtered_results['mole'],
                        "Mole %": st.session_state.filtered_results['mole_percent']
                    }[data_type]

                    display_numbers = st.sidebar.checkbox("Display Numbers on Heatmap", value=True)
                    font_size = st.sidebar.slider("Font Size for Numbers on Heatmap", min_value=5, max_value=30, value=14)


                    combinations, mole_percent_data_combinations, mass_data_combinations, mole_data_combinations, sd_df = get_combinations_and_related_data_isotrack(
                        selected_data, st.session_state.filtered_results['mass'], 
                        st.session_state.filtered_results['mole'], 
                        st.session_state.filtered_results['mole_percent'])

                    colorscale_options = [
                        'aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance', 'blackbody', 'bluered', 'blues', 'blugrn',
                        'bluyl', 'brbg', 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl', 'darkmint', 'deep',
                        'delta', 'dense', 'earth', 'edge', 'electric', 'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens',
                        'greys', 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet', 'magenta', 'magma', 'matter',
                        'mint', 'mrybm', 'mygbm', 'oranges', 'orrd', 'oryel', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg',
                        'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 'puor', 'purd', 'purp', 'purples', 'purpor',
                        'rainbow', 'rdbu', 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar', 'spectral', 'speed',
                        'sunset', 'sunsetdark', 'teal', 'tealgrn', 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
                        'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd'
                    ]

                    st.sidebar.title('Heatmap Parameters')
                    selected_colorscale = st.sidebar.selectbox('Select a colorscale:', colorscale_options, index=89)

                    if combinations:
                        if len(combinations) <= 1:
                            st.warning("At least two combinations are needed to create a heatmap. Please select more elements.")
                        else:
                            max_idx = max(1, len(combinations) - 1)  # Ensure max_value is at least 1
                            start = st.sidebar.number_input('Start from combination:', min_value=1, max_value=max_idx, value=1)
                            end = st.sidebar.number_input('End at combination:', min_value=2, max_value=len(combinations), value=min(2, len(combinations)))
                            end = max(end, start + 1)

                            try:
                                if data_type == "Mass":
                                    heatmap_df = prepare_heatmap_data_isotrack(mass_data_combinations, combinations, start, end)
                                elif data_type == "Mole":
                                    heatmap_df = prepare_heatmap_data_isotrack(mole_data_combinations, combinations, start, end)
                                elif data_type == "Mole %":
                                    heatmap_df = prepare_heatmap_data_isotrack(mole_percent_data_combinations, combinations, start, end)

                                if heatmap_df is not None:
                                    plot = plot_heatmap(heatmap_df, sd_df, selected_colorscale=selected_colorscale, 
                                                    display_numbers=display_numbers, font_size=font_size)
                                    st.plotly_chart(plot)
                            except Exception as e:
                                st.error(f"Error preparing heatmap: {str(e)}")
                                
                st.sidebar.title('Element Distribution')
                perform_element_distribution = st.sidebar.checkbox("Perform Element Distribution")
                if perform_element_distribution:

                    selected_data = st.session_state.filtered_results['mass']
                    combinations, mole_percent_data_combinations, mass_data_combinations, mole_data_combinations, sd_df = get_combinations_and_related_data_isotrack(
                        selected_data, st.session_state.filtered_results['mass'], 
                        st.session_state.filtered_results['mole'], 
                        st.session_state.filtered_results['mole_percent'])
                    
                    elements_to_analyze = st.sidebar.multiselect('Select elements to analyze:',
                                                                st.session_state.filtered_results['mass'].columns.tolist())
                    elements_to_exclude = st.sidebar.multiselect('Select elements to exclude from combinations:',
                                                                st.session_state.filtered_results['mass'].columns.tolist(), default=[])
                    percentage_threshold = st.sidebar.slider(
                            'Minimum percentage threshold for display (%):', 
                            min_value=0.0, 
                            max_value=100.0, 
                            value=0.5, 
                            step=0.1
                        )
                    if elements_to_analyze:
                        plot_combination_distribution_by_counts(combinations, elements_to_analyze,
                                                            elements_to_exclude, percentage_threshold)


                st.sidebar.title('Element Flow Analysis')
                perform_parallel = st.sidebar.checkbox("Perform Element Flow Analysis?")
                if perform_parallel:
                    try:

                        show_concentration = st.sidebar.checkbox("Show Particle Concentration (P/mL)")
                        

                        parallel_fig = create_parallel_sets(st.session_state.filtered_results, particles_per_ml, show_concentration)
                        st.plotly_chart(parallel_fig)
                        
                    except Exception as e:
                        st.error(f"Error creating element flow diagram: {str(e)}")


                st.sidebar.title('Clustering Analysis')
                perform_clustering = st.sidebar.checkbox('Perform Clustering Analysis?')
                if perform_clustering:
                    clustering_method = st.sidebar.selectbox(
                        'Select clustering method:',
                        ['K-Means', 'Hierarchical', 'Spectral', 'Gaussian', 'Mini-Batch K-Means', 'Mean Shift']
                    )
                    
                    st.sidebar.title('Analysis Options')
                    perform_clustering_number = st.sidebar.checkbox('Evaluate optimal number of clusters')
                    perform_heatmap = st.sidebar.checkbox('Generate clustering heatmap')
                    
                
                    data_type_for_clustering = st.sidebar.selectbox(
                        'Select data type for clustering:',
                        ['Mass', 'Mole', 'Count','Mole Percent', 'Diameter']
                    )
                    
                    if data_type_for_clustering == 'Mass':
                        data_for_clustering = st.session_state.filtered_results['mass']
                    elif data_type_for_clustering == 'Mole':
                        data_for_clustering = st.session_state.filtered_results['mole']
                    elif data_type_for_clustering == 'Count':
                        data_for_clustering = st.session_state.filtered_results['counts']
                    elif data_type_for_clustering == 'Mole Percent':
                        data_for_clustering = st.session_state.filtered_results['mole_percent']
                    elif data_type_for_clustering == 'Diameter':
                        data_for_clustering = st.session_state.filtered_results['diameter']
                        
                    if perform_clustering_number:
                        if clustering_method != 'Mean Shift':
                            max_clusters = st.sidebar.slider('Maximum number of clusters to evaluate:', 
                                                        min_value=5, max_value=100, value=20)
                            fig, optimal_clusters = evaluate_clusters(data_for_clustering, clustering_method, max_clusters)
                            st.plotly_chart(fig)
                            st.write("Suggested number of clusters based on different metrics:")
                            for metric, n_clusters in optimal_clusters.items():
                                st.write(f"- {metric}: {n_clusters}")
                    
                    if perform_heatmap:
                        num_clusters = st.sidebar.slider('Select number of clusters:', 
                                                    min_value=2, max_value=100, value=5)
                        
                        labels = apply_clustering(data_for_clustering, clustering_method, num_clusters)
                        if labels is not None:
                            colorbar_title = "Mass (fg)" if data_type_for_clustering == 'Mass' else \
                                            "Count" if data_type_for_clustering == 'Count' else \
                                            "Mole (fmol)" if data_type_for_clustering == 'Mole' else \
                                            "Mole %" if data_type_for_clustering == 'Mole Percent' else \
                                            "Diameter (nm)"
                            
                            use_log = st.sidebar.checkbox("Use square root scale for Mass data", 
                                                        value=True if data_type_for_clustering == 'Mass' else False)
                                                        
                            plot_heatmap_cluster(data_for_clustering, labels, 
                                            f"{clustering_method} Clustering Results for {data_type_for_clustering}",
                                            colorbar_title,
                                            use_log=use_log)
                                                                                                            
                                                                                            
                                                                                




def combine_replicate_files(files):
    """
    Combines multiple replicate files into a single file before processing.
    """
    if not files:
        return None
        
    try:
        first_file = files[0]
        if 'csv' in first_file.name.lower():
            content = first_file.getvalue().decode('utf-8').splitlines()
        else:
            return None

        event_start_idx = None
        for idx, line in enumerate(content):
            if 'event number' in line.lower():
                event_start_idx = idx
                break
                
        if event_start_idx is None:
            return None
            

        combined_lines = content[:event_start_idx + 1]
        last_event_num = 0
        for file in files:
            if 'csv' in file.name.lower():
                current_content = file.getvalue().decode('utf-8').splitlines()
            else:
                continue
                

            current_event_start = None
            for idx, line in enumerate(current_content):
                if 'event number' in line.lower():
                    current_event_start = idx + 1
                    break
                    
            if current_event_start is not None:
              
                events = current_content[current_event_start:]
                
                if file != files[0]:
                    for line in events:
                        if line.strip():
                            parts = line.split(',')
                            try:
                                event_num = float(parts[0])
                                parts[0] = str(event_num + last_event_num)
                                combined_lines.append(','.join(parts))
                            except (ValueError, IndexError):
                                continue
                else:
                    combined_lines.extend(events)
                    
              
                try:
                    last_line = [line for line in events if line.strip()][-1]
                    last_event_num = float(last_line.split(',')[0])
                except:
                    pass
                    

        combined_content = '\n'.join(combined_lines).encode('utf-8')
    
        file_obj = BytesIO(combined_content)
        class CombinedFile:
            def __init__(self, file_obj, name):
                self.file_obj = file_obj
                self.name = name
                
            def getvalue(self):
                return self.file_obj.getvalue()
                
            def read(self):
                return self.file_obj.read()
                
            def seek(self, *args, **kwargs):
                return self.file_obj.seek(*args, **kwargs)
        
        return CombinedFile(file_obj, "Combined_Files.csv")
        
    except Exception as e:
        st.error(f"Error combining files: {str(e)}")
        return None
    
def combine_isotrack_files(files):
    """
    Combines multiple Isotrack files without cleaning the data.
    """
    if not files:
        return None
        
    try:
        first_file = files[0]
        if 'csv' in first_file.name.lower():
            content = first_file.getvalue().decode('utf-8').splitlines()
        else:
            return None

        data_start_idx = None
        for idx, line in enumerate(content):
            if 'Particle ID' in line:
                data_start_idx = idx
                break
                
        if data_start_idx is None:
            return None

        
        combined_lines = content[:data_start_idx + 1]
        
        for file in files:
            if 'csv' in file.name.lower():
                current_content = file.getvalue().decode('utf-8').splitlines()
            else:
                continue

           
            current_start = None
            current_end = None
            for idx, line in enumerate(current_content):
                if 'Particle ID' in line:
                    current_start = idx + 1
                elif current_start is not None and (not line.strip() or all(cell.strip() == '' for cell in line.split(','))):
                    current_end = idx
                    break
                    
            if current_start is not None:
                if current_end is None:
                    current_end = len(current_content)
                data_lines = current_content[current_start:current_end]
                combined_lines.extend(data_lines)

     
        combined_lines.append('')
        
        combined_content = '\n'.join(combined_lines).encode('utf-8')
        file_obj = BytesIO(combined_content)
        
        class CombinedFile:
            def __init__(self, file_obj, name):
                self.file_obj = file_obj
                self.name = name
                
            def getvalue(self):
                return self.file_obj.getvalue()
                
            def read(self):
                return self.file_obj.read()
                
            def seek(self, *args, **kwargs):
                return self.file_obj.seek(*args, **kwargs)
        
        return CombinedFile(file_obj, "Combined_Isotrack_Files.csv")
        
    except Exception as e:
        st.error(f"Error combining files: {str(e)}")
        return None
    
if 'combined_files' not in st.session_state:
    st.session_state.combined_files = []
    
if 'processed_groups' not in st.session_state:
    st.session_state.processed_groups = set()

with tab2:
    data_type = st.radio(
        "Select data type:",
        ["NuQuant", "IsoTrack"],
        key='data_type_tab2'
    )

    combine_files = st.checkbox('Combine multiple files?', key='combine_files_tab2')
    
    new_files = st.file_uploader(
        ':file_folder: Upload files',
        type=['csv'],
        accept_multiple_files=True,
        key='multiple_files_tab2'
    )

    if new_files and combine_files and len(new_files) > 1:
     
        file_groups = {}
        for file in new_files:
            match = re.search(r'\d+', file.name)
            if match:
                base_num = match.group()
                if base_num not in file_groups:
                    file_groups[base_num] = []
                file_groups[base_num].append(file)

    
        for base_num, group_files in file_groups.items():
           
            group_key = f"{base_num}_{len(group_files)}"
            if group_key not in st.session_state.processed_groups and len(group_files) > 1:
                if data_type == "IsoTrack":
                    st.info(f"Combining files for group: {base_num}")
                    combined_file = combine_isotrack_files(group_files)
                    if combined_file:
                        combined_file.name = f"Combined_{base_num}_{len(st.session_state.combined_files) + 1}.csv"
                        st.session_state.combined_files.append(combined_file)
                        st.session_state.processed_groups.add(group_key)
                else:
                    for i in range(0, len(group_files), 3):
                        sub_group = group_files[i:i+3]
                        if len(sub_group) > 1:
                            st.info(f"Combining files for group: {base_num}")
                            combined_file = combine_replicate_files(sub_group)
                            if combined_file:
                                combined_file.name = f"Combined_{base_num}_{len(st.session_state.combined_files) + 1}.csv"
                                st.session_state.combined_files.append(combined_file)
                                st.session_state.processed_groups.add(group_key)

  
    uploaded_files = st.session_state.combined_files if combine_files else new_files


    if st.session_state.combined_files:
        st.write("Combined files:", [f.name for f in st.session_state.combined_files])

    if st.button("Clear all combined files"):
        st.session_state.combined_files = []
        st.session_state.processed_groups = set()
        st.rerun()
                

    def preprocess_csv_file(file):
        lines = file.getvalue().decode('utf-8').splitlines()
        max_fields = max([line.count(',') for line in lines]) + 1
        cleaned_lines = []

        for line in lines:
            fields = line.split(',')
            if len(fields) < max_fields:
                fields.extend([''] * (max_fields - len(fields)))
            cleaned_lines.append(','.join(fields))

        cleaned_file_content = "\n".join(cleaned_lines)
        return cleaned_file_content


    def count_rows_after_keyword_until_no_data(data, keyword, column_index=1):
        keyword_found = False
        count = 0
        for value in data.iloc[:, column_index]:
            if keyword_found:
                if pd.notna(value):
                    count += 1
                else:
                    break
            elif keyword.lower() in str(value).lower():
                keyword_found = True
        return count


    def extract_numeric_value_from_string(s):
        match = re.search(r"(\d+(\.\d+)?)", s)
        return float(match.group(1)) if match else None


    def find_value_at_keyword(data, keyword, column_index=1):
        for value in data.iloc[:, column_index]:
            if keyword.lower() in str(value).lower():
                return extract_numeric_value_from_string(str(value))
        return None


    def calculate_particles_per_ml(event_number, q_plasma, acquisition_time, dilution_factor):
        try:
            value = (float(event_number) * dilution_factor) / ((float(q_plasma) / 1000) * (float(acquisition_time)))
            return f"{value:.2e}"
        except ValueError:
            return None


    dilution_factors = {}
    acquisition_times = {}


    def find_start_index(df, keyword, column_index=0):
        for i, value in enumerate(df.iloc[:, column_index]):
            if keyword.lower() in str(value).lower():
                return i
        return None


    @st.cache_data
    def process_data(df, keyword='event number'):
        start_index = find_start_index(df, keyword)
        if start_index is not None:
            new_header = df.iloc[start_index]
            data = df.iloc[start_index + 1:].reset_index(drop=True)
            data.columns = [str(col) for col in new_header]

            mass_data_cols = [col for col in data.columns if
                            'mass' in col and 'total' not in col and not col.endswith('mass %')]
            mole_data_cols = [col for col in data.columns if
                            'mole' in col and 'total' not in col and not col.endswith('mole %')]
            count_cols = [col for col in data.columns if col.endswith('counts')]

            mass_data = data[mass_data_cols].rename(columns=lambda x: re.sub(r'\d+', '', x.split(' ')[0]).strip())
            mole_data = data[mole_data_cols].rename(columns=lambda x: re.sub(r'\d+', '', x.split(' ')[0]).strip())
            counts_data = data[count_cols].rename(columns=lambda x: re.sub(r'\d+', '', x.split(' ')[0]).strip())

            mass_data = mass_data.apply(pd.to_numeric, errors='coerce').fillna(0)
            counts_data = counts_data.apply(pd.to_numeric, errors='coerce').fillna(0)
            mole_data = mole_data.apply(pd.to_numeric, errors='coerce').fillna(0)

            mass_data = mass_data.loc[~(mass_data == 0).all(axis=1)]
            mole_data = mole_data.loc[~(mole_data == 0).all(axis=1)]
            counts_data = counts_data.loc[~(counts_data == 0).all(axis=1)]

            mass_percent_data = mass_data.div(mass_data.sum(axis=1), axis=0)
            mass_percent_data = mass_percent_data.rename(columns=lambda x: re.sub(r'\d+', '', x.split(' ')[0]).strip())

            mole_percent_data = mole_data.div(mole_data.sum(axis=1), axis=0)
            mole_percent_data = mole_percent_data.rename(columns=lambda x: re.sub(r'\d+', '', x.split(' ')[0]).strip())

            return mass_data, mass_percent_data, mole_data, mole_percent_data, counts_data
        else:
            return None, None, None, None, None
        
    def process_isotrack(df):
        """Process IsoTrack format data with specific structure for counts, mass, moles, and diameter"""
        try:
           
            transport_rate = None
            for idx, row in df.iterrows():
                if 'Transport Rate:' in str(row.iloc[0]):
                    try:
                        match = re.search(r'(\d+\.?\d*)', str(row.iloc[0]))
                        if match:
                            transport_rate = float(match.group(1))
                    except (ValueError, AttributeError):
                        continue
                    break
      
            start_idx = None
            for idx, row in df.iterrows():
                if 'Particle ID' in str(row.iloc[0]):
                    start_idx = idx
                    break
                    
            if start_idx is None:
                st.write("Could not find 'Particle ID' in data")
                return None, None, None, None, None
            
          
            def find_end_of_data(df, start_idx):
                data_section = df.iloc[start_idx + 1:]
                for idx, row in data_section.iterrows():
                    if row.isna().all() or row.astype(str).str.strip().eq('').all():
                        return idx
                return len(df)
            
            end_idx = find_end_of_data(df, start_idx)
                    
        
            data = df.iloc[start_idx:end_idx].copy()
            data.columns = [str(x).strip() for x in data.iloc[0]]
            data = data.iloc[1:].reset_index(drop=True)
                
           
            counts_cols = [col for col in data.columns if 'counts' in col]
            fg_cols = [col for col in data.columns if '(fg)' in col and 'Total' not in col and not col.endswith('Mass%')]
            fmol_cols = [col for col in data.columns if '(fmol)' in col and 'Total' not in col and not col.endswith('Mole%')]
            nm_cols = [col for col in data.columns if '(nm)' in col]
            
            
            counts_data = data[counts_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            mass_data = data[fg_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            mole_data = data[fmol_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            diameter_data = data[nm_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            def clean_element_name(col):
               
                col = col.split('(')[0].strip()
                return ''.join(c for c in col if not c.isdigit()).strip()
            
          
            counts_data.columns = [clean_element_name(col) for col in counts_data.columns]
            mass_data.columns = [clean_element_name(col) for col in mass_data.columns]
            mole_data.columns = [clean_element_name(col) for col in mole_data.columns]
            diameter_data.columns = [clean_element_name(col) for col in diameter_data.columns]

       
            mole_percent_data = mole_data.div(mole_data.sum(axis=1), axis=0) * 100
            
            return {
                'counts': counts_data,
                'mass': mass_data,
                'mole': mole_data,
                'mole_percent': mole_percent_data,
                'diameter': diameter_data,
                'transport_rate': transport_rate,
                'particle_count': len(data)
            }
                
        except Exception as e:
            st.error(f'Error processing IsoTrack data: {str(e)}')
            return None


    @st.cache_data
    def clean_data(df, keyword='event number'):
        start_index = find_start_index(df, keyword)

        if start_index is not None:
            new_header = df.iloc[start_index]
            data = df.iloc[start_index + 1:].reset_index(drop=True)
            data.columns = [str(col) for col in new_header]

            def make_unique(column_names):
                counts = {}
                for i, col in enumerate(column_names):
                    if col in counts:
                        counts[col] += 1
                        column_names[i] = f"{col}_{counts[col]}"
                    else:
                        counts[col] = 0
                return column_names

            data.columns = make_unique(data.columns.tolist())

            elements = set(col.split(' ')[0] for col in data.columns if 'fwhm' in col)

            for element in elements:
                count_col = f'{element} counts'
                mole_col = f'{element} moles [fmol]'
                mass_col = f'{element} mass [fg]'
                fwhm_col = f'{element} fwhm'
                mole_per = f'{element} mole %'
                mass_per = f'{element} mass %'

                if all(col in data.columns for col in [count_col, mole_col, mass_col, fwhm_col, mole_per, mass_per]):
                    data.loc[data[fwhm_col].isna(), [count_col, mole_col, mass_col, mass_per, mole_per]] = 0

            cleaned_df = df.copy()
            cleaned_df.iloc[start_index + 1:, :] = data.values

            return cleaned_df
        else:
            st.error("Header row with 'fwhm' not found.")
            return None


    def display_summary_table(data_dict, file_letter_map, dilution_factors, acquisition_times):
        """
        Displays a summary table of the processed data and provides a download option.
        """
        summary_data = {
            "Filename": [],
            "Custom Name": [],
            "Total Particles Count": [],
            "Calibrated Transport Rate (µL/s)": [],
            "Dilution Factor": [],
            "Acquisition Time (s)": [],
            "Particles per mL": []
        }

        for filename, data in data_dict.items():
            df = data['df']
            event_number_cell = data['event_number_cell']
            transport_rate = data['transport_rate']
            particles_per_ml = data['particles_per_ml']

            summary_data["Filename"].append(filename)
            summary_data["Custom Name"].append(file_letter_map[filename])
            summary_data["Total Particles Count"].append(event_number_cell if event_number_cell is not None else 'N/A')
            summary_data["Calibrated Transport Rate (µL/s)"].append(transport_rate if transport_rate is not None else 'N/A')
            summary_data["Dilution Factor"].append(dilution_factors.get(filename, 'N/A'))
            summary_data["Acquisition Time (s)"].append(acquisition_times.get(filename, 'N/A'))
            summary_data["Particles per mL"].append(particles_per_ml if particles_per_ml is not None else 'N/A')

        summary_df = pd.DataFrame(summary_data)
        csv = summary_df.to_csv(index=False)

        st.table(summary_df)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='infofiles.csv',
            mime='text/csv'
        )


    if 'processed_data_dict' not in st.session_state:
        st.session_state.processed_data_dict = {}
    
    if 'file_letter_map' not in st.session_state:
        st.session_state.file_letter_map = {}
        
    if 'dilution_factors' not in st.session_state:
        st.session_state.dilution_factors = {}
        
    if 'acquisition_times' not in st.session_state:
        st.session_state.acquisition_times = {}

    if uploaded_files:

        files_to_process = []
        file_info = []
        
        for i, file in enumerate(uploaded_files):
            filename = file.name
            if filename not in st.session_state.file_letter_map:

                default_name = os.path.splitext(filename)[0]
                st.session_state.file_letter_map[filename] = default_name
            
            file_info.append({
                'filename': filename,
                'custom_name': st.session_state.file_letter_map[filename],
                'dilution_factor': st.session_state.dilution_factors.get(filename, 1.0),
                'acquisition_time': st.session_state.acquisition_times.get(filename, 60.0)
            })
            
            if filename not in st.session_state.processed_data_dict:
                files_to_process.append(file)

        st.write('Enter custom names, dilution factors, and acquisition times for each file:')
        file_info_df = pd.DataFrame(file_info)
        updated_file_info_df = st.data_editor(file_info_df, key="file_info_editor")


        for i, row in updated_file_info_df.iterrows():
            filename = row['filename']
            st.session_state.file_letter_map[filename] = row['custom_name']
            st.session_state.dilution_factors[filename] = row['dilution_factor']
            st.session_state.acquisition_times[filename] = row['acquisition_time']


        if updated_file_info_df.to_dict() != file_info_df.to_dict():
            st.rerun()


        if files_to_process:
            st.info(f"Processing {len(files_to_process)} new files...")
            
            for file in files_to_process:
                filename = file.name
                try:
                    if 'csv' in filename:
                        cleaned_file_content = preprocess_csv_file(file)
                        df = pd.read_csv(StringIO(cleaned_file_content))
                    else:
                        st.error('File format not supported. Please upload a CSV file.')
                        continue

                    if df is not None:
                        if data_type == "IsoTrack":
                            try:
                                processed_data = process_isotrack(df)
                                if processed_data is not None:
                                    mass_data = processed_data['mass']
                                    mole_data = processed_data['mole'] 
                                    mole_percent_data = processed_data['mole_percent']
                                    counts_data = processed_data['counts']
                                    transport_rate = processed_data['transport_rate']
                                    event_number_cell = processed_data['particle_count']
                                    
                                    particles_per_ml = calculate_particles_per_ml(
                                        event_number_cell,
                                        transport_rate,
                                        st.session_state.acquisition_times[filename],
                                        st.session_state.dilution_factors[filename]
                                    )
                                    
                                    st.session_state.processed_data_dict[filename] = {
                                        'df': df,
                                        'event_number_cell': event_number_cell,
                                        'transport_rate': transport_rate,
                                        'particles_per_ml': particles_per_ml,
                                        'mass_data': mass_data,
                                        'mole_data': mole_data,
                                        'mass_percent_data': None,  
                                        'mole_percent_data': mole_percent_data,
                                        'counts_data': counts_data
                                    }
                            except Exception as e:
                                st.error(f"Error processing IsoTrack file {filename}: {str(e)}")
                                continue
                        else:
                            try:
                                df = clean_data(df)
                                if df is not None:
                                    event_number_cell = count_rows_after_keyword_until_no_data(df, 'event number', column_index=0)
                                    transport_rate_cell = find_value_at_keyword(df, 'calibrated transport rate', column_index=1)
                                    transport_rate = extract_numeric_value_from_string(str(transport_rate_cell))
                                    
                                    particles_per_ml = calculate_particles_per_ml(
                                        event_number_cell, 
                                        transport_rate, 
                                        st.session_state.acquisition_times[filename],
                                        st.session_state.dilution_factors[filename]
                                    )

                                    mass_data, mass_percent_data, mole_data, mole_percent_data, counts_data = process_data(df)
                                    
                                    st.session_state.processed_data_dict[filename] = {
                                        'df': df,
                                        'event_number_cell': event_number_cell,
                                        'transport_rate': transport_rate,
                                        'particles_per_ml': particles_per_ml,
                                        'mass_data': mass_data,
                                        'mole_data': mole_data,
                                        'mass_percent_data': mass_percent_data,
                                        'mole_percent_data': mole_percent_data,
                                        'counts_data': counts_data
                                    }
                            except Exception as e:
                                st.error(f'An error occurred with file {filename}: {str(e)}')
                                st.write("File content:", df.head())  
                                continue
                except Exception as e:
                    st.error(f"An error occurred with file {filename}: {e}")
                    continue

        for filename in st.session_state.processed_data_dict:
            data = st.session_state.processed_data_dict[filename]
            particles_per_ml = calculate_particles_per_ml(
                data['event_number_cell'],
                data['transport_rate'],
                st.session_state.acquisition_times[filename],
                st.session_state.dilution_factors[filename]
            )
            st.session_state.processed_data_dict[filename]['particles_per_ml'] = particles_per_ml

      
        data_dict = st.session_state.processed_data_dict
        file_letter_map = st.session_state.file_letter_map

        if st.button("Clear all processed data"):
            st.session_state.processed_data_dict = {}
            st.session_state.file_letter_map = {}
            st.session_state.dilution_factors = {}
            st.session_state.acquisition_times = {}
            st.session_state.combined_files = []
            st.session_state.processed_groups = set()
            st.rerun()


    def plot_mass_distribution(df_dict, element, detection_type, bin_size, x_max, title, file_letter_map, plot_type):
        if plot_type == "Histogram":
            
            if isinstance(element, list):
                element = element[0]  
            plot_histogram_for_element(df_dict, element, detection_type, bin_size, x_max, title, file_letter_map)
        else:  
            
            plot_boxplot_for_elements(df_dict, element, detection_type, x_max, title, file_letter_map)

    def plot_boxplot_for_elements(df_dict, elements, detection_type, x_max, title, file_letter_map):
        filenames = list(df_dict.keys())
        
      
        use_log_scale = st.sidebar.checkbox('Use Logarithmic Scale for Box Plot', value=True, key='box_plot_log_scale')
        
        
        show_mean = st.sidebar.checkbox('Show Mean Line', value=True, key='box_plot_show_mean')
        show_points = st.sidebar.checkbox('Show Individual Points', value=True, key='box_plot_show_points')
        show_notch = st.sidebar.checkbox('Show Notched Box Plots', value=False, key='box_plot_show_notch')
        
        
        box_width = st.sidebar.slider('Box Width (All)', 
                            min_value=0.1, 
                            max_value=0.9, 
                            value=0.5, 
                            step=0.05, 
                            key='box_width_global_control')
        
    
        if not isinstance(elements, list):
            elements = [elements]
        
      
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        all_data = []
        labels = []
        
   
        if len(elements) == 1:
            elements_per_row = 1  
            num_rows = 1
            fig_width = 800 
        elif len(elements) == 2:
            elements_per_row = 2 
            num_rows = 1
            fig_width = 1200
        else:
            elements_per_row = 2  
            num_rows = (len(elements) + elements_per_row - 1) // elements_per_row  
            fig_width = 1200
        
   
        fig = make_subplots(
            rows=num_rows, 
            cols=elements_per_row,
            subplot_titles=elements,  
            horizontal_spacing=0.10,  
            vertical_spacing=0.15      
        )
        
       
        stats_data = []
        
  
        for elem_idx, element in enumerate(elements):
           
            row_idx = elem_idx // elements_per_row + 1
            col_idx = elem_idx % elements_per_row + 1
            
            element_data = {}
            for file_idx, filename in enumerate(filenames):
                data = df_dict[filename]
                mass_data = data['mass_data']
                
                if mass_data is None or element not in mass_data.columns:
                    st.warning(f"Element {element} not found in {filename}")
                    continue
                    
                mass_data = mass_data.dropna(subset=[element])
                mass_data[element] = mass_data[element][mass_data[element] > 0]

                if detection_type == 'Single':
                    mass_data = mass_data.loc[mass_data.apply(lambda row: (row > 0).sum(), axis=1) == 1]
                elif detection_type == 'Multiple':
                    mass_data = mass_data.loc[mass_data.apply(lambda row: (row > 0).sum(), axis=1) > 1]

                color = st.sidebar.color_picker(f'Color for {file_letter_map[filename]}',
                                        default_colors[file_idx % len(default_colors)], 
                                        key=f"BoxMass_{filename}_{element}")
                
                
                y_data = mass_data[element]
                if use_log_scale and len(y_data) > 0:
                    y_data = y_data[y_data > 0]  
                
              
                if len(y_data) > 0:
                    stats_data.append({
                        'Element': element,
                        'File': file_letter_map[filename],
                        'Mean': y_data.mean(),
                        'Median': y_data.median(),
                        'StdDev': y_data.std(),
                        'Min': y_data.min(),
                        'Max': y_data.max(),
                        'Count': len(y_data),
                        '25th Percentile': y_data.quantile(0.25),
                        '75th Percentile': y_data.quantile(0.75)
                    })
                
               
                fig.add_trace(
                    go.Box(
                        y=y_data,
                        name=f"{file_letter_map[filename]}",
                        marker_color=color,
                        boxmean=show_mean,
                        boxpoints='all' if show_points else False,
                        jitter=1,
                        pointpos=-1.8,
                        notched=show_notch,
                        marker=dict(
                            opacity=0.6,
                            size=5,
                            color=color,
                        ),
                        line=dict(width=2),
                        showlegend=elem_idx == 0,
                        legendgroup=file_letter_map[filename],
                        whiskerwidth=1,
                        width=box_width,  # Using the global box width here
                    ),
                    row=row_idx,
                    col=col_idx
                )
                
                
                all_data.append(y_data.tolist())
                labels.append(f"{file_letter_map[filename]} - {element}")
        
      
        for i in range(1, num_rows + 1):
            for j in range(1, elements_per_row + 1):
               
                subplot_idx = (i-1) * elements_per_row + (j-1)
                if subplot_idx >= len(elements):
                    continue
                    
               
                if use_log_scale:
                   
                    major_ticks = [1, 10, 100, 1000, 10000]
                    minor_ticks = [2, 3, 4, 5, 6, 7, 8, 9, 
                                20, 30, 40, 50, 60, 70, 80, 90,
                                200, 300, 400, 500, 600, 700, 800, 900,
                                2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
                    
                   
                    all_ticks = major_ticks + minor_ticks
                    all_ticks.sort()
                    
                   
                    tick_text = []
                    for val in all_ticks:
                        if val in major_ticks:
                           
                            tick_text.append(f'<span style="font-size:60px">{val}</span>')
                        else:
                        
                            tick_text.append(f'<span style="font-size:10px">{val}</span>')
                    
                    fig.update_yaxes(
                        type='log',
                        title="Mass (fg)" if j == 1 else None,
                        title_font=dict(size=60, family='Times New Roman', color='black'),
                        tickvals=all_ticks,
                        ticktext=tick_text,
                        tickfont=dict(family='Times New Roman', color='black'),
                        gridcolor='rgba(211, 211, 211, 0.3)',
                        gridwidth=1,
                        row=i, col=j
                    )
                else:
                    fig.update_yaxes(
                        range=[0, x_max],
                        title="Mass (fg)" if j == 1 else None,  
                        title_font=dict(size=50, family='Times New Roman', color='black'),
                        tickfont=dict(size=50, family='Times New Roman', color='black'),
                        gridcolor='rgba(211, 211, 211, 0.3)',  
                        gridwidth=1,
                        row=i, col=j
                    )
                
               
                fig.update_xaxes(
                    showticklabels=False,
                    showline=True,
                    linecolor='black',
                    linewidth=1.5,
                    row=i, col=j
                )
        
       
        for i, elem in enumerate(elements):
            if i < len(fig.layout.annotations):
                fig.layout.annotations[i].font.size = 50
                fig.layout.annotations[i].font.color = 'black'
                fig.layout.annotations[i].font.family = 'Times New Roman'
                fig.layout.annotations[i].font.weight = 'bold'
                
        
        
       
        fig.update_layout(
            title=f" ",
            title_font=dict(size=40, family='Times New Roman', color='black', weight='bold'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=450 * num_rows,  
            width=fig_width,  
            margin=dict(l=130, r=50, t=80, b=50),  
            boxmode='group',
            boxgap=0.3,  
            boxgroupgap=0.3,  
            legend=dict(
                font=dict(
                    size=50,
                    color='black',
                    family='Times New Roman'
                ),
                orientation="h",  
                yanchor="bottom",
                y=1.05,  
                xanchor="center",
                x=0.5,
                bgcolor='rgba(255, 255, 255, 0.8)',  
                bordercolor='black',
                borderwidth=1,
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        
        if stats_data:
            st.subheader(f"Statistical Summary for {', '.join(elements)}")
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df.style.format({
                'Mean': '{:.2f}',
                'Median': '{:.2f}',
                'StdDev': '{:.2f}',
                'Min': '{:.2f}',
                'Max': '{:.2f}',
                '25th Percentile': '{:.2f}',
                '75th Percentile': '{:.2f}'
            }))
            
           
            csv_stats = stats_df.to_csv(index=False)
            st.download_button(
                label="Download statistical summary as CSV",
                data=csv_stats,
                file_name='element_statistics.csv',
                mime='text/csv',
                key='stats_download'
            )
        
       
        if st.sidebar.button('Download Figure as PNG', key='box_download_png'):
            img_bytes = fig.to_image(
                format="png",
                width=2000 if len(elements) == 1 else 3000,  
                height=1000 * num_rows,
                scale=3
            )
            
            st.sidebar.download_button(
                label="Click to Download PNG",
                data=img_bytes,
                file_name=f"box_{'_'.join(elements)}.png",
                mime="image/png",
                key=f"png_download_{'_'.join(elements)}"
            )
        
   
        if all_data:
            max_len = max(len(data_list) for data_list in all_data)
            export_data = {}
            for i, label in enumerate(labels):
                padded_data = all_data[i] + [None] * (max_len - len(all_data[i]))
                export_data[label] = padded_data
                
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="Download boxplot data as CSV",
                data=csv,
                file_name=f'boxplot_{"_".join(elements)}_data.csv',
                mime='text/csv',
                key='boxplot_download'
            )
    
    
    
    def plot_histogram_for_element(df_dict, element, detection_type, bin_size, x_max, title, file_letter_map):
        def format_power_notation(v):
            if abs(v) < 1e-10:
                return '0'
            
            exponent = int(np.floor(np.log10(abs(v))))
            mantissa = v / (10**exponent)
            return f'{mantissa:.1f}×10<sup>{exponent}</sup>'

        filenames = list(df_dict.keys())
        num_files = len(filenames)
        plots_per_figure = st.sidebar.checkbox('Show two plots per figure', value=True, key='plots_per_figure')
        
        if plots_per_figure:
            num_subplots = (num_files + 1) // 2
        else:
            num_subplots = num_files

        fig = make_subplots(rows=num_subplots, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        
        subplot_y_values = [[] for _ in range(num_subplots)]
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        bin_edges = np.arange(0, x_max + bin_size, bin_size)
        all_summary_data = []

        display_mode = st.sidebar.radio(
            "Display mode:",
            ["Particles/mL", "Percentage (%)", "Particle Count"],
            key='display_mode_control'
        )

        for i, filename in enumerate(filenames):
            if plots_per_figure:
                subplot_index = i // 2 + 1
            else:
                subplot_index = i + 1
                
            data = df_dict[filename]
            particles_per_ml = data['particles_per_ml']
            mass_data = data['mass_data']  
                    
            if mass_data is None or element not in mass_data.columns:
                st.warning(f"Element {element} not found in {filename}")
                continue
                
            mass_data = mass_data.dropna(subset=[element])
            mass_data[element] = mass_data[element][mass_data[element] > 0]

            if detection_type == 'Single':
                mass_data = mass_data.loc[mass_data.apply(lambda row: (row > 0).sum(), axis=1) == 1]
            elif detection_type == 'Multiple':
                mass_data = mass_data.loc[mass_data.apply(lambda row: (row > 0).sum(), axis=1) > 1]

            if particles_per_ml is None and display_mode == "Particles/mL":
                st.warning(f"Particles per mL not available for file {filename}. Skipping.")
                continue

            particles_per_ml_value = float(particles_per_ml.replace('e', 'E')) if particles_per_ml else 0
            
            color = st.sidebar.color_picker(f'Pick a color for {file_letter_map[filename]}',
                                        default_colors[i % len(default_colors)], 
                                        key=f"Mass_{filename}")

            hist_data = mass_data[element]
            hist_data = hist_data[hist_data > 0]

            total_particles = len(hist_data)

            hist_counts, _ = np.histogram(hist_data, bins=bin_edges)

            if display_mode == "Percentage (%)":
                hist_density = [(count / total_particles) * 100 for count in hist_counts] if total_particles > 0 else [0] * len(hist_counts)
            elif display_mode == "Particle Count":
                hist_density = hist_counts  # Use raw counts without normalization
            else:  # Particles/mL
                hist_density = [(count / total_particles) * particles_per_ml_value for count in hist_counts] if total_particles > 0 else [0] * len(hist_counts)
                
            subplot_y_values[subplot_index-1].extend(hist_density)
            

            fig.add_trace(
                go.Bar(
                    x=bin_edges[:-1] + bin_size / 2,
                    y=hist_density,
                    name=f'{file_letter_map[filename]}',
                    marker=dict(
                        color=color,
                        line=dict(color='black', width=2)
                    ),
                    width=bin_size,
                    showlegend=False,
            
                ),
                row=subplot_index, col=1
            )
            y_max_local = np.max(hist_density) if len(hist_density) > 0 else 1

            
            fig.add_annotation(
                x=x_max * 0.9,
                y=y_max_local * 0.9,
                xref=f"x{subplot_index}" if subplot_index > 1 else "x",
                yref=f"y{subplot_index}" if subplot_index > 1 else "y",
                text=f"{total_particles}",
                showarrow=False,
                font=dict(size=80, color="black", weight='bold'),
                bgcolor="rgba(255,255,255,0)",
                align="right",
                row=subplot_index, col=1
            )
            
            


            file_summary_data = []
            file_summary_data.append(['File name: ' + filename, '', ''])
            file_summary_data.append(['Mass (fg)', display_mode, ''])
            for edge, density in zip(bin_edges[:-1], hist_density):
                file_summary_data.append([edge, density, ''])
            all_summary_data.append(file_summary_data)
            

        middle_subplot = (num_subplots + 1) // 2
        for i in range(1, num_subplots + 1):
            y_vals = subplot_y_values[i-1]
            if y_vals:
                y_max = max(y_vals)
                tick_vals = np.linspace(0, y_max, 6)
                

                if display_mode == "Percentage (%)":
                    y_axis_title = "Frequency (%)"
                elif display_mode == "Particle Count":
                    y_axis_title = "Particle Count"
                else:
                    y_axis_title = "Frequency"
                

                if display_mode == "Percentage (%)":
                    tick_labels = [f'{v:.1f}' for v in tick_vals]
                elif display_mode == "Particle Count":
                    tick_labels = [f'{int(v)}' if v.is_integer() else f'{v:.0f}' for v in tick_vals]
                else:  # Particles/mL
                    tick_labels = [f'{v:.1f}' for v in tick_vals]
                
                fig.update_yaxes(

                    ticktext=tick_labels,
                    tickvals=tick_vals,
                    title_font=dict(size=80, color='black', weight='bold'),
                    tickfont=dict(size=80, color='black', weight='bold'),
                    linecolor='black',
                    linewidth=2,
                    row=i, col=1,
                    showgrid=False
                )
            
            fig.update_xaxes(
                range=[0, x_max],
                linecolor='black',
                linewidth=2,
                row=i, col=1
            )

        fig.update_xaxes(
            range=[0, x_max],
            title_text="Mass (fg)",
            title_font=dict(size=80, color='black', weight='bold'),
            tickfont=dict(size=80, color='black', weight='bold'),
            linecolor='black',
            linewidth=2,
            row=num_subplots, col=1,
            showgrid=False
        )
        

        fig.update_layout(
            title=f"{title}: {element}",
            barmode='overlay',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=80, color="black", weight='bold'),



            height=300 * num_subplots,
            margin=dict(l=400, r=50, t=200, b=200)
        )

        st.plotly_chart(fig, use_container_width=True)

        if st.sidebar.button('Download Figure as PNG'):
            img_bytes = fig.to_image(
                format="png",
                width=1200 * 4,
                height=(200 * num_subplots) * 4,
                scale=3
            )
            
            st.sidebar.download_button(
                label="Click to Download PNG",
                data=img_bytes,
                file_name=f"histogram_{element}_{title}.png",
                mime="image/png",
                key=f"png_download_{element}"
            )

        if all_summary_data:
            max_rows = max(len(data) for data in all_summary_data)
            for file_data in all_summary_data:
                while len(file_data) < max_rows:
                    file_data.append(['', '', ''])

            merged_summary_data = []
            for row_idx in range(max_rows):
                merged_row = []
                for file_data in all_summary_data:
                    merged_row.extend(file_data[row_idx])
                    merged_row.append('')
                merged_summary_data.append(merged_row)

            summary_df = pd.DataFrame(merged_summary_data)
            csv = summary_df.to_csv(index=False, header=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='histogram_data.csv',
                mime='text/csv',
                key='multipmass'
            )
                
                
    def analyze_single_multiple_element(data_dict, file_letter_map):
        """
        Analyzes and visualizes single vs multiple element particles in the provided data.
        Creates pie charts or heatmaps showing the distribution of single and multiple element particles.
        """
        
        percentage_threshold_Multiple = st.sidebar.slider('Multiple Elements Threshold (%)', 
                                                    min_value=0.0, 
                                                    max_value=10.0, 
                                                    value=0.5, 
                                                    step=0.1, 
                                                    key='multi_percent_thresh_tab2')

        percentage_threshold_Single = st.sidebar.slider('Single Elements Threshold (%)', 
                                                    min_value=0.0, 
                                                    max_value=10.0, 
                                                    value=0.5, 
                                                    step=0.1, 
                                                    key='single_percent_thresh_tab2')
        

        visualization_type = st.sidebar.radio(
            "Visualization type:",
            ["Pie Charts", "Heatmaps"],
            key='single_multiple_visualization_type'
        )
        
        charts_container = st.container()
        
        with charts_container:

            all_files_data = {}
            
            for filename, data in data_dict.items():
                combinations, _, _, _, _, _ = get_combinations_and_related_data(data_dict, filename)
                
                if combinations:
                    sorted_combinations = sorted(combinations.items(), key=lambda item: item[1]['counts'], reverse=True)
                    total_counts_all = sum(details['counts'] for _, details in sorted_combinations)
                    
                    filtered_combinations_Multiple = []
                    filtered_combinations_Single = []
                    
                    for combination, details in sorted_combinations:
                        percentage = (details['counts'] / total_counts_all) * 100
                        
                        if len(combination.split(', ')) == 1:
                            if percentage >= percentage_threshold_Single:
                                filtered_combinations_Single.append((combination, details, percentage))
                        else:
                            if percentage >= percentage_threshold_Multiple:
                                filtered_combinations_Multiple.append((combination, details, percentage))
                    

                    all_files_data[filename] = {
                        'single': filtered_combinations_Single,
                        'multiple': filtered_combinations_Multiple,
                        'total_counts': total_counts_all
                    }
            

            if visualization_type == "Pie Charts":

                for filename, file_data in all_files_data.items():
                    st.markdown(f"## {file_letter_map[filename]}", unsafe_allow_html=True)
                    
                    single_element_combinations = file_data['single']
                    multiple_element_combinations = file_data['multiple']
                    total_counts_all = file_data['total_counts']

                    col1, col2 = st.columns(2)
                    
                    summary_data_single = [['Combination', 'Count', 'Percentage']]
                    
                    single_percentage_sum = sum(percentage for _, _, percentage in single_element_combinations)
                    multiple_percentage_sum = sum(percentage for _, _, percentage in multiple_element_combinations)

                    others_single_percent = max(0, 100 - single_percentage_sum)
                    others_single_count = int((others_single_percent * total_counts_all) / 100) if others_single_percent > 0 else 0

                    others_multiple_percent = max(0, 100 - multiple_percentage_sum)
                    others_multiple_count = int((others_multiple_percent * total_counts_all) / 100) if others_multiple_percent > 0 else 0


                    if single_element_combinations or others_single_percent > 0:
                        labels = [f"{combination} ({details['counts']})" for combination, details, percentage in single_element_combinations]
                        values = [details['counts'] for _, details, _ in single_element_combinations]

                        if others_single_percent > 0:
                            labels.append(f"Others ({others_single_count})")
                            values.append(others_single_count)
                            summary_data_single.append(['Others', others_single_count, f"{others_single_percent:.2f}%"])

                        colors = px.colors.sequential.RdBu[:len(labels)]
                        if others_single_percent > 0:
                            colors[-1] = '#777777'
                        
                        fig_single = px.pie(
                            values=values, 
                            names=labels, 
                            title=f" ",
                            color_discrete_sequence=colors 
                        )

                        fig_single.update_traces(
                            textinfo='percent+label',
                            textfont=dict(size=25, color='black'),
                            hoverinfo='label+percent+value',
                            textposition='inside',
                            pull=[0.05] * len(values),
                            marker=dict(line=dict(color='#000000', width=2))
                        )
                        
                        fig_single.update_layout(
                            title=dict(
                                text=f" ",
                                font=dict(size=40, family='Arial', color='black')
                            ),
                            legend=dict(
                                font=dict(size=30, color='black')
                            ),
                            font=dict(size=25, color='black'),
                            height=800,
                            margin=dict(t=100, b=50)
                        )
                        
                        with col1:
                            st.plotly_chart(fig_single, use_container_width=True)
                            st.markdown(f"<h3 style='font-size:25px;'>Total particle count: {total_counts_all}</h3>", unsafe_allow_html=True)
                        
                        for combination, details, percentage in single_element_combinations:
                            summary_data_single.append([combination, details['counts'], f"{percentage:.2f}%"])
                    else:
                        with col1:
                            st.markdown(f"<h3 style='font-size:30px;'>No single elements above {percentage_threshold_Single}% threshold</h3>", unsafe_allow_html=True)
                    
                    summary_data_multiple = [['Combination', 'Count', 'Percentage']]
                    

                    if multiple_element_combinations or others_multiple_percent > 0:
                        labels = [f"{combination} ({details['counts']})" for combination, details, percentage in multiple_element_combinations]
                        values = [details['counts'] for _, details, _ in multiple_element_combinations]
                        
                        if others_multiple_percent > 0:
                            labels.append(f"Others ({others_multiple_count})")
                            values.append(others_multiple_count)
                            summary_data_multiple.append(['Others', others_multiple_count, f"{others_multiple_percent:.2f}%"])
                        
                        colors = px.colors.sequential.RdBu[:len(labels)]
                        if others_multiple_percent > 0:
                            colors[-1] = '#777777'  
                        
                        fig_multiple = px.pie(
                            values=values, 
                            names=labels, 
                            title=f" ",
                            color_discrete_sequence=colors 
                        )
                        
                        fig_multiple.update_traces(
                            textinfo='percent+label',
                            textfont=dict(size=25, color='black'),
                            hoverinfo='label+percent+value',
                            textposition='inside',
                            pull=[0.05] * len(values),
                            marker=dict(line=dict(color='#000000', width=2))
                        )
                        
                        fig_multiple.update_layout(
                            title=dict(
                                text=f" ",
                                font=dict(size=40, family='Arial', color='black')
                            ),
                            legend=dict(
                                font=dict(size=30, color='black')
                            ),
                            font=dict(size=25, color='black'),
                            height=800,
                            margin=dict(t=100, b=50)
                        )
                        
                        with col2:
                            st.plotly_chart(fig_multiple, use_container_width=True)
                            st.markdown(f"<h3 style='font-size:25px;'>Total particle count: {total_counts_all}</h3>", unsafe_allow_html=True)
                        
                        for combination, details, percentage in multiple_element_combinations:
                            summary_data_multiple.append([combination, details['counts'], f"{percentage:.2f}%"])
                    else:
                        with col2:
                            st.markdown(f"<h3 style='font-size:30px;'>No multiple elements above {percentage_threshold_Multiple}% threshold</h3>", unsafe_allow_html=True)
                    

                    csv_container = st.container()
                    with csv_container:
                        col1, col2 = st.columns(2)
                        
                        if summary_data_single:
                            summary_df_single = pd.DataFrame(summary_data_single)
                            csv_single = summary_df_single.to_csv(index=False, header=False)
                            with col1:
                                st.download_button(
                                    label=f"Download {file_letter_map[filename]} Single Element Data",
                                    data=csv_single,
                                    file_name=f'{file_letter_map[filename]}_single_element_data.csv',
                                    mime='text/csv',
                                    key=f'single_{filename}'
                                )
                        
                        if summary_data_multiple:
                            summary_df_multiple = pd.DataFrame(summary_data_multiple)
                            csv_multiple = summary_df_multiple.to_csv(index=False, header=False)
                            with col2:
                                st.download_button(
                                    label=f"Download {file_letter_map[filename]} Multiple Element Data",
                                    data=csv_multiple,
                                    file_name=f'{file_letter_map[filename]}_multiple_element_data.csv',
                                    mime='text/csv',
                                    key=f'multiple_{filename}'
                                )
                    
                    download_container = st.container()
                    with download_container:
                        col1, col2 = st.columns(2)
                        
                        if single_element_combinations or others_single_percent > 0:
                            with col1:
                                if st.button(f"Download {file_letter_map[filename]} Single Element Chart", key=f"dl_single_{filename}"):
                                    img_bytes = fig_single.to_image(format="png", width=1200, height=900, scale=2)
                                    st.download_button(
                                        label="Click to Download PNG",
                                        data=img_bytes,
                                        file_name=f"{file_letter_map[filename]}_single_elements.png",
                                        mime="image/png",
                                        key=f"png_single_{filename}"
                                    )
                        
                        if multiple_element_combinations or others_multiple_percent > 0:
                            with col2:
                                if st.button(f"Download {file_letter_map[filename]} Multiple Element Chart", key=f"dl_multiple_{filename}"):
                                    img_bytes = fig_multiple.to_image(format="png", width=1200, height=900, scale=2)
                                    st.download_button(
                                        label="Click to Download PNG",
                                        data=img_bytes,
                                        file_name=f"{file_letter_map[filename]}_multiple_elements.png",
                                        mime="image/png",
                                        key=f"png_multiple_{filename}"
                                    )
                    
                    st.markdown("<hr style='height:2px;border-width:0;color:gray;background-color:gray'>", unsafe_allow_html=True)
            
            else:  # Heatmaps visualization with graph_objects

                st.sidebar.subheader("Heatmap Appearance")
                use_log_scale = st.sidebar.checkbox("Use logarithmic scale", value=True, key="single_multiple_log_scale")


                colorscale_options = [
                    'aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance', 'blackbody', 'bluered', 'blues', 'blugrn',
                    'bluyl', 'brbg', 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl', 'darkmint', 'deep',
                    'delta', 'dense', 'earth', 'edge', 'electric', 'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens',
                    'greys', 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet', 'magenta', 'magma', 'matter',
                    'mint', 'mrybm', 'mygbm', 'oranges', 'orrd', 'oryel', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg',
                    'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 'puor', 'purd', 'purp', 'purples', 'purpor',
                    'rainbow', 'rdbu', 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar', 'spectral', 'speed',
                    'sunset', 'sunsetdark', 'teal', 'tealgrn', 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
                    'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd'
                ]

                selected_colorscale = st.sidebar.selectbox(
                    "Select colorscale:",
                    colorscale_options,
                    index=0,
                    key="single_multiple_colorscale"
                )


                reverse_colors = st.sidebar.checkbox("Reverse color scale", key="reverse_colorscale")
                colorscale = selected_colorscale.lower()
                if reverse_colors:
                    colorscale = colorscale + "_r"


                single_element_data = {}
                multiple_element_data = {}
                
                for filename, file_data in all_files_data.items():
                    file_label = file_letter_map[filename]
                    particles_per_ml = data_dict[filename]['particles_per_ml']
                    if particles_per_ml is None:
                        particles_per_ml = 0
                    else:
                        particles_per_ml = float(particles_per_ml.replace('e', 'E'))
                    

                    single_elements = {}
                    for combination, details, percentage in file_data['single']:
                        element = combination  # For single elements, the combination is just the element name
                        count = details['counts']
                        element_particles_per_ml = (count / file_data['total_counts']) * particles_per_ml
                        single_elements[element] = element_particles_per_ml
                    
                    single_element_data[file_label] = single_elements
                    

                    multiple_elements = {}
                    for combination, details, percentage in file_data['multiple']:
                        count = details['counts']
                        element_particles_per_ml = (count / file_data['total_counts']) * particles_per_ml
                        multiple_elements[combination] = element_particles_per_ml
                    
                    multiple_element_data[file_label] = multiple_elements
                

                all_single_elements = set()
                for elements_dict in single_element_data.values():
                    all_single_elements.update(elements_dict.keys())
                
                single_df = pd.DataFrame(index=single_element_data.keys(), columns=sorted(all_single_elements))
                
                for file_label, elements_dict in single_element_data.items():
                    for element, value in elements_dict.items():
                        single_df.loc[file_label, element] = value
                
                single_df = single_df.fillna(0)
                

                all_multiple_combinations = set()
                for combinations_dict in multiple_element_data.values():
                    all_multiple_combinations.update(combinations_dict.keys())
                

                top_combinations = sorted(all_multiple_combinations, 
                                        key=lambda x: sum(multiple_element_data[file].get(x, 0) for file in multiple_element_data.keys()),
                                        reverse=True)[:15]  # Show only top 15 combinations
                
                multiple_df = pd.DataFrame(index=multiple_element_data.keys(), columns=top_combinations)
                
                for file_label, combinations_dict in multiple_element_data.items():
                    for combination, value in combinations_dict.items():
                        if combination in top_combinations:
                            multiple_df.loc[file_label, combination] = value
                
                multiple_df = multiple_df.fillna(0)
                

                if use_log_scale:

                    min_nonzero = 1.0
                    for df in [single_df, multiple_df]:
                        nonzero_min = df[df > 0].min().min() if (df > 0).any().any() else min_nonzero
                        min_nonzero = min(min_nonzero, nonzero_min / 10)
                    

                    single_df_plot = single_df.copy()
                    multiple_df_plot = multiple_df.copy()
                    
                    single_df_plot[single_df_plot == 0] = min_nonzero
                    multiple_df_plot[multiple_df_plot == 0] = min_nonzero
                    
                    single_df_plot = np.log10(single_df_plot)
                    multiple_df_plot = np.log10(multiple_df_plot)
                    

                    log_min = min(single_df_plot.min().min(), multiple_df_plot.min().min())
                    log_max = max(single_df_plot.max().max(), multiple_df_plot.max().max())
                    

                    log_range = np.arange(np.floor(log_min), np.ceil(log_max) + 1)
                    tick_vals = log_range
                    tick_text = [f"10<sup>{int(i)}</sup>" for i in log_range]
                else:

                    single_df_plot = single_df
                    multiple_df_plot = multiple_df
                    

                    all_values = np.concatenate([single_df.values.flatten(), multiple_df.values.flatten()])
                    vmin = np.min(all_values[all_values > 0]) if np.any(all_values > 0) else 0
                    vmax = np.max(all_values)
                    
                    log_min = vmin
                    log_max = vmax
                    tick_vals = None
                    tick_text = None
                

                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Single Elements", "Multiple Elements"),
                    horizontal_spacing=0.05,  # reduced spacing
                    column_widths=[0.45, 0.55]  # Give more space to multiple elements
                )
                
         
                heatmap_single = go.Heatmap(
                    z=single_df_plot.values,
                    x=single_df_plot.columns,
                    y=single_df_plot.index,
                    customdata=single_df.values if use_log_scale else None,
                    colorscale=colorscale,
                    showscale=False, 
                    zmin=log_min,
                    zmax=log_max,
                    hovertemplate='Sample: %{y}<br>Element: %{x}<br>Particles/mL: %{customdata:.2e}<extra></extra>' if use_log_scale else 
                                'Sample: %{y}<br>Element: %{x}<br>Particles/mL: %{z:.2e}<extra></extra>',
                    name="Single Elements"
                )
            
                colorbar_title = "Particles/mL (log scale)" if use_log_scale else "Particles/mL"
                heatmap_multiple = go.Heatmap(
                    z=multiple_df_plot.values,
                    x=multiple_df_plot.columns,
                    y=multiple_df_plot.index,
                    customdata=multiple_df.values if use_log_scale else None,
                    colorscale=colorscale,
                    showscale=True,
                    zmin=log_min,
                    zmax=log_max,
                    colorbar=dict(
                        title=colorbar_title,
                        titleside="right",
                        x=1.02,  # Position colorbar
                        lenmode="fraction",
                        len=0.9,
                        thickness=20,
                        tickfont=dict(size=30, family='Times New Roman', color='black'),
                        titlefont=dict(size=30, family='Times New Roman', color='black'),
                        tickvals=tick_vals,
                        ticktext=tick_text
                    ),
                    hovertemplate='Sample: %{y}<br>Combination: %{x}<br>Particles/mL: %{customdata:.2e}<extra></extra>' if use_log_scale else 
                                'Sample: %{y}<br>Combination: %{x}<br>Particles/mL: %{z:.2e}<extra></extra>',
                    name="Multiple Elements"
                )
                
           
                fig.add_trace(heatmap_single, row=1, col=1)
                fig.add_trace(heatmap_multiple, row=1, col=2)
                
           
                scale_info = " (Log Scale)" if use_log_scale else ""
                colorscale_info = f" - {selected_colorscale}" if selected_colorscale != "Viridis" else ""
                
                fig.update_layout(
                    title=dict(
                        text=f"Distribution of Single and Multiple Element Particles{scale_info}",
                        font=dict(size=30, family='Times New Roman', color='black'),
                        x=0.5,
                        xanchor="center"
                    ),
                    height=max(1000, 100 + 40 * max(len(single_df), len(multiple_df))),
                    width=1200,
                    font=dict(size=30, family='Times New Roman', color='black'),
                    margin=dict(l=110, r=100, t=100, b=200),
                    plot_bgcolor="white",
                    paper_bgcolor="white"
                )
                

                for i in range(1, 3):

                    fig.update_xaxes(
                        title=dict(
                            text="Element" if i == 1 else "Element Combination",
                            font=dict(size=30, family='Times New Roman', color='black')
                        ),
                        tickangle=45,
                        tickfont=dict(size=30, family='Times New Roman', color='black'),
                        gridcolor="lightgray",
                        row=1, 
                        col=i
                    )
                    

                    fig.update_yaxes(
                        title=dict(
                            text="Sample" if i == 1 else None,
                            font=dict(size=30, family='Times New Roman', color='black')
                        ),
                        tickfont=dict(size=30, family='Times New Roman', color='black'),
                        gridcolor="lightgray",
                        showticklabels=i == 1,  # Only show tick labels for the first heatmap
                        row=1, 
                        col=i
                    )
                

                st.plotly_chart(fig, use_container_width=True)
                

                col1, col2 = st.columns(2)
                
                with col1:

                    csv_single = single_df.to_csv()
                    st.download_button(
                        label="Download Single Elements Data (CSV)",
                        data=csv_single,
                        file_name='single_elements_heatmap_data.csv',
                        mime='text/csv',
                        key='single_heatmap_csv'
                    )
                
                with col2:

                    csv_multiple = multiple_df.to_csv()
                    st.download_button(
                        label="Download Multiple Elements Data (CSV)",
                        data=csv_multiple,
                        file_name='multiple_elements_heatmap_data.csv',
                        mime='text/csv',
                        key='multiple_heatmap_csv'
                    )
                

                if st.sidebar.button("Download Heatmap Figure as PNG"):
                    img_bytes = fig.to_image(format="png", width=1200, height=800, scale=3)
                    st.sidebar.download_button(
                        label="Click to Download PNG",
                        data=img_bytes,
                        file_name="single_multiple_elements_heatmap.png",
                        mime="image/png",
                        key="heatmap_png_download"
                    )

    def plot_ternary_heatmap(df_dict, elements, title, file_letter_map):
        num_files = len(df_dict)
        num_cols = 3
        num_rows = (num_files // num_cols) + (num_files % num_cols > 0)
        fig = sp.make_subplots(rows=num_rows, cols=num_cols,
                            specs=[[{'type': 'ternary'}] * num_cols for _ in range(num_rows)])
        all_summary_data = []

        default_file_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        element_colors = ['#000000', '#5CA7A8', '#B36366']

        headers = []
        max_rows = 0

        for I, (filename, data) in enumerate(df_dict.items()):
            row = (I // num_cols) + 1
            col = (I % num_cols) + 1
            
            if data_type == "IsoTrack":
                mole_data = data['mole_data']  
            else:
                df = data['df']
                _, _, mole_data, _, _ = process_data(df)
                
            if all(element in mole_data.columns for element in elements):
                mole_data = mole_data.dropna(subset=elements)

                file_color = st.sidebar.color_picker(f'Pick a color for file {file_letter_map[filename]}',
                                                    default_file_colors[I % len(default_file_colors)], key=f"ternary_{filename}")

                fig.add_trace(go.Scatterternary(
                    a=mole_data[elements[0]],
                    b=mole_data[elements[1]],
                    c=mole_data[elements[2]],
                    mode='markers',
                    name=f'{file_letter_map[filename]}',
                    marker=dict(color=file_color, size=6, line=dict(width=1))
                ), row=row, col=col)

                fig.update_layout(
                    **{f'ternary{(row - 1) * num_cols + col}.sum': 1,
                    f'ternary{(row - 1) * num_cols + col}.aaxis.title': elements[0],
                    f'ternary{(row - 1) * num_cols + col}.baxis.title': elements[1],
                    f'ternary{(row - 1) * num_cols + col}.caxis.title': elements[2],
                    f'ternary{(row - 1) * num_cols + col}.aaxis.titlefont.size': 25,
                    f'ternary{(row - 1) * num_cols + col}.aaxis.titlefont.color': 'black',
                    f'ternary{(row - 1) * num_cols + col}.baxis.titlefont.size': 25,
                    f'ternary{(row - 1) * num_cols + col}.baxis.titlefont.color': "black",
                    f'ternary{(row - 1) * num_cols + col}.caxis.titlefont.size': 25,
                    f'ternary{(row - 1) * num_cols + col}.caxis.titlefont.color': "black",
                    f'ternary{(row - 1) * num_cols + col}.aaxis.tickfont.size': 18,
                    f'ternary{(row - 1) * num_cols + col}.aaxis.tickfont.color': "black",
                    f'ternary{(row - 1) * num_cols + col}.baxis.tickfont.size': 18,
                    f'ternary{(row - 1) * num_cols + col}.baxis.tickfont.color': "black",
                    f'ternary{(row - 1) * num_cols + col}.caxis.tickfont.size': 18,
                    f'ternary{(row - 1) * num_cols + col}.caxis.tickfont.color': "black",
                    f'ternary{(row - 1) * num_cols + col}.aaxis.gridcolor': element_colors[0],
                    f'ternary{(row - 1) * num_cols + col}.baxis.gridcolor': element_colors[1],
                    f'ternary{(row - 1) * num_cols + col}.caxis.gridcolor': element_colors[2],
                    f'ternary{(row - 1) * num_cols + col}.aaxis.linecolor': element_colors[0],
                    f'ternary{(row - 1) * num_cols + col}.baxis.linecolor': element_colors[1],
                    f'ternary{(row - 1) * num_cols + col}.caxis.linecolor': element_colors[2]}
                )
                
                file_summary_data = []
                file_summary_data.append(['File name: ' + filename, '', ''])
                file_summary_data.append([elements[0], elements[1], elements[2]])
                for a, b, c in zip(mole_data[elements[0]], mole_data[elements[1]], mole_data[elements[2]]):
                    file_summary_data.append([a, b, c])
                all_summary_data.append(file_summary_data)
                max_rows = max(max_rows, len(file_summary_data))

        for file_data in all_summary_data:
            while len(file_data) < max_rows:
                file_data.append(['', '', ''])

        merged_summary_data = []
        for row_idx in range(max_rows):
            merged_row = []
            for file_data in all_summary_data:
                merged_row.extend(file_data[row_idx] + [''])
            merged_summary_data.append(merged_row)

        fig.update_layout(
            title_text=title,
            showlegend=True,
            height=600 * num_rows,
            plot_bgcolor='lightgray',
            font=dict(
                size=18,
                color="black"
            ),
            legend=dict(
                font=dict(
                    size=30,
                    color="black"
                )
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        if merged_summary_data:
            summary_df = pd.DataFrame(merged_summary_data)
            csv = summary_df.to_csv(index=False, header=False)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='ternary_data.csv',
                mime='text/csv'
            )













    def plot_mole_ratio_histogram_for_files(df_dict, element1, element2, bin_size, x_max, title, file_letter_map):
        filenames = list(df_dict.keys())
        n_subplots = len(filenames)
        
        fig = make_subplots(rows=n_subplots, cols=1, vertical_spacing=0.02)
        
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        all_summary_data = []

        for i, filename in enumerate(filenames):
            data = df_dict[filename]
            if data_type == "IsoTrack":
                mole_data = data['mole_data']  
            else:
                df = data['df']
                _, _, mole_data, _, _ = process_data(df)
                
            if all(element in mole_data.columns for element in [element1, element2]):
                mole_data = mole_data.dropna(subset=[element1, element2])

                filtered_data = mole_data[(mole_data[element1] > 0) & (mole_data[element2] > 0)]
                ratios = filtered_data[element1] / filtered_data[element2]
                ratios = ratios.dropna()
                
                total_particles = len(filtered_data)

                if ratios.empty:
                    st.error(f'No valid data available for plotting in file {filename}.')
                    continue
                
                avg_ratio = ratios.mean()

                color = st.sidebar.color_picker(f'Pick a color for {file_letter_map[filename]}',
                                                default_colors[i % len(default_colors)], key=f"colo_{filename}")
                hist, bin_edges = np.histogram(ratios, bins=np.arange(0, x_max + bin_size, bin_size))
                y_max = max(hist) if len(hist) > 0 else 1

                fig.add_trace(
                    go.Histogram(
                        x=ratios,
                        xbins=dict(size=bin_size, start=0, end=x_max),
                        name=f'{file_letter_map[filename]}',
                        marker=dict(color=color, line=dict(width=1.5)),
                        opacity=1,
                        showlegend=False
                    ),
                    row=i+1, col=1
                )
                fig.add_annotation(
                    x=x_max * 0.9,  
                    y=y_max *0.9,          
                    xref="x",
                    yref="y",      
                    text=f"Avg: {avg_ratio:.1f}<br>n: {total_particles}",
                    showarrow=False,
                    font=dict(size=80, color="black", weight='bold'),
                    bgcolor="rgba(255,255,255,0)",
                    align="right",
                    row=i+1, col=1
                )

    
                file_summary_data = []
                file_summary_data.append(['File name: ' + filename, ''])
                file_summary_data.append([f'{element1}/{element2}', 'Frequency'])
                hist, bin_edges = np.histogram(ratios, bins=np.arange(0, x_max + bin_size, bin_size))
                for edge, count in zip(bin_edges[:-1], hist):
                    file_summary_data.append([edge, count])
                all_summary_data.append(file_summary_data)
        for i in range(1, n_subplots + 1):
            is_last_subplot = (i == n_subplots)
            
            fig.update_xaxes(
                range=[0, x_max],
                linecolor='black',
                linewidth=2,
                showticklabels=is_last_subplot, 
                row=i, col=1
            )
            
            if is_last_subplot:

                if x_max <= 10:
                    tick_interval = 1
                elif x_max <= 20:
                    tick_interval = 5
                elif x_max <= 50:
                    tick_interval = 10
                else:
                    tick_interval = 20
                    
                fig.update_xaxes(
                    title_text=f"{element1}/{element2} Ratio",
                    title_font=dict(size=80, color='black', weight='bold'),
                    tickfont=dict(size=80, color='black', weight='bold'),
                    tickmode='array',
                    tickvals=list(range(0, int(x_max) + 1, tick_interval)),
                    ticktext=[str(x) for x in range(0, int(x_max) + 1, tick_interval)],
                    row=i, col=1,
                    showgrid=False
                )
            
            fig.update_yaxes(
                tickfont=dict(size=80, color='black', weight='bold'), 
                linecolor='black',
                linewidth=2,
                row=i, col=1,
                showgrid=False 
            )
        

        fig.add_annotation(
            text="Particle count",
            font=dict(size=80, color='black', weight='bold'),
            showarrow=False,
            xref="paper",
            yref="paper",
            x=-0.07, 
            y=0.5,
            textangle=-90
        )


        fig.update_layout(
            title=title,
            barmode='overlay', 
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=80, color="black", weight='bold'),
            legend=dict(
                font=dict(size=50, color="black"),
            ),
            height=300 * n_subplots,
            margin=dict(l=500, r=70, t=200, b=200)  
        )


        st.plotly_chart(fig, use_container_width=True)
        

        if st.sidebar.button('Download Figure as PNG', key='download_mole_ratio_btn'):
            img_bytes = fig.to_image(
                format="png",
                width=1200 * 4,
                height=(200 * n_subplots) * 4,
                scale=3
            )
            
            st.sidebar.download_button(
                label="Click to Download PNG",
                data=img_bytes,
                file_name=f"mole_ratio_{element1}_{element2}_{title}.png",
                mime="image/png",
                key=f"png_download_mole_ratio" 
            )

        if all_summary_data:
            max_rows = max(len(data) for data in all_summary_data)
            for file_data in all_summary_data:
                while len(file_data) < max_rows:
                    file_data.append(['', ''])

            merged_summary_data = []
            for row_idx in range(max_rows):
                merged_row = []
                for file_data in all_summary_data:
                    merged_row.extend(file_data[row_idx])
                    merged_row.append('') 
                merged_summary_data.append(merged_row)

            summary_df = pd.DataFrame(merged_summary_data)
            csv = summary_df.to_csv(index=False, header=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='mole_ratio_histogram_data.csv',
                mime='text/csv',
                key='moleratio'
            )


    def get_combinations_and_related_data(df_dict, filename):
        start_time = time.time()
        
        if data_type == "IsoTrack":
            processed_data = process_isotrack(df_dict[filename]['df'])
            if processed_data is not None:
                mass_data = processed_data['mass']
                mole_data = processed_data['mole']
                mole_percent_data = processed_data['mole_percent']
            else:
                st.error("Could not process IsoTrack data")
                return None, None, None, None, None, None
        else:
            df = df_dict[filename]['df']
            mass_data, mass_percent_data, mole_data, mole_percent_data, _ = process_data(df)

        combination_data = {}
        mole_percent_data = mole_percent_data.apply(pd.to_numeric, errors='coerce')

        for index, row in mole_percent_data.iterrows():
            elements = row[row > 0].index.tolist()
            combination_key = ', '.join(sorted(elements))
            combination_data.setdefault(combination_key, []).append(index)

        related_data = {
            'mass_data': {},
            'mass_percent_data': {},
            'mole_data': {},
            'mole_percent_data': {}
        }
        combinations = {}

        for combination_key, indices in combination_data.items():
            indices = pd.Index(indices)

            related_data['mass_data'][combination_key] = mass_data.loc[indices]
            related_data['mole_data'][combination_key] = mole_data.loc[indices]
            related_data['mole_percent_data'][combination_key] = mole_percent_data.loc[indices]

            if indices.size > 0:
                filtered_data = mole_percent_data.loc[indices]
                sums = filtered_data.sum()
                counts = indices.size
                average = sums / counts
                squared_diffs = ((filtered_data - average) ** 2).sum()

                combinations[combination_key] = {
                    'sums': sums,
                    'counts': counts,
                    'average': average,
                    'squared_diffs': squared_diffs,
                    'sd': np.sqrt(squared_diffs / counts)
                }

        sd_data = {key: value['sd'] for key, value in combinations.items()}
        sd_df = pd.DataFrame(sd_data).transpose()

        elapsed_time = time.time() - start_time
        st.write(f"Time taken: {elapsed_time} seconds")

        return combinations, related_data['mole_percent_data'], related_data['mass_data'], None, related_data['mole_data'], sd_df

    def read_and_process_files(uploaded_files):
        data_dict = {}
        elements_set = set()
        for file in uploaded_files:
            df = pd.read_csv(file)
            _, _, _, mole_percent_data, _ = process_data(df)
            if mole_percent_data is not None:
                data_dict[file.name] = mole_percent_data
                elements_set.update(mole_percent_data.columns.tolist())
            else:
                st.error(f"Failed to process {file.name}.")
        return data_dict, list(elements_set)


    def text_color_based_on_background(value, min_val, max_val):
        norm_value = (value - min_val) / (max_val - min_val)
        return 'white' if norm_value > 0.5 else 'black'

    


    def summarize_data_with_count_threshold(data, count_threshold):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")


        total_data = data.sum()
        

        percentages = total_data * 100 / total_data.sum()
        

        element_counts = (data > 0).sum(axis=0)
        

        main = percentages[element_counts >= count_threshold]
        others = percentages[element_counts < count_threshold]
        main_counts = element_counts[element_counts >= count_threshold]
        

        if others.sum() > 0:
            main = main.copy()  
            main['Others'] = others.sum()
            main_counts['Others'] = element_counts[element_counts < count_threshold].sum()

        return main, main_counts
        
    def plot_mass_correlation(data_dict, x_element, y_element, file_letter_map, data_type, marker_size=8, opacity=0.6, use_log_scale=False):
        """
        Creates a scatter plot showing mass correlation between two selected elements with regression lines and R² values.
        """
        fig = go.Figure()
        

        colors = [
            '#1E88E5',  # rich blue
            '#FF5252',  # coral red
            '#00BFA5',  # emerald green
            '#8E24AA',  # deep purple
            '#FFB300',  # amber gold
            '#00ACC1',  # teal blue
            '#EC407A',  # rose pink
            '#F57C00',  # deep orange
            '#5C6BC0',  # indigo blue
            '#7CB342'   # lime green
        ]
        
        y_offset = 0.95  
        color_index = 0  
        
        for filename, data in data_dict.items():
            df = data['df']
            if data_type == "IsoTrack":
                processed_data = process_isotrack(df)
                if processed_data is not None:
                    mass_data = processed_data['mass']
                else:
                    continue
            else:
                mass_data, _, _, _, _ = process_data(df)
                

            valid_data = mass_data[(mass_data[x_element] > 0) & (mass_data[y_element] > 0)]
            
            if not valid_data.empty:

                current_color = colors[color_index % len(colors)]
                

                fig.add_trace(go.Scatter(
                    x=valid_data[x_element],
                    y=valid_data[y_element],
                    mode='markers',
                    name=file_letter_map[filename],
                    marker=dict(
                        size=marker_size,
                        opacity=opacity,
                        color=current_color,
                        line=dict(width=1, color='black')
                    )
                ))
                

                x = valid_data[x_element]
                y = valid_data[y_element]
                
                if use_log_scale:
                    x = np.log10(x)
                    y = np.log10(y)
                
                slope, intercept = np.polyfit(x, y, 1)
                r_squared = np.corrcoef(x, y)[0,1]**2
                
        
                x_range = np.linspace(min(x), max(x), 100)
                y_range = slope * x_range + intercept
                
                if use_log_scale:
                    x_range = 10**x_range
                    y_range = 10**y_range

                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    name=f'{file_letter_map[filename]} fit',
                    line=dict(
                        color=current_color,
                        dash='dash',
                    ),
                    showlegend=False
                ))
                
                fig.add_annotation(
                    text=f"{file_letter_map[filename]}: R² = {r_squared:.3f}",
                    xref="paper",
                    yref="paper",
                    x=0.02,
                    y=y_offset,
                    showarrow=False,
                    font=dict(
                        size=20,
                        color=current_color,
                        family='Times New Roman'
                    ),
                    align="left"
                )
                y_offset -= 0.05  
                color_index += 1  
        
        fig.update_layout(
            title=dict(
                text=f'Mass Correlation: {y_element} vs {x_element}',
                font=dict(size=40, family='Times New Roman', color='black')
            ),
            xaxis_title=dict(
                text=f'{x_element} (fg)',
                font=dict(size=40, family='Times New Roman', color='black')
            ),
            yaxis_title=dict(
                text=f'{y_element} (fg)',
                font=dict(size=40, family='Times New Roman', color='black')
            ),
            height=800,
            showlegend=True,
            plot_bgcolor='white',
            legend=dict(
                font=dict(size=40, family='Times New Roman', color='black'),
                itemsizing='constant'
            ),
            font=dict(size=40, color='black'),
            margin=dict(l=110, r=80, t=100, b=100) 
        )
        
        fig.update_xaxes(
            type='log' if use_log_scale else 'linear',
            gridcolor='lightgray',
            zeroline=False,
            title_font=dict(size=60, family='Times New Roman', color='black'),
            tickfont=dict(size=60, family='Times New Roman', color='black'),
            linecolor='black',
            linewidth=2,
            showgrid=False
        )
        
        fig.update_yaxes(
            type='log' if use_log_scale else 'linear',
            gridcolor='lightgray',
            zeroline=False,
            title_font=dict(size=60, family='Times New Roman', color='black'),
            tickfont=dict(size=60, family='Times New Roman', color='black'),
            linecolor='black',
            linewidth=2,
            showgrid=False
        )
    
        return fig



    def visualize_mass_and_mole_percentages_pie_charts(data_dict, file_letter_map, color_map, percent_threshold, particle_count_threshold, show_mass, show_mole, use_particle_threshold):
        """
        Analyzes and visualizes mass and mole percentages in the provided data.
        Now includes options for both percentage threshold and particle count threshold.
        """
        chart_container = st.container()
    
        with chart_container:
            st.empty()

            num_files = len(data_dict)
            charts_per_file = sum([show_mass, show_mole])
            
            if charts_per_file == 0:
                st.warning("Please select at least one type of percentage to display (Mass or Mole)")
                return

            num_cols = charts_per_file
            num_rows = num_files

            fig = sp.make_subplots(
                rows=num_rows, 
                cols=num_cols,
                specs=[[{'type': 'pie'}] * num_cols] * num_rows,
                subplot_titles=[f"{letter}" 
                            for letter in file_letter_map.values() 
                            for i in range(charts_per_file)],
                horizontal_spacing=0.025,
                vertical_spacing=0.025,
            
            )
            fig.update_annotations(font_size=20) 

            summary_data = []
            current_row = 1

            for filename, data in data_dict.items():
                df = data['df']
                if data_type == "IsoTrack":
                    processed_data = process_isotrack(df)
                    if processed_data is not None:
                        mass_data = processed_data['mass']
                        mole_data = processed_data['mole']
                    else:
                        continue
                else:
                    mass_data, _, mole_data, _, _ = process_data(df)

                current_col = 1

                if show_mass:

                    if use_particle_threshold:
                        mass_percent, mass_counts = summarize_data_with_count_threshold(mass_data, particle_count_threshold)
                    else:
                        mass_percent, mass_counts = summarize_data(mass_data, percent_threshold)
                        
                    colors_mass = [color_map.get(index, '#CCCCCC') for index in mass_percent.index]
                    
                    fig.add_trace(
                        go.Pie(
                            labels=mass_percent.index,
                            values=mass_percent.values,
                            marker=dict(colors=colors_mass, line=dict(color='#000000', width=1)),
                            textinfo='label+percent',
                            texttemplate='%{label}: %{value:.1f}% (%{customdata})',
                            customdata=[mass_counts.get(index, 0) for index in mass_percent.index],
                            hoverinfo='label+percent+value',
                            hovertext=[f"{label}: {value:.1f}% ({mass_counts.get(label, 0):d} counts)" 
                                    for label, value in mass_percent.items()],
                            textfont=dict(size=25, color='black'),
                            showlegend=False
                        ),
                        row=current_row,
                        col=current_col
                    )
                    current_col += 1

                    mass_summary_data = [['File name: ' + filename, '', ''],
                                    ['Element', 'Mass Percent (%)', 'Counts']]
                    for element, value in mass_percent.items():
                        mass_summary_data.append([element, f'{value:.2f}', mass_counts.get(element, 0)])
                    summary_data.append(mass_summary_data)

                if show_mole:

                    if use_particle_threshold:
                        mole_percent, mole_counts = summarize_data_with_count_threshold(mole_data, particle_count_threshold)
                    else:
                        mole_percent, mole_counts = summarize_data(mole_data, percent_threshold)
                        
                    colors_mole = [color_map.get(index, '#CCCCCC') for index in mole_percent.index]
                    
                    fig.add_trace(
                        go.Pie(
                            labels=mole_percent.index,
                            values=mole_percent.values,
                            marker=dict(colors=colors_mole, line=dict(color='#000000', width=1)),
                            textinfo='label+percent',
                            texttemplate='%{label}: %{value:.1f}% (%{customdata})',
                            customdata=[mole_counts.get(index, 0) for index in mole_percent.index],
                            hoverinfo='label+percent+value',
                            hovertext=[f"{label}: {value:.1f}% ({mole_counts.get(label, 0):d} counts)" 
                                    for label, value in mole_percent.items()],
                            textfont=dict(size=25, color='black'),
                            showlegend=False
                        ),
                        row=current_row,
                        col=current_col
                    )

                    mole_summary_data = [['File name: ' + filename, '', ''],
                                    ['Element', 'Mole Percent (%)', 'Counts']]
                    for element, value in mole_percent.items():
                        mole_summary_data.append([element, f'{value:.2f}', mole_counts.get(element, 0)])
                    summary_data.append(mole_summary_data)

                current_row += 1

            fig.update_layout(
                height=300 * num_rows,  
                width=800 * num_cols,   
                title_text=" ",
                title_x=0.5,
                showlegend=False,
                font=dict(size=1, color='black')
            )

            st.plotly_chart(fig, use_container_width=True, key="pie_charts_fixed")
            st.sidebar.markdown("### Download Chart")
            if st.sidebar.button("Generate High Quality PNG"):
                img_bytes = fig.to_image(
                    format="png",
                    width=1200 * num_cols,
                    height=800 * num_rows,
                    scale=1
                )
                
                st.sidebar.download_button(
                    label="Download PNG",
                    data=img_bytes,
                    file_name="pie_charts_high_quality.png",
                    mime="image/png",
                    key="download_png"
                )

            if summary_data:
                max_rows = max(len(data) for data in summary_data)
                for data in summary_data:
                    while len(data) < max_rows:
                        data.append(['', '', ''])

                merged_summary_data = []
                for row_idx in range(max_rows):
                    merged_row = []
                    for data in summary_data:
                        merged_row.extend(data[row_idx])
                    merged_summary_data.append(merged_row)

                summary_df = pd.DataFrame(merged_summary_data)
                csv = summary_df.to_csv(index=False, header=False)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='mass_and_mole_percentages.csv',
                    mime='text/csv',
                    key="download_pie_charts_fixed"
                )


    def create_color_map(_elements, base_colors):
        color_map = {}
        for i, element in enumerate(sorted(_elements)):
            default_color = base_colors[i % len(base_colors)]
            color = st.sidebar.color_picker(f"Color {element}", value=default_color, key=f"colo_{element}")
            color_map[element] = color
        color_map['Others'] = st.sidebar.color_picker(f"Color Others", '#777777')
        return color_map

    def prepare_heatmap_data(data_combinations, combinations, start, end, file_letters=None, combined_mode=False):
        heatmap_df = pd.DataFrame()
        combo_counts = {combo: info['counts'] for combo, info in combinations.items()}

        for combo, df in data_combinations.items():
            df = df.apply(pd.to_numeric, errors='coerce')
            avg_percents = df.mean().to_frame().T
            avg_percents = avg_percents.div(avg_percents.sum(axis=1), axis=0)

            if combined_mode:

                file_info = combo.split('(')[-1].replace(')', '').strip()
                base_combo = combo.rsplit(' (', 1)[0]  
                count = combo_counts[combo]
                

                combo_with_count = f"{file_info}- {base_combo} ({count})"
            else:
                count = combo_counts[combo]
                combo_with_count = f"{combo} ({count})"
                
            avg_percents.index = [combo_with_count]
            heatmap_df = pd.concat([heatmap_df, avg_percents])

        heatmap_df['Counts'] = heatmap_df.index.map(lambda x: int(x.split('(')[1].split(')')[0].strip()))
        heatmap_df = heatmap_df.sort_values(by='Counts', ascending=False)

        heatmap_df = heatmap_df.iloc[start - 1:end]
        heatmap_df.drop(columns=['Counts'], inplace=True)

        return heatmap_df

    def get_combinations_and_related_data(data_dict, selected_file):
        start_time = time.time()
        
        if data_type == "IsoTrack":
            processed_data = process_isotrack(data_dict[selected_file]['df'])
            if processed_data is not None:
                mass_data = processed_data['mass']
                mole_data = processed_data['mole']
                mole_percent_data = processed_data['mole_percent'] / 100 
            else:
                st.error("Could not process IsoTrack data")
                return None, None, None, None, None, None
        else:
            df = data_dict[selected_file]['df']
            mass_data, _, mole_data, mole_percent_data, _ = process_data(df)

        if mole_percent_data is None:
            st.error("Could not process mole percent data")
            return None, None, None, None, None, None

        combination_data = {}
        mole_percent_data = mole_percent_data.apply(pd.to_numeric, errors='coerce')

        for index, row in mole_percent_data.iterrows():
            elements = row[row > 0].index.tolist()
            combination_key = ', '.join(sorted(elements))
            combination_data.setdefault(combination_key, []).append(index)

        related_data = {
            'mass_data': {},
            'mass_percent_data': {},
            'mole_data': {},
            'mole_percent_data': {}
        }
        combinations = {}

        for combination_key, indices in combination_data.items():
            indices = pd.Index(indices)

            related_data['mass_data'][combination_key] = mass_data.loc[indices]
            related_data['mole_data'][combination_key] = mole_data.loc[indices]
            related_data['mole_percent_data'][combination_key] = mole_percent_data.loc[indices]

            if indices.size > 0:
                filtered_data = mole_percent_data.loc[indices]
                sums = filtered_data.sum()
                counts = indices.size
                average = sums / counts
                squared_diffs = ((filtered_data - average) ** 2).sum()

                combinations[combination_key] = {
                    'sums': sums,
                    'counts': counts,
                    'average': average,
                    'squared_diffs': squared_diffs,
                    'sd': np.sqrt(squared_diffs / counts)
                }

        sd_data = {key: value['sd'] for key, value in combinations.items()}
        sd_df = pd.DataFrame(sd_data).transpose()

        elapsed_time = time.time() - start_time
        st.write(f"Time taken: {elapsed_time} seconds")

        return combinations, related_data['mole_percent_data'], related_data['mass_data'], None, related_data['mole_data'], sd_df

    def plot_heatmap(heatmap_df, sd_df, selected_colorscale='ylgnbu', display_numbers=True, font_size=14, combined_mode=False):
        heatmap_df = heatmap_df.fillna(0)
        sd_df = sd_df.fillna(0)
       

        elements_with_data = [col for col in heatmap_df.columns if heatmap_df[col].any()]
        heatmap_df = heatmap_df[elements_with_data]
        sd_df = sd_df[elements_with_data]

        elements = heatmap_df.columns.tolist()
        combinations_with_counts = heatmap_df.index.tolist()
        
        def extract_count(combo_string):
            try:
                match = re.search(r'\((\d+)\)', combo_string)
                if match:
                    return int(match.group(1))
                return 0
            except:
                return 0

        total_count = sum(extract_count(comb) for comb in combinations_with_counts)


        z_values = heatmap_df.values * 100
        z_values = np.nan_to_num(z_values, nan=0.0)  # Convert any remaining NaNs to 0
        min_val = z_values.min()
        max_val = z_values.max()


        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=elements,
            y=combinations_with_counts,
            colorscale=selected_colorscale,
            colorbar=dict(
                title='Mole %',
                titlefont=dict(size=30, family='Times New Roman', color='black', weight='bold'),
                tickfont=dict(size=30, family='Times New Roman', color='black', weight='bold'),
                ticks='outside',
                ticklen=5,
                tickwidth=2,
                tickcolor='black'
            )
        ))
        

        if combined_mode:

            file_identifiers = []
            cleaned_labels = []
            
            for combo in combinations_with_counts:
                if '(' in combo and ')' in combo:

                    parts = combo.split('-', 1)
                    file_id = parts[0].strip()
                    file_identifiers.append(file_id)
                    

                    if len(parts) > 1:
                        cleaned_label = parts[1].strip()
                    else:
                        cleaned_label = combo  # Fallback
                    cleaned_labels.append(cleaned_label)
                else:
                    file_identifiers.append(None)
                    cleaned_labels.append(combo)
            

            unique_files = sorted(list(set([f for f in file_identifiers if f is not None])))
            

            file_colors = {}
            default_colors = px.colors.qualitative.Plotly  # Use Plotly's default qualitative colors
            for i, file_id in enumerate(unique_files):
                file_colors[file_id] = default_colors[i % len(default_colors)]
            

            colored_labels = []
            for i, (combo, clean_label) in enumerate(zip(combinations_with_counts, cleaned_labels)):
                file_id = file_identifiers[i]
                if file_id and file_id in file_colors:
                    color = file_colors[file_id]

                    colored_labels.append(f'<span style="color:{color};font-weight:bold;">{clean_label}</span>')
                else:
                    colored_labels.append(clean_label)
            

            fig.update_layout(
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(combinations_with_counts))),
                    ticktext=colored_labels,
                    tickangle=0,
                    autorange='reversed',
                    title='Particle (Frequency)',
                    titlefont=dict(size=30, family='Times New Roman', color='black', weight='bold'),
                    tickfont=dict(size=30, family='Times New Roman', color='black', weight='bold'),
                    showgrid=False
                )
            )
            

            for file_id, color in file_colors.items():
                fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(size=15, color=color),
                    name=f'File: {file_id}',
                    showlegend=True
                ))
        

        if display_numbers:
            for y, comb_with_count in enumerate(combinations_with_counts):
                base_comb = comb_with_count.split(' (')[0]
                if '-' in base_comb and combined_mode:
                    base_comb = base_comb.split('- ', 1)[1] if '- ' in base_comb else base_comb
                
                for x, elem in enumerate(elements):

                    avg_value = heatmap_df.loc[comb_with_count, elem] * 100
                    if np.isnan(avg_value):
                        avg_value = 0.0
                    

                    sd_key = base_comb
                    if '(' in comb_with_count and ')' in comb_with_count:
                        file_letter = comb_with_count.split(')')[-1].strip()
                        if file_letter:
                            sd_key = f"{base_comb} ({file_letter})"
                    

                    if elem in sd_df.columns and sd_key in sd_df.index:
                        sd_value = sd_df.loc[sd_key, elem] * 100
                        if np.isnan(sd_value):
                            sd_value = 0.0
                    else:
                        sd_value = 0.0
                    

                    color = text_color_based_on_background(avg_value, min_val, max_val)


                    if avg_value != 0:
                        annotation_text = f"{avg_value:.1f}"
                        fig.add_annotation(
                            x=x, 
                            y=y, 
                            text=annotation_text, 
                            showarrow=False,
                            font=dict(size=font_size, color=color, family='Times New Roman')
                        )


        fig.update_layout(
            title=f'Molar Percentage After Treatment - Total Particles: {total_count}',
            xaxis=dict(
                title='Elements', 
                tickangle=0,
                titlefont=dict(size=30, family='Times New Roman', color='black', weight='bold'),
                tickfont=dict(size=30, family='Times New Roman', color='black', weight='bold'),
            ),
            height=max(600, 40 * len(combinations_with_counts)),
            plot_bgcolor='white',
            legend=dict(
                title=" ",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=30, family='Times New Roman', color='black', weight='bold'),
                itemsizing='constant'
            )
        )

        return fig
    
    def calculate_sample_correlations(combined_mole_percent_combinations, file_letter_map, method='pearson', missing_penalty=0.5):
        """
        Calculate correlations between samples based on their element combinations.
        
        Args:
            combined_mole_percent_combinations: Dictionary with combination data
            file_letter_map: Mapping of filenames to sample labels
            method: Correlation method to use ('pearson', 'spearman', 'aitchison', 'cosine', 'clustering')
            missing_penalty: Penalty factor applied when a combination exists in one sample but not in another
        
        Returns:
            Correlation DataFrame and list of sample labels
        """
        from scipy.spatial.distance import pdist, squareform
        from scipy.cluster import hierarchy
        import sklearn.metrics.pairwise as pairwise
        

        samples = set()
        for combo in combined_mole_percent_combinations.keys():
            if '(' in combo and ')' in combo:
                sample = combo.split('(')[-1].replace(')', '').strip()
                samples.add(sample)
        
        samples = sorted(list(samples))
        


        all_elements = set()
        for df in combined_mole_percent_combinations.values():
            all_elements.update(df.columns)
        all_elements = sorted(list(all_elements))
        

        sample_element_sums = {sample: {element: 0.0 for element in all_elements} for sample in samples}
        sample_counts = {sample: 0 for sample in samples}
        

        sample_combinations = {sample: set() for sample in samples}
        all_combinations = set()
        

        for combo, df in combined_mole_percent_combinations.items():
            if '(' in combo and ')' in combo:
                base_combo = combo.split('(')[0].strip()
                sample = combo.split('(')[-1].replace(')', '').strip()
                particle_count = len(df)
                sample_counts[sample] += particle_count
                

                sample_combinations[sample].add(base_combo)
                all_combinations.add(base_combo)
                
                for element in all_elements:
                    if element in df.columns:

                        sample_element_sums[sample][element] += df[element].mean() * particle_count
        

        for sample in samples:
            if sample_counts[sample] > 0:
                for element in all_elements:
                    sample_element_sums[sample][element] /= sample_counts[sample]
        

        sample_element_df = pd.DataFrame(sample_element_sums).T
        sample_element_df = sample_element_df.fillna(0)
        

        if method == 'pearson':
            correlation_matrix = sample_element_df.T.corr(method='pearson')
            title_suffix = "Pearson Correlation"
            
        elif method == 'spearman':
            correlation_matrix = sample_element_df.T.corr(method='spearman')
            title_suffix = "Spearman Rank Correlation"
            
        elif method == 'aitchison':


            epsilon = 1e-10
            comp_data = sample_element_df.replace(0, epsilon)
            

            from scipy.stats import gmean
            clr_data = comp_data.apply(lambda x: np.log(x) - np.log(gmean(x)), axis=1)
            

            dist_matrix = pd.DataFrame(
                1 - squareform(pdist(clr_data, metric='euclidean')),
                index=comp_data.index,
                columns=comp_data.index
            )
            

            min_val = dist_matrix.values.min()
            max_val = dist_matrix.values.max()
            correlation_matrix = dist_matrix.copy()
            correlation_matrix = 2 * (correlation_matrix - min_val) / (max_val - min_val) - 1
            
            title_suffix = "Aitchison Distance (CLR-transformed)"
            
        elif method == 'cosine':

            cosine_sim = pd.DataFrame(
                pairwise.cosine_similarity(sample_element_df),
                index=sample_element_df.index,
                columns=sample_element_df.index
            )
            correlation_matrix = cosine_sim
            title_suffix = "Cosine Similarity"
            
        elif method == 'clustering':


            from sklearn.preprocessing import StandardScaler
            std_data = pd.DataFrame(
                StandardScaler().fit_transform(sample_element_df),
                index=sample_element_df.index,
                columns=sample_element_df.columns
            )
            

            distance = pd.DataFrame(
                squareform(pdist(std_data, metric='euclidean')),
                index=std_data.index,
                columns=std_data.index
            )
            

            max_dist = distance.values.max()
            correlation_matrix = 1 - (distance / max_dist)
            
            title_suffix = "Hierarchical Clustering Similarity"
        
        else:

            correlation_matrix = sample_element_df.T.corr(method='pearson')
            title_suffix = "Pearson Correlation"
        


        penalty_matrix = np.ones((len(samples), len(samples)))
        
        for i, sample1 in enumerate(samples):
            for j, sample2 in enumerate(samples):
                if i != j:  # No penalty on diagonal

                    s1_combos = sample_combinations[sample1]
                    s2_combos = sample_combinations[sample2]
                    
                    if s1_combos or s2_combos:
                        jaccard = len(s1_combos.intersection(s2_combos)) / len(s1_combos.union(s2_combos))
                        



                        penalty_matrix[i, j] = 1 - ((1 - jaccard) * missing_penalty)
        

        penalty_df = pd.DataFrame(penalty_matrix, index=correlation_matrix.index, columns=correlation_matrix.columns)
        correlation_matrix = correlation_matrix * penalty_df
        

        if missing_penalty > 0:
            title_suffix += f" with Missing Combination Penalty"
        

        sample_labels = {}
        for filename, label in file_letter_map.items():
            for sample in samples:
                if sample in label:
                    sample_labels[sample] = label
        

        labeled_correlation = correlation_matrix.copy()
        if sample_labels:
            new_index = [sample_labels.get(idx, idx) for idx in labeled_correlation.index]
            labeled_correlation.index = new_index
            labeled_correlation.columns = new_index
        
        return labeled_correlation, samples, title_suffix

    def plot_sample_correlation_matrix(correlation_df, title_suffix="Pearson Correlation"):
        """
        Create a heatmap visualization of the sample correlation matrix.
        
        Args:
            correlation_df: DataFrame containing correlation values
            title_suffix: The correlation method used (added to title)
        
        Returns:
            Plotly figure object
        """
        title = f"Sample Similarity Analysis: {title_suffix}"
        import plotly.graph_objects as go
        import numpy as np
        

        colorscale = [
            [0.0, 'rgb(49,54,149)'],
            [0.1, 'rgb(69,117,180)'],
            [0.2, 'rgb(116,173,209)'],
            [0.3, 'rgb(171,217,233)'],
            [0.4, 'rgb(224,243,248)'],
            [0.5, 'rgb(255,255,255)'],
            [0.6, 'rgb(254,224,144)'],
            [0.7, 'rgb(253,174,97)'],
            [0.8, 'rgb(244,109,67)'],
            [0.9, 'rgb(215,48,39)'],
            [1.0, 'rgb(165,0,38)']
        ]
        

        fig = go.Figure(data=go.Heatmap(
            z=correlation_df.values,
            x=correlation_df.columns,
            y=correlation_df.index,
            colorscale=colorscale,
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title="Correlation",
                titleside="right",
                titlefont=dict(size=25, family='Times New Roman', color='black',weight='bold'),
                tickfont=dict(size=20, family='Times New Roman', color='black', weight='bold'),
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"]
            ),
            hovertemplate='Sample 1: %{y}<br>Sample 2: %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        

        for i, row in enumerate(correlation_df.index):
            for j, col in enumerate(correlation_df.columns):
                value = correlation_df.iloc[i, j]
                text_color = 'black' if abs(value) < 0.7 else 'white'
                
                fig.add_annotation(
                    x=col,
                    y=row,
                    text=f"{value:.2f}",
                    showarrow=False,
                    font=dict(
                        color=text_color,
                        size=25,
                        family='Times New Roman'
                    )
                )
        

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=35, family='Times New Roman', color='black', weight='bold'),
                x=0.5
            ),
            height=max(600, 80 * len(correlation_df)),
            width=max(700, 80 * len(correlation_df)),
            xaxis=dict(
                title="",
                tickfont=dict(size=30, family='Times New Roman', color='black', weight='bold'),
                side='bottom',
                tickangle=0
            ),
            yaxis=dict(
                title="",
                tickfont=dict(size=30, family='Times New Roman', color='black', weight='bold'),
                autorange='reversed'
            ),
            plot_bgcolor='white',
            margin=dict(l=150, r=100, t=150, b=150)
        )
        
        return fig
    
    def create_combination_difference_heatmap(combined_mole_percent_combinations, correlation_df, n_pairs=10, max_combinations=15, sort_by_count=False, sort_by_correlation=False, highlighted_elements=None, correlation_method='pearson', missing_combination_penalty=-0.8):
        """
        Create a heatmap showing how element combinations contribute to correlation between sample pairs,
        with specific similarity values adjusted for particle count differences for all combinations.
        
        Args:
            combined_mole_percent_combinations: Dictionary with combination data
            correlation_df: DataFrame containing correlation values between samples
            n_pairs: Number of sample pairs to show
            max_combinations: Maximum number of combinations to show
            sort_by_count: Whether to sort by particle count within frequency groups
            sort_by_correlation: Whether to sort by overall correlation contribution
            highlighted_elements: List of elements to highlight in the combinations
            correlation_method: Method to use for element correlations (pearson, spearman, cosine, etc.)
            missing_combination_penalty: Penalty value when only one sample has a combination (-1.0 to 0.0)
        """
        import statistics
        import re
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.stats import pearsonr, spearmanr
        

        if highlighted_elements is None:
            highlighted_elements = []
        

        combo_to_samples = {}
        combo_sample_data = {}
        
        for full_combo in combined_mole_percent_combinations.keys():
            if '(' in full_combo and ')' in full_combo:
                combo_type = full_combo.split('(')[0].strip()
                sample = full_combo.split('(')[-1].replace(')', '').strip()
                
                if combo_type not in combo_to_samples:
                    combo_to_samples[combo_type] = set()
                    combo_sample_data[combo_type] = {}
                    
                combo_to_samples[combo_type].add(sample)
                combo_sample_data[combo_type][sample] = combined_mole_percent_combinations[full_combo]
        

        all_combo_types = sorted(list(combo_to_samples.keys()))
        all_elements = set()
        for combo in all_combo_types:
            elements = [e.strip() for e in combo.split(',')]
            all_elements.update(elements)
        

        combo_sample_counts = {}
        combo_particle_counts = {}
        single_sample_combos = set()
        
        for combo_type in all_combo_types:
            samples_with_combo = combo_to_samples.get(combo_type, set())
            sample_count = len(samples_with_combo)
            combo_sample_counts[combo_type] = sample_count
            
            if sample_count == 1:
                single_sample_combos.add(combo_type)
            
            total_count = 0
            for full_combo, df in combined_mole_percent_combinations.items():
                if '(' in full_combo:
                    current_combo = full_combo.split('(')[0].strip()
                    if combo_type == current_combo:
                        total_count += len(df)
                    combo_particle_counts[combo_type] = total_count
        

        sample_pairs = []
        for i in range(len(correlation_df.index)):
            for j in range(i+1, len(correlation_df.columns)):
                sample1 = correlation_df.index[i]
                sample2 = correlation_df.columns[j]
                corr_value = correlation_df.iloc[i, j]
                
                pair_name = f"{sample1} vs {sample2}"
                sample_pairs.append((pair_name, corr_value, sample1, sample2))
        
        sample_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        selected_pairs = sample_pairs[:n_pairs]
        

        multi_sample_combos = [(combo, freq) for combo, freq in combo_sample_counts.items() if freq > 1]
        single_sample_combos_list = [(combo, 1) for combo in single_sample_combos]
        
        multi_sample_combos.sort(key=lambda x: x[1], reverse=True)
        
        if multi_sample_combos:
            frequencies = [count for _, count in multi_sample_combos]
            median_frequency = statistics.median(frequencies)
        else:
            median_frequency = 0
        

        if sort_by_correlation:

            combo_correlation_scores = {}
            

            contribution_matrix = []

            for pair_name, corr_value, sample1, sample2 in selected_pairs:
                row = []
                for combo_type in final_combo_order:
                    samples_with_this_combo = combo_to_samples.get(combo_type, set())
                    sample1_has_combo = sample1 in samples_with_this_combo
                    sample2_has_combo = sample2 in samples_with_this_combo
                    
                    if combo_type in single_sample_combos:

                        if sample1_has_combo or sample2_has_combo:
                            row.append({"specific": missing_combination_penalty, "overall": corr_value})
                        else:
                            row.append({"specific": 0, "overall": corr_value})
                    else:

                        if sample1_has_combo and sample2_has_combo:

                            combo_elements = [elem.strip() for elem in combo_type.split(',')]
                            
                            try:

                                sample1_data = None
                                sample2_data = None
                                
                                for full_combo, df in combined_mole_percent_combinations.items():
                                    combo_part = full_combo.split('(')[0].strip()
                                    sample_part = full_combo.split('(')[-1].replace(')', '').strip()
                                    
                                    if combo_part == combo_type and sample_part == sample1:
                                        sample1_data = df
                                    elif combo_part == combo_type and sample_part == sample2:
                                        sample2_data = df
                                
                                if sample1_data is not None and sample2_data is not None:

                                    valid_elements = [e for e in combo_elements if e in sample1_data.columns and e in sample2_data.columns]
                                    

                                    if len(valid_elements) > 1:

                                        if correlation_method == 'pearson':

                                            from scipy.stats import pearsonr
                                            

                                            mean1 = sample1_data[valid_elements].mean().values
                                            mean2 = sample2_data[valid_elements].mean().values
                                            
                                            if len(mean1) >= 2:  # Need at least 2 points for correlation
                                                corr_matrix = np.corrcoef(mean1, mean2)
                                                specific_sim = corr_matrix[0, 1]
                                            else:

                                                max_val = max(mean1[0], mean2[0]) if mean1[0] != 0 or mean2[0] != 0 else 1
                                                rel_diff = 1 - abs(mean1[0] - mean2[0]) / max_val
                                                specific_sim = 2 * rel_diff - 1  # Map [0,1] to [-1,1]
                                        
                                        elif correlation_method == 'spearman':

                                            from scipy.stats import spearmanr
                                            

                                            if len(valid_elements) >= 2:
                                                specific_sim, _ = spearmanr(
                                                    sample1_data[valid_elements].mean(), 
                                                    sample2_data[valid_elements].mean()
                                                )
                                                

                                                if np.isnan(specific_sim):
                                                    specific_sim = 0
                                            else:

                                                if len(sample1_data) > 1 and len(sample2_data) > 1:
                                                    series1 = sample1_data[valid_elements[0]].dropna()
                                                    series2 = sample2_data[valid_elements[0]].dropna()
                                                    if len(series1) > 1 and len(series2) > 1:
                                                        specific_sim, _ = spearmanr(series1, series2)
                                                        if np.isnan(specific_sim):
                                                            specific_sim = 0
                                                    else:
                                                        specific_sim = 0
                                                else:
                                                    specific_sim = 0
                                        
                                        elif correlation_method == 'cosine':

                                            from sklearn.metrics.pairwise import cosine_similarity
                                            mean1 = sample1_data[valid_elements].mean().values.reshape(1, -1)
                                            mean2 = sample2_data[valid_elements].mean().values.reshape(1, -1)
                                            specific_sim = cosine_similarity(mean1, mean2)[0][0]
                                        
                                        else:  # Default to cosine for other methods or fallback
                                            from sklearn.metrics.pairwise import cosine_similarity
                                            mean1 = sample1_data[valid_elements].mean().values.reshape(1, -1)
                                            mean2 = sample2_data[valid_elements].mean().values.reshape(1, -1)
                                            specific_sim = cosine_similarity(mean1, mean2)[0][0]
                                    
                                    else:

                                        element = valid_elements[0] if valid_elements else combo_elements[0]
                                        
                                        if correlation_method == 'spearman' and element in sample1_data.columns and element in sample2_data.columns:

                                            from scipy.stats import spearmanr
                                            

                                            if len(sample1_data) > 1 and len(sample2_data) > 1:
                                                series1 = sample1_data[element].dropna()
                                                series2 = sample2_data[element].dropna()
                                                if len(series1) > 1 and len(series2) > 1:
                                                    specific_sim, _ = spearmanr(series1, series2)
                                                    if np.isnan(specific_sim):
                                                        specific_sim = 0
                                                else:

                                                    mean1 = sample1_data[element].mean()
                                                    mean2 = sample2_data[element].mean()
                                                    max_val = max(mean1, mean2) if mean1 != 0 or mean2 != 0 else 1
                                                    rel_diff = 1 - abs(mean1 - mean2) / max_val
                                                    specific_sim = 2 * rel_diff - 1  # Map [0,1] to [-1,1]
                                            else:
                                                mean1 = sample1_data[element].mean()
                                                mean2 = sample2_data[element].mean()
                                                max_val = max(mean1, mean2) if mean1 != 0 or mean2 != 0 else 1
                                                rel_diff = 1 - abs(mean1 - mean2) / max_val
                                                specific_sim = 2 * rel_diff - 1  # Map [0,1] to [-1,1]
                                        
                                        elif correlation_method == 'pearson' and element in sample1_data.columns and element in sample2_data.columns:

                                            from scipy.stats import pearsonr
                                            

                                            if len(sample1_data) > 1 and len(sample2_data) > 1:
                                                series1 = sample1_data[element].dropna()
                                                series2 = sample2_data[element].dropna()
                                                if len(series1) > 1 and len(series2) > 1:
                                                    specific_sim, _ = pearsonr(series1, series2)
                                                    if np.isnan(specific_sim):
                                                        specific_sim = 0
                                                else:

                                                    mean1 = sample1_data[element].mean()
                                                    mean2 = sample2_data[element].mean()
                                                    max_val = max(mean1, mean2) if mean1 != 0 or mean2 != 0 else 1
                                                    rel_diff = 1 - abs(mean1 - mean2) / max_val
                                                    specific_sim = 2 * rel_diff - 1  # Map [0,1] to [-1,1]
                                            else:
                                                mean1 = sample1_data[element].mean()
                                                mean2 = sample2_data[element].mean()
                                                max_val = max(mean1, mean2) if mean1 != 0 or mean2 != 0 else 1
                                                rel_diff = 1 - abs(mean1 - mean2) / max_val
                                                specific_sim = 2 * rel_diff - 1  # Map [0,1] to [-1,1]
                                        
                                        else:

                                            if element in sample1_data.columns and element in sample2_data.columns:
                                                mean1 = sample1_data[element].mean()
                                                mean2 = sample2_data[element].mean()
                                                max_val = max(mean1, mean2) if mean1 != 0 or mean2 != 0 else 1
                                                rel_diff = 1 - abs(mean1 - mean2) / max_val
                                                specific_sim = 2 * rel_diff - 1  # Map [0,1] to [-1,1]
                                            else:
                                                specific_sim = 0
                                    

                                    if np.isnan(specific_sim):
                                        specific_sim = 0
                                    

                                    particle_count1 = len(sample1_data)
                                    particle_count2 = len(sample2_data)
                                    count_ratio = min(particle_count1, particle_count2) / max(particle_count1, particle_count2)
                                    adjusted_sim = specific_sim * (count_ratio ** 0.5)
                                    

                                    if len(combo_elements) == 1 and combo_elements[0] in sample1_data.columns and combo_elements[0] in sample2_data.columns:
                                        mean1 = sample1_data[combo_elements[0]].mean()
                                        mean2 = sample2_data[combo_elements[0]].mean()
                                        element_info = f"{mean1:.2f}/{mean2:.2f}"
                                    else:
                                        element_info = None
                                    

                                    cell_data = {
                                        "specific": adjusted_sim,
                                        "original_sim": specific_sim,
                                        "count_ratio": count_ratio,
                                        "counts": f"{particle_count1}/{particle_count2}",
                                        "overall": corr_value,
                                        "method": correlation_method
                                    }
                                    
                                    if element_info:
                                        cell_data["element_vals"] = element_info
                                    
                                    row.append(cell_data)
                                else:
                                    row.append({"specific": corr_value, "overall": corr_value})
                            except Exception as e:
                                row.append({"specific": corr_value, "overall": corr_value})
                        elif sample1_has_combo or sample2_has_combo:

                            row.append({"specific": missing_combination_penalty, "overall": corr_value})
                        else:

                            row.append({"specific": 0, "overall": corr_value})
                
                contribution_matrix.append(row)
                    

            combo_avg_contributions = {}
            for combo_type, scores in combo_correlation_scores.items():
                if scores:  # Only consider combos with actual scores


                    combo_avg_contributions[combo_type] = sum(abs(score) for score in scores) / len(scores)
                else:
                    combo_avg_contributions[combo_type] = 0
            

            sorted_combos = sorted(combo_avg_contributions.keys(), 
                                key=lambda x: combo_avg_contributions.get(x, 0), 
                                reverse=False)
            

            sorted_combos = [combo for combo in sorted_combos if combo not in single_sample_combos]
            

            final_combo_order = sorted_combos[:max_combinations]
        else:

            frequent_combos = []
            infrequent_combos = []
            
            for combo, frequency in multi_sample_combos:
                if frequency >= median_frequency:
                    frequent_combos.append((combo, frequency))
                else:
                    infrequent_combos.append((combo, frequency))
            
            infrequent_combos.extend(single_sample_combos_list)
            

            if sort_by_count:
                frequent_combos.sort(key=lambda x: combo_particle_counts.get(x[0], 0), reverse=True)
                infrequent_combos.sort(key=lambda x: combo_particle_counts.get(x[0], 0), reverse=True)
            

            max_frequent = min(len(frequent_combos), max_combinations // 2)
            max_infrequent = min(len(infrequent_combos), max_combinations // 2)
            
            if max_frequent < max_combinations // 2:
                max_infrequent = min(len(infrequent_combos), max_combinations - max_frequent)
            elif max_infrequent < max_combinations // 2:
                max_frequent = min(len(frequent_combos), max_combinations - max_infrequent)
            
            frequent_combos = frequent_combos[:max_frequent]
            infrequent_combos = infrequent_combos[:max_infrequent]
            
            final_combo_order = [c[0] for c in frequent_combos + infrequent_combos]
        

        contribution_matrix = []
        
        for pair_name, corr_value, sample1, sample2 in selected_pairs:
            row = []
            for combo_type in final_combo_order:
                samples_with_this_combo = combo_to_samples.get(combo_type, set())
                sample1_has_combo = sample1 in samples_with_this_combo
                sample2_has_combo = sample2 in samples_with_this_combo
                
                if combo_type in single_sample_combos:

                    if sample1_has_combo or sample2_has_combo:
                        row.append({"specific": missing_combination_penalty, "overall": corr_value})
                    else:
                        row.append({"specific": 0, "overall": corr_value})
                else:

                    if sample1_has_combo and sample2_has_combo:

                        combo_elements = [elem.strip() for elem in combo_type.split(',')]
                        
                        try:

                            sample1_data = None
                            sample2_data = None
                            
                            for full_combo, df in combined_mole_percent_combinations.items():
                                combo_part = full_combo.split('(')[0].strip()
                                sample_part = full_combo.split('(')[-1].replace(')', '').strip()
                                
                                if combo_part == combo_type and sample_part == sample1:
                                    sample1_data = df
                                elif combo_part == combo_type and sample_part == sample2:
                                    sample2_data = df
                            
                            if sample1_data is not None and sample2_data is not None:

                                if len(combo_elements) > 1:

                                    mean1 = sample1_data[combo_elements].mean().values.reshape(1, -1)
                                    mean2 = sample2_data[combo_elements].mean().values.reshape(1, -1)
                                    specific_sim = cosine_similarity(mean1, mean2)[0][0]
                                else:

                                    element = combo_elements[0]
                                    mean1 = sample1_data[element].mean()
                                    mean2 = sample2_data[element].mean()

                                    max_val = max(mean1, mean2)
                                    min_val = min(mean1, mean2)
                                    if max_val == 0:  
                                        specific_sim = 1.0 
                                    else:
                                        relative_diff = 1 - (max_val - min_val) / max_val

                                        specific_sim = 2 * relative_diff - 1  
                                

                                particle_count1 = len(sample1_data)
                                particle_count2 = len(sample2_data)
                                

                                count_ratio = min(particle_count1, particle_count2) / max(particle_count1, particle_count2)
                                

                                adjusted_sim = specific_sim * (count_ratio ** 0.5)
                                

                                if len(combo_elements) == 1:
                                    element_info = f"{mean1:.2f}/{mean2:.2f}"
                                else:
                                    element_info = None
                                

                                cell_data = {
                                    "specific": adjusted_sim, 
                                    "original_sim": specific_sim,
                                    "count_ratio": count_ratio,
                                    "counts": f"{particle_count1}/{particle_count2}",
                                    "overall": corr_value
                                }
                                
                                if element_info:
                                    cell_data["element_vals"] = element_info
                                    
                                row.append(cell_data)
                            else:
                                row.append({"specific": corr_value, "overall": corr_value})
                        except Exception as e:
                            row.append({"specific": corr_value, "overall": corr_value})
                    elif sample1_has_combo or sample2_has_combo:

                        row.append({"specific": missing_combination_penalty, "overall": corr_value})
                    else:

                        row.append({"specific": 0, "overall": corr_value})
            
            contribution_matrix.append(row)
        

        pair_labels = [f"{pair[0]} (r={pair[1]:.2f})" for pair in selected_pairs]
        

        if not sort_by_correlation and frequent_combos and infrequent_combos:
            separator_index = len(frequent_combos) - 0.5
        else:
            separator_index = None
        

        x_labels = []
        highlighted_indices = []
        
        for i, combo_type in enumerate(final_combo_order):
            sample_count = combo_sample_counts.get(combo_type, 0)
            

            if sort_by_correlation and combo_type in combo_correlation_scores and combo_correlation_scores[combo_type]:
                avg_contribution = sum(abs(s) for s in combo_correlation_scores[combo_type]) / len(combo_correlation_scores[combo_type])
                label_prefix = f"{combo_type} ({sample_count}S, r={avg_contribution:.2f})"
            else:
                label_prefix = f"{combo_type} ({sample_count}S)"
            

            combo_elements = [e.strip() for e in combo_type.split(',')]
            contains_highlighted = any(elem in combo_elements for elem in highlighted_elements)
            
            if contains_highlighted:
                highlighted_indices.append(i)
                stars = '*' * len([e for e in highlighted_elements if e in combo_elements])
                label = f"{label_prefix} {stars}"
            else:
                label = label_prefix
                
            x_labels.append(label)
        

        colorscale = [
            [0.0, 'rgb(49,54,149)'],
            [0.1, 'rgb(69,117,180)'],
            [0.2, 'rgb(116,173,209)'],
            [0.3, 'rgb(171,217,233)'],
            [0.4, 'rgb(224,243,248)'],
            [0.5, 'rgb(255,255,255)'],
            [0.6, 'rgb(254,224,144)'],
            [0.7, 'rgb(253,174,97)'],
            [0.8, 'rgb(244,109,67)'],
            [0.9, 'rgb(215,48,39)'],
            [1.0, 'rgb(165,0,38)']
        ]
        

        z_values = [[cell["specific"] if isinstance(cell, dict) else cell for cell in row] for row in contribution_matrix]
        

        fig = go.Figure()
        

        fig.add_trace(go.Heatmap(
            z=z_values,
            x=x_labels,
            y=pair_labels,
            colorscale=colorscale,
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title="Correlation Contribution",
                titleside="right",
                titlefont=dict(size=15, family='Times New Roman', color='black', weight='bold'),
                tickfont=dict(size=20, family='Times New Roman', color='black', weight='bold'),
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"]
            ),
            hovertemplate='Sample Pair: %{y}<br>Combination: %{x}<br>Contribution: %{z:.3f}<extra></extra>'
        ))
        

        if not sort_by_correlation and frequent_combos and separator_index is not None:

            fig.add_shape(
                type="rect", 
                x0=-0.5, 
                y0=-0.5, 
                x1=separator_index, 
                y1=len(pair_labels)-0.5,
                line=dict(width=0),
                fillcolor="rgba(144, 238, 144, 0.2)", 
                layer="below"
            )
            

            fig.add_shape(
                type="rect", 
                x0=separator_index, 
                y0=-0.5, 
                x1=len(final_combo_order)-0.5, 
                y1=len(pair_labels)-0.5,
                line=dict(width=0),
                fillcolor="rgba(255, 182, 193, 0.2)", 
                layer="below"
            )
        

        if not sort_by_correlation:
            if frequent_combos:
                fig.add_annotation(
                    x=len(frequent_combos)//2,
                    y=len(pair_labels) + 0.01,
                    text="FREQUENTLY FOUND COMBINATIONS",
                    showarrow=False,
                    xref="x",
                    yref="y",
                    xanchor="center",
                    yanchor="bottom",
                    font=dict(size=15, family='Times New Roman', color='black', weight='bold')
                )
            
            if infrequent_combos:
                fig.add_annotation(
                    x=len(frequent_combos) + len(infrequent_combos)//2,
                    y=len(pair_labels) + 0.01,
                    text="LESS FREQUENTLY FOUND COMBINATIONS",
                    showarrow=False,
                    xref="x",
                    yref="y",
                    xanchor="center",
                    yanchor="bottom",
                    font=dict(size=15, family='Times New Roman', color='black', weight='bold')
                )
        else:

            fig.add_annotation(
                x=len(final_combo_order)//2,
                y=len(pair_labels) + 0.01,
                text="COMBINATIONS SORTED BY CORRELATION CONTRIBUTION",
                showarrow=False,
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="bottom",
                font=dict(size=15, family='Times New Roman', color='black', weight='bold')
            )
        

        if not sort_by_correlation and separator_index is not None:
            fig.add_shape(
                type="line",
                x0=separator_index,
                y0=-0.5,
                x1=separator_index,
                y1=len(pair_labels) - 0.5,
                line=dict(
                    color="black",
                    width=5,
                    dash="solid",
                ),
                xref="x",
                yref="y"
            )
        

        for idx in highlighted_indices:
            fig.add_shape(
                type="rect",
                x0=idx - 0.45,
                x1=idx + 0.45,
                y0=-0.5,
                y1=len(pair_labels) - 0.5,
                line=dict(width=0),
                fillcolor="rgba(255, 255, 0, 0.2)", 
                layer="below"
            )
        
        

        if sort_by_correlation:
            sort_method = "by Overall Correlation Contribution"
        elif sort_by_count:
            sort_method = "by Particle Count within Groups"
        else:
            sort_method = "by Sample Frequency"


        fig.update_layout(
            title=dict(
                text=f"Element Combination Contribution to Sample {correlation_method.title()} Correlation (Sorted {sort_method})",
                font=dict(size=22, family='Times New Roman', color='black', weight='bold')
            ),
            xaxis=dict(
                title="Combinations (sample counts)" if not sort_by_correlation else "Combinations (sample counts, Avg. Correlation)",
                titlefont=dict(size=25, family='Times New Roman', color='black', weight='bold'),
                tickfont=dict(size=20, family='Times New Roman', color='black', weight='bold'),
                tickangle=45,
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(


                tickfont=dict(size=15, family='Times New Roman', color='black', weight='bold'),
                showgrid=False,
                zeroline=False
            ),
            width=max(2000, 200 * len(final_combo_order)),
            height=max(1000, 80 * len(pair_labels)),
            plot_bgcolor='white',
            margin=dict(l=225, r=30, t=100, b=250)
        )
        

        for i, trace in enumerate(fig.data):
            if hasattr(trace, 'hovertemplate'):
                fig.data[i].hovertemplate = (
                    'Sample Pair: %{y}<br>' +
                    'Combination: %{x}<br>' +
                    f'{correlation_method.title()} Contribution: %{{z:.3f}}<br>' +
                    '<extra></extra>'
                )
        
        return fig, sorted(list(all_elements))
                                                                            
                    
    
    def plot_diameter_distribution(df_dict, element, bin_size, x_min, x_max, title, file_letter_map):
        """
        Creates enhanced histogram plots for diameter data with overlaid lines and average values,
        with adjustable x-axis minimum and maximum values
        """
        filenames = list(df_dict.keys())
        num_files = len(filenames)
        plots_per_figure = st.sidebar.checkbox('Show two plots per figure', value=True, key='plots_per_figure_diameter')
        
        if plots_per_figure:
            num_subplots = (num_files + 1) // 2
        else:
            num_subplots = num_files

        fig = make_subplots(rows=num_subplots, cols=1, shared_xaxes=True, vertical_spacing=0.08)
        
        subplot_y_values = [[] for _ in range(num_subplots)]
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

        bin_edges = np.arange(x_min, x_max + bin_size, bin_size)
        all_summary_data = []

        avg_data = {
            "File": [],
            "Average Diameter (nm)": [],
            "Standard Deviation (nm)": [],
            "Median (nm)": [],
            "Count": []
        }

        for i, filename in enumerate(filenames):
            if plots_per_figure:
                subplot_index = i // 2 + 1
            else:
                subplot_index = i + 1
                
            data = df_dict[filename]
            processed_data = process_isotrack(data['df'])
            
            if processed_data is None or 'diameter' not in processed_data:
                st.warning(f"Diameter data not found in {filename}")
                continue
                
            diameter_data = processed_data['diameter']
            
            if element not in diameter_data.columns:
                st.warning(f"Element {element} diameter not found in {filename}")
                continue
                
            diameter_data = diameter_data.dropna(subset=[element])
            
            color = st.sidebar.color_picker(f'Pick a color for {file_letter_map[filename]}',
                                        default_colors[i % len(default_colors)], 
                                        key=f"Diameter_{filename}")

            hist_data = diameter_data[element]
            hist_data_filtered = hist_data[(hist_data >= x_min) & (hist_data <= x_max)]
            
            if len(hist_data_filtered) == 0:
                st.warning(f"No data within the selected range for {filename}")
                continue
                
            hist_counts, bin_edges_hist = np.histogram(hist_data_filtered, bins=bin_edges)
            total_particles = len(hist_data_filtered)
            total_original = len(hist_data)

            avg_diameter = hist_data_filtered.mean()
            std_diameter = hist_data_filtered.std()
            median_diameter = hist_data_filtered.median()
            
            avg_data["File"].append(file_letter_map[filename])
            avg_data["Average Diameter (nm)"].append(f"{avg_diameter:.2f}")
            avg_data["Standard Deviation (nm)"].append(f"{std_diameter:.2f}")
            avg_data["Median (nm)"].append(f"{median_diameter:.2f}")
            avg_data["Count"].append(f"{total_particles} ({total_particles/total_original:.1%} of total)")

            hist_density = [(count / total_particles) * 100 for count in hist_counts]
            subplot_y_values[subplot_index-1].extend(hist_density)

            fig.add_trace(
                go.Bar(
                    x=bin_edges[:-1] + bin_size / 2,
                    y=hist_density,
                    name=f'{file_letter_map[filename]}',
                    marker=dict(
                        color=color,
                        line=dict(color='black', width=1.5),
                        opacity=0.8,
                    ),
                    width=bin_size * 0.9,  
                    opacity=0.7,
                    showlegend=False
                ),
                row=subplot_index, col=1
            )

            x_smooth = bin_edges[:-1] + bin_size / 2
            if len(hist_density) > 5:
                try:
                    from scipy.signal import savgol_filter
                    window_size = min(7, len(hist_density) - 2 if len(hist_density) % 2 == 0 else len(hist_density) - 1)
                    if window_size >= 3:
                        y_smooth = savgol_filter(hist_density, window_size, 2)
                    else:
                        from scipy.ndimage import gaussian_filter1d
                        y_smooth = gaussian_filter1d(hist_density, sigma=1)
                except:
                    from scipy.ndimage import gaussian_filter1d
                    y_smooth = gaussian_filter1d(hist_density, sigma=1)
            else:
                from scipy.ndimage import gaussian_filter1d
                y_smooth = gaussian_filter1d(hist_density, sigma=1)
            
            fig.add_trace(
                go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode='lines',
                    name=f'{file_letter_map[filename]}',
                    line=dict(
                        color=color,
                        width=4,
                        shape='spline',
                        smoothing=1.3
                    )
                ),
                row=subplot_index, col=1
            )
            
            fig.add_shape(
                type="line",
                x0=avg_diameter, y0=0,
                x1=avg_diameter, y1=max(hist_density) if hist_density else 0,
                line=dict(
                    color=color,
                    width=3,
                    dash="dash",
                ),
                row=subplot_index, col=1
            )
    
            max_y_value = max(y_smooth) if len(y_smooth) > 0 else (max(hist_density) if hist_density else 0)
            annotation_y_pos = max_y_value * 0.85 
            
            fig.add_annotation(
                x=avg_diameter,
                y=annotation_y_pos,
                text=f"Avg: {avg_diameter:.1f} ± {std_diameter:.1f} nm",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=color,
                bgcolor="rgba(255, 255, 255, 0.8)",  
                bordercolor=color,
                borderwidth=2,
                borderpad=4,
                font=dict(size=60, color=color, family='Times New Roman'),
                row=subplot_index, col=1
            )

            file_summary_data = []
            file_summary_data.append(['File name: ' + filename, '', ''])
            file_summary_data.append(['Diameter (nm)', 'Percentage (%)', ''])
            for edge, density in zip(bin_edges[:-1], hist_density):
                file_summary_data.append([edge, density, ''])
            all_summary_data.append(file_summary_data)
   
        middle_subplot = (num_subplots + 1) // 2
        for i in range(1, num_subplots + 1):
            y_vals = subplot_y_values[i-1]
            if y_vals:
                y_max = max(y_vals) * 1.3  
                tick_vals = np.linspace(0, y_max, 6)
                
                fig.update_yaxes(
                    title_text="Frequency (%)" if i == middle_subplot else None,
                    ticktext=[f'{v:.1f}' for v in tick_vals],
                    tickvals=tick_vals,
                    range=[0, y_max],
                    title_font=dict(size=60, family='Times New Roman', color='black'),
                    tickfont=dict(size=60, family='Times New Roman', color='black'),
                    linecolor='black',
                    linewidth=2,
                    gridcolor='rgba(211, 211, 211, 0.3)', 
                    gridwidth=1,
                    row=i, col=1,
                    showgrid=True
                )
            
            fig.update_xaxes(
                range=[x_min, x_max],  
                linecolor='black',
                linewidth=2,
                showgrid=False,
                row=i, col=1
            )

        fig.update_xaxes(
            range=[x_min, x_max],  
            title_text="Diameter (nm)",
            title_font=dict(size=60, family='Times New Roman', color='black'),
            tickfont=dict(size=60, family='Times New Roman', color='black'),
            linecolor='black',
            linewidth=2,
            gridcolor='rgba(211, 211, 211, 0.3)',
            gridwidth=1,
            row=num_subplots, col=1,
            showgrid=True
        )

        fig.update_layout(
            title=f"{title}: {element} (Range: {x_min}-{x_max} nm)",
            barmode='overlay',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=60, family='Times New Roman', color='black'),
            legend=dict(font=dict(size=60, color="black", family='Times New Roman')),
            height=350 * num_subplots, 
            margin=dict(l=400, r=80, t=200, b=200) 
        )

        st.plotly_chart(fig, use_container_width=True)

        if avg_data["File"]:
            st.subheader(f"Average {element} Diameter Summary (Range: {x_min}-{x_max} nm)")
            avg_df = pd.DataFrame(avg_data)
     
            st.table(avg_df)

            csv_avg = avg_df.to_csv(index=False)
            st.download_button(
                label="Download average diameter data as CSV",
                data=csv_avg,
                file_name=f'average_diameter_{element}_data_{x_min}to{x_max}nm.csv',
                mime='text/csv',
                key='avg_diameter_csv'
            )

        if st.sidebar.button('Download Figure as PNG', key='download_diameter_png'):
            img_bytes = fig.to_image(
                format="png",
                width=1200 * 4,
                height=(350 * num_subplots) * 4,
                scale=2 
            )
            
            st.sidebar.download_button(
                label="Click to Download PNG",
                data=img_bytes,
                file_name=f"diameter_histogram_{element}_{x_min}to{x_max}nm.png",
                mime="image/png",
                key=f"png_download_diameter_{element}"
            )

        if all_summary_data:
            max_rows = max(len(data) for data in all_summary_data)
            for file_data in all_summary_data:
                while len(file_data) < max_rows:
                    file_data.append(['', '', ''])

            merged_summary_data = []
            for row_idx in range(max_rows):
                merged_row = []
                for file_data in all_summary_data:
                    merged_row.extend(file_data[row_idx])
                    merged_row.append('')
                merged_summary_data.append(merged_row)

            summary_df = pd.DataFrame(merged_summary_data)
            csv = summary_df.to_csv(index=False, header=False)
            st.download_button(
                label="Download histogram data as CSV",
                data=csv,
                file_name=f'diameter_histogram_data_{x_min}to{x_max}nm.csv',
                mime='text/csv',
                key='diameter_csv'
            )
            
    def create_comparative_element_correlation_network(data_dict, file_letter_map, min_correlation=0.3):
        import networkx as nx
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np
        from itertools import combinations
        
        st.subheader("Comparative Element Correlation Networks")
        

        selected_samples = st.multiselect(
            "Select samples to compare (2-3 recommended):",
            options=[file_letter_map[filename] for filename in data_dict.keys()],
            default=[file_letter_map[filename] for filename in list(data_dict.keys())[:min(2, len(data_dict))]],
            key="corr_network_samples"
        )
        
        if len(selected_samples) < 2:
            st.warning("Please select at least 2 samples to compare.")
            return
        

        display_mode = st.radio(
            "Select display mode:",
            ["Differential Network", "Side-by-Side Networks", "Animation"],
            key="corr_network_mode"
        )
        

        col1, col2 = st.columns(2)
        with col1:
            min_correlation = st.slider(
                "Minimum correlation strength:", 
                min_value=0.1, 
                max_value=0.9,
                value=min_correlation,
                step=0.05,
                key="min_correlation"
            )
        
        with col2:
            show_negative = st.checkbox("Show negative correlations", value=True, key="show_negative")
        

        sample_correlations = {}
        sample_element_counts = {}
        all_elements = set()
        
        for filename, data in data_dict.items():
            sample_name = file_letter_map[filename]
            
            if sample_name not in selected_samples:
                continue
                

            if data_type == "IsoTrack":
                processed_data = process_isotrack(data_dict[filename]['df'])
                if processed_data is not None:
                    mass_data = processed_data['mass']
                else:
                    st.error(f"Could not process data for {sample_name}")
                    continue
            else:
                mass_data, _, _, _, _ = process_data(data_dict[filename]['df'])
            

            corr_matrix = mass_data.corr(method='spearman')
            element_counts = {}
            for element in corr_matrix.columns:

                element_counts[element] = (mass_data[element] > 0).sum() / len(mass_data) * 100
                all_elements.add(element)
            
            sample_correlations[sample_name] = corr_matrix
            sample_element_counts[sample_name] = element_counts
        
        if not sample_correlations:
            st.error("Could not calculate correlations for the selected samples.")
            return
        

        all_elements = sorted(list(all_elements))
        

        if display_mode == "Side-by-Side Networks":

            num_samples = len(selected_samples)
            cols = min(2, num_samples)
            rows = (num_samples + cols - 1) // cols  # Ceiling division
            

            fig = go.Figure()
            

            x_ranges = []
            y_ranges = []
            

            network_layouts = {}
            
            for sample_name in selected_samples:

                G = nx.Graph()
                
                corr_matrix = sample_correlations[sample_name]
                element_counts = sample_element_counts[sample_name]
                

                for element in corr_matrix.columns:
                    if element in element_counts:
                        G.add_node(element, abundance=element_counts[element])
                

                for i, elem1 in enumerate(corr_matrix.columns):
                    for j, elem2 in enumerate(corr_matrix.columns):
                        if i < j:  
                            if elem1 in G.nodes and elem2 in G.nodes:
                                corr = corr_matrix.loc[elem1, elem2]
                                if abs(corr) >= min_correlation and (show_negative or corr >= 0):
                                    G.add_edge(elem1, elem2, weight=abs(corr), 
                                            original_corr=corr)
                

                pos = nx.spring_layout(G, seed=42, k=0.5) 
                network_layouts[sample_name] = {
                    'graph': G,
                    'positions': pos
                }
                

                x_coords = [coord[0] for coord in pos.values()]
                y_coords = [coord[1] for coord in pos.values()]
                
                if x_coords and y_coords:
                    x_ranges.append((min(x_coords), max(x_coords)))
                    y_ranges.append((min(y_coords), max(y_coords)))
            

            if x_ranges and y_ranges:
                x_min = min(r[0] for r in x_ranges)
                x_max = max(r[1] for r in x_ranges)
                y_min = min(r[0] for r in y_ranges)
                y_max = max(r[1] for r in y_ranges)
                

                x_padding = (x_max - x_min) * 0.1
                y_padding = (y_max - y_min) * 0.1
                
                x_min -= x_padding
                x_max += x_padding
                y_min -= y_padding
                y_max += y_padding
            else:
                x_min, x_max = -1, 1
                y_min, y_max = -1, 1
            

            fig = go.Figure()
            

            subplot_width = 1.0 / cols
            subplot_height = 1.0 / rows
            
            for idx, sample_name in enumerate(selected_samples):
                row = idx // cols
                col = idx % cols
                

                x_domain = [col * subplot_width, (col + 1) * subplot_width - 0.05]
                y_domain = [1 - (row + 1) * subplot_height + 0.05, 1 - row * subplot_height - 0.05]
                
                if sample_name in network_layouts:
                    G = network_layouts[sample_name]['graph']
                    pos = network_layouts[sample_name]['positions']
                    

                    for edge in G.edges(data=True):
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        
                        corr = edge[2]['original_corr']
                        

                        if corr < 0:
                            color = f'rgba(255,0,0,{abs(corr)})'  
                        else:
                            color = f'rgba(0,0,255,{corr})'  
                            
                        width = 1 + 4 * abs(corr)  
                        
                        edge_trace = go.Scatter(
                            x=[x0, x1, None], 
                            y=[y0, y1, None],
                            line=dict(width=width, color=color),
                            hoverinfo='text',
                            text=f"{edge[0]} — {edge[1]}: {corr:.2f}",
                            mode='lines',
                            showlegend=False,
                            xaxis=f'x{idx+1}',
                            yaxis=f'y{idx+1}'
                        )
                        fig.add_trace(edge_trace)
                    

                    node_x = []
                    node_y = []
                    node_text = []
                    node_size = []
                    
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        

                        abundance = G.nodes[node]['abundance']
                        size = 20 + abundance * 1.5  # Scale for visibility
                        node_size.append(size)
                        

                        node_text.append(f"{node}: {abundance:.1f}%")
                    
                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=list(G.nodes()),
                        textposition="top center",
                        textfont=dict(size=12, color='black'),
                        marker=dict(
                            size=node_size,
                            color=[G.nodes[node]['abundance'] for node in G.nodes()],
                            colorscale='Viridis',
                            line=dict(width=2, color='black')
                        ),
                        hoverinfo='text',
                        hovertext=node_text,
                        showlegend=False,
                        xaxis=f'x{idx+1}',
                        yaxis=f'y{idx+1}'
                    )
                    fig.add_trace(node_trace)
                

                x_center = (x_domain[0] + x_domain[1]) / 2
                y_top = y_domain[1] + 0.05


                fig.add_annotation(
                    text=sample_name,
                    x=x_center,
                    y=y_top,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16, color='black', family='Arial, bold'),
                    align='center'
                )
                

                fig.update_layout(**{
                    f'xaxis{idx+1}': dict(
                        domain=x_domain,
                        range=[x_min, x_max],
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False
                    ),
                    f'yaxis{idx+1}': dict(
                        domain=y_domain,
                        range=[y_min, y_max],
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False
                    )
                })
            

            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='rgba(0,0,255,0.8)', width=4),
                name='Positive Correlation'
            ))
            
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='rgba(255,0,0,0.8)', width=4),
                name='Negative Correlation'
            ))
            

            fig.update_layout(
                title='Element Correlation Networks by Sample',
                height=300 * rows,
                width=600 * cols,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.1,
                    xanchor="center",
                    x=0.5
                ),
                plot_bgcolor='rgb(248,248,248)'
            )
        
        elif display_mode == "Differential Network":

            

            if len(selected_samples) > 2:
                col1, col2 = st.columns(2)
                with col1:
                    reference_sample = st.selectbox(
                        "Select reference sample:",
                        options=selected_samples,
                        index=0,
                        key="ref_sample"
                    )
                
                with col2:

                    comparison_samples = st.multiselect(
                        "Select comparison samples:",
                        options=[s for s in selected_samples if s != reference_sample],
                        default=[s for s in selected_samples if s != reference_sample][:min(2, len(selected_samples)-1)],
                        key="comp_samples"
                    )
                    
                if not comparison_samples:
                    st.warning("Please select at least one comparison sample.")
                    return
            else:

                reference_sample = selected_samples[0]
                comparison_samples = [selected_samples[1]]
            


            ref_corr = sample_correlations[reference_sample]
            

            comp_corrs = [sample_correlations[sample] for sample in comparison_samples]
            

            common_elements = sorted(list(set(ref_corr.columns) & set().union(*[corr.columns for corr in comp_corrs])))
            
            if not common_elements:
                st.error("No common elements between the selected samples.")
                return
            

            ref_corr = ref_corr.loc[common_elements, common_elements]
            aligned_comp_corrs = []
            
            for corr in comp_corrs:
                aligned_comp_corrs.append(corr.loc[common_elements, common_elements])
            

            if aligned_comp_corrs:
                avg_comp_corr = sum(aligned_comp_corrs) / len(aligned_comp_corrs)
                

                diff_corr = avg_comp_corr - ref_corr
                

                ref_abundance = sample_element_counts[reference_sample]
                comp_abundance = {element: 0 for element in common_elements}
                
                for sample in comparison_samples:
                    for element in common_elements:
                        if element in sample_element_counts[sample]:
                            comp_abundance[element] += sample_element_counts[sample][element]
                

                for element in comp_abundance:
                    comp_abundance[element] /= len(comparison_samples)
                

                abundance_change = {element: comp_abundance.get(element, 0) - ref_abundance.get(element, 0)
                                for element in common_elements}
                

                G = nx.Graph()
                

                for element in common_elements:
                    G.add_node(element, 
                            ref_abundance=ref_abundance.get(element, 0),
                            comp_abundance=comp_abundance.get(element, 0),
                            change=abundance_change.get(element, 0))
                

                for i, elem1 in enumerate(common_elements):
                    for j, elem2 in enumerate(common_elements):
                        if i < j:  # Avoid duplicates
                            ref_value = ref_corr.loc[elem1, elem2]
                            comp_value = avg_comp_corr.loc[elem1, elem2]
                            diff_value = diff_corr.loc[elem1, elem2]
                            

                            if (abs(ref_value) >= min_correlation or abs(comp_value) >= min_correlation):
                                G.add_edge(elem1, elem2, 
                                        ref_corr=ref_value,
                                        comp_corr=comp_value,
                                        diff=diff_value)
                

                pos = nx.spring_layout(G, seed=42, k=0.5)
                

                fig = go.Figure()
                

                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    

                    ref_corr = edge[2]['ref_corr']
                    comp_corr = edge[2]['comp_corr']
                    diff = edge[2]['diff']
                    


                    if abs(diff) >= 0.1:  # Significant difference
                        if diff > 0:
                            color = f'rgba(0,0,255,{min(1, abs(diff)*1.5)})'  # Blue, stronger correlation in comparison
                            dash = 'solid'
                        else:
                            color = f'rgba(255,0,0,{min(1, abs(diff)*1.5)})'  # Red, weaker correlation in comparison
                            dash = 'solid'
                        
                        width = 1 + 4 * abs(diff)  # Width by difference magnitude
                    else:
                        color = 'rgba(100,100,100,0.5)'  # Gray, similar correlation
                        width = 1
                        dash = 'dot'
                    
                    edge_trace = go.Scatter(
                        x=[x0, x1, None], 
                        y=[y0, y1, None],
                        line=dict(width=width, color=color, dash=dash),
                        hoverinfo='text',
                        text=f"{edge[0]} — {edge[1]}<br>Reference: {ref_corr:.2f}<br>Comparison: {comp_corr:.2f}<br>Difference: {diff:.2f}",
                        mode='lines',
                        showlegend=False
                    )
                    fig.add_trace(edge_trace)
                

                node_x = []
                node_y = []
                node_text = []
                node_size = []
                node_color = []
                
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    

                    avg_abundance = (G.nodes[node]['ref_abundance'] + G.nodes[node]['comp_abundance']) / 2
                    size = 20 + avg_abundance * 1.5  # Scale for visibility
                    node_size.append(size)
                    

                    change = G.nodes[node]['change']
                    if change > 0:
                        node_color.append('rgba(0,255,0,0.7)')  # Green for increase
                    elif change < 0:
                        node_color.append('rgba(255,0,0,0.7)')  # Red for decrease
                    else:
                        node_color.append('rgba(100,100,100,0.7)')  # Gray for no change
                    

                    node_text.append(f"{node}<br>Reference: {G.nodes[node]['ref_abundance']:.1f}%<br>" +
                                f"Comparison: {G.nodes[node]['comp_abundance']:.1f}%<br>" +
                                f"Change: {change:+.1f}%")
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=list(G.nodes()),
                    textposition="top center",
                    textfont=dict(size=12, color='black'),
                    marker=dict(
                        size=node_size,
                        color=node_color,
                        line=dict(width=2, color='black')
                    ),
                    hoverinfo='text',
                    hovertext=node_text
                )
                fig.add_trace(node_trace)
                


                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='lines',
                    line=dict(color='rgba(0,0,255,0.8)', width=4),
                    name='Stronger correlation in comparison'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='lines',
                    line=dict(color='rgba(255,0,0,0.8)', width=4),
                    name='Weaker correlation in comparison'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='lines',
                    line=dict(color='rgba(100,100,100,0.5)', width=1, dash='dot'),
                    name='Similar correlation'
                ))
                

                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(color='rgba(0,255,0,0.7)', size=15, line=dict(width=2, color='black')),
                    name='Higher abundance in comparison'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(color='rgba(255,0,0,0.7)', size=15, line=dict(width=2, color='black')),
                    name='Lower abundance in comparison'
                ))
                

                title_text = f"Differential Element Correlation: "
                if len(comparison_samples) == 1:
                    title_text += f"{comparison_samples[0]} vs {reference_sample}"
                else:
                    title_text += f"{len(comparison_samples)} samples vs {reference_sample}"
                    
                fig.update_layout(
                    title=title_text,
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=800,
                    width=900,
                    plot_bgcolor='rgb(248,248,248)',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.15,
                        xanchor="center",
                        x=0.5
                    )
                )
            else:
                st.error("Error processing comparison correlations.")
                return
        
        elif display_mode == "Animation":

            

            common_elements = set(all_elements)
            for sample_name in selected_samples:
                if sample_name in sample_correlations:
                    common_elements &= set(sample_correlations[sample_name].columns)
            
            common_elements = sorted(list(common_elements))
            
            if not common_elements:
                st.error("No common elements across all selected samples.")
                return
            

            all_G = nx.Graph()
            

            for element in common_elements:

                avg_abundance = sum(sample_element_counts[sample].get(element, 0) 
                                for sample in selected_samples) / len(selected_samples)
                all_G.add_node(element, avg_abundance=avg_abundance)
            

            for sample_name in selected_samples:
                corr_matrix = sample_correlations[sample_name]
                
                for i, elem1 in enumerate(common_elements):
                    for j, elem2 in enumerate(common_elements):
                        if i < j:  # Avoid duplicates
                            if elem1 in corr_matrix.columns and elem2 in corr_matrix.columns:
                                corr = corr_matrix.loc[elem1, elem2]
                                if abs(corr) >= min_correlation and (show_negative or corr >= 0):

                                    if not all_G.has_edge(elem1, elem2):
                                        all_G.add_edge(elem1, elem2)
            

            pos = nx.spring_layout(all_G, seed=42, k=0.5)
            

            frames = []
            
            for sample_name in selected_samples:
                if sample_name in sample_correlations:
                    corr_matrix = sample_correlations[sample_name]
                    element_counts = sample_element_counts[sample_name]
                    

                    frame_data = []
                    

                    for i, elem1 in enumerate(common_elements):
                        for j, elem2 in enumerate(common_elements):
                            if i < j and all_G.has_edge(elem1, elem2):
                                x0, y0 = pos[elem1]
                                x1, y1 = pos[elem2]
                                

                                if (elem1 in corr_matrix.columns and 
                                    elem2 in corr_matrix.columns):
                                    corr = corr_matrix.loc[elem1, elem2]
                                    if abs(corr) >= min_correlation and (show_negative or corr >= 0):

                                        if corr < 0:
                                            color = f'rgba(255,0,0,{abs(corr)})'  # Red
                                        else:
                                            color = f'rgba(0,0,255,{corr})'  # Blue
                                        
                                        width = 1 + 4 * abs(corr)
                                        visible = True
                                    else:
                                        color = 'rgba(200,200,200,0.1)'  # Very light gray
                                        width = 1
                                        visible = 'legendonly'
                                else:
                                    color = 'rgba(200,200,200,0.1)'
                                    width = 1
                                    visible = 'legendonly'
                                
                                edge_trace = go.Scatter(
                                    x=[x0, x1, None], 
                                    y=[y0, y1, None],
                                    line=dict(width=width, color=color),
                                    hoverinfo='text',
                                    text=f"{elem1} — {elem2}: {corr:.2f}" if 'corr' in locals() else "",
                                    mode='lines',
                                    showlegend=False,
                                    visible=visible
                                )
                                frame_data.append(edge_trace)
                    

                    node_x = []
                    node_y = []
                    node_text = []
                    node_size = []
                    
                    for node in common_elements:
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        

                        abundance = element_counts.get(node, 0)
                        size = 20 + abundance * 1.5  # Scale for visibility
                        node_size.append(size)
                        

                        node_text.append(f"{node}: {abundance:.1f}%")
                    
                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=common_elements,
                        textposition="top center",
                        textfont=dict(size=12, color='black'),
                        marker=dict(
                            size=node_size,
                            color=[element_counts.get(node, 0) for node in common_elements],
                            colorscale='Viridis',
                            line=dict(width=2, color='black')
                        ),
                        hoverinfo='text',
                        hovertext=node_text,
                        showlegend=False
                    )
                    frame_data.append(node_trace)
                    

                    frames.append(go.Frame(
                        data=frame_data,
                        name=sample_name
                    ))
            

            fig = go.Figure(
                data=frames[0].data,
                frames=frames
            )
            

            sliders = [{
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 16},
                    'prefix': 'Sample: ',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 500, 'easing': 'cubic-in-out'},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': [
                    {
                        'args': [
                            [sample_name],
                            {
                                'frame': {'duration': 500, 'redraw': True},
                                'mode': 'immediate',
                                'transition': {'duration': 500}
                            }
                        ],
                        'label': sample_name,
                        'method': 'animate'
                    }
                    for sample_name in selected_samples
                ]
            }]
            

            updatemenus = [{
                'buttons': [
                    {
                        'args': [
                            None,
                            {
                                'frame': {'duration': 500, 'redraw': True},
                                'fromcurrent': True,
                                'transition': {'duration': 500, 'easing': 'quadratic-in-out'}
                            }
                        ],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [
                            [None],
                            {
                                'frame': {'duration': 0, 'redraw': True},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }
                        ],
                        'label': 'Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }]
            

            fig.update_layout(
                title='Animated Element Correlation Network',
                showlegend=True,
                updatemenus=updatemenus,
                sliders=sliders,
                margin=dict(b=100, l=20, r=20, t=50),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=700,
                width=900,
                plot_bgcolor='rgb(248,248,248)'
            )
            

            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='rgba(0,0,255,0.8)', width=4),
                name='Positive Correlation'
            ))
            
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='rgba(255,0,0,0.8)', width=4),
                name='Negative Correlation'
            ))
        

        if display_mode == "Differential Network":
            st.info("""
            **How to interpret the Differential Network:**
            - **Red edges** show correlations that are weaker in the comparison sample(s)
            - **Blue edges** show correlations that are stronger in the comparison sample(s)
            - **Dotted gray edges** show correlations that are similar in both groups
            - **Node colors**: green = higher abundance in comparison, red = lower abundance
            - **Node size** represents average element abundance across samples
            - Hover over nodes and edges for detailed information about the differences
            """)
        elif display_mode == "Side-by-Side Networks":
            st.info("""
            **How to interpret Side-by-Side Networks:**
            - **Blue edges** show positive correlations (elements appear together)
            - **Red edges** show negative correlations (elements rarely occur together)
            - **Edge thickness** indicates correlation strength
            - **Node size** represents element frequency in that specific sample
            - All networks use the same scale for direct comparison
            - Hover over nodes and edges for detailed information
            """)
        elif display_mode == "Animation":
            st.info("""
            **How to interpret the Animated Network:**
            - Use the slider or play button to transition between samples
            - **Edge appearance/disappearance** shows correlation changes between samples
            - **Edge color/thickness changes** show correlation strength differences
            - **Node size changes** indicate differences in element abundance
            - The node positions stay consistent to help track changes
            - Hover over nodes and edges for detailed information
            """)
        

        st.plotly_chart(fig, use_container_width=True)
        

        img_bytes = fig.to_image(format="png", width=1200, height=900, scale=2)
        st.download_button(
            label="Download Network as PNG",
            data=img_bytes,
            file_name=f"element_correlation_comparison.png",
            mime="image/png"
        )
        

        if display_mode == "Differential Network":
            st.subheader("Tabular Comparison of Key Correlations")
            

            comparison_data = []
            

            for i, elem1 in enumerate(common_elements):
                for j, elem2 in enumerate(common_elements):
                    if i < j:  # Only look at unique pairs
                        try:

                            ref_value = float(ref_corr.at[elem1, elem2]) if hasattr(ref_corr, 'at') else 0.0
                            comp_value = float(avg_comp_corr.at[elem1, elem2]) if hasattr(avg_comp_corr, 'at') else 0.0
                            diff_value = comp_value - ref_value
                            

                            if (abs(ref_value) >= min_correlation or abs(comp_value) >= min_correlation) and abs(diff_value) >= 0.1:
                                comparison_data.append({
                                    'Element Pair': f"{elem1} — {elem2}",
                                    'Reference': f"{ref_value:.2f}",
                                    'Comparison': f"{comp_value:.2f}",
                                    'Difference': f"{diff_value:+.2f}"
                                })
                        except Exception:

                            pass
            
            if comparison_data:

                comparison_data.sort(key=lambda x: abs(float(x['Difference'].replace('+', ''))), reverse=True)
                

                comparison_df = pd.DataFrame(comparison_data[:20])  # Top 20
                

                st.dataframe(comparison_df, height=400)
                

                csv = pd.DataFrame(comparison_data).to_csv(index=False)
                st.download_button(
                    label="Download Correlation Comparison as CSV",
                    data=csv,
                    file_name="correlation_comparison.csv",
                    mime="text/csv"
                )
            else:
                st.write("No significant correlation differences found.")
                    
                
    def create_multi_element_radar_chart(data_dict, file_letter_map):
        import numpy as np
        import plotly.graph_objects as go
        
        st.subheader("Multi-Element Radar Chart Comparison")
        

        all_elements = set()
        for filename, data in data_dict.items():
            if data_type == "IsoTrack":
                processed_data = process_isotrack(data['df'])
                if processed_data is not None:
                    mole_percent_data = processed_data['mole_percent']
                else:
                    continue
            else:
                _, _, _, mole_percent_data, _ = process_data(data['df'])
            
            all_elements.update(mole_percent_data.columns)
        

        all_elements = sorted(list(all_elements))
        

        selected_elements = st.multiselect(
            "Select elements to include in radar chart:",
            options=all_elements,
            default=all_elements[:min(10, len(all_elements))],
            key="radar_elements"
        )
        
        if not selected_elements:
            st.warning("Please select at least 3 elements for the radar chart.")
            return
        
        if len(selected_elements) < 3:
            st.warning("Please select at least 3 elements for a meaningful radar chart.")
            return
        

        norm_method = st.radio(
            "Select normalization method:",
            options=["Min-Max Scaling", "Z-Score", "Log Transform", "Rank Transform"],
            key="norm_method"
        )
        

        selected_samples = st.multiselect(
            "Select samples to compare:",
            options=[file_letter_map[filename] for filename in data_dict.keys()],
            default=[file_letter_map[filename] for filename in list(data_dict.keys())[:min(5, len(data_dict))]],
            key="radar_samples"
        )
        
        if not selected_samples:
            st.warning("Please select at least one sample.")
            return
        

        sample_data = {}
        
        for filename, data in data_dict.items():
            sample_name = file_letter_map[filename]
            
            if sample_name not in selected_samples:
                continue
            
            if data_type == "IsoTrack":
                processed_data = process_isotrack(data['df'])
                if processed_data is not None:
                    mole_percent_data = processed_data['mole_percent']
                else:
                    continue
            else:
                _, _, _, mole_percent_data, _ = process_data(data['df'])
            

            sample_values = {}
            sample_std_values = {}
            
            for element in selected_elements:
                if element in mole_percent_data.columns:
                    values = mole_percent_data[element].values
                    values = values[~np.isnan(values)]  # Remove NaN
                    
                    if len(values) > 0:
                        sample_values[element] = np.mean(values)
                        sample_std_values[element] = np.std(values)
                    else:
                        sample_values[element] = 0
                        sample_std_values[element] = 0
                else:
                    sample_values[element] = 0
                    sample_std_values[element] = 0
            
            sample_data[sample_name] = {
                'values': sample_values,
                'std_values': sample_std_values
            }
        

        element_min = {element: float('inf') for element in selected_elements}
        element_max = {element: float('-inf') for element in selected_elements}
        element_mean = {element: 0 for element in selected_elements}
        element_std = {element: 0 for element in selected_elements}
        element_all_values = {element: [] for element in selected_elements}
        

        for sample_name, data in sample_data.items():
            for element, value in data['values'].items():
                element_min[element] = min(element_min[element], value)
                element_max[element] = max(element_max[element], value)
                element_all_values[element].append(value)
        

        for element in selected_elements:
            if element_all_values[element]:
                element_mean[element] = np.mean(element_all_values[element])
                element_std[element] = np.std(element_all_values[element])
        

        normalized_data = {}
        
        for sample_name, data in sample_data.items():
            normalized_values = {}
            
            for element, value in data['values'].items():
                if norm_method == "Min-Max Scaling":

                    if element_max[element] > element_min[element]:
                        normalized_values[element] = (value - element_min[element]) / (element_max[element] - element_min[element])
                    else:
                        normalized_values[element] = 0.5  # Default if all values are the same
                
                elif norm_method == "Z-Score":

                    if element_std[element] > 0:
                        normalized_values[element] = (value - element_mean[element]) / element_std[element]

                        normalized_values[element] = (normalized_values[element] + 3) / 6
                        normalized_values[element] = max(0, min(1, normalized_values[element]))  # Clip to [0, 1]
                    else:
                        normalized_values[element] = 0.5
                
                elif norm_method == "Log Transform":


                    if element_max[element] > 0:
                        log_value = np.log1p(value)
                        log_max = np.log1p(element_max[element])
                        normalized_values[element] = log_value / log_max if log_max > 0 else 0
                    else:
                        normalized_values[element] = 0
                
                elif norm_method == "Rank Transform":

                    sorted_values = sorted(element_all_values[element])
                    rank = sorted_values.index(value) if value in sorted_values else 0
                    normalized_values[element] = rank / (len(sorted_values) - 1) if len(sorted_values) > 1 else 0.5
            
            normalized_data[sample_name] = normalized_values
        

        fig = go.Figure()
        

        colors = px.colors.qualitative.Plotly
        

        for i, sample_name in enumerate(selected_samples):
            if sample_name in normalized_data:
                color = colors[i % len(colors)]
                

                values = [normalized_data[sample_name].get(element, 0) for element in selected_elements]
                

                values.append(values[0])
                labels = selected_elements + [selected_elements[0]]
                

                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=labels,
                    fill='toself',
                    name=sample_name,
                    line=dict(color=color, width=3),
                    fillcolor=f'rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.2)'
                ))
        

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    showticklabels=False,
                    ticks=''
                )
            ),
            showlegend=True,
            title=f"Element Profile Comparison ({norm_method})",
            height=700,
            width=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        

        st.info("""
        **How to interpret this visualization:**
        - Each axis represents one element
        - Values are normalized using the selected method to allow comparison
        - Larger areas indicate higher relative concentrations
        - Similar shapes indicate similar compositional patterns
        - Different normalization methods highlight different aspects of the data
        """)
        

        img_bytes = fig.to_image(format="png", width=1200, height=1200, scale=2)
        st.download_button(
            label="Download Radar Chart as PNG",
            data=img_bytes,
            file_name=f"element_profile_radar.png",
            mime="image/png"
        )
            
    
    if 'data_dict' in globals():
        display_summary_table(data_dict, file_letter_map, st.session_state.dilution_factors, st.session_state.acquisition_times)
        
        default_colors = [
            '#FF6347', '#FFD700', '#FFA500', '#20B2AA', '#00BFFF',
            '#F0E68C', '#E0FFFF', '#AFEEEE', '#DDA0DD', '#FFE4E1',
            '#FAEBD7', '#D3D3D3', '#90EE90', '#FFB6C1', '#FFA07A',
            '#C71585', '#DB7093', '#FF1493', '#FF69B4', '#FFA07A',
            '#FFB6C1', '#87CEEB', '#98FB98', '#FFFFE0', '#FFDAB9',
            '#E6E6FA', '#FFF0F5', '#B0E0E6', '#FFC0CB', '#F5DEB3',
        ]

        st.sidebar.title('Elemental Distribution')

        if st.sidebar.checkbox("Show Mass and Mole", value=False, key='show_mass_mole_counts_distribution'):
            st.sidebar.subheader('Choose combination')
            show_mass = st.sidebar.checkbox('Show Mass Percentages', value=False, key='show_mass')
            show_mole = st.sidebar.checkbox('Show Mole Percentages', value=True, key='show_mole')
            
            sample_filename = next(iter(data_dict))
            df = data_dict[sample_filename]['df']
            
            if data_type == "IsoTrack":
                processed_data = process_isotrack(df)
                if processed_data is not None:
                    mass_data = processed_data['mass']
                    mole_data = processed_data['mole']
                    available_elements = mass_data.columns
                else:
                    st.error("Could not process IsoTrack data")
                    available_elements = []
            else:
                mass_data, _, mole_data, _, _ = process_data(df)
                available_elements = mole_data.columns

            color_map = create_color_map(available_elements, default_colors)
            

            threshold_type = st.sidebar.radio(
                "Threshold type:",
                ["Percentage", "Particle Count"],
                key='threshold_type'
            )
            
            use_particle_threshold = threshold_type == "Particle Count"
            
            if use_particle_threshold:

                particle_count_threshold = st.sidebar.number_input(
                    'Minimum particle count threshold', 
                    min_value=1, 
                    max_value=1000, 
                    value=10, 
                    step=1,
                    key='particle_count_threshold'
                )
                percent_threshold = 0  # Not used
            else:

                percent_threshold = st.sidebar.number_input(
                    'Threshold for Others (%)', 
                    format="%f", 
                    key='threshold_others'
                )
                particle_count_threshold = 0  # Not used
        
            visualize_mass_and_mole_percentages_pie_charts(
                data_dict=data_dict,
                file_letter_map=file_letter_map,
                color_map=color_map,
                percent_threshold=percent_threshold,
                particle_count_threshold=particle_count_threshold,
                show_mass=show_mass,
                show_mole=show_mole,
                use_particle_threshold=use_particle_threshold
            )
                        
        st.sidebar.title('Mass Correlation Analysis')
        perform_mass_correlation = st.sidebar.checkbox('Show Mass Correlation?', key='perform_mass_correlation')

        if perform_mass_correlation:
        
            sample_filename = next(iter(data_dict))
            df = data_dict[sample_filename]['df']
            
            if data_type == "IsoTrack":
                processed_data = process_isotrack(df)
                if processed_data is not None:
                    mass_data = processed_data['mass']
                    available_elements = mass_data.columns.tolist()
                else:
                    st.error("Could not process IsoTrack data")
                    available_elements = []
            else:
                mass_data, _, _, _, _ = process_data(df)
                available_elements = mass_data.columns.tolist()

            st.sidebar.subheader('Plot Parameters')
            x_element = st.sidebar.selectbox('Select element for X-axis:', 
                                        available_elements, 
                                        key='mass_correlation_x')
            y_element = st.sidebar.selectbox('Select element for Y-axis:', 
                                        available_elements,
                                        index=1 if len(available_elements) > 1 else 0,
                                        key='mass_correlation_y')
            
            marker_size = st.sidebar.slider('Marker Size', 
                                        min_value=1, 
                                        max_value=20, 
                                        value=8, 
                                        key='marker_size')
            
            opacity = st.sidebar.slider('Marker Opacity', 
                                    min_value=0.1, 
                                    max_value=1.0, 
                                    value=0.6, 
                                    key='marker_opacity')
            
            st.subheader('Mass Correlation Plot')
            use_log_scale = st.sidebar.checkbox('Use Logarithmic Scale', value=False, key='use_log_scale')
    
            correlation_fig = plot_mass_correlation(data_dict, x_element, y_element, 
                                                file_letter_map, data_type,
                                                marker_size, opacity, use_log_scale)
            st.plotly_chart(correlation_fig, use_container_width=True)
            st.sidebar.markdown("### Download Chart")
            if st.sidebar.button("Generate High Quality PNG"):
                img_bytes = correlation_fig.to_image(
                    format="png",
                    width=1200,
                    height=800,
                    scale=1
                )
                
                st.sidebar.download_button(
                    label="Download PNG",
                    data=img_bytes,
                    file_name="correlation_high_quality.png",
                    mime="image/png",
                    key="download_png"
                )
            
        st.sidebar.title('Mass Distribution Analysis')
        perform_mass_distribution_analysis = st.sidebar.checkbox('Perform Mass Distribution Analysis?', key='perform_mass_distribution_analysis')

        if perform_mass_distribution_analysis:
            sample_filename = next(iter(data_dict))
            df = data_dict[sample_filename]['df']
            
            if data_type == "IsoTrack":
                processed_data = process_isotrack(df)
                if processed_data is not None:
                    mass_data = processed_data['mass']
                else:
                    mass_data = None
            else:
                mass_data, _, _, _, _ = process_data(df)

            if mass_data is not None:
                plot_type = st.sidebar.radio(
                    "Select plot type:",
                    ["Histogram", "Box Plot"],
                    key='mass_plot_type'
                )
            
                if plot_type == "Histogram":
                    element_to_plot = st.sidebar.selectbox('Select an element to view the distribution:', 
                                                        mass_data.columns, 
                                                        key='mass_element_to_plot')
                else: 
                    elements_to_plot = st.sidebar.multiselect('Select elements to view in box plot:', 
                                                        mass_data.columns, 
                                                        default=[mass_data.columns[0]],
                                                        key='mass_elements_to_plot')
                    if not elements_to_plot:
                        st.warning("Please select at least one element for the box plot.")
                        elements_to_plot = [mass_data.columns[0]]
                
                detection_type = st.sidebar.selectbox('Select detection type Mass:', 
                                                    ['All', 'Single', 'Multiple'], 
                                                    index=0, 
                                                    key='mass_detection_type')
                
                bin_size = st.sidebar.slider('Select bin size:', 
                                            min_value=0.001, 
                                            max_value=10.0, 
                                            value=0.01, 
                                            step=0.01, 
                                            key='mass_bin_size', 
                                            disabled=(plot_type=="Box Plot"))
                
                x_max = st.sidebar.slider('Select max value for y-axis (Mass (fg)):', 
                                        min_value=0, 
                                        max_value=1000, 
                                        value=10, 
                                        step=1, 
                                        key='mass_x_max')


                if plot_type == "Histogram":
                    plot_mass_distribution(data_dict, element_to_plot, detection_type, bin_size, x_max, 
                                        "Mass Distribution", file_letter_map, plot_type)
                else:  # Box Plot
                    plot_mass_distribution(data_dict, elements_to_plot, detection_type, bin_size, x_max, 
                                        "Mass Distribution", file_letter_map, plot_type)
            else:
                st.error("Could not process mass data from the selected file.")

        st.sidebar.title('Ternary Diagrams')
        perform_ternary_analysis = st.sidebar.checkbox('Perform Ternary Analysis?', key='perform_ternary_analysis')

        if perform_ternary_analysis:
            sample_filename = next(iter(data_dict))
            df = data_dict[sample_filename]['df']
            

            if data_type == "IsoTrack":
                processed_data = process_isotrack(df)
                if processed_data is not None:
                    mole_data = processed_data['mole']
                else:
                    st.error("Could not process IsoTrack data")
                    mole_data = None
            else:
                _, _, mole_data, _, _ = process_data(df)
                
            if mole_data is not None:
                available_elements = mole_data.columns.tolist()
                elements_to_plot = st.sidebar.multiselect('Select three elements for the ternary plot:', 
                                                        available_elements, 
                                                        default=available_elements[:3], 
                                                        key='ternary_elements_to_plot')
                
                if len(elements_to_plot) == 3:
                    plot_ternary_heatmap(data_dict, elements_to_plot, "Ternary Diagrams", file_letter_map)
                else:
                    st.warning("Please select exactly three elements for the ternary plot.")

        st.sidebar.title("Mole Ratio Analysis")
        perform_mole_ratio_analysis = st.sidebar.checkbox('Perform Mole Ratio Analysis?', key='perform_mole_ratio_analysis')

        if perform_mole_ratio_analysis:
            sample_filename = next(iter(data_dict))
            df = data_dict[sample_filename]['df']
            
          
            if data_type == "IsoTrack":
                processed_data = process_isotrack(df)
                if processed_data is not None:
                    mole_data = processed_data['mole']
                else:
                    st.error("Could not process IsoTrack data")
                    mole_data = None
            else:
                _, _, mole_data, _, _ = process_data(df)
            
            if mole_data is not None:
                elements = mole_data.columns.tolist()
                element1_to_plot = st.sidebar.selectbox('Select Element 1 for the Ratio', 
                                                    elements, 
                                                    key='mole_ratio_element1')
                element2_to_plot = st.sidebar.selectbox('Select Element 2 for the Ratio', 
                                                    elements, 
                                                    index=1 if len(elements) > 1 else 0, 
                                                    key='mole_ratio_element2')
                bin_size = st.sidebar.slider('Bin Size', 
                                        min_value=0.001, 
                                        max_value=25.0, 
                                        value=0.01, 
                                        step=0.01, 
                                        key='mole_ratio_bin_size')
                x_max = st.sidebar.slider('Max X-axis Value', 
                                        min_value=0, 
                                        max_value=1000, 
                                        value=25, 
                                        step=1, 
                                        key='mole_ratio_x_max')
                title = "Mole Ratio Analysis"
                
                plot_mole_ratio_histogram_for_files(data_dict, element1_to_plot, element2_to_plot, bin_size, x_max, title, file_letter_map)
                
                
        st.sidebar.title('Heatmap Analysis')
        perform_heatmap_analysis = st.sidebar.checkbox('Perform Heatmap Analysis?', key='perform_heatmap_analysis')

        if perform_heatmap_analysis:
            display_mode = st.sidebar.radio(
                "Select heatmap display mode:",
                ["Individual Files", "Combined Files"],
                key='heatmap_display_mode'
            )

            colorscale_options = [
                'aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance', 'blackbody', 'bluered', 'blues', 'blugrn',
                'bluyl', 'brbg', 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl', 'darkmint', 'deep',
                'delta', 'dense', 'earth', 'edge', 'electric', 'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens',
                'greys', 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet', 'magenta', 'magma', 'matter',
                'mint', 'mrybm', 'mygbm', 'oranges', 'orrd', 'oryel', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg',
                'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 'puor', 'purd', 'purp', 'purples', 'purpor',
                'rainbow', 'rdbu', 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar', 'spectral', 'speed',
                'sunset', 'sunsetdark', 'teal', 'tealgrn', 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
                'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd'
            ]

            st.sidebar.title('Heatmap Parameters')
            display_numbers = st.sidebar.checkbox("Display Numbers on Heatmap", value=True, key='heatmap_display_numbers')
            font_size = st.sidebar.slider("Font Size for Numbers on Heatmap", min_value=5, max_value=30, value=14, key='heatmap_font_size')
            selected_colorscale = st.sidebar.selectbox('Select a colorscale:', colorscale_options, index=89, key='heatmap_colorscale')

            if display_mode == "Individual Files":
                selected_file = st.sidebar.selectbox('Select file for heatmap:', 
                                                list(data_dict.keys()), 
                                                key='heatmap_selected_file')
                
                combinations, mole_percent_combinations, mass_data, _, mole_data, sd_df = get_combinations_and_related_data(data_dict, selected_file)
                
                if combinations is not None:
        
                    if data_type == "IsoTrack":
                        processed_data = process_isotrack(data_dict[selected_file]['df'])
                        if processed_data is not None:
                            available_elements = processed_data['mole'].columns.tolist()
                        else:
                            st.error("Could not process IsoTrack data")
                            available_elements = []
                    else:
                        _, _, _, mole_percent_data, _ = process_data(data_dict[selected_file]['df'])
                        available_elements = mole_percent_data.columns.tolist()

                    total_combinations = len(mole_percent_combinations)
                    start = st.sidebar.number_input('Start from combination:', min_value=1,
                                                max_value=total_combinations - 1, value=1,  
                                                key='heatmap_start_combination')
                    end = st.sidebar.number_input('End at combination:', min_value=2, 
                                                max_value=total_combinations, 
                                                key='heatmap_end_combination')
                    end = max(end, start + 1)

                    heatmap_df = prepare_heatmap_data(mole_percent_combinations, combinations,
                                                    start, end,
                                                    file_letters=file_letter_map[selected_file],
                                                    combined_mode=False)

                    heatmap_fig = plot_heatmap(heatmap_df, sd_df, selected_colorscale,
                                            display_numbers, font_size)
                    st.plotly_chart(heatmap_fig)

            elif display_mode == "Combined Files":
                combined_mole_percent_combinations = {}
                combined_combinations = {}
                combined_sd_df = pd.DataFrame()

                for filename in data_dict.keys():
                    combinations, mole_percent_combinations, _, _, _, sd_df = get_combinations_and_related_data(data_dict, filename)
                    
                    if combinations is not None:
                        for combo in mole_percent_combinations:
                            new_combo = f"{combo} ({file_letter_map[filename]})"
                            combined_mole_percent_combinations[new_combo] = mole_percent_combinations[combo].fillna(0)
                            combined_combinations[new_combo] = combinations[combo]
                        
                        sd_df = sd_df.fillna(0)
                        sd_df.index = [f"{idx} ({file_letter_map[filename]})" for idx in sd_df.index]
                        combined_sd_df = pd.concat([combined_sd_df, sd_df])

                if combined_combinations:
                    total_combinations = len(combined_mole_percent_combinations)
                    start = st.sidebar.number_input('Start from combination:', min_value=1,
                                                max_value=total_combinations - 1, value=1,  
                                                key='heatmap_start_combination')
                    end = st.sidebar.number_input('End at combination:', min_value=2, 
                                                max_value=total_combinations, 
                                                key='heatmap_end_combination')
                    end = max(end, start + 1)

                    heatmap_df = prepare_heatmap_data(combined_mole_percent_combinations,
                                        combined_combinations,
                                        start, end,
                                        combined_mode=True)


                    heatmap_fig = plot_heatmap(heatmap_df, combined_sd_df, selected_colorscale,
                                            display_numbers, font_size, combined_mode=True)
                    st.plotly_chart(heatmap_fig)
                    
                    if display_mode == "Combined Files" and combined_combinations:
                        show_correlation = st.sidebar.checkbox(
                            "Show Sample Correlation Analysis", 
                            value=False,
                            key='show_sample_correlation'
                        )
                        

                        if show_correlation:
                                st.subheader("Sample Correlation Analysis")
                                

                                correlation_methods = {
                                    'pearson': 'Pearson Correlation (Linear relationship)',
                                    'spearman': 'Spearman Rank Correlation (Monotonic relationship)',
                                    'cosine': 'Cosine Similarity (Pattern similarity)',
                                    'aitchison': 'Aitchison Distance (Compositional data)',
                                    'clustering': 'Hierarchical Clustering Similarity'
                                }
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    selected_method = st.selectbox(
                                        "Select correlation method:",
                                        options=list(correlation_methods.keys()),
                                        format_func=lambda x: correlation_methods[x],
                                        key='correlation_method'
                                    )
                                
                                with col2:

                                    missing_combo_penalty = st.slider(
                                        "Missing combination penalty:",
                                        min_value=0.0,
                                        max_value=1.0,
                                        value=0.5,
                                        step=0.1,
                                        help="Reduces correlation when samples don't share the same element combinations"
                                    )
                                

                                method_explanations = {
                                    'pearson': """
                                        **Pearson Correlation** measures linear relationships between samples.
                                        - Strengths: Detects linear relationships, widely understood
                                        - Limitations: Sensitive to outliers, assumes normal distribution
                                        - When to use: For normally distributed data with linear relationships
                                    """,
                                    'spearman': """
                                        **Spearman Rank Correlation** measures monotonic relationships between samples.
                                        - Strengths: Robust against outliers, works with non-linear data
                                        - Limitations: Only detects monotonic relationships
                                        - When to use: When data isn't normally distributed or contains outliers
                                    """,
                                    'cosine': """
                                        **Cosine Similarity** measures the cosine of the angle between sample vectors.
                                        - Strengths: Focuses on pattern similarity regardless of magnitude
                                        - Limitations: Doesn't account for compositional nature of data
                                        - When to use: When pattern of elements matters more than their absolute values
                                    """,
                                    'aitchison': """
                                        **Aitchison Distance** is designed specifically for compositional data.
                                        - Strengths: Mathematically appropriate for percentage/compositional data
                                        - Limitations: More complex to interpret
                                        - When to use: When analyzing true compositional data (percentages that sum to 100%)
                                    """,
                                    'clustering': """
                                        **Hierarchical Clustering Similarity** groups samples by overall similarity.
                                        - Strengths: Focuses on natural groupings in the data
                                        - Limitations: Distance-based rather than true correlation
                                        - When to use: When identifying groups of similar samples is the main goal
                                    """
                                }
                                
                                with st.expander("About this correlation method", expanded=True):
                                    st.markdown(method_explanations[selected_method])
                                    
                                    if missing_combo_penalty > 0:
                                        st.markdown(f"""
                                        **Missing Combination Penalty ({missing_combo_penalty:.1f})**: Reduces correlation scores between samples that don't 
                                        share the same element combinations. Higher values apply a stronger penalty for dissimilar combination profiles.
                                        """)
                                
                                with st.spinner(f"Calculating {correlation_methods[selected_method]}..."):
                             
                                    correlation_df, sample_list, title_suffix = calculate_sample_correlations(
                                        combined_mole_percent_combinations, 
                                        file_letter_map,
                                        method=selected_method,
                                        missing_penalty=missing_combo_penalty
                                    )
                                    
                                    
                                    correlation_fig = plot_sample_correlation_matrix(
                                        correlation_df,
                                        title_suffix=title_suffix
                                    )
                                    
                                    st.plotly_chart(correlation_fig)
                                    
                              
                                st.subheader("Element Combination Analysis")
                                
                                show_combination_heatmap = st.checkbox("Show Element Combination Similarity Heatmap", value=True)
                                
                                if show_combination_heatmap:
                                 
                                    st.markdown("### Combination Heatmap Parameters")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                  
                                        n_pairs = st.slider("Number of sample pairs to show", 
                                                        min_value=3, max_value=50, value=10, 
                                                        key="n_pairs")
                                        
                                
                                        n_combinations = st.slider("Maximum number of combinations to show",
                                                            min_value=5, max_value=100, value=15,
                                                            key="n_combinations")

                                    with col2:
                                     
                                        sort_options = st.radio("Sort combinations by:",
                                                            ["Sample Frequency", "Particle Count", "Correlation Contribution"],
                                                            key="sort_options",
                                                            help="Choose how to order the element combinations in the heatmap")
                                        

                                        one_sided_penalty = st.slider(
                                            "One-sided combination penalty:",
                                            min_value=-1.0,
                                            max_value=0.0,
                                            value=-0.8,
                                            step=0.1,
                                            help="Penalty applied when only one sample has a particular element combination"
                                        )

                                  
                                    sort_by_count = sort_options == "Particle Count"
                                    sort_by_correlation = sort_options == "Correlation Contribution"

                                  
                                    combo_diff_fig, all_elements = create_combination_difference_heatmap(
                                        combined_mole_percent_combinations,
                                        correlation_df,
                                        n_pairs=n_pairs,
                                        max_combinations=n_combinations,
                                        sort_by_count=sort_by_count,
                                        sort_by_correlation=sort_by_correlation,
                                        highlighted_elements=[],
                                        correlation_method=selected_method,
                                        missing_combination_penalty=one_sided_penalty
                                    )

                                    
                                    highlighted_elements = st.sidebar.multiselect(
                                        "Highlight elements in combinations:", 
                                        options=all_elements,
                                        help="Selected elements will be highlighted with '*' and yellow background"
                                    )

                                  
                                    if highlighted_elements:
                                        combo_diff_fig, _ = create_combination_difference_heatmap(
                                            combined_mole_percent_combinations,
                                            correlation_df,
                                            n_pairs=n_pairs,
                                            max_combinations=n_combinations,
                                            sort_by_count=sort_by_count,
                                            sort_by_correlation=sort_by_correlation,
                                            highlighted_elements=highlighted_elements,
                                            correlation_method=selected_method,
                                            missing_combination_penalty=one_sided_penalty
                                        )

                                    
                                    st.plotly_chart(combo_diff_fig)
                                    
                                   
                                    if st.button("Download Combination Similarity Heatmap as PNG"):
                                        img_bytes = combo_diff_fig.to_image(
                                            format="png",
                                            width=1200,
                                            height=1000,
                                            scale=4
                                        )
                                        
                                        st.download_button(
                                            label="Click to Download PNG",
                                            data=img_bytes,
                                            file_name="combination_similarity_heatmap.png",
                                            mime="image/png",
                                            key="combo_diff_heatmap_png"
                                        )
                                   
                                    try:
                                        diff_data = combo_diff_fig.data[0]
                                        if hasattr(diff_data, 'z') and hasattr(diff_data, 'x') and hasattr(diff_data, 'y'):
                                            diff_df = pd.DataFrame(
                                                diff_data.z,
                                                columns=diff_data.x,
                                                index=diff_data.y
                                            )
                                            
                                            st.download_button(
                                                label="Download Combination Similarity Data (CSV)",
                                                data=diff_df.to_csv(),
                                                file_name="combination_similarities.csv",
                                                mime='text/csv',
                                                key="combo_diff_csv"
                                            )
                                    except Exception as e:
                                        st.warning(f"Could not extract combination similarity data: {e}")
                                                                                                                                            
        st.sidebar.title("Single and Multiple Element Analysis")
        perform_single_multiple_analysis = st.sidebar.checkbox('Compare Single vs Multiple Element Particles?', key='perform_single_multiple_analysis')

        if perform_single_multiple_analysis:
            analyze_single_multiple_element(data_dict, file_letter_map)
                                                
                    
        is_isotrack = data_type == "IsoTrack"
    
        if is_isotrack:
            st.sidebar.title('Diameter Analysis (IsoTrack)')
            perform_diameter_analysis = st.sidebar.checkbox('Perform Diameter Analysis?', key='perform_diameter_analysis')

            if perform_diameter_analysis:
                sample_filename = next(iter(data_dict))
                processed_data = process_isotrack(data_dict[sample_filename]['df'])
                
                if processed_data is not None and 'diameter' in processed_data:
                    diameter_data = processed_data['diameter']
                    available_elements = diameter_data.columns.tolist()
                    
                    element_to_plot = st.sidebar.selectbox('Select an element to view the diameter distribution:', 
                                                        available_elements, 
                                                        key='diameter_element_to_plot')
                    
                    bin_size = st.sidebar.slider('Select bin size (nm):', 
                                                min_value=0.1, 
                                                max_value=50.0, 
                                                value=1.0, 
                                                step=0.1, 
                                                key='diameter_bin_size')
                    
                    x_min = st.sidebar.slider('Select min value for x-axis (nm):', 
                                            min_value=0, 
                                            max_value=500, 
                                            value=10, 
                                            step=1, 
                                            key='diameter_x_min')
                    
                    x_max = st.sidebar.slider('Select max value for x-axis (nm):', 
                                            min_value=x_min + 10, 
                                            max_value=1000, 
                                            value=50, 
                                            step=1, 
                                            key='diameter_x_max')

                    plot_diameter_distribution(data_dict, element_to_plot, bin_size, x_min, x_max, 
                                            "Diameter Distribution", file_letter_map)
                else:
                    st.error("No diameter data found in the IsoTrack files.")
                    
                    
        st.sidebar.title('Download Filtered Data')
        if 'data_dict' in globals() and data_dict:
            download_data = st.sidebar.checkbox('Download specific data types', key='download_specific_data')
            
            if download_data:
           
                files_to_export = st.sidebar.multiselect(
                    'Select files to export:',
                    options=list(data_dict.keys()),
                    default=list(data_dict.keys())[0:1],  
                    format_func=lambda x: file_letter_map.get(x, x),
                    key='files_to_export'
                )
         
                data_types_to_export = st.sidebar.multiselect(
                    'Select data types to export:',
                    options=['Mass Data', 'Mole Data', 'Mass Percent Data', 'Mole Percent Data', 'Counts Data'],
                    default=['Mass Data', 'Mole Data'],
                    key='data_types_to_export'
                )
                
                element_filter = st.sidebar.checkbox('Filter by specific elements', key='element_filter')
                selected_elements = []
                
                if element_filter and data_types_to_export:
                  
                    available_elements = set()
                    for filename in files_to_export:
                        data = data_dict[filename]
                        for data_type_name in ['mass_data', 'mole_data', 'mass_percent_data', 'mole_percent_data', 'counts_data']:
                            if data_type_name.replace('_', ' ').title() in data_types_to_export and data[data_type_name] is not None:
                                available_elements.update(data[data_type_name].columns)
                                break
                    
                 
                    selected_elements = st.sidebar.multiselect(
                        'Select elements to include:',
                        options=sorted(list(available_elements)),
                        default=sorted(list(available_elements))[:5] if len(available_elements) > 5 else sorted(list(available_elements)),
                        key='elements_to_export'
                    )
                
         
                st.sidebar.subheader("Download Files")
                for filename in files_to_export:
                    data = data_dict[filename]
                    file_label = file_letter_map.get(filename, filename)
                    
                 
                    file_container = st.sidebar.container()
                    file_container.markdown(f"**{file_label}**")
                    
                 
                    file_dfs = []
                    
                    for data_type_name in ['mass_data', 'mole_data', 'mass_percent_data', 'mole_percent_data', 'counts_data']:
                        display_name = data_type_name.replace('_', ' ').title()
                        if display_name in data_types_to_export and data[data_type_name] is not None:
                            df = data[data_type_name]
                            

                            if element_filter and selected_elements:
                                selected_cols = [col for col in df.columns if col in selected_elements]
                                if not selected_cols:  # Skip if no selected columns for this data type
                                    continue
                                df = df[selected_cols]
                            
                            if not df.empty:
                                file_dfs.append((display_name, df))
                    

                    for data_type_label, df in file_dfs:
                        csv = df.to_csv(index=False)
                        
                        file_container.download_button(
                            label=f"Download {data_type_label}",
                            data=csv,
                            file_name=f"{file_label}_{data_type_label.replace(' ', '_')}.csv",
                            mime="text/csv",
                            key=f"download_{filename}_{data_type_label}"
                        )
                    
              
                    file_container.markdown("---")
                    
        st.sidebar.title('Advanced Visualizations')
        show_advanced_vis = st.sidebar.checkbox('Show Advanced Visualizations', key='show_advanced_vis')

        if show_advanced_vis and 'data_dict' in globals() and data_dict:
            vis_type = st.sidebar.selectbox(
                "Select visualization type:",
                ["Element Correlation Network", 
                "Particle Size-Composition Chart",
                "Multi-Element Radar Chart"],
                key='adv_vis_type'
            )
            
            if vis_type == "Element Correlation Network":
                create_comparative_element_correlation_network(data_dict, file_letter_map)
            elif vis_type == "Multi-Element Radar Chart":
                create_multi_element_radar_chart(data_dict, file_letter_map)

                                                        
                                        






with tab3:
    data_type = st.radio(
        "Select data type:",
        ["NuQuant", "IsoTrack"],
        key='data_type_tab3'
    )

    ff = st.file_uploader(':file_folder: Upload a file', type=['csv'], key='isotopic')
    
    if ff:
        filename = ff.name
        st.write('Uploaded File: ', filename)

        try:
            if 'csv' in filename:
                cleaned_file_content = preprocess_csv_file(ff)
                dd = pd.read_csv(StringIO(cleaned_file_content))
            else:
                st.error('File format not supported. Please upload a CSV file.')
                dd = None

            if dd is not None:
                dilution_factor = st.number_input(
                    'Enter Dilution Factor:', 
                    format="%f", 
                    value=1.0, 
                    key='dilution_factor_tab3'
                )
                acquisition_time = st.number_input(
                    'Enter Total Acquisition Time (in seconds):', 
                    format="%f", 
                    value=60.0, 
                    key='acquisition_time_tab3'
                )

                if data_type == "IsoTrack":
                    try:
                        processed_data = process_isotrack(dd)
                        if processed_data is not None:
                            mass_data = processed_data['mass']
                            mole_data = processed_data['mole']
                            counts_data = processed_data['counts']
                            event_number_cell = processed_data['particle_count']
                            transport_rate = processed_data['transport_rate']
                        else:
                            st.error("Could not process IsoTrack data")
                            mass_data = None
                            mole_data = None
                            counts_data = None
                            event_number_cell = None
                            transport_rate = None
                    except Exception as e:
                        st.error(f"Error processing IsoTrack data: {str(e)}")
                        mass_data = None
                        mole_data = None
                        counts_data = None
                        event_number_cell = None
                        transport_rate = None
                else:
                    try:
                        mass_data, _, mole_data, _, counts_data = process_data(dd)
                        event_number_cell = count_rows_after_keyword_until_no_data(dd, 'event number', column_index=0)
                        transport_rate_cell = find_value_at_keyword(dd, 'calibrated transport rate', column_index=1)
                        transport_rate = extract_numeric_value_from_string(str(transport_rate_cell))
                    except Exception as e:
                        st.error(f"Error processing NuQuant data: {str(e)}")
                        mass_data = None
                        mole_data = None
                        counts_data = None
                        event_number_cell = None
                        transport_rate = None


        except Exception as e:
            st.error(f'An error occurred while processing the file: {str(e)}')
            dd = None
    
    
    
    
    def preprocess_csv_file(ff):
        lines = ff.getvalue().decode('utf-8').splitlines()
        max_fields = max([line.count(',') for line in lines]) + 1
        cleaned_lines = []

        for line in lines:
            fields = line.split(',')
            if len(fields) < max_fields:
                fields.extend([''] * (max_fields - len(fields)))
            cleaned_lines.append(','.join(fields))
        
        cleaned_file_content = "\n".join(cleaned_lines)
        return cleaned_file_content

    def count_rows_after_keyword_until_no_data(data, keyword, column_index=1):
        keyword_found = False
        count = 0
        for value in data.iloc[:, column_index]:
            if keyword_found:
                if pd.notna(value):
                    count += 1
                else:
                    break
            elif keyword.lower() in str(value).lower():
                keyword_found = True
        return count

    def extract_numeric_value_from_string(s):
        match = re.search(r"(\d+(\.\d+)?)", s)
        return float(match.group(1)) if match else None

    def find_value_at_keyword(data, keyword, column_index=1):
        for value in data.iloc[:, column_index]:
            if keyword.lower() in str(value).lower():
                return extract_numeric_value_from_string(str(value))
        return None

    def calculate_particles_per_ml(event_number, q_plasma, acquisition_time, dilution_factor):
        try:
            value = (float(event_number) * dilution_factor) / ((float(q_plasma) / 1000) * (float(acquisition_time)))
            return f"{value:.2e}"
        except ValueError:
            return None


    def find_start_index(dd, keyword, column_index=0):
        for i, value in enumerate(dd.iloc[:, column_index]):
            if keyword.lower() in str(value).lower():
                return i
        return None
    
    def extract_sensitivity_calibration(dd):
        """
        extract sensitivity calibration information for isotopes from the dataframe.
        Returns a dictionary with isotope sensitivities.
        """
        sensitivity_dict = {}
        sensitivity_start = None
        
        for i, row in dd.iterrows():
            if any('sensitivity' in str(val).lower() for val in row):
                sensitivity_start = i
                break
        
        if sensitivity_start is None:
            return None
        current_row = sensitivity_start + 1
        while current_row < len(dd):
            row = dd.iloc[current_row]
            
     
            if pd.isna(row.iloc[0]) and pd.isna(row.iloc[1]):
                break
  
            isotope = str(row.iloc[0])
            if any(char.isdigit() for char in isotope):  
                sensitivity = row.iloc[1]  
                if pd.notna(sensitivity):
                    try:
                        sensitivity = float(sensitivity)
                        sensitivity_dict[isotope.strip()] = sensitivity
                    except (ValueError, TypeError):
                        pass
            
            current_row += 1

        
        return sensitivity_dict

    @st.cache_data
    @st.cache_resource
    def process_data(dd, keyword='event number'):
        start_index = find_start_index(dd, keyword)
        if start_index is not None:
            new_header = dd.iloc[start_index]
            data = dd.iloc[start_index + 1:].reset_index(drop=True)
            data.columns = [str(col) for col in new_header]

            mass_data_cols = [col for col in data.columns if
                            'mass' in col and 'total' not in col and not col.endswith('mass %')]
            mole_data_cols = [col for col in data.columns if
                            'mole' in col and 'total' not in col and not col.endswith('mole %')]
            count_cols = [col for col in data.columns if col.endswith('counts')]


            mass_data = data[mass_data_cols].rename(columns=lambda x: x.split(' ')[0])
            mole_data = data[mole_data_cols].rename(columns=lambda x: x.split(' ')[0])
            counts_data = data[count_cols].rename(columns=lambda x: x.split(' ')[0])

            mass_data = mass_data.apply(pd.to_numeric, errors='coerce').fillna(0)
            counts_data = counts_data.apply(pd.to_numeric, errors='coerce').fillna(0)
            mole_data = mole_data.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            mass_data = mass_data.loc[~(mass_data == 0).all(axis=1)]
            mole_data = mole_data.loc[~(mole_data == 0).all(axis=1)]
            counts_data = counts_data.loc[~(counts_data == 0).all(axis=1)]

            mass_percent_data = mass_data.div(mass_data.sum(axis=1), axis=0) * 100
            mass_percent_data = mass_percent_data.rename(columns=lambda x: x.split(' ')[0])

            mole_percent_data = mole_data.div(mole_data.sum(axis=1), axis=0) * 100
            mole_percent_data = mole_percent_data.rename(columns=lambda x: x.split(' ')[0])

            return mass_data, mass_percent_data, mole_data, mole_percent_data, counts_data
        else:
            return None, None, None, None, None
        
        
    @st.cache_data
    @st.cache_resource
    def clean_data(dd, keyword='event number'):
        start_index = find_start_index(dd, keyword)

        if start_index is not None:
            new_header = dd.iloc[start_index]
            data = dd.iloc[start_index + 1:].reset_index(drop=True)
            data.columns = [str(col) for col in new_header]

            def make_unique(column_names):
                counts = {}
                for i, col in enumerate(column_names):
                    if col in counts:
                        counts[col] += 1
                        column_names[i] = f"{col}_{counts[col]}"
                    else:
                        counts[col] = 0
                return column_names

            data.columns = make_unique(data.columns.tolist())



            elements = set(col.split(' ')[0] for col in data.columns if 'fwhm' in col)

            for element in elements:
                count_col = f'{element} counts'
                mole_col = f'{element} moles [fmol]'
                mass_col = f'{element} mass [fg]'
                fwhm_col = f'{element} fwhm'
                mole_per = f'{element} mole %'
                mass_per = f'{element} mass %'

                if all(col in data.columns for col in [count_col, mole_col, mass_col, fwhm_col, mole_per, mass_per]):

                    data.loc[data[fwhm_col].isna(), [count_col, mole_col, mass_col, mass_per, mole_per]] = 0



            cleaned_dd = dd.copy()
            cleaned_dd.iloc[start_index + 1:, :] = data.values

            return cleaned_dd
        else:
            st.error("Header row with 'fwhm' not found.")
            return None
    
    def process_isotrack(df):
        """Process IsoTrack format data with specific structure for counts, mass, moles, and diameter"""
        try:
            transport_rate = None
            for idx, row in df.iterrows():
                if 'Transport Rate:' in str(row.iloc[0]):
                    try:
                        match = re.search(r'(\d+\.?\d*)', str(row.iloc[0]))
                        if match:
                            transport_rate = float(match.group(1))
                    except (ValueError, AttributeError):
                        continue
                    break
                
            start_idx = None
            for idx, row in df.iterrows():
                if 'Particle ID' in str(row.iloc[0]):
                    start_idx = idx
                    break
                    
            if start_idx is None:
                st.write("Could not find 'Particle ID' in data")
                return None
            
            def find_end_of_data(df, start_idx):
                data_section = df.iloc[start_idx + 1:]
                for idx, row in data_section.iterrows():
                    if row.isna().all() or row.astype(str).str.strip().eq('').all():
                        return idx
                return len(df)
            
            end_idx = find_end_of_data(df, start_idx)
                    

            data = df.iloc[start_idx:end_idx].copy()
            data.columns = [str(x).strip() for x in data.iloc[0]]
            data = data.iloc[1:].reset_index(drop=True)
                

            counts_cols = [col for col in data.columns if 'counts' in col]
            fg_cols = [col for col in data.columns if '(fg)' in col and 'Total' not in col and not col.endswith('Mass%')]
            fmol_cols = [col for col in data.columns if '(fmol)' in col and 'Total' not in col and not col.endswith('Mole%')]
            nm_cols = [col for col in data.columns if '(nm)' in col]
            

            counts_data = data[counts_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            mass_data = data[fg_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            mole_data = data[fmol_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            diameter_data = data[nm_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            def clean_element_name(col):

                col = col.split('(')[0].strip()
                return col.strip()


            counts_data.columns = [clean_element_name(col) for col in counts_data.columns]
            mass_data.columns = [clean_element_name(col) for col in mass_data.columns]
            mole_data.columns = [clean_element_name(col) for col in mole_data.columns]
            diameter_data.columns = [clean_element_name(col) for col in diameter_data.columns]

            mole_percent_data = mole_data.div(mole_data.sum(axis=1), axis=0) * 100
            
            return {
                'counts': counts_data,
                'mass': mass_data,
                'mole': mole_data,
                'mole_percent': mole_percent_data,
                'diameter': diameter_data,
                'transport_rate': transport_rate,
                'particle_count': len(data)
            }
                
        except Exception as e:
            st.error(f'Error processing IsoTrack data: {str(e)}')
            return None

    if ff is not None:
        filename = ff.name
        st.write('Uploaded File: ', filename)

        try:
            if 'csv' in filename:
                cleaned_file_content = preprocess_csv_file(ff)
                dd = pd.read_csv(StringIO(cleaned_file_content))
            else:
                st.error('File format not supported. Please upload a CSV or Excel file.')
                dd = None

            if dd is not None:



                dd = clean_data(dd)
                






                dilution_factor = st.number_input('Enter Dilution Factor:', format="%f", value=1.0, key='dilution_tab3')
                acquisition_time = st.number_input('Enter Total Acquisition Time (in seconds):', format="%f", value=60.0, key='acq_time_tab3')

                if data_type == "IsoTrack":
                    processed_data = process_isotrack(dd)
                    if processed_data is not None:
                        event_number_cell = processed_data['particle_count']
                        transport_rate = processed_data['transport_rate']
                        mass_data = processed_data['mass']
                        mole_data = processed_data['mole']
                        counts_data = processed_data['counts']
                    else:
                        st.error("Could not process IsoTrack data")
                        event_number_cell = None
                        transport_rate = None
                        mass_data = None
                        mole_data = None
                        counts_data = None
                else:
                    event_number_cell = count_rows_after_keyword_until_no_data(dd, 'event number', column_index=0)
                    transport_rate_cell = find_value_at_keyword(dd, 'calibrated transport rate', column_index=1)
                    transport_rate = extract_numeric_value_from_string(str(transport_rate_cell))

                    mass_data, _, mole_data, _, counts_data = process_data(dd)

                if event_number_cell is not None:
                    st.write(f'Total Particles Count: {event_number_cell} Particles')
                else:
                    st.write('Event number not found or no valid count available.')

                if transport_rate is not None:
                    st.write(f'Calibrated Transport Rate: {transport_rate} µL/s')
                else:
                    st.write('Calibrated transport rate not found or no valid rate available.')

                if event_number_cell is not None and transport_rate is not None:
                    particles_per_ml = calculate_particles_per_ml(event_number_cell, transport_rate,
                                                                acquisition_time, dilution_factor)
                    if particles_per_ml is not None:
                        st.write(f'Particles per ml: {particles_per_ml} Particles/mL')
                    else:
                        st.write('Error in calculation. Please check input values.')
                else:
                    st.write('Required data not found in file.')

        except Exception as e:
            st.error(f'An error occurred: {e}')
            dd = None
            
    def get_isotopic_abundance(element_symbol):
        el = element(element_symbol)
        isotopes = {f"{iso.mass_number}{el.symbol}": iso.abundance for iso in el.isotopes if iso.abundance is not None}
        return isotopes





    def plot_counts_ratio_histogram(count_data, element1, element2, color, bin_size, x_max, title, 
                                line_x_value=None, line_x_value_2=None, count_threshold=None, 
                                adjust_to_natural=False):
        filtered_data = count_data.dropna(subset=[element1, element2])
        filtered_data = filtered_data[(filtered_data[element1] > 0) & (filtered_data[element2] > 0)]
        
        if count_threshold is not None:
            filtered_data = filtered_data[(filtered_data[element1] >= count_threshold) & 
                                        (filtered_data[element2] >= count_threshold)]
        
        ratios = filtered_data[element1] / filtered_data[element2]
        ratios = ratios.dropna()
        
        if ratios.empty:
            st.error('No valid data available for plotting.')
            return

        adjustment_factor = None
        if line_x_value is not None and line_x_value_2 is not None:
            adjustment_factor = line_x_value / line_x_value_2
            st.write(f"Adjustment factor (Natural/Standard): {adjustment_factor:.4f}")
            
            if adjust_to_natural:
                ratios = ratios * adjustment_factor
                if line_x_value_2 is not None:
                    line_x_value_2 = line_x_value_2 * adjustment_factor
                st.write("Data adjusted to natural abundance scale")

        total_count = len(ratios)

        fig = px.histogram(ratios, x=ratios, nbins=int(x_max / bin_size), title=title,
                        color_discrete_sequence=[color],
                        labels={'x': f"Count Ratio {element1}/{element2}", 'y': 'Frequency'})

        fig.update_traces(marker_line_color='black', 
                        marker_line_width=1.5)

        if line_x_value is not None:
            fig.add_vline(x=line_x_value, line_dash="dot", line_width=5,
                        line_color='blue')
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='blue', dash='dot'),
                showlegend=True,
                name='Natural Abundance'
            ))
                
        if line_x_value_2 is not None:
            fig.add_vline(x=line_x_value_2, line_dash="dot", line_width=5,
                        line_color='green')
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='green', dash='dot'),
                showlegend=True,
                name='Standard Ratio' if not adjust_to_natural else 'Adjusted Standard Ratio'
            ))

        fig.update_layout(
            xaxis_title=f"{element1}/{element2}" + (" (Adjusted to Natural Abundance)" if adjust_to_natural else ""),
            yaxis_title="Frequency",
            xaxis=dict(
                range=[0, x_max],
                title_font=dict(size=35, color='black'),
                tickfont=dict(size=35, color='black'),
                linecolor='black',
                linewidth=3
            ),
            yaxis=dict(
                title_font=dict(size=35, color='black'),
                showgrid=False,
                tickfont=dict(size=35, color='black'),
                linecolor='black',
                linewidth=3
            ),
            legend_title_font=dict(size=24),
            legend_font=dict(size=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        fig.add_annotation(
            x=0.9, y=0.95,
            xref="paper", yref="paper",
            text=f"Total NPs: {total_count}",
            showarrow=False,
            font=dict(size=24, color="black"),
            align="center",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )

        st.plotly_chart(fig)

        summary_data = [[f'File name: {element1}/{element2}, count ratio']]
        if adjustment_factor is not None:
            summary_data.append([f'Adjustment factor (Natural/Standard): {adjustment_factor:.4f}'])
            summary_data.append([f'Data adjusted to natural abundance: {"Yes" if adjust_to_natural else "No"}'])
        summary_data.append([''])  
        
        for value in ratios:
            summary_data.append([value, ''])
        
        max_rows = len(summary_data)
        while len(summary_data) < max_rows:
            summary_data.append(['', ''])

        if summary_data:
            summary_dd = pd.DataFrame(summary_data)
            csv = summary_dd.to_csv(index=False, header=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='count_ratio_histogram_data.csv',
                mime='text/csv'
            )

    def plot_mole_ratio_histogram(mole_data, element1, element2, color, bin_size, x_max, title):

        filtered_data = mole_data.dropna(subset=[element1, element2])
        filtered_data = filtered_data[(filtered_data[element1] > 0) & (filtered_data[element2] > 0)]

        ratios = filtered_data[element1] / filtered_data[element2]
        ratios = ratios.dropna()

        if ratios.empty:
            st.error('No valid data available for plotting.')
            return

        total_count = len(ratios)

        fig = px.histogram(ratios, x=ratios, nbins=int(x_max / bin_size), title=title,
                        color_discrete_sequence=[color],
                        labels={'x': f"Mole Ratio {element1}/{element2}", 'y': 'Frequency'})

        fig.update_traces(marker_line_color='black',  
                        marker_line_width=1.5)

        fig.update_layout(
            xaxis_title=f"{element1}/{element2}",
            yaxis_title="Frequency",
            xaxis=dict(
                range=[0, x_max],
                title_font=dict(size=30, color='black'),
                tickfont=dict(size=30, color='black'),
                linecolor='black',
                linewidth=3
            ),
            yaxis=dict(
                title_font=dict(size=30, color='black'),
                showgrid=False,
                tickfont=dict(size=30, color='black'),
                linecolor='black',
                linewidth=3
            ),
            legend_title_font=dict(size=24),
            legend_font=dict(size=16),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        fig.add_annotation(
            x=0.9, y=0.95,
            xref="paper", yref="paper",
            text=f"Total NPs: {total_count}",
            showarrow=False,
            font=dict(size=24, color="black"),
            align="center",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )

        st.plotly_chart(fig)

        summary_data = [[f'File name: {element1}/{element2}, mole ratio']]
        for value in ratios:
            summary_data.append([value, ''])
        
        max_rows = len(summary_data)
        while len(summary_data) < max_rows:
            summary_data.append(['', ''])

        if summary_data:
            summary_dd = pd.DataFrame(summary_data)
            csv = summary_dd.to_csv(index=False, header=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='mole_ratio_histogram_data.csv',
                mime='text/csv',
                key='isotopic_button'
            )
            
    def get_ratio_confidence_interval(counts_A, counts_B, confidence_level=0.95):
        """
        Calculate confidence intervals for ratio of two Poisson variables (A/B)
        using improved statistical approach from the paper
        """
        ratio = counts_A / counts_B
        z = norm.ppf((1 + confidence_level) / 2)  
        
        std_error = ratio * np.sqrt(1/counts_A + 1/counts_B)
        
        lower = ratio - z * std_error
        upper = ratio + z * std_error
        
        return lower, upper


 



    def plot_isotopic_ratio(counts_data, mass_data, element1, element2, x_axis_element, color, x_max, title, 
                      line_y_value=None, line_y_value_2=None, adjust_to_natural=False,
                      use_counts_x_axis=False, log_x=False, log_y=False):
        """
        Plot isotopic ratio data with Poisson confidence intervals
        """
        counts_data[element1] = pd.to_numeric(counts_data[element1], errors='coerce')
        counts_data[element2] = pd.to_numeric(counts_data[element2], errors='coerce')
        mass_data[element2] = pd.to_numeric(mass_data[element2], errors='coerce')
        mass_data[element1] = pd.to_numeric(mass_data[element1], errors='coerce')

        filtered_counts = counts_data.dropna(subset=[element1, element2])
        filtered_counts = filtered_counts[(filtered_counts[element1] > 0) & (filtered_counts[element2] > 0)]
        
        filtered_mass = mass_data.loc[filtered_counts.index]
        ratios = filtered_counts[element1] / filtered_counts[element2]
        ratios = ratios.dropna()

        if ratios.empty:
            st.error('No valid data available for plotting.')
            return

        x_axis_data = filtered_counts[x_axis_element] if use_counts_x_axis else filtered_mass[x_axis_element]
        x_axis_label = f"Counts of {x_axis_element}" if use_counts_x_axis else f"Mass of {x_axis_element} (fg)"

        adjustment_factor = None
        if line_y_value is not None and line_y_value_2 is not None:
            adjustment_factor = line_y_value / line_y_value_2
            st.write(f"Adjustment factor (Natural/Standard): {adjustment_factor:.4f}")
            
            if adjust_to_natural:
                ratios = ratios * adjustment_factor
                if line_y_value_2 is not None:
                    line_y_value_2 = line_y_value_2 * adjustment_factor
                st.write("Data adjusted to natural abundance scale")

        total_count = len(ratios)
        mean_ratio = ratios.mean()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_axis_data, y=ratios,
            mode='markers', marker=dict(size=12, color=color, line=dict(width=2)),
            name=f"Ratio {element1}/{element2}"
        ))

        num_bins = 50  
        if log_x:
            bins = np.logspace(np.log10(max(x_axis_data.min(), 1e-10)), np.log10(x_max), num=num_bins)
        else:
            bins = np.linspace(0, x_max, num=num_bins)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        poisson_ci = {
            'lower_95': np.zeros(len(bin_centers)),
            'upper_95': np.zeros(len(bin_centers))
        }

        for idx, (b_start, b_end) in enumerate(zip(bins[:-1], bins[1:])):
            bin_mask = (x_axis_data >= b_start) & (x_axis_data < b_end)
            bin_counts = filtered_counts[bin_mask]
            
            if len(bin_counts) > 0:
                try:
                    counts_1 = bin_counts[element1].mean()
                    counts_2 = bin_counts[element2].mean()
                    
                    if counts_1 > 0 and counts_2 > 0:
                        ratio = counts_1 / counts_2
    
                        std_error = ratio * np.sqrt(1/counts_1 + 1/counts_2)
  
                        z = 1.96
                        factor = adjustment_factor if adjust_to_natural else 1
                        poisson_ci['lower_95'][idx] = (ratio - z * std_error) * factor
                        poisson_ci['upper_95'][idx] = (ratio + z * std_error) * factor
                except Exception as e:
                    poisson_ci['lower_95'][idx] = np.nan
                    poisson_ci['upper_95'][idx] = np.nan
            else:
                poisson_ci['lower_95'][idx] = np.nan
                poisson_ci['upper_95'][idx] = np.nan

        valid_mask = ~np.isnan(poisson_ci['lower_95'])
        valid_centers = bin_centers[valid_mask]
        
        if len(valid_centers) > 0:
            valid_data_95_upper = poisson_ci['upper_95'][valid_mask]
            valid_data_95_lower = poisson_ci['lower_95'][valid_mask]
            
            if len(valid_centers) > 2:

                smoothed_upper_95 = lowess(valid_data_95_upper, valid_centers, frac=0.3, it=2)
                smoothed_lower_95 = lowess(valid_data_95_lower, valid_centers, frac=0.3, it=2)
                
                fig.add_trace(go.Scatter(
                    x=smoothed_upper_95[:, 0],
                    y=smoothed_upper_95[:, 1],
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=smoothed_lower_95[:, 0],
                    y=smoothed_lower_95[:, 1],
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),
                    fill='tonexty',
                    fillcolor='rgba(221,160,221,0.5)',
                    name='Poisson 95% CI'
                ))

        if line_y_value is not None:
            x_min = x_axis_data.min() if log_x else 0
            fig.add_trace(go.Scatter(
                x=[x_min, x_max], y=[line_y_value, line_y_value],
                mode='lines', line=dict(dash='dot', width=5, color='blue'),
                name='Natural Abundance'
            ))

        if line_y_value_2 is not None:
            x_min = x_axis_data.min() if log_x else 0
            fig.add_trace(go.Scatter(
                x=[x_min, x_max], y=[line_y_value_2, line_y_value_2],
                mode='lines', line=dict(dash='dot', width=5, color='green'),
                name='Standard Ratio' if not adjust_to_natural else 'Adjusted Standard Ratio'
            ))

        x_min = x_axis_data.min() if log_x else 0
        fig.add_trace(go.Scatter(
            x=[x_min, x_max], y=[mean_ratio, mean_ratio],
            mode='lines', line=dict(dash='solid', width=3, color='red'),
            name='Mean Ratio' + (" (Adjusted)" if adjust_to_natural else "")
        ))

        fig.update_layout(
            title=title,
            xaxis_title=x_axis_label,
            yaxis_title=f"Ratio {element1}/{element2}" + (" (Adjusted to Natural Abundance)" if adjust_to_natural else ""),
            xaxis=dict(
                type='log' if log_x else 'linear',
                range=[np.log10(x_axis_data.min()) if log_x else 0, 
                        np.log10(x_max) if log_x else x_max],
                title_font=dict(size=30, color='black'),
                tickfont=dict(size=30, color='black'),
                linecolor='black',
                linewidth=3
            ),
            yaxis=dict(
                type='log' if log_y else 'linear',
                title_font=dict(size=30, color='black'),
                showgrid=False,
                tickfont=dict(size=30, color='black'),
                linecolor='black',
                linewidth=3
            ),
            legend_title_font=dict(size=24),
            legend_font=dict(size=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        fig.add_annotation(
            x=0.9, y=0.95,
            xref="paper", yref="paper",
            text=f"Total NPs: {total_count}",
            showarrow=False,
            font=dict(size=24, color="black"),
            align="center",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )

        st.plotly_chart(fig)

        summary_data = [['File name: ' + title, '', '', '']]
        if adjustment_factor is not None:
            summary_data.append([f'Adjustment factor (Natural/Standard): {adjustment_factor:.4f}'])
            summary_data.append([f'Data adjusted to natural abundance: {"Yes" if adjust_to_natural else "No"}'])
        summary_data.append([f'{"Counts" if use_counts_x_axis else "Mass (fg)"} {x_axis_element}', f'{element1}/{element2} Ratio', ''])
        
        for x_val, ratio in zip(x_axis_data, ratios):
            summary_data.append([x_val, ratio, ''])
        
        if summary_data:
            summary_dd = pd.DataFrame(summary_data)
            csv = summary_dd.to_csv(index=False, header=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='isotopic_ratio_data.csv',
                mime='text/csv'
            )
            

    def plot_mass_correlation_single(data, x_element, y_element, color, marker_size=8, opacity=0.6, use_log_scale=False, data_type="Mass"):
        """
        Creates a scatter plot showing correlation between two selected elements with regression line and R² value.
        data_type parameter allows switching between mass and counts data.
        """

        valid_data = data[(data[x_element] > 0) & (data[y_element] > 0)]
        
        if valid_data.empty:
            st.error('No valid data available for plotting.')
            return
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=valid_data[x_element],
            y=valid_data[y_element],
            mode='markers',
            name='Data Points',
            marker=dict(
                size=marker_size,
                color=color,
                opacity=opacity,
                line=dict(width=1, color='black')
            )
        ))
   
        x = valid_data[x_element]
        y = valid_data[y_element]
        
        if use_log_scale:
            x = np.log10(x)
            y = np.log10(y)
        
        slope, intercept = np.polyfit(x, y, 1)
        r_squared = np.corrcoef(x, y)[0,1]**2
        

        x_range = np.linspace(min(x), max(x), 100)
        y_range = slope * x_range + intercept
        
        if use_log_scale:
            x_range = 10**x_range
            y_range = 10**y_range
    
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            name=f'Fit',
            line=dict(
                color=color,
                dash='dash'
            )
        ))
     
        fig.add_annotation(
            text=f"R² = {r_squared:.3f}",
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.95,
            showarrow=False,
            font=dict(
                size=20,
                color=color
            ),
            align="left"
        )

        units = "(fg)" if data_type == "Mass" else "(counts)"
        
        fig.update_layout(
            title=dict(
                text=f'{data_type} Correlation: {y_element} vs {x_element}',
                font=dict(size=40, color='black')
            ),
            xaxis_title=dict(
                text=f'{x_element} {units}',
                font=dict(size=40, color='black')
            ),
            yaxis_title=dict(
                text=f'{y_element} {units}',
                font=dict(size=40, color='black')
            ),
            height=800,
            showlegend=True,
            plot_bgcolor='white',
            legend=dict(
                font=dict(size=40, color='black'),
                itemsizing='constant'
            ),
            font=dict(size=40, color='black')
        )
        
        fig.update_xaxes(
            type='log' if use_log_scale else 'linear',
            gridcolor='lightgray',
            zeroline=False,
            tickfont=dict(size=40, color='black'),
            linecolor='black',
            linewidth=2,
            showgrid=False
        )
        
        fig.update_yaxes(
            type='log' if use_log_scale else 'linear',
            gridcolor='lightgray',
            zeroline=False,
            tickfont=dict(size=40, color='black'),
            linecolor='black',
            linewidth=2,
            showgrid=False
        )
        
        st.plotly_chart(fig)

        summary_data = [[f'File name: {data_type} Correlation {x_element} vs {y_element}']]
        summary_data.append([f'{x_element} {units}', f'{y_element} {units}', ''])
        
        for x_val, y_val in zip(valid_data[x_element], valid_data[y_element]):
            summary_data.append([x_val, y_val, ''])
        
        if summary_data:
            summary_dd = pd.DataFrame(summary_data)
            csv = summary_dd.to_csv(index=False, header=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f'{data_type.lower()}_correlation_{x_element}_{y_element}.csv',
                mime='text/csv'
            )
                                        
                
        
    if 'dd' in globals():
            mass_data = None
            mole_data = None 
            counts_data = None

            if data_type == "IsoTrack":
        
                processed_data = process_isotrack(dd)
                if processed_data is not None:
                    mass_data = processed_data['mass']
                    mole_data = processed_data['mole']
                    counts_data = processed_data['counts']
                else:
                    st.error("Could not process IsoTrack data")
            else:
                mass_data, _, mole_data, _, counts_data = process_data(dd)

            if mole_data is not None:
                st.sidebar.title("Mole Ratio")
                perform_mole_ratio_analysis = st.sidebar.checkbox('Perform Mole Ratio Analysis?', key='perform_ratio_analysis')

                if perform_mole_ratio_analysis:
                    element1_to_plot = st.sidebar.selectbox('Select Element 1 for the Ratio', mole_data.columns, key='mole_element1')
                    element2_to_plot = st.sidebar.selectbox('Select Element 2 for the Ratio', mole_data.columns, 
                                                        index=1 if len(mole_data.columns) > 1 else 0, key='mole_element2')
                    color = st.sidebar.color_picker('Pick a Color for the Histogram', '#DCAD7A', key='mole_color')
                    bin_size = st.sidebar.slider('Bin Size', min_value=0.001, max_value=10.0, value=0.01, step=0.01, key='mole_bin_size')
                    x_max = st.sidebar.slider('Max X-axis Value', min_value=0, max_value=100, value=25, step=1, key='mole_x_max')
                    title = "Molar Ratio"

                    plot_mole_ratio_histogram(mole_data, element1_to_plot, element2_to_plot, color, bin_size, x_max, title)

            if counts_data is not None:
                st.sidebar.title("Isotopic Ratio")
                perform_isotopic_ratio_analysis = st.sidebar.checkbox('Perform Isotopic Ratio Analysis?', key='perform_isotopic_ratio_analysis')

                if perform_isotopic_ratio_analysis:

                    sensitivity_dict = extract_sensitivity_calibration(dd)

                    element1_to_plot = st.sidebar.selectbox('Select Element 1 for the Isotopic Ratio', counts_data.columns, key='isotopic_element1')
                    element2_to_plot = st.sidebar.selectbox('Select Element 2 for the Isotopic Ratio', counts_data.columns, 
                                                        index=1 if len(counts_data.columns) > 1 else 0, key='isotopic_element2')

                    x_axis_element = st.sidebar.radio('Select Element for X-axis:', [element1_to_plot, element2_to_plot], 
                                                    key='isotopic_x_axis_element')

                    color = st.sidebar.color_picker('Pick a Color for the Isotopic Histogram', '#BB75C1', key='isotopic_color')
                    x_max = st.sidebar.slider('Max X-axis Value Isotopic', min_value=0, max_value=50000, value=1000, step=1, 
                                            key='isotopic_x_max')
                    title = "Isotopic Ratio"

                    if sensitivity_dict and element1_to_plot in sensitivity_dict and element2_to_plot in sensitivity_dict:
                        line_value_2 = sensitivity_dict[element1_to_plot] / sensitivity_dict[element2_to_plot]
                        st.sidebar.write(f"Standard ratio from sensitivities ({element1_to_plot}/{element2_to_plot}): {line_value_2:.4f}")
                    else:
                        line_value_2 = st.sidebar.number_input('Set Y-axis Line Value Standard', 
                                                            min_value=0.00, max_value=100.00, value=1.00, step=0.10, 
                                                            key='isotopic_line_value_2')
                        if not sensitivity_dict:
                            st.sidebar.write("No sensitivity data found in file")
                        else:
                            st.sidebar.write("Sensitivity data not found for selected isotopes")

                    element1_symbol = ''.join([i for i in element1_to_plot if not i.isdigit()])
                    element2_symbol = ''.join([i for i in element2_to_plot if not i.isdigit()])
                    
                    isotopic_abundances_element1 = get_isotopic_abundance(element1_symbol)
                    isotopic_abundances_element2 = get_isotopic_abundance(element2_symbol)

                    abundance1 = isotopic_abundances_element1.get(element1_to_plot, 'Not found')
                    abundance2 = isotopic_abundances_element2.get(element2_to_plot, 'Not found')

                    if abundance1 != 'Not found' and abundance2 != 'Not found':
                        line_value = abundance1 / abundance2
                        st.sidebar.write(f"Natural abundance ratio ({element1_to_plot}/{element2_to_plot}): {line_value:.4f}")
                    else:
                        line_value = None

                    adjust_to_natural = st.sidebar.checkbox('Adjust Data to Natural Abundance Scale?', 
                                                key='adjust_to_natural_isotopic')
            
                    if line_value is not None and line_value_2 is not None:
                        adjustment_factor = line_value / line_value_2
                        st.sidebar.write(f"Adjustment factor (Natural/Standard): {adjustment_factor:.4f}")

                    use_counts_x_axis = st.sidebar.checkbox('Use Counts for X-axis', False)
                    log_x = st.sidebar.checkbox('Use Logarithmic X-axis', False)
                    log_y = st.sidebar.checkbox('Use Logarithmic Y-axis', False)

                    plot_isotopic_ratio(counts_data, mass_data, element1_to_plot, element2_to_plot, 
                                        x_axis_element, color, x_max, title, 
                                        line_y_value=line_value, line_y_value_2=line_value_2,
                                        adjust_to_natural=adjust_to_natural,
                                        use_counts_x_axis=use_counts_x_axis,
                                        log_x=log_x,
                                        log_y=log_y)

                st.sidebar.title("Ratio Counts")
                perform_counts_ratio_analysis = st.sidebar.checkbox('Perform Counts Ratio Analysis Bars?', key='perform_ratio_counts_analysis')

                if perform_counts_ratio_analysis:
                    sensitivity_dict = extract_sensitivity_calibration(dd)

                    element1_to_plot_counts = st.sidebar.selectbox('Select Element 1 for the Counts Ratio', 
                                                                counts_data.columns, key='counts_element1')
                    element2_to_plot_counts = st.sidebar.selectbox('Select Element 2 for the Counts Ratio', 
                                                                counts_data.columns, 
                                                                index=1 if len(counts_data.columns) > 1 else 0, 
                                                                key='counts_element2')
                    
                    color = st.sidebar.color_picker('Pick a Color for the Histogram counts', '#6ECEB4', key='counts_color')
                    bin_size = st.sidebar.slider('Bin Size counts', min_value=0.001, max_value=10.0, value=0.01, step=0.01, 
                                                key='counts_bin_size')
                    x_max = st.sidebar.slider('Max X-axis Value counts', min_value=0, max_value=100, value=25, step=1, 
                                            key='counts_x_max')
                    title = "Ratio Counts"

                    if sensitivity_dict and element1_to_plot_counts in sensitivity_dict and element2_to_plot_counts in sensitivity_dict:
                        line_value_2 = sensitivity_dict[element1_to_plot_counts] / sensitivity_dict[element2_to_plot_counts]
                        st.sidebar.write(f"Standard ratio from sensitivities ({element1_to_plot_counts}/{element2_to_plot_counts}): {line_value_2:.4f}")
                    else:
                        line_value_2 = st.sidebar.number_input('Set X-axis Line Value standard', 
                                                            min_value=0.00, max_value=100.00, value=1.00, step=0.10, 
                                                            key='counts_line_value_2')
                        if not sensitivity_dict:
                            st.sidebar.write("No sensitivity data found in file")
                        else:
                            st.sidebar.write("Sensitivity data not found for selected isotopes")
                    
                    count_threshold = st.sidebar.number_input('Set Count Threshold', min_value=0, max_value=10000, value=500, step=50, 
                                                            key='counts_threshold')
                    
                    element1_symbol = ''.join([i for i in element1_to_plot_counts if not i.isdigit()])
                    element2_symbol = ''.join([i for i in element2_to_plot_counts if not i.isdigit()])
                    
                    isotopic_abundances_element1 = get_isotopic_abundance(element1_symbol)
                    isotopic_abundances_element2 = get_isotopic_abundance(element2_symbol)

                    abundance1 = isotopic_abundances_element1.get(element1_to_plot_counts, 'Not found')
                    abundance2 = isotopic_abundances_element2.get(element2_to_plot_counts, 'Not found')
                    
                    st.sidebar.write(f"Isotopic abundances for count {element1_to_plot_counts}: {abundance1}")
                    st.sidebar.write(f"Isotopic abundances for count {element2_to_plot_counts}: {abundance2}")

                    if abundance1 != 'Not found' and abundance2 != 'Not found':
                        line_value = abundance1 / abundance2
                        st.sidebar.write(f"Natural abundance ratio ({element1_to_plot_counts}/{element2_to_plot_counts}): {line_value:.4f}")
                    else:
                        line_value = None
                        
                    adjust_to_natural = st.sidebar.checkbox('Adjust Data to Natural Abundance Scale?', 
                                                key='adjust_to_natural')
            
                    if line_value is not None and line_value_2 is not None:
                        adjustment_factor = line_value / line_value_2
                        st.sidebar.write(f"Adjustment factor (Natural/Standard): {adjustment_factor:.4f}")

        
                    plot_counts_ratio_histogram(counts_data, element1_to_plot_counts, element2_to_plot_counts, 
                                    color, bin_size, x_max, title, line_x_value=line_value, 
                                    line_x_value_2=line_value_2, count_threshold=count_threshold,
                                    adjust_to_natural=adjust_to_natural)
            else:
                st.error("No valid data available for analysis")
                
    
            st.sidebar.title("Correlation Analysis")
            perform_correlation = st.sidebar.checkbox('Perform Correlation Analysis?', key='perform_correlation')

            if perform_correlation and (mass_data is not None or counts_data is not None):
        
                correlation_data_type = st.sidebar.radio("Select data type for correlation:", 
                                                    ["Mass", "Counts"], 
                                                    key='correlation_data_type')
                
                if correlation_data_type == "Mass" and mass_data is not None:
                    correlation_data = mass_data
                    available_columns = mass_data.columns
                elif correlation_data_type == "Counts" and counts_data is not None:
                    correlation_data = counts_data
                    available_columns = counts_data.columns
                else:
                    st.sidebar.error(f"No {correlation_data_type} data available for analysis")
                    available_columns = []
                    correlation_data = None
                
                if correlation_data is not None and len(available_columns) > 0:
                    x_element = st.sidebar.selectbox(f'Select X-axis Element ({correlation_data_type})', 
                                                available_columns, key='corr_x')
                    y_element = st.sidebar.selectbox(f'Select Y-axis Element ({correlation_data_type})', 
                                                available_columns, 
                                                index=1 if len(available_columns) > 1 else 0, key='corr_y')
                    
                    color = st.sidebar.color_picker('Pick a Color for the Plot', '#1E90FF', key='corr_color')
                    marker_size = st.sidebar.slider('Marker Size', min_value=4, max_value=20, value=8, step=1, key='corr_size')
                    opacity = st.sidebar.slider('Marker Opacity', min_value=0.1, max_value=1.0, value=0.6, step=0.1, key='corr_opacity')
                    use_log_scale = st.sidebar.checkbox('Use Logarithmic Scale', True, key='corr_log')

                    plot_mass_correlation_single(correlation_data, x_element, y_element, color, 
                                            marker_size, opacity, use_log_scale, correlation_data_type)
    
