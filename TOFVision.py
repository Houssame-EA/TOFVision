"""


Université de Montréal


Département de Chimie


Groupe Prof. Kevin J. Wilkinson
Boiphysicochimie de l'environnement


Aut: H-E Ahabchane, Amanda Wu

Date : 01/01/2024
Update :20/11/2024
"""

###
import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
import numpy as np
import \
    plotly.graph_objects as go
import re
import time
from io import StringIO
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from statsmodels.nonparametric.smoothers_lowess import lowess
import plotly.subplots as sp
from io import BytesIO
import warnings
from mendeleev import element
from scipy.stats import poisson, norm, expon, binom, lognorm, gamma, weibull_min
from scipy.interpolate import CubicSpline

warnings.filterwarnings('ignore')

###                             running the script                          streamlit run tofapp.py

st.set_page_config(page_title='TOFVision', page_icon=':atom_symbol:', layout='wide')

# Groupe Prof. Kevin J. Wilkinson Biophysicochimie de l'environnement - Département de chimie - Université de Montréal - logo

logo_url = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBwgHBgkIBwgKCgkLDRYPDQwMDRsUFRAWIB0iIiAdHx8kKDQsJCYxJx8fLT0tMTU3Ojo6Iys/RD84QzQ5OjcBCgoKDQwNGg8PGjclHyU3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3N//AABEIAFwAXAMBEQACEQEDEQH/xAAcAAEAAgIDAQAAAAAAAAAAAAAABQcEBgIDCAH/xAA6EAABAwMCBAQDBAgHAAAAAAABAgMEAAURBiESEzFBByJRYRQykRVxgaEWM0JSYqKxwSMkQ3OS4fD/xAAbAQEAAgMBAQAAAAAAAAAAAAAAAQQDBQYCB//EACsRAAICAgEDAwIGAwAAAAAAAAABAgMEETEFEiETQVEyYQYUI3GRoSKBwf/aAAwDAQACEQMRAD8AvCvJIoBQCgIyDqC0T7hJt8S4R3JsZZQ9H4sLSR18p3I9xtUgk6gCgFAKAUAoBQCgFAQeo9X2LTK2G71PEdb4UptPLUskDqcJBxUgkLTdIV4gNTrZIRIjOpCkrT/cdQfY70B5q8VmHIXiLeS0VNrLqHm1pOCCptJyD9+akgs6/eLH6PM2lgWpU1cq2MS+aZHAPOCMfKe6agksPT1z+2bFb7oGuV8ZGbe5fFxcHEkHGe+M0BIUAqAKAUAoBQCgKQ8StBaq1Drl+VBiB2C8ltLTy5ACGgEAHIJyNwThIOc+uakgn9Fw7d4VW+Y3qi7xW350lJbDalKCkBIAIRjiG5Vk4x0oCv8AxrUxJ1excYTqHo023tOtuoOQsZUMj6CpBFavWJFg0dJG5NsXHJ/2nVJ/vQk3lnxJlaR0hpSNHtzMsP2/iUXHSgp4FcGBgH0oQWzpm6G96ft91UyGTLYS8Wwri4cjOM96hkknUAUAoBQCgFAKkHnTx1gmLrsyMHhlxG3AfcZSR/KPrQg1yY45eLFYI8Vtx+VCRIYdCUnyoLgWjJ6dFKH4VkhXOf0oxW31VebJJGXLtl0l6etNv+BKXIC5B41OpwpLikqAAz2IP1rOsK/4Kb6tiJ/UYl/TNNrs8aRCfb+z47jK1kApPE6pYwQT2I64rFOiyH1RZYqzKLvokj0X4dFP6B2DhUCBAayQc78IzWEtE+l5paihDqFKHVIUCRQHZQCoBxUkk5CsUByoCG1bqOHpayPXOdlSUEJbaT8zqz0SP/bDJqSCp7Nr5nVE/kagvFyty3VcLMeIv4eON9k8xJ4yfdRA9hWp6nZn1xcsZLS/ky1qD+ol9UeHbV3baeYuM1yRGB5Tc+Qt9tQzkpJUeIA47H8K0eJ+JLYWJZEU1/BksxlKLUHpmvwVDlKZ+H+GdYWWnY+McpY6j/vuDX1LCyKsilWVcM+e9Qx7aL3G17fyZFWyicVqShClrOEpGVE9hXmTSXk9VxbklHkhoi2UJJmTHYMOceNuAh5bbTmP2lgHBUc5x9c1x+ZlStsl6K8HZU+rXUq9ttcs6r+m1Wb4QM2xlLzq8JWyOWtABGSCMHO+29VMd22Nvu4MlLsnt93BZOh9TzYt1Ysd5krlMSciFKdOXErAzy1n9rIBIV12wc7Vax7/AFVp8lim71OeSyasFgVAFAVJ432+5Xy76bs9vTxB7nueY4SCngHET7An61hyMivGqdtnCJjFyekV3rDw/n6YgtTHJLUyMpQQ4ptBSW1HoMEnIPr+Va/p/Wqc2bhFaa+T3ZU4rZtnh5cNTGY1bFSFSm2eAy/iBlMRoZw2CNy6fQ54QPvxq+s04SrdjWm+Nct/P7GWqU96OzVj8aFrG5LcdbabVGjqWVHGXPMPxPCE/lXRfg+3twW7H434Od/EFErrIKtbfk7IVq1BcI5kwbDKVHxlKpC0sKcH8KVHP1xXRS6jBPSWzWw6FbKO5SSZCXhxT1muTfKdakMoUh5hxPC42oDdJH3fWstlqux5OBXpxZYubCNvycdTWkXi1tri+Z1pPEyAdlpI6f0ri6LfSsal7nQVWenNqRENRlzk2z7YdUy9CWtLoV1CUgLBV+A6+lWHJQ7uz3M7ko93Z52bZHcVMudkEZDiHXLmwW+NPCSAoKUcdflCutYsWLVpjx4tWF51szYCoAoCA1dZn7pFjyLa4hu5wXedFU58ijjCm1fwqTke2x7VhyaIZFTqnwyYtxezVLhfrVPhvWq8v/YlxUjdqaEpW0rstBV5VYO4UD9K4t9MzcG9TjHvivj3Rb9SE48mozhYLO7CRpy5zprafLNh26Y4nnerpcSeEL9id+m1b/plWZmTk8qja9m1x9v2KmRfTRHbnosTRVv0bKJuVhaakTB+tdlKU7KbPormEqSa3agoLtS1r2PKl3LuT2Yd+iarOtG5cR2U1p0Ox/ikplNgqwCStIUPK2NuMZyrG1SDVL/cIt61ZcLlbyFwi03GS4PlfUji4lj1Hm4c9+H0rb9Orfa5PhnNdcvi5xhHlEHypUBsR20vuwc+XkKAdaH7u/VP3bjpWuzekS73ZSt/YnG6hVYkrPEv6ZDXDmyL3bm4bUyQXnUtJbebKFrOdkcSsBXfqdsneqcce2EGpx0biiUZxai1/osvwo+zrldH586U19tx+Y0i2k+aIgK4VK/iUcDKhsBt60ppVUdIsVVKtaRatZjKKgCgFAVn4rQ+VebRcVISWHm3IbhIzhWy0fXCxV7Aklbp+5qur1ylj90eUawAEjCQAPQVvVFLg46UpPlmHcGYKQJkvhZW38shKyhafuUCDWC+ulrutLuHflxl2UNt/BiLcRLb/wAdm+T2AflkOPOI/wCK1b/StR+b6dCWmzofyXW7a960Z8KXGkpKI6sFvZTZSUKR7FJ3Fbii6q2O62c3l4uRjy1dFpmTWcqGRp6EbnrWxspGREcXNcP7qUp4R/MpP51q+pSWoxOh6DW+6c/bggNd2ib4da5YvNoymK+6X4x34Qf9RpXtufwPtWpOlL305eomobLFukBWWZCM8J6oV3SfcHaoJJKoAoBQEXqaysags0i3SVFAcAKHEjdtYOUrHuCAalNp+DzKKktMpz/MRJr1sujYZuMf9Y32cT2cR6pP5dDuK6HGyY3R+5xfUMCWNNtfT7ETIeTy5N0eWjLb3wsEOfKhWeFTmPXOfuCa5rqmTO7J9FcLk7XoODXi4X5iS/ykZouAbt/IsMeRLUk8AfCMo4s+ZXErHEep271pXVuzuuaX2Ok9bVfbQm/ucpcF1+Hz22XmpsZJUy88tJW53KVYJ2Pp27dKy4uXLGuUovx/RXzcCGZjuuyPnXPufftBj4FmXuUvJSW0JGVLUrolI7k9MV3rvjGv1JHyeOHZK90xXlMs7w8049Z4Ts65ICbnPwp1GQeQ2Plbz7ZJPuTWgutds+5naYuPHHqVcST1npuNqqwSLXJIQpQ4mXsZLTg+VQ/ofUE1iLJCeF2jbjo6DNYuFxbkiS6FoZZB4GyBgkE9zt27CgN3qAKAUAoCD1Tpa2anipauLakut5LEllXC6yfVKv7HapTae0eZRUlpooG/6Ju0O+S7OZyXhGVzWC+Snmtr34wBkdcg+4+6q998KX3SXPuW8fGlfHsi/C9jOtsW+MsN25dxRHdaRltAYSpK2wQNl+u+OnfvWttnjyfqKO0za015MV6blpr7f9MxSFwmHpcgy/ikeVKFqS6HVHZKUHh7nbAwa8x/UkoQS0zJL9KDsm3tfPn+Cy9CaBh6ejRJM5Spl0baCQ47jhj7bpbT0HpxdT69q30pyaSb4OZjXCLbiuTda8GQUAoBQCgFAKAUBCam0xA1Ey0JXMZksEmPLYPC60T1we4PcHINRKKktSXg9RnKD3FmoOaB1ChzDN2tjyRsl16KtKwPcJVg/hiqT6dU+GzYLqlyXlLZNad0Izbpzdyu0xVynNbs5QEMsH1Qjfze5JPpirNVEKVqCKd+RZe9zZuFZTCKAUAoBQCgFAKAUAoBUgUAqAKAUAoBQH//2Q=="

# Université de Montréal - logo

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

st.title(""":atom_symbol: TOFVission""")
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


################## Tab1 analyse one file 

#Data structure selection before file upload
with tab1:
    
    data_type = st.radio(
        "Select data structure type:",
        ["NuQuant", "SPCal"],
        key='data_type_selection'
    )
    combine_files = st.checkbox('Combine multiple files?', key='combine_files')
    
    if combine_files:
        fl = st.file_uploader(
            ':file_folder: Upload files',
            type=['csv'],
            accept_multiple_files=True,
            key='multiple_files'
        )
    else:
        fl = st.file_uploader(
            ':file_folder: Upload a file',
            type=['csv'],
            key='single_file'
        )
    
    if fl:
        with st.sidebar:
            st.title("Analysis Options")
            
    


### extract data


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
            
            for col in range(1, len(headers.columns), 2):  
                element = str(headers.iloc[0, col])
                if pd.notna(element):
           
                    counts_column = pd.to_numeric(data.iloc[:, col], errors='coerce')
                    counts_data[element] = counts_column.fillna(0)  
                    
              
                    if col + 1 < len(headers.columns):
                        fg_column = pd.to_numeric(data.iloc[:, col + 1], errors='coerce')
                        fg_data[element] = fg_column.fillna(0)  
    
            #st.write("### Counts Data")
            #st.dataframe(counts_data)
            
            #st.write("### Mass Data (fg)")
            #st.dataframe(fg_data)
            
   
            particle_counts = {}
            for col in counts_data.columns:
                particle_count = (counts_data[col] > 0).sum()
                particle_counts[col] = particle_count
            
            #st.write("### Particle Numbers")
            #for element, count in particle_counts.items():
                #st.write(f"- {element}: {count} particles")
            
            return counts_data, fg_data
            
        except Exception as e:
            st.error(f"Error processing SPCal data: {str(e)}")
            return None, None


    @st.cache_data
    @st.cache_resource
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


    @st.cache_data
    @st.cache_resource
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

            # st.write("Column names in the DataFrame after header adjustment:", data.columns.tolist())

            elements = set(col.split(' ')[0] for col in data.columns if 'fwhm' in col)

            for element in elements:
                count_col = f'{element} counts'
                mole_col = f'{element} moles [fmol]'
                mass_col = f'{element} mass [fg]'
                fwhm_col = f'{element} fwhm'
                mole_per = f'{element} mole %'
                mass_per = f'{element} mass %'

                if all(col in data.columns for col in [count_col, mole_col, mass_col, fwhm_col, mole_per, mass_per]):
                    # st.write(f"Found all required columns for {element}. Cleaning data...")
                    data.loc[data[fwhm_col].isna(), [count_col, mole_col, mass_col, mass_per, mole_per]] = 0
                # else:
                #     st.write(f"Required columns for {element} not found in DataFrame.")

            cleaned_df = df.copy()
            cleaned_df.iloc[start_index + 1:, :] = data.values

            return cleaned_df
        else:
            st.error("Header row with 'fwhm' not found.")
            return None


    def process_uploaded_files(files, data_type):
        """process single or multiple files based on combination selection"""
        if isinstance(files, list):
            if not files:
                return None

            try:
                if 'csv' in files[0].name:
                    cleaned_content = preprocess_csv_file(files[0])
                    base_df = pd.read_csv(StringIO(cleaned_content))
                    st.error('File format not supported')
                    return None

                keyword = "Time" if data_type == "SPCal" else "event number"
                start_index = find_start_index(base_df, keyword)
                
                if start_index is None:
                    st.error(f"Could not find {keyword} in the first file")
                    return None
                    
        
                header = base_df.iloc[:start_index + 1]
                data_frames = [base_df.iloc[start_index + 1:]]
                

                for additional_file in files[1:]:
                    try:
                        if 'csv' in additional_file.name:
                            content = preprocess_csv_file(additional_file)
                            df = pd.read_csv(StringIO(content))
                        else:
                            df = pd.read_excel(additional_file)
                            
                        add_start_index = find_start_index(df, keyword)
                        if add_start_index is not None:
                            data_frames.append(df.iloc[add_start_index + 1:])
                        else:
                            st.warning(f"Skipping file {additional_file.name}: Could not find {keyword}")
                            
                    except Exception as e:
                        st.error(f"Error processing {additional_file.name}: {str(e)}")
                        continue
                
                # Combine all data
                combined_data = pd.concat(data_frames, ignore_index=True)
                final_df = pd.concat([header, combined_data], ignore_index=True)
                
                st.success(f"Successfully combined {len(files)} files")
                return final_df
                    
            except Exception as e:
                st.error(f'Error processing combined files: {str(e)}')
                return None
                
        else: 
            try:
                if 'csv' in files.name:
                    cleaned_content = preprocess_csv_file(files)
                    return pd.read_csv(StringIO(cleaned_content))
                else:
                    st.error('File format not supported')
                    return None
            except Exception as e:
                st.error(f'Error processing file: {str(e)}')
                return None

    if fl is not None:
        if combine_files:
            df = process_uploaded_files(fl, data_type)
            if df is not None:
                st.write('Processing combined files...')
        else:
            df = process_uploaded_files(fl, data_type)
            if df is not None:
                st.write('Uploaded File: ', fl.name)
                    
        if df is not None:
            try:
                if data_type == "SPCal":
                    headers, data = process_and_display_spcal(df)
                    if headers is not None and data is not None:
                        st.write("### Data Summary")
                        st.write(f"Number of particles: {len(data)}")
                        st.write(f"Number of elements: {len(data.columns) - 1}")
                else:
                    # Nu Quant processing code
                    df = clean_data(df)
                    if df is not None:
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

            except Exception as e:
                st.error(f'An error occurred: {str(e)}')
                st.write("Error details:", type(e).__name__)
                
    #@st.cache_data
    #@st.cache_resource
    
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
        
        

    #@st.cache_data
    #@st.cache_resource
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

        return combinations, related_data['mole_percent_data'], related_data['mass_data'], related_data[
            'mass_percent_data'], related_data['mole_data'], sd_df


    #@st.cache_data
    #@st.cache_resource


    def prepare_heatmap_data(data_combinations, combinations, start, end):
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


    def text_color_based_on_background(avg_value, min_val, max_val):
        norm_value = (avg_value - min_val) / (max_val - min_val)
        return "black" if norm_value < 0.5 else "white"


    def text_color_based_on_background_2(avg_value, min_val, max_val):
        avg_value *= 100
        norm_value = (avg_value - min_val) / (max_val - min_val)
        return "black" if norm_value < 0.5 else "white"


    #@st.cache_data
    #@st.cache_resource
    def plot_heatmap(heatmap_df, sd_df, selected_colorscale='ylgnbu', display_numbers=True, font_size=14):
        st.sidebar.header("Element Selection")
        all_elements = sorted(heatmap_df.columns.tolist())
        selected_elements = st.sidebar.multiselect(
            "Select elements to display (leave empty to show all):",
            options=all_elements,
            default=[]
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
                            annotation_text = f"{avg_value * 100:.1f}\n±{sd_value:.1f}" if not np.isnan(sd_value) else f"{avg_value * 100:.1f}"
                        else:  
                            color = text_color_based_on_background(avg_value, min_val, max_val)
                            annotation_text = f"{avg_value:.1f}\n±{sd_value:.1f}" if not np.isnan(sd_value) else f"{avg_value:.1f}"
                        
                        fig.add_annotation(
                            x=x, y=y, 
                            text=annotation_text, 
                            showarrow=False,
                            font=dict(size=font_size, color=color)
                        )

        selection_info = f" (Filtered by: {', '.join(selected_elements)})" if selected_elements else ""
        
        fig.update_layout(
            title=f'{title_prefix} After Treatment - Total Particles: {total_count}{selection_info}',
            xaxis=dict(
                title='Elements', 
                tickangle=0,
                title_font=dict(size=33, color='black'),
                tickfont=dict(size=33, color='black')
            ),
            yaxis=dict(
                title='Particle (Frequency)', 
                autorange='reversed',
                title_font=dict(size=33, color='black'),
                showgrid=False,
                tickfont=dict(size=33, color='black')
            ), 
            height=max(600, 60 * len(combinations_with_counts)),
            width=3175
        )

        return fig

    def display_aggregated_data(aggregated_data, data_type):
        st.header(f"All {data_type.replace('_', ' ').title()}")
        st.dataframe(aggregated_data)


    #@st.cache_data
    #@st.cache_resource
    def aggregate_combination_data(data_dict):
        aggregated_df = pd.concat(data_dict.values(), keys=data_dict.keys())
        aggregated_df.reset_index(level=0, inplace=True)
        aggregated_df.rename(columns={'level_0': 'Combination'}, inplace=True)
        return aggregated_df


    #@st.cache_data
    #@st.cache_resource
    
    def plot_combination_distribution_by_counts(combinations, elements_to_analyze, elements_to_exclude, count_threshold):
        filtered_combinations = {
            combo: info for combo, info in combinations.items()
            if info['counts'] >= count_threshold
            and any(elem in combo.split(', ') for elem in elements_to_analyze)
            and not any(elem in combo.split(', ') for elem in elements_to_exclude)
        }

        other_counts = sum(
            info['counts'] for combo, info in combinations.items()
            if info['counts'] < count_threshold
            and any(elem in combo.split(', ') for elem in elements_to_analyze)
            and not any(elem in combo.split(', ') for elem in elements_to_exclude)
        )

        if other_counts > 0:
            filtered_combinations['Others'] = {'counts': other_counts}

        if not filtered_combinations:
            st.write(f"No valid combinations found after filtering out {' or '.join(elements_to_exclude)}.")
            return

        total_counts = sum(info['counts'] for info in filtered_combinations.values())
        sorted_combinations = sorted(filtered_combinations.items(), key=lambda item: item[1]['counts'], reverse=True)

        labels = [
            f"{combo} ({info['counts']}) : ({info['counts'] / total_counts * 100:.2f}%)"
            for combo, info in sorted_combinations
        ]
        values = [info['counts'] for _, info in sorted_combinations]
        texts = [
            f"{combo} ({info['counts'] / total_counts * 100:.2f}%)"
            for combo, info in sorted_combinations
        ]
        if other_counts > 0:
            texts.append(f"Others: {other_counts} ({other_counts / total_counts * 100:.2f}%)")

        hover_texts = [
            f"{combo}: {info['counts']}<br>({(info['counts'] / total_counts * 100):.2f}%)"
            for combo, info in sorted_combinations
        ]
        if other_counts > 0:
            hover_texts.append(f"Others: {other_counts}<br>({other_counts / total_counts * 100:.2f}%)")

        default_colors = [
            '#FF6347', '#FFD700', '#FFA500', '#20B2AA', '#00BFFF',
            '#F0E68C', '#E0FFFF', '#AFEEEE', '#DDA0DD', '#FFE4E1',
            '#FAEBD7', '#D3D3D3', '#90EE90', '#FFB6C1', '#FFA07A',
            '#C71585', '#DB7093', '#FF1493', '#FF69B4', '#FFA07A',
            '#FFB6C1', '#87CEEB', '#98FB98', '#FFFFE0', '#FFDAB9',
            '#E6E6FA', '#FFF0F5', '#B0E0E6', '#FFC0CB', '#F5DEB3'
        ]

        st.sidebar.title("Customize Combination Colors")
        color_selections = []
        for i, (combo, _) in enumerate(sorted_combinations):
            if combo != 'Others':
                color = st.sidebar.color_picker(f"Pick color for {combo}", default_colors[i % len(default_colors)])
                color_selections.append(color)
            else:
                color_selections.append('#abcdef')  
        pie_chart_colors = color_selections

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            text=texts,
            hoverinfo="label+percent+value",
            hovertext=hover_texts,
            hovertemplate='%{hovertext}',
            textfont=dict(size=25),
            marker=dict(colors=pie_chart_colors, line=dict(color='#000000', width=1)),
            pull=[0.1 if label == 'Others' else 0 for label in labels], 
        )])
        fig.update_traces(textinfo='text')
        
        def generate_title(elements_to_analyze, elements_to_exclude):
            title = f"Distribution of particles containing {' or '.join(elements_to_analyze)}"
            if elements_to_exclude:
                title += f" excluding {' and '.join(elements_to_exclude)}"
            return title

        fig.update_layout(
            legend=dict(font=dict(size=16, color='black')),
            title=generate_title(elements_to_analyze, elements_to_exclude),
            title_font_size=20,
            title_x=0.4,
            title_y=0.95,
            title_xanchor='center',
            title_yanchor='top',
            annotations=[
                #dict(
                 #   x=0.95,
                  #  y=1,
                   # xref="paper",
                    #yref="paper",
                    #text=f"Others: all particles below {count_threshold}",
                    #showarrow=False,
                    #font=dict(size=28, color="black"),
                #),
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

        st.plotly_chart(fig, use_container_width=True)

        summary_data = [['Combination', 'Count', 'Percentage']]
        for combo, info in sorted_combinations:
            percentage = info['counts'] / total_counts * 100
            summary_data.append([combo, info['counts'], f"{percentage:.2f}%"])

        summary_df = pd.DataFrame(summary_data)
        csv = summary_df.to_csv(index=False, header=False)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='combination_distribution_data.csv',
            mime='text/csv'
        )
    


    #@st.cache_data
    #@st.cache_resource
    def prepare_data(mole_percent_data):
        mole_percent_data = mole_percent_data.apply(pd.to_numeric, errors='coerce').dropna()
        if mole_percent_data.empty:
            st.error("No data available after cleaning.")
            return None
        return mole_percent_data


    #@st.cache_data
    #@st.cache_resource
    def apply_clustering(mole_percent_data, method, n_clusters):
        mole_percent_data = prepare_data(mole_percent_data)
        if mole_percent_data is None:
            return None, None

        # scaler = StandardScaler()
        scaled_data = (mole_percent_data)  # scaler.fit_transform

        # select clustering model
        clustering_models = {
            'Hierarchical': AgglomerativeClustering(n_clusters=n_clusters),
            'Spectral': SpectralClustering(
                n_clusters=n_clusters,
                affinity='nearest_neighbors',
                assign_labels='kmeans',
                n_jobs=-1
            ),
            'Gaussian': GaussianMixture(n_components=n_clusters),
            'K-Means': KMeans(n_clusters=n_clusters),
            'Mini-Batch K-Means': MiniBatchKMeans(n_clusters=n_clusters),
            'Mean Shift': MeanShift()
        }

        cluster_model = clustering_models.get(method)
        if not cluster_model:
            st.error(f"Unsupported clustering method: {method}")
            return None, None

        labels = cluster_model.fit_predict(scaled_data)
        return labels, mole_percent_data


    #@st.cache_data
    #@st.cache_resource
    def plot_heatmap_cluster(mole_percent_data, labels, title):
    
        if labels is None:
            st.error("No labels available for plotting.")
            return

 
        mole_percent_data['Cluster'] = labels

   
        cluster_summary = mole_percent_data.groupby('Cluster').mean()
        cluster_counts = mole_percent_data.groupby('Cluster').size()

        if cluster_summary.empty:
            st.error("Failed to compute cluster summary.")
            return

        cluster_labels = []
        for i in cluster_summary.index:
     
            top_elements = cluster_summary.loc[i][cluster_summary.loc[i] > 0.01].sort_values(ascending=False).head(
                5).index.tolist()
            top_elements = ', '.join(top_elements) 
            count = cluster_counts[i]
            label = f"{top_elements} ({count})"
            cluster_labels.append(label)

        sorted_indices = cluster_counts.sort_values(ascending=True).index
        sorted_labels = [cluster_labels[i] for i in sorted_indices]
        sorted_summary = cluster_summary.loc[sorted_indices]

        z_values = sorted_summary.values
        z_values = np.where(z_values == 0, np.nan, z_values)

        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=sorted_summary.columns,
            y=sorted_labels,
            colorscale='ylGnBu',
            colorbar=dict(
                title='Mole %',
                title_font=dict(
                    color='black'))))

        fig.update_layout(
            title=title,
            xaxis_title='Elements',
            yaxis_title='Clusters',
            height=800,
            width=800
        )

        fig.update_layout(
            xaxis=dict(title='Elements', tickangle=-45,
                    title_font=dict(size=20, color='black'),
                    tickfont=dict(size=20, color='black'),

                    ),
            yaxis=dict(title='Clusters (Frequency)',
                    title_font=dict(size=20, color='black'),
                    tickfont=dict(size=20, color='black'),
                    showgrid=False, 
                    zeroline=False,  
                    ),
            height=max(600, 40 * len(sorted_labels))
        )

        st.plotly_chart(fig, use_container_width=True)


    #@st.cache_data(experimental_allow_widgets=True)
    #@st.cache_resource(experimental_allow_widgets=True)
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


    #@st.cache_data
    #@st.cache_resource
    def plot_pie_chart(combinations, title_suffix, total_counts, particles_per_ml_value):
        labels = [combination for combination, _ in combinations]
        values = [details['counts'] for _, details in combinations]

        fig = px.pie(values=values, names=labels, title=f"{title_suffix}",
                    color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_traces(textinfo='percent+label')
        return fig


    md = None


    #@st.cache_data(experimental_allow_widgets=True)
    #@st.cache_resource(experimental_allow_widgets=True)
    def create_color_map(_elements, base_colors):
        color_map = {}
        for i, element in enumerate(_elements):
            default_color = base_colors[i % len(base_colors)]  
            color = st.sidebar.color_picker(f"Color for {element}", value=default_color, key=f"color_{element}")
            color_map[element] = color
        color_map['Others'] = st.sidebar.color_picker(f"Color for Others", '#777777')  
        return color_map

    #@st.cache_data
    #@st.cache_resource
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
        #colors_counts = [color_map.get(index, '#CCCCCC') for index in element_counts.index]

        fig.add_trace(go.Pie(labels=mass_percent.index, values=mass_percent.values,
                            marker=dict(colors=colors_mass, line=dict(color='#000000', width=1)),
                            textinfo='label+percent',
                            texttemplate='%{label}(%{customdata:d}): %{value:.1f}%',
                            customdata=[element_counts.get(index, 0) for index in mass_percent.index],
                            hoverinfo='label+percent+value',
                            hovertext=[f"{label}: {element_counts.get(label, 0):d} counts" for label in
                                        mass_percent.index],
                            textfont=dict(size=25, color='black'),
                            title=dict(text='', font=dict(size=18, color='black'))), row=1, col=1)

        #fig.add_trace(go.Pie(labels=element_counts.index, values=element_counts.values,
                            #marker=dict(colors=colors_counts),
                            #textinfo='label+value+percent',
                            #texttemplate='%{label}: %{value:d}',
                            #hoverinfo='label+value+percent',
                            #hovertext=[f"{label}: {value:d} counts" for label, value in element_counts.items()],
                            #textfont=dict(size=17, color='black'),
                            #title=dict(text='Element Counts', font=dict(size=18, color='black'))), row=1, col=2)

        fig.add_trace(go.Pie(labels=mole_percent.index, values=mole_percent.values,
                            marker=dict(colors=colors_mole, line=dict(color='#000000', width=1)),
                            textinfo='label+percent',
                            texttemplate='%{label}(%{customdata:d}): %{value:.1f}%',
                            customdata=[element_counts.get(index, 0) for index in mole_percent.index],
                            hoverinfo='label+percent+value',
                            hovertext=[f"{label}: {element_counts.get(label, 0):d} counts" for label in
                                        mole_percent.index],
                            textfont=dict(size=25, color='black'),
                            title=dict(text='', font=dict(size=18, color='black'))), row=1, col=2)

        fig.update_layout(title_text='Mass and Mole Percentages', title_x=0.4, height=600, width=2500,
                        legend=dict(font=dict(size=16, color='black')))

        st.plotly_chart(fig)

        # prepare summary data for export
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
                    
                    color_map = create_color_map(fg_data.columns, default_colors)
                    threshold = st.sidebar.number_input('Threshold for Others (%)', format="%f")
                    visualize_mass_percentages_pie_chart_spcal(fg_data, color_map, threshold)
                    
                    
                    
                st.sidebar.title('Mass Distribution')
                perform_mass_distribution_analysis = st.sidebar.checkbox('Perform Mass distribution Analysis?')

                if perform_mass_distribution_analysis:
                    elements_to_plot = st.sidebar.multiselect(
                        'Select up to 3 elements to view the histogram:', 
                        fg_data.columns,
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
                        fg_data, 
                        elements_to_plot, 
                        all_color, 
                        single_color, 
                        multiple_color, 
                        bin_size, 
                        x_max, 
                        "Mass Distribution"
                    )
                    
                combinations, mass_data_combinations, sd_df = get_combinations_and_related_data_spcal(fg_data)
            
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
                        fg_data.columns.tolist()
                    )
                    elements_to_exclude = st.sidebar.multiselect(
                        'Select elements to exclude from combinations:',
                        fg_data.columns.tolist(),
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

                combinations, mass_data_combinations, sd_df = get_combinations_and_related_data_spcal(fg_data)

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
                    start = st.sidebar.number_input('Start from combination:', min_value=1, max_value=len(combinations) - 1, value=1)
                    end = st.sidebar.number_input('End at combination:', min_value=2, max_value=len(combinations), value=2)
                    end = max(end, start + 1)

                    try:
                        heatmap_df = prepare_heatmap_data_spcal(mass_data_combinations, combinations, start, end)
                        if heatmap_df is not None:
                            plot = plot_heatmap(heatmap_df, sd_df, selected_colorscale=selected_colorscale, 
                                              display_numbers=display_numbers, font_size=font_size)
                            st.plotly_chart(plot)
                    except Exception as e:
                        st.error(f"Error preparing heatmap: {str(e)}")

        
        elif data_type == "NuQuant":
            mass_data, mass_percent_data, mole_data, mole_percent_data, counts_data = process_data(df)
            combinations, related_mole_percent_data, related_mass_data, related_mass_percent_data, related_mole_data, sd_df = get_combinations_and_related_data(
                mass_data, mass_data, mass_percent_data, mole_data, mole_percent_data)

            elements = mass_data.columns

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
                visualize_mass_and_mole_percentages_pie_charts(mass_data, mole_data, color_map, threshold)

            md, _, _, _, _ = process_data(df)

            # initialize session state variables if not already set
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
                    "Mass": mass_data,
                    "Mole": mole_data,
                    "Mole %": mole_percent_data
                }[data_type]

                # Clear previous data if data type changes
                # if 'data_type' not in st.session_state or st.session_state['data_type'] != data_type:
                #     st.session_state['combinations'] = None
                #     st.session_state['mass_data_combinations'] = None
                #     st.session_state['mole_data_combinations'] = None
                #     st.session_state['mole_percent_data_combinations'] = None
                #     st.session_state['sd_df'] = None
                #     st.session_state['data_loaded'] = False
                #     st.session_state['data_type'] = data_type
                display_numbers = st.sidebar.checkbox("Display Numbers on Heatmap", value=True)
                font_size = st.sidebar.slider("Font Size for Numbers on Heatmap", min_value=5, max_value=30, value=14)

                combinations, mole_percent_data_combinations, mass_data_combinations, mass_percent_data_combination, mole_data_combinations, sd_df = get_combinations_and_related_data(
                    selected_data, mass_data, mass_percent_data, mole_data, mole_percent_data)
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

                start = st.sidebar.number_input('Start from combination:', min_value=1, max_value=len(combinations) - 1,
                                                value=1)
                end = st.sidebar.number_input('End at combination:', min_value=2, max_value=len(combinations), value=2)
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
                elements_to_plot = st.sidebar.multiselect('Select up to 3 elements to view the histogram:', mass_data.columns,
                                                        max_selections=3)
                only_selected_elements = st.checkbox("Show only particles with exactly the selected elements")

                bin_size = st.sidebar.slider('Select bin size:', min_value=0.001, max_value=10.0, value=0.01, step=0.01)
                x_max = st.sidebar.slider('Select max value for x-axis (Mass (fg)):', min_value=0, max_value=1000, value=10,
                                        step=1)

                all_color = st.sidebar.color_picker('Pick a color for All data', '#00f900')  # Default color is light green
                single_color = st.sidebar.color_picker('Pick a color for Single detections', '#0000ff')  # Default blue
                multiple_color = st.sidebar.color_picker('Pick a color for Multiple detections', '#fff000')  # Default red

                plot_histogram_for_elements(mass_data, elements_to_plot, all_color, single_color, multiple_color, bin_size,
                                            x_max, "Mass Distribution", related_mass_data, only_selected_elements)

            st.sidebar.title("Single and Multiple Element Analysis")
            show_s_m = st.sidebar.checkbox("Single and Multiple Element Analysis?")
            if show_s_m:
                visualize_pie_chart_single_and_multiple(st.session_state['combinations'], ppm)

            st.sidebar.title('Element Distribution')
            perform_element_distribution = st.sidebar.checkbox("Perform Element Distribution")
            if perform_element_distribution:
                elements_to_analyze = st.sidebar.multiselect('Select elements to analyze:',
                                                            st.session_state['heatmap_df'].columns.tolist())
                elements_to_exclude = st.sidebar.multiselect('Select elements to exclude from combinations:',
                                                            st.session_state['heatmap_df'].columns.tolist(), default=[])
                count_threshold = st.sidebar.number_input('Set a count threshold for display:', min_value=0, value=10, step=1)
                if elements_to_analyze:
                    plot_combination_distribution_by_counts(st.session_state['combinations'], elements_to_analyze,
                                                            elements_to_exclude, count_threshold)

            st.sidebar.title('Raw Data after combination')

            Raw_data_after_combination = st.sidebar.checkbox('Export Raw Data?')
            if Raw_data_after_combination:

                _, mole_percent_data_after_combination, mass_data_after_combination, mass_percent_data_after_combination, mole_data_after_combination, sd_df = get_combinations_and_related_data(
                    mole_percent_data, mass_data, mass_percent_data, mole_data, mole_percent_data)

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
                _, _, _, mole_percent_data, _ = process_data(df)
                clustering_method = st.sidebar.selectbox('Select a clustering method:',
                                                        ['K-Means', 'Hierarchical', 'Spectral', 'Gaussian',
                                                        'Mini-Batch K-Means', 'Mean Shift'])
                num_clusters = st.sidebar.slider('Select number of clusters:', min_value=1, max_value=100, value=5)

                labels, processed_data = apply_clustering(mole_percent_data, clustering_method, num_clusters)
                plot_heatmap_cluster(processed_data, labels, f"{clustering_method} Clustering Results")
            


############################ Tab2 analyse multiple files 

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
                    
                # Update last event number
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
    
with tab2:

    combine_files = st.checkbox('Combine multiple Nu Quant Vitesse files?', key='combine_files_tab2')
    if 'all_files' not in st.session_state:
        st.session_state.all_files = []
    if 'current_files' not in st.session_state:
        st.session_state.current_files = None

    new_files = st.file_uploader(
        ':file_folder: Upload files',
        type=['csv'],
        accept_multiple_files=True,
        key='multiple_files_tab2'
    )

 
    uploaded_files = []

 
    current_file_names = [f.name for f in new_files] if new_files else []
    previous_file_names = [f.name for f in st.session_state.current_files] if st.session_state.current_files else []
    
    files_changed = current_file_names != previous_file_names
    if new_files:
        if combine_files and len(new_files) > 1:
            if files_changed:
                file_groups = [new_files[i:i + 3] for i in range(0, len(new_files), 3)]
                
                combined_files = []
                for i, group in enumerate(file_groups):
                    if len(group) > 1:
                        st.info(f"Combining {len(group)} files")
                        combined_file = combine_replicate_files(group)
                        if combined_file:
                            combined_file.name = f"Combined_Files_{i}.csv"  
                            combined_files.append(combined_file)
                
                if combined_files:
                    st.session_state.all_files = combined_files
                    uploaded_files = combined_files
                    st.session_state.current_files = new_files  
            else:
            
                uploaded_files = st.session_state.all_files
        else:
            st.session_state.all_files = new_files
            uploaded_files = new_files
            st.session_state.current_files = new_files

    if st.session_state.all_files:
        with st.sidebar:
            st.title("Multi Analysis Options")
                
    #### Logic
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
        summary_data = {
            "Filename": [],
            "Assigned Letter": [],
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
            summary_data["Assigned Letter"].append(file_letter_map[filename])
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
            mime='text/csv')
        
    
    if uploaded_files:
            data_dict = {}
            file_letter_map = {}
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            dilution_factors = {}
            acquisition_times = {}

            file_info = []
            
            st.write('Enter the dilustion factor and the total acquisition time in seconds for each file :')
            for i, file in enumerate(uploaded_files):
                filename = file.name
                letter = letters[i % len(letters)]
                file_letter_map[filename] = letter
                file_info.append({
                    'filename': filename,
                    'letter': letter,
                    'dilution_factor': 1.0,  # default value
                    'acquisition_time': 60.0
                })

            # dataframe for user inputs
            file_info_df = pd.DataFrame(file_info)
            # st.dataframe(file_info_df)
            updated_file_info_df = st.data_editor(file_info_df, key="file_info_editor")

            for i, row in updated_file_info_df.iterrows():
                filename = row['filename']
                dilution_factors[filename] = row['dilution_factor']
                acquisition_times[filename] = row['acquisition_time']

            for file in uploaded_files:
                filename = file.name
                try:
                    if 'csv' in filename:
                        cleaned_file_content = preprocess_csv_file(file)
                        df = pd.read_csv(StringIO(cleaned_file_content))
                    else:
                        st.error('File format not supported. Please upload a CSV or Excel file.')
                        df = None

                    if df is not None:
                        # st.write(f"Before Cleaning ({filename}):")
                        # st.dataframe(df)

                        df = clean_data(df)

                        if df is not None:
                            # st.write(f"After Cleaning ({filename}):")
                            # st.dataframe(df)

                            event_number_cell = count_rows_after_keyword_until_no_data(df, 'event number', column_index=0)
                            # if event_number_cell is not None:
                            # st.write(f'Total Particles Count: {event_number_cell} Particles')
                            # else:
                            # st.write('Event number not found or no valid count available.')
                            transport_rate_cell = find_value_at_keyword(df, 'calibrated transport rate', column_index=1)
                            transport_rate = extract_numeric_value_from_string(str(transport_rate_cell))
                            # if transport_rate is not None:
                            # st.write(f'Calibrated Transport Rate: {transport_rate} µL/s')
                            # else:
                            # st.write('Calibrated transport rate not found or no valid rate available.')

                            particles_per_ml = None
                            dilution_factor = dilution_factors[filename]
                            acquisition_time = acquisition_times[filename]

                            particles_per_ml = calculate_particles_per_ml(event_number_cell, transport_rate, acquisition_time,
                                                                        dilution_factor)
                            # if particles_per_ml is not None:
                            # st.write(f'Particles per ml: {particles_per_ml} Particles/mL')
                            # else:
                            # st.write('Error in calculation. Please check input values.')

                            data_dict[filename] = {
                                'df': df,
                                'event_number_cell': event_number_cell,
                                'transport_rate': transport_rate,
                                'particles_per_ml': particles_per_ml
                            }

                except Exception as e:
                    st.error(f'An error occurred with file {filename}: {e}')
                    df = None

    def plot_distribution(df_dict, element, detection_type, distribution_type, title, file_letter_map):
        fig = go.Figure()
        all_summary_data = []

        # default colors
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        color_dict = {}
        max_rows = 0

        for i, (filename, data) in enumerate(df_dict.items()):
            df = data['df']
            mass_data, _, _, _, _ = process_data(df)
            if element in mass_data.columns:
                mass_data = mass_data.dropna(subset=[element])
                mass_data[element] = mass_data[element][mass_data[element] > 0]

                if detection_type == 'Single':
                    mass_data = mass_data.loc[mass_data.apply(lambda row: (row > 0).sum(), axis=1) == 1]
                elif detection_type == 'Multiple':
                    mass_data = mass_data.loc[mass_data.apply(lambda row: (row > 0).sum(), axis=1) > 1]

                if not mass_data.empty:
                    if distribution_type == 'Poisson':
                        lambda_ = np.mean(mass_data[element])
                        k = np.arange(0, np.max(mass_data[element]) + 1)
                        dist = poisson.pmf(k, lambda_)
                    elif distribution_type == 'Gaussian':
                        mean = np.mean(mass_data[element])
                        std_dev = np.std(mass_data[element])
                        k = np.linspace(mean - 3 * std_dev, mean + 3 * std_dev, 100)
                        dist = norm.pdf(k, mean, std_dev)
                    elif distribution_type == 'Exponential':
                        lambda_ = 1 / np.mean(mass_data[element])
                        k = np.linspace(0, np.max(mass_data[element]), 100)
                        dist = expon.pdf(k, scale=1 / lambda_)
                    elif distribution_type == 'Binomial':
                        n = 100  # adjust this value
                        p = np.mean(mass_data[element]) / n
                        k = np.arange(0, n + 1)
                        dist = binom.pmf(k, n, p)
                    elif distribution_type == 'Log-normal':
                        clean_data = mass_data[element].replace([np.inf, -np.inf], np.nan).dropna()
                        if len(clean_data) > 0:
                            shape, loc, scale = lognorm.fit(clean_data)
                            k = np.linspace(0, np.max(clean_data), 100)
                            dist = lognorm.pdf(k, shape, loc, scale)
                        else:
                            st.warning(f"Not enough valid data for log-normal distribution for {filename}.")
                            continue
                    elif distribution_type == 'Gamma':
                        clean_data = mass_data[element].replace([np.inf, -np.inf], np.nan).dropna()
                        if len(clean_data) > 0:
                            shape, loc, scale = gamma.fit(clean_data)
                            k = np.linspace(0, np.max(clean_data), 100)
                            dist = gamma.pdf(k, shape, loc, scale)
                        else:
                            st.warning(f"Not enough valid data for gamma distribution for {filename}.")
                            continue
                    elif distribution_type == 'Weibull':
                        clean_data = mass_data[element].replace([np.inf, -np.inf], np.nan).dropna()
                        if len(clean_data) > 0:
                            shape, loc, scale = weibull_min.fit(clean_data)
                            k = np.linspace(0, np.max(clean_data), 100)
                            dist = weibull_min.pdf(k, shape, loc, scale)
                        else:
                            st.warning(f"Not enough valid data for Weibull distribution for {filename}.")
                            continue
                    elif distribution_type == 'Cubic Spline':
                        mass_data_finite = mass_data[element][np.isfinite(mass_data[element])]
                        k = np.linspace(0, np.max(mass_data_finite), 100)
                        cs = CubicSpline(np.arange(len(mass_data_finite)), mass_data_finite)
                        dist = cs(k)
                    else:
                        continue

                    color = st.sidebar.color_picker(f'Pick a color for line {file_letter_map[filename]}',
                                                    default_colors[i % len(default_colors)], key=f"Mass_D_{filename}")
                    color_dict[filename] = color

                    fig.add_trace(go.Scatter(
                        x=k,
                        y=dist,
                        mode='markers+lines',
                        name=f"{file_letter_map[filename]}",
                        marker=dict(size=10),
                        line=dict(width=2, color=color)
                    ))

                    file_summary_data = []
                    file_summary_data.append(['File name: ' + filename, '', ''])
                    file_summary_data.append(['k', f'{element} Probability', ''])
                    for xi, yi in zip(k, dist):
                        file_summary_data.append([xi, yi, ''])
                    all_summary_data.append(file_summary_data)
                    max_rows = max(max_rows, len(file_summary_data))

        for file_data in all_summary_data:
            while len(file_data) < max_rows:
                file_data.append(['', '', ''])

        # merge all file data side by side
        merged_summary_data = []
        for row_idx in range(max_rows):
            merged_row = []
            for file_data in all_summary_data:
                merged_row.extend(file_data[row_idx])
            merged_summary_data.append(merged_row)

        fig.update_layout(
            title=f"{title}: {element} ({distribution_type} Distribution)",
            xaxis_title="Index k" if distribution_type in ['Poisson', 'Binomial'] else "Value",
            yaxis_title=f"{distribution_type} Probability",
            xaxis=dict(
                title_font=dict(size=30, color='black'),
                tickfont=dict(size=30, color='black'),
                linecolor='black',
                linewidth=3
            ),
            yaxis=dict(
                title_font=dict(size=30, color='black'),
                tickfont=dict(size=30, color='black'),
                linecolor='black',
                linewidth=3
            ),
            legend_title_font=dict(size=24, color='black'),
            legend_font=dict(size=20, color='black'),
        )

        st.plotly_chart(fig, use_container_width=True)

        # export summary data as CSV
        if merged_summary_data:
            summary_df = pd.DataFrame(merged_summary_data)
            csv = summary_df.to_csv(index=False, header=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='distribution_data.csv',
                mime='text/csv'
            )
            
    


    def plot_histogram_for_element(df_dict, element, detection_type, bin_size, x_max, title, file_letter_map):
        filenames = list(df_dict.keys())
        num_files = len(filenames)
        num_subplots = (num_files + 1) // 2  #

        fig = make_subplots(rows=num_subplots, cols=1, shared_xaxes=True, vertical_spacing=0.02)

        # default colors
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

        bin_edges = np.arange(0, x_max + bin_size, bin_size)
        all_summary_data = []

        for i in range(num_files):
            subplot_index = i // 2 + 1
            filename = filenames[i]
            data = df_dict[filename]
            df = data['df']
            particles_per_ml = data['particles_per_ml']
            if particles_per_ml is None:
                st.warning(f"Particles per mL not available for file {filename}. Skipping.")
                continue

            particles_per_ml_value = float(particles_per_ml.replace('e', 'E'))
            mass_data, _, _, _, _ = process_data(df)
            if element in mass_data.columns:
                mass_data = mass_data.dropna(subset=[element])
                mass_data[element] = mass_data[element][mass_data[element] > 0]

                if detection_type == 'Single':
                    mass_data = mass_data.loc[mass_data.apply(lambda row: (row > 0).sum(), axis=1) == 1]
                elif detection_type == 'Multiple':
                    mass_data = mass_data.loc[mass_data.apply(lambda row: (row > 0).sum(), axis=1) > 1]

                color = st.sidebar.color_picker(f'Pick a color for {file_letter_map[filename]}',
                                                default_colors[i % len(default_colors)], key=f"Mass_{filename}")

                hist_data = mass_data[element]
                hist_counts, _ = np.histogram(hist_data, bins=bin_edges)

                # the particles per mL for each bin
                hist_density = [(count / len(hist_data)) * particles_per_ml_value for count in hist_counts]

                fig.add_trace(
                    go.Bar(
                        x=bin_edges[:-1] + bin_size / 2,
                        y=hist_density,
                        name=f'{file_letter_map[filename]}',
                        marker_color=color,
                        width=bin_size,
                        opacity=0.7  
                    ),
                    row=subplot_index, col=1
                )

                file_summary_data = []
                file_summary_data.append(['File name: ' + filename, '', ''])
                file_summary_data.append(['Mass (fg)', 'Particles/mL', ''])
                for edge, density in zip(bin_edges[:-1], hist_density):
                    file_summary_data.append([edge, density, ''])
                all_summary_data.append(file_summary_data)

        for i in range(1, num_subplots + 1):
            fig.update_yaxes(
                title_text="Frequency" if i == num_subplots // 1.3 else None,  
                tickformat='.1e',
                title_font=dict(size=30, color='black'),
                tickfont=dict(size=25, color='black'),
                linecolor='black',
                linewidth=2,
                row=i, col=1
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
            title_font=dict(size=30, color='black'),
            tickfont=dict(size=30, color='black'),
            linecolor='black',
            linewidth=2,
            row=num_subplots, col=1
        )

        fig.update_layout(
            title=f"{title}: {element}",
            barmode='overlay', 
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=30, color="black"),
            legend=dict(font=dict(size=30, color="black")),
            height=300 * num_subplots,  
            margin=dict(l=50, r=50, t=50, b=50)  
        )

        st.plotly_chart(fig, use_container_width=True)

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
        # st.write("Recalculated Mole Percent Data for Selected Elements")
        # st.dataframe(all_mole_percent_data)

        # Provide option to download the recalculated data
        # csv = all_mole_percent_data.to_csv(index=False)
        # st.download_button(
        # label="Download data as CSV",
        # data=csv,
        # file_name='recalculated_mole_percent_data.csv',
        # mime='text/csv',
        # )


    def plot_mole_ratio_histogram_for_files(df_dict, element1, element2, bin_size, x_max, title, file_letter_map):
        filenames = list(df_dict.keys())
        n_subplots = len(filenames)
        
        fig = make_subplots(rows=n_subplots, cols=1, vertical_spacing=0.02)
        
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        all_summary_data = []

        for i, filename in enumerate(filenames):
            data = df_dict[filename]
            df = data['df']
            _, _, mole_data, _, _ = process_data(df)
            
            if all(element in mole_data.columns for element in [element1, element2]):
                mole_data = mole_data.dropna(subset=[element1, element2])

                filtered_data = mole_data[(mole_data[element1] > 0) & (mole_data[element2] > 0)]
                ratios = filtered_data[element1] / filtered_data[element2]
                ratios = ratios.dropna()

                if ratios.empty:
                    st.error(f'No valid data available for plotting in file {filename}.')
                    continue

                color = st.sidebar.color_picker(f'Pick a color for {file_letter_map[filename]}',
                                                default_colors[i % len(default_colors)], key=f"colo_{filename}")

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
                fig.update_xaxes(
                    title_text=f"{element1}/{element2} Ratio",
                    title_font=dict(size=30, color='black'),
                    tickfont=dict(size=30, color='black'),
                    tickmode='array',
                    tickvals=list(range(0, int(x_max) + 1, 10)),
                    ticktext=[str(x) for x in range(0, int(x_max) + 1, 10)], 
                    row=i, col=1
                )
            
            fig.update_yaxes(
                title_text=f"Frequency",
                title_font=dict(size=30, color='black'),
                tickfont=dict(size=30, color='black'),
                linecolor='black',
                linewidth=2,
                row=i, col=1
            )

        fig.update_layout(
            title_text=title,
            height=300 * n_subplots,
            showlegend=False,
            font=dict(size=18, color="black")
        )

        st.plotly_chart(fig, use_container_width=True)

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


    def get_combinations(df_dict, elements):
        start_time = time.time()

        combination_data = {}
        particle_counts = {}
        processed_data = {}
        for filename, data in df_dict.items():
            df = data['df']
            _, _, _, mole_percent_data, _ = process_data(df)
            processed_data[filename] = mole_percent_data

            mole_percent_data = mole_percent_data.dropna(subset=elements, how='all')
            selected_data = mole_percent_data.apply(pd.to_numeric, errors='coerce')

            for index, row in selected_data.iterrows():
                elements_in_row = row[row > 0].index.tolist()
                top_elements = row.loc[elements_in_row].sort_values(ascending=False).head(4).index.tolist()
                combination_key = ', '.join(sorted(top_elements))
                if any(elem in elements_in_row for elem in elements): 
                    combination_data.setdefault(combination_key, []).append((filename, index))
                    particle_counts[(combination_key, filename)] = particle_counts.get((combination_key, filename), 0) + 1

        elapsed_time = time.time() - start_time
        st.write(f"Time taken: {elapsed_time} seconds")
        return combination_data, processed_data, particle_counts

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

    


    def summarize_data(data, threshold):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")

        total_data = data.sum()

        percentages = total_data * 100 / total_data.sum()

        main = percentages[percentages > threshold]
        others = percentages[percentages <= threshold]

        if others.sum() > 0:
            main['Others'] = others.sum()

        element_counts = (data > 0).sum(axis=0)
        total_counts = element_counts.sum()

        count_percentages = element_counts * 100 / total_counts

        main_counts = element_counts[count_percentages > threshold]
        others_counts = element_counts[count_percentages <= threshold].sum()

        if others_counts > 0:
            main_counts['Others'] = others_counts

        return main, main_counts


    # @st.cache_data
    def visualize_mass_and_mole_percentages_pie_charts(mass_data, mole_data, color_map, threshold, show_mass, show_mole,
                                             file_letter, filename):
        mass_percent, mass_counts = summarize_data(mass_data, threshold)
        mole_percent, mole_counts = summarize_data(mole_data, threshold)

        num_cols = sum([show_mass, show_mole])
        fig = sp.make_subplots(rows=1, cols=num_cols, specs=[[{'type': 'pie'}, ] * num_cols],
                            column_widths=[1] * num_cols, horizontal_spacing=0.001)

        col = 1
        summary_data = []

        if show_mass:
            colors_mass = [color_map.get(index, '#CCCCCC') for index in mass_percent.index]
            fig.add_trace(go.Pie(labels=mass_percent.index, values=mass_percent.values,
                                marker=dict(colors=colors_mass, line=dict(color='#000000', width=1)),
                                textinfo='label+percent',
                                texttemplate='%{label}: %{value:.1f}% (%{customdata})',
                                customdata=[mass_counts.get(index, 0) for index in mass_percent.index],
                                hoverinfo='label+percent+value',
                                hovertext=[f"{label}: {value:.1f}% ({mass_counts.get(label, 0):d} counts)" for label, value in mass_percent.items()],
                                textfont=dict(size=20, color='black'),
                                title=dict(text=f'Mass Percentages', font=dict(size=18, color='black'))), row=1, col=col)

            mass_summary_data = [['File name: ' + filename, '', ''], ['Element', 'Mass Percent (%)', 'Counts']]
            for element, value in mass_percent.items():
                mass_summary_data.append([element, f'{value:.2f}', mass_counts.get(element, 0)])
            summary_data.append(mass_summary_data)
            col += 1


        if show_mole:
            colors_mole = [color_map.get(index, '#CCCCCC') for index in mole_percent.index]
            fig.add_trace(go.Pie(labels=mole_percent.index, values=mole_percent.values,
                                marker=dict(colors=colors_mole, line=dict(color='#000000', width=1)),
                                textinfo='label+percent',
                                texttemplate='%{label}: %{value:.1f}% (%{customdata})',
                                customdata=[mole_counts.get(index, 0) for index in mole_percent.index],
                                hoverinfo='label+percent+value',
                                hovertext=[f"{label}: {value:.1f}% ({mole_counts.get(label, 0):d} counts)" for label, value in mole_percent.items()],
                                textfont=dict(size=20, color='black'),
                                title=dict(text=f'Mole Percentages', font=dict(size=18, color='black'))), row=1, col=col)

            mole_summary_data = [['File name: ' + filename, '', ''], ['Element', 'Mole Percent (%)', 'Counts']]
            for element, value in mole_percent.items():
                mole_summary_data.append([element, f'{value:.2f}', mole_counts.get(element, 0)])
            summary_data.append(mole_summary_data)
            col += 1

        fig.update_layout(title_text=f'Mass and Mole Percentages for File {file_letter}', title_x=0.385,
                        title_y=0.95, height=600, width=2000 * num_cols,
                        legend=dict(font=dict(size=16, color='black')))

        st.plotly_chart(fig)

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

        if merged_summary_data:
            summary_df = pd.DataFrame(merged_summary_data)
            csv = summary_df.to_csv(index=False, header=False)
            st.download_button(
                label=f"Download data as CSV {file_letter}",
                data=csv,
                file_name='mass_and_mole_percentages.csv',
                mime='text/csv'
            )


    def create_color_map(_elements, base_colors):
        color_map = {}
        for i, element in enumerate(sorted(_elements)):
            default_color = base_colors[i % len(base_colors)]
            color = st.sidebar.color_picker(f"Color {element}", value=default_color, key=f"colo_{element}")
            color_map[element] = color
        color_map['Others'] = st.sidebar.color_picker(f"Color Others", '#777777')
        return color_map

    def prepare_heatmap_data(data_combinations, combinations, start, end):
        heatmap_df = pd.DataFrame()
        combo_counts = {combo: info['counts'] for combo, info in combinations.items()}

        for combo, df in data_combinations.items():
            df = df.apply(pd.to_numeric, errors='coerce')
            avg_percents = df.mean().to_frame().T

            avg_percents = avg_percents.div(avg_percents.sum(axis=1), axis=0)

            combo_with_count = f"{combo} ({combo_counts[combo]})"
            avg_percents.index = [combo_with_count]
            heatmap_df = pd.concat([heatmap_df, avg_percents])

        heatmap_df['Counts'] = heatmap_df.index.map(lambda x: combo_counts[x.split(' (')[0]])
        heatmap_df = heatmap_df.sort_values(by='Counts', ascending=False)

        heatmap_df = heatmap_df.iloc[start - 1:end]
        heatmap_df.drop(columns=['Counts'], inplace=True)

        return heatmap_df

    def get_combinations_and_related_data(mole_percent_data, mass_data, mass_percent_data, mole_data):
        start_time = time.time()
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
            related_data['mass_percent_data'][combination_key] = mass_percent_data.loc[indices]
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
        st.write(f"Time taken for  {elapsed_time} seconds")

        return combinations, related_data['mole_percent_data'], related_data['mass_data'], related_data[
            'mass_percent_data'], related_data['mole_data'], sd_df

    def plot_heatmap(heatmap_df, sd_df, selected_colorscale='ylgnbu', display_numbers=True, font_size=14):
        elements_with_data = [col for col in heatmap_df.columns if heatmap_df[col].any()]
        heatmap_df = heatmap_df[elements_with_data]
        sd_df = sd_df[elements_with_data]

        elements = heatmap_df.columns.tolist()
        combinations_with_counts = heatmap_df.index.tolist()

        total_count = sum(int(comb.split('(')[-1].replace(')', '').strip()) for comb in combinations_with_counts)

        z_values = heatmap_df.values * 100
        min_val = z_values.min()
        max_val = z_values.max()

        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=elements,
            y=combinations_with_counts,
            colorscale=selected_colorscale,
            colorbar=dict(
                title='Mole %',
                titlefont=dict(size=20, color='black'),
                tickfont=dict(size=20, color='black'),
                ticks='outside',
                ticklen=5,
                tickwidth=2,
                tickcolor='black'
            )
        ))
        
        
        if display_numbers:

            for y, comb_with_count in enumerate(combinations_with_counts):
                comb = comb_with_count.split(' (')[0]
                for x, elem in enumerate(elements):
                    avg_value = heatmap_df.loc[comb_with_count, elem] * 100
                    sd_value = sd_df.loc[comb, elem] * 100 if elem in sd_df.columns and comb in sd_df.index else np.nan
                    color = text_color_based_on_background(avg_value, min_val, max_val)

                    if avg_value != 0:
                        annotation_text = f"{avg_value:.1f}\n±{sd_value:.1f}" if not np.isnan(sd_value) else f"{avg_value:.2f}"
                        fig.add_annotation(x=x, y=y, text=annotation_text, showarrow=False,
                                        font=dict(size=font_size, color=color))

        fig.update_layout(
            title=f'Molar Percentage After Treatment - Total Particles: {total_count}',
            xaxis=dict(title='Elements', tickangle=0,
                    title_font=dict(size=24, color='black'),
                    tickfont=dict(size=24, color='black'),
                    ),
            yaxis=dict(title='Particle (Frequency)', autorange='reversed',
                    title_font=dict(size=24, color='black'),
                    showgrid=False,
                    tickfont=dict(size=24, color='black')),
            height=max(600, 40 * len(combinations_with_counts)),
        )

        return fig

    if 'data_dict' in globals():
        display_summary_table(data_dict, file_letter_map, dilution_factors, acquisition_times)

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
            mass_data, _, mole_data, _, _ = process_data(df)
            available_elements = mole_data.columns

            color_map = create_color_map(available_elements, default_colors)
            threshold = st.sidebar.number_input('Threshold for Others (%)', format="%f", key='threshold_others')

            for filename, data in data_dict.items():
                df = data['df']
                mass_data, _, mole_data, _, _ = process_data(df)
                visualize_mass_and_mole_percentages_pie_charts(mass_data, mole_data, color_map, threshold, show_mass,
                                                            show_mole, file_letter_map[filename], filename)

        st.sidebar.title('Mass Distribution Analysis')
        perform_mass_distribution_analysis = st.sidebar.checkbox('Perform Mass Distribution Analysis?', key='perform_mass_distribution_analysis')

        if perform_mass_distribution_analysis:
            sample_filename = next(iter(data_dict))
            df = data_dict[sample_filename]['df']
            mass_data, _, _, _, _ = process_data(df)

            element_to_plot = st.sidebar.selectbox('Select an element to view the histogram:', mass_data.columns, key='mass_element_to_plot')
            detection_type = st.sidebar.selectbox('Select detection type Mass:', ['All', 'Single', 'Multiple'], index=0, key='mass_detection_type')
            bin_size = st.sidebar.slider('Select bin size:', min_value=0.001, max_value=10.0, value=0.01, step=0.01, key='mass_bin_size')
            x_max = st.sidebar.slider('Select max value for x-axis (Mass (fg)):', min_value=0, max_value=1000, value=10, step=1, key='mass_x_max')

            plot_histogram_for_element(data_dict, element_to_plot, detection_type, bin_size, x_max, "Mass Distribution", file_letter_map)

        st.sidebar.title('Distribution Analysis')
        perform_distribution_analysis = st.sidebar.checkbox('Perform Distribution Analysis?', key='perform_distribution_analysis')

        if perform_distribution_analysis:
            sample_filename = next(iter(data_dict))
            df = data_dict[sample_filename]['df']
            mass_data, _, _, _, _ = process_data(df)

            element_to_plot = st.sidebar.selectbox('Select an element to view:', mass_data.columns, key='distribution_element_to_plot')
            detection_type = st.sidebar.selectbox('Select detection type:', ['All', 'Single', 'Multiple'], index=0, key='distribution_detection_type')
            distribution_type = st.sidebar.selectbox('Select distribution type:', 
                                                    ['Poisson', 'Gaussian', 'Exponential', 'Binomial', 'Log-normal',
                                                    'Gamma', 'Weibull', 'Cubic Spline'], index=0, key='distribution_type')

            plot_distribution(data_dict, element_to_plot, detection_type, distribution_type, "Mass Distribution", file_letter_map)

        st.sidebar.title('Ternary Diagrams')
        perform_ternary_analysis = st.sidebar.checkbox('Perform Ternary Analysis?', key='perform_ternary_analysis')

        if perform_ternary_analysis:
            sample_filename = next(iter(data_dict))
            df = data_dict[sample_filename]['df']
            _, _, mole_data, _, _ = process_data(df)
            available_elements = mole_data.columns.tolist()

            elements_to_plot = st.sidebar.multiselect('Select three elements for the ternary plot:', available_elements, default=available_elements[:3], key='ternary_elements_to_plot')

            if len(elements_to_plot) == 3:
                plot_ternary_heatmap(data_dict, elements_to_plot, "Ternary Diagrams", file_letter_map)
            else:
                st.warning("Please select exactly three elements for the ternary plot.")

        st.sidebar.title("Mole Ratio Analysis")
        perform_mole_ratio_analysis = st.sidebar.checkbox('Perform Mole Ratio Analysis?', key='perform_mole_ratio_analysis')

        if perform_mole_ratio_analysis:
            sample_filename = next(iter(data_dict))
            df = data_dict[sample_filename]['df']
            _, _, mole_data, _, _ = process_data(df)

            elements = mole_data.columns.tolist()
            element1_to_plot = st.sidebar.selectbox('Select Element 1 for the Ratio', elements, key='mole_ratio_element1')
            element2_to_plot = st.sidebar.selectbox('Select Element 2 for the Ratio', elements, index=1 if len(elements) > 1 else 0, key='mole_ratio_element2')
            bin_size = st.sidebar.slider('Bin Size', min_value=0.001, max_value=25.0, value=0.01, step=0.01, key='mole_ratio_bin_size')
            x_max = st.sidebar.slider('Max X-axis Value', min_value=0, max_value=1000, value=25, step=1, key='mole_ratio_x_max')
            title = "Mole Ratio Analysis"

            plot_mole_ratio_histogram_for_files(data_dict, element1_to_plot, element2_to_plot, bin_size, x_max, title, file_letter_map)

        st.sidebar.title('Heatmap Analysis')
        perform_heatmap_analysis = st.sidebar.checkbox('Perform Heatmap Analysis?', key='perform_heatmap_analysis')

        if perform_heatmap_analysis:
            selected_file = st.sidebar.selectbox('Select file for heatmap:', list(data_dict.keys()), key='heatmap_selected_file')

            if selected_file:
                df = data_dict[selected_file]['df']
                mass_data, mass_percent_data, mole_data, mole_percent_data, counts_data = process_data(df)
                available_elements = mole_percent_data.columns.tolist()
                combinations, mole_percent_combinations, _, _, _, sd_df = get_combinations_and_related_data(mole_percent_data, mass_data, mass_percent_data, mole_data)

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
                start = st.sidebar.number_input('Start from combination:', min_value=1,
                                                    max_value=len(mole_percent_data) - 1, value=1,  key='heatmap_start_combination')
                end = st.sidebar.number_input('End at combination:', min_value=2, max_value=len(mole_percent_data), key='heatmap_end_combination')
                end = max(end, start + 1)

                heatmap_df = prepare_heatmap_data(mole_percent_combinations, combinations,
                                                start, end)

                heatmap_fig = plot_heatmap(heatmap_df, sd_df, selected_colorscale=selected_colorscale, display_numbers=display_numbers, font_size=font_size)
                st.plotly_chart(heatmap_fig, use_container_width=True)


        # st.sidebar.title('3D Heatmap Analysis')
        #
        # perform_3d_heatmap_analysis = st.sidebar.checkbox('Perform 3D Heatmap Analysis?')
        #
        # if perform_3d_heatmap_analysis:
        #     sample_filename = next(iter(data_dict))
        #     df = data_dict[sample_filename]['df']
        #     _, _, mole_percent_data, _, _ = process_data(df)
        #     available_elements = mole_percent_data.columns.tolist()
        #
        #     elements_to_plot = st.sidebar.multiselect('Select elements for the 3D heatmap:', available_elements,
        #                                               default=available_elements[:1])
        #
        #     start = st.sidebar.number_input('Start combination (1-based index):', min_value=1, value=1)
        #     end = st.sidebar.number_input('End combination (1-based index):', min_value=1, value=10)
        #     threshold = st.sidebar.number_input('Threshold for number of data points:', min_value=0, value=0, step=1)
        #
        #     if elements_to_plot:
        #         plot_3d_stacked_heatmap(data_dict, elements_to_plot, file_letter_map, start, end, threshold)
        #     else:
        #         st.warning("Please select at least one element for the 3D heatmap.")
        


################ Tab3 analyse isotopic ratio 


with tab3:
    ff = st.file_uploader(':file_folder: Upload a file', type=['csv'], key='isotopic')
    if ff :
        with st.sidebar:
            st.title("Isotopic Ratio Options")
    
    ### Logic
    
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
            #print(sensitivity_dict)
        
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

            # st.write("Column names in the DataFrame after header adjustment:", data.columns.tolist())

            elements = set(col.split(' ')[0] for col in data.columns if 'fwhm' in col)

            for element in elements:
                count_col = f'{element} counts'
                mole_col = f'{element} moles [fmol]'
                mass_col = f'{element} mass [fg]'
                fwhm_col = f'{element} fwhm'
                mole_per = f'{element} mole %'
                mass_per = f'{element} mass %'

                if all(col in data.columns for col in [count_col, mole_col, mass_col, fwhm_col, mole_per, mass_per]):
                    # st.write(f"Found all required columns for {element}. Cleaning data...")
                    data.loc[data[fwhm_col].isna(), [count_col, mole_col, mass_col, mass_per, mole_per]] = 0
                # else:
                #     st.write(f"Required columns for {element} not found in DataFrame.")

            cleaned_dd = dd.copy()
            cleaned_dd.iloc[start_index + 1:, :] = data.values

            return cleaned_dd
        else:
            st.error("Header row with 'fwhm' not found.")
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
                #st.write("Before Cleaning:")
                #st.dataframe(dd)

                dd = clean_data(dd)
                
                #sensitivity_data = extract_sensitivity_calibration(dd)
                #if sensitivity_data:
                    #st.write("Found sensitivity calibration data:", sensitivity_data)
                #else:
                    #st.write("No sensitivity calibration data found in the file")

                if dd is not None:
                    #st.write("After Cleaning:")
                    #st.dataframe(dd)
                    
                    dilution_factor = st.number_input('Enter Dilution Factor:', format="%f", value=1.0, key='single_dilution_isotope')
                    acquisition_time = st.number_input('Enter Total Acquisition Time (in seconds):', format="%f", value=60.0, key='single_acquisition_isotope')

                    
                    event_number_cell = count_rows_after_keyword_until_no_data(dd, 'event number', column_index=0)
                    if event_number_cell is not None:
                        st.write(f'Total Particles Count: {event_number_cell} Particles')
                    else:
                        st.write('Event number not found or no valid count available.')

                    transport_rate_cell = find_value_at_keyword(dd, 'calibrated transport rate', column_index=1)
                    transport_rate = extract_numeric_value_from_string(str(transport_rate_cell))
                    if transport_rate is not None:
                        st.write(f'Calibrated Transport Rate: {transport_rate} µL/s')
                    else:
                        st.write('Calibrated transport rate not found or no valid rate available.')

                    if event_number_cell is not None and transport_rate_cell is not None:
                        particles_per_ml = calculate_particles_per_ml(event_number_cell, transport_rate_cell,
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


    def get_poisson_confidence_interval(counts, confidence_level):
        lower_bound = poisson.ppf((1 - confidence_level) / 2, counts)
        upper_bound = poisson.ppf(1 - (1 - confidence_level) / 2, counts)
        return lower_bound, upper_bound

    #@st.cache_resource
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
    #@st.cache_resource
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

    #@st.cache_data
    #@st.cache_resource
    def plot_isotopic_ratio(counts_data, mass_data, element1, element2, x_axis_element, color, x_max, title, 
                       line_y_value=None, line_y_value_2=None, adjust_to_natural=False):
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
            x=filtered_mass[x_axis_element], y=ratios,
            mode='markers', marker=dict(size=12, color=color, line=dict(width=2)),
            name=f"Ratio {element1}/{element2}"
        ))

        num_bins = 100 
        bins = np.linspace(0, x_max, num=num_bins)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        lower_68_list = []
        upper_68_list = []
        lower_95_list = []
        upper_95_list = []

        for b_start, b_end in zip(bins[:-1], bins[1:]):
            bin_mask = (filtered_mass[x_axis_element] >= b_start) & (filtered_mass[x_axis_element] < b_end)
            bin_counts = filtered_counts[bin_mask]
            if len(bin_counts) > 0:
                bin_ratios = bin_counts[element1] / bin_counts[element2]
                lower_68, upper_68 = get_poisson_confidence_interval(bin_counts[element1], 0.68)
                lower_95, upper_95 = get_poisson_confidence_interval(bin_counts[element1], 0.95)
                
                if adjust_to_natural and adjustment_factor is not None:
                    lower_68_list.append((lower_68.mean() / bin_counts[element2].mean()) * adjustment_factor)
                    upper_68_list.append((upper_68.mean() / bin_counts[element2].mean()) * adjustment_factor)
                    lower_95_list.append((lower_95.mean() / bin_counts[element2].mean()) * adjustment_factor)
                    upper_95_list.append((upper_95.mean() / bin_counts[element2].mean()) * adjustment_factor)
                else:
                    lower_68_list.append(lower_68.mean() / bin_counts[element2].mean())
                    upper_68_list.append(upper_68.mean() / bin_counts[element2].mean())
                    lower_95_list.append(lower_95.mean() / bin_counts[element2].mean())
                    upper_95_list.append(upper_95.mean() / bin_counts[element2].mean())
            else:
                lower_68_list.append(np.nan)
                upper_68_list.append(np.nan)
                lower_95_list.append(np.nan)
                upper_95_list.append(np.nan)

        bin_centers_nonan = bin_centers[~np.isnan(lower_68_list)]
        lower_68_list_nonan = np.array(lower_68_list)[~np.isnan(lower_68_list)]
        upper_68_list_nonan = np.array(upper_68_list)[~np.isnan(upper_68_list)]
        lower_95_list_nonan = np.array(lower_95_list)[~np.isnan(lower_95_list)]
        upper_95_list_nonan = np.array(upper_95_list)[~np.isnan(upper_95_list)]

        smoothed_lower_68 = lowess(lower_68_list_nonan, bin_centers_nonan, frac=0.7, missing='drop')[:, 1]
        smoothed_upper_68 = lowess(upper_68_list_nonan, bin_centers_nonan, frac=0.7, missing='drop')[:, 1]
        smoothed_lower_95 = lowess(lower_95_list_nonan, bin_centers_nonan, frac=0.7, missing='drop')[:, 1]
        smoothed_upper_95 = lowess(upper_95_list_nonan, bin_centers_nonan, frac=0.7, missing='drop')[:, 1]

        fig.add_trace(go.Scatter(
            x=bin_centers_nonan, y=smoothed_upper_68,
            mode='lines', line=dict(color='rgba(128,0,128,0)'), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=bin_centers_nonan, y=smoothed_lower_68,
            mode='lines', line=dict(color='rgba(128,0,128,0)'), fill='tonexty', 
            fillcolor='rgba(128,0,128,0.3)', 
            name='68% CI' + (" (Adjusted)" if adjust_to_natural else "")
        ))
        fig.add_trace(go.Scatter(
            x=bin_centers_nonan, y=smoothed_upper_95,
            mode='lines', line=dict(color='rgba(255,165,0,0)'), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=bin_centers_nonan, y=smoothed_lower_95,
            mode='lines', line=dict(color='rgba(255,165,0,0)'), fill='tonexty', 
            fillcolor='rgba(255,165,0,0.3)', 
            name='95% CI' + (" (Adjusted)" if adjust_to_natural else "")
        ))

        if line_y_value is not None:
            fig.add_trace(go.Scatter(
                x=[0, x_max], y=[line_y_value, line_y_value],
                mode='lines', line=dict(dash='dot', width=5, color='blue'),
                name='Natural Abundance'
            ))

        if line_y_value_2 is not None:
            fig.add_trace(go.Scatter(
                x=[0, x_max], y=[line_y_value_2, line_y_value_2],
                mode='lines', line=dict(dash='dot', width=5, color='green'),
                name='Standard Ratio' if not adjust_to_natural else 'Adjusted Standard Ratio'
            ))

        fig.add_trace(go.Scatter(
            x=[0, x_max], y=[mean_ratio, mean_ratio],
            mode='lines', line=dict(dash='solid', width=3, color='red'),
            name='Mean Ratio' + (" (Adjusted)" if adjust_to_natural else "")
        ))

        fig.update_layout(
            title=title,
            xaxis_title=f"Mass of {x_axis_element} (fg)",
            yaxis_title=f"Ratio {element1}/{element2}" + (" (Adjusted to Natural Abundance)" if adjust_to_natural else ""),
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
        summary_data.append([f'Mass (fg) {x_axis_element}', f'{element1}/{element2} Ratio', ''])
        
        for mass, ratio in zip(filtered_mass[x_axis_element], ratios):
            summary_data.append([mass, ratio, ''])
        
        if summary_data:
            summary_dd = pd.DataFrame(summary_data)
            csv = summary_dd.to_csv(index=False, header=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='isotopic_ratio_data.csv',
                mime='text/csv'
            )
        
    if 'dd' in globals():
        
        mass_data, _, mole_data, _, counts_data = process_data(dd)

        st.sidebar.title("Mole Ratio")
        perform_mole_ratio_analysis = st.sidebar.checkbox('Perform Mole Ratio Analysis?', key='perform_ratio_analysis')

        if perform_mole_ratio_analysis:
            element1_to_plot = st.sidebar.selectbox('Select Element 1 for the Ratio', mole_data.columns, key='mole_element1')
            element2_to_plot = st.sidebar.selectbox('Select Element 2 for the Ratio', mole_data.columns, index=1 if len(mole_data.columns) > 1 else 0, key='mole_element2')
            color = st.sidebar.color_picker('Pick a Color for the Histogram', '#DCAD7A', key='mole_color')
            bin_size = st.sidebar.slider('Bin Size', min_value=0.001, max_value=10.0, value=0.01, step=0.01, key='mole_bin_size')
            x_max = st.sidebar.slider('Max X-axis Value', min_value=0, max_value=100, value=25, step=1, key='mole_x_max')
            title = "Molar Ratio"

            plot_mole_ratio_histogram(mole_data, element1_to_plot, element2_to_plot, color, bin_size, x_max, title)


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
            x_max = st.sidebar.slider('Max X-axis Value Isotopic', min_value=0, max_value=1000, value=25, step=1, 
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


            plot_isotopic_ratio(counts_data, mass_data, element1_to_plot, element2_to_plot, 
                       x_axis_element, color, x_max, title, 
                       line_y_value=line_value, line_y_value_2=line_value_2,
                       adjust_to_natural=adjust_to_natural)


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
            
            
    
