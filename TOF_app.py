"""


Université de Montréal


Département de Chimie 


Groupe Prof. Kevin J. Wilkinson 
Boiphysicochimie de l'environnement 


Aut: H-E Ahabchane

Date : 01/04/2024
Update : 
"""

###
import streamlit as st 
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import re  
import time
warnings.filterwarnings('ignore')
###


st.set_page_config(page_title='TOF-ICP-MS Analysis', page_icon=':atom_symbol:', layout='wide')

# Groupe Prof. Kevin J. Wilkinson Biophysicochimie de l'environnement - Département de chimie - Université de Montréal - logo 

logo_url = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBwgHBgkIBwgKCgkLDRYPDQwMDRsUFRAWIB0iIiAdHx8kKDQsJCYxJx8fLT0tMTU3Ojo6Iys/RD84QzQ5OjcBCgoKDQwNGg8PGjclHyU3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3N//AABEIAFwAXAMBEQACEQEDEQH/xAAcAAEAAgIDAQAAAAAAAAAAAAAABQcEBgIDCAH/xAA6EAABAwMCBAQDBAgHAAAAAAABAgMEAAURBiESEzFBByJRYRQykRVxgaEWM0JSYqKxwSMkQ3OS4fD/xAAbAQEAAgMBAQAAAAAAAAAAAAAAAQQDBQYCB//EACsRAAICAgEDAwIGAwAAAAAAAAABAgMEETEFEiETQVEyYQYUI3GRoSKBwf/aAAwDAQACEQMRAD8AvCvJIoBQCgIyDqC0T7hJt8S4R3JsZZQ9H4sLSR18p3I9xtUgk6gCgFAKAUAoBQCgFAQeo9X2LTK2G71PEdb4UptPLUskDqcJBxUgkLTdIV4gNTrZIRIjOpCkrT/cdQfY70B5q8VmHIXiLeS0VNrLqHm1pOCCptJyD9+akgs6/eLH6PM2lgWpU1cq2MS+aZHAPOCMfKe6agksPT1z+2bFb7oGuV8ZGbe5fFxcHEkHGe+M0BIUAqAKAUAoBQCgKQ8StBaq1Drl+VBiB2C8ltLTy5ACGgEAHIJyNwThIOc+uakgn9Fw7d4VW+Y3qi7xW350lJbDalKCkBIAIRjiG5Vk4x0oCv8AxrUxJ1excYTqHo023tOtuoOQsZUMj6CpBFavWJFg0dJG5NsXHJ/2nVJ/vQk3lnxJlaR0hpSNHtzMsP2/iUXHSgp4FcGBgH0oQWzpm6G96ft91UyGTLYS8Wwri4cjOM96hkknUAUAoBQCgFAKkHnTx1gmLrsyMHhlxG3AfcZSR/KPrQg1yY45eLFYI8Vtx+VCRIYdCUnyoLgWjJ6dFKH4VkhXOf0oxW31VebJJGXLtl0l6etNv+BKXIC5B41OpwpLikqAAz2IP1rOsK/4Kb6tiJ/UYl/TNNrs8aRCfb+z47jK1kApPE6pYwQT2I64rFOiyH1RZYqzKLvokj0X4dFP6B2DhUCBAayQc78IzWEtE+l5paihDqFKHVIUCRQHZQCoBxUkk5CsUByoCG1bqOHpayPXOdlSUEJbaT8zqz0SP/bDJqSCp7Nr5nVE/kagvFyty3VcLMeIv4eON9k8xJ4yfdRA9hWp6nZn1xcsZLS/ky1qD+ol9UeHbV3baeYuM1yRGB5Tc+Qt9tQzkpJUeIA47H8K0eJ+JLYWJZEU1/BksxlKLUHpmvwVDlKZ+H+GdYWWnY+McpY6j/vuDX1LCyKsilWVcM+e9Qx7aL3G17fyZFWyicVqShClrOEpGVE9hXmTSXk9VxbklHkhoi2UJJmTHYMOceNuAh5bbTmP2lgHBUc5x9c1x+ZlStsl6K8HZU+rXUq9ttcs6r+m1Wb4QM2xlLzq8JWyOWtABGSCMHO+29VMd22Nvu4MlLsnt93BZOh9TzYt1Ysd5krlMSciFKdOXErAzy1n9rIBIV12wc7Vax7/AFVp8lim71OeSyasFgVAFAVJ432+5Xy76bs9vTxB7nueY4SCngHET7An61hyMivGqdtnCJjFyekV3rDw/n6YgtTHJLUyMpQQ4ptBSW1HoMEnIPr+Va/p/Wqc2bhFaa+T3ZU4rZtnh5cNTGY1bFSFSm2eAy/iBlMRoZw2CNy6fQ54QPvxq+s04SrdjWm+Nct/P7GWqU96OzVj8aFrG5LcdbabVGjqWVHGXPMPxPCE/lXRfg+3twW7H434Od/EFErrIKtbfk7IVq1BcI5kwbDKVHxlKpC0sKcH8KVHP1xXRS6jBPSWzWw6FbKO5SSZCXhxT1muTfKdakMoUh5hxPC42oDdJH3fWstlqux5OBXpxZYubCNvycdTWkXi1tri+Z1pPEyAdlpI6f0ri6LfSsal7nQVWenNqRENRlzk2z7YdUy9CWtLoV1CUgLBV+A6+lWHJQ7uz3M7ko93Z52bZHcVMudkEZDiHXLmwW+NPCSAoKUcdflCutYsWLVpjx4tWF51szYCoAoCA1dZn7pFjyLa4hu5wXedFU58ijjCm1fwqTke2x7VhyaIZFTqnwyYtxezVLhfrVPhvWq8v/YlxUjdqaEpW0rstBV5VYO4UD9K4t9MzcG9TjHvivj3Rb9SE48mozhYLO7CRpy5zprafLNh26Y4nnerpcSeEL9id+m1b/plWZmTk8qja9m1x9v2KmRfTRHbnosTRVv0bKJuVhaakTB+tdlKU7KbPormEqSa3agoLtS1r2PKl3LuT2Yd+iarOtG5cR2U1p0Ox/ikplNgqwCStIUPK2NuMZyrG1SDVL/cIt61ZcLlbyFwi03GS4PlfUji4lj1Hm4c9+H0rb9Orfa5PhnNdcvi5xhHlEHypUBsR20vuwc+XkKAdaH7u/VP3bjpWuzekS73ZSt/YnG6hVYkrPEv6ZDXDmyL3bm4bUyQXnUtJbebKFrOdkcSsBXfqdsneqcce2EGpx0biiUZxai1/osvwo+zrldH586U19tx+Y0i2k+aIgK4VK/iUcDKhsBt60ppVUdIsVVKtaRatZjKKgCgFAVn4rQ+VebRcVISWHm3IbhIzhWy0fXCxV7Aklbp+5qur1ylj90eUawAEjCQAPQVvVFLg46UpPlmHcGYKQJkvhZW38shKyhafuUCDWC+ulrutLuHflxl2UNt/BiLcRLb/wAdm+T2AflkOPOI/wCK1b/StR+b6dCWmzofyXW7a960Z8KXGkpKI6sFvZTZSUKR7FJ3Fbii6q2O62c3l4uRjy1dFpmTWcqGRp6EbnrWxspGREcXNcP7qUp4R/MpP51q+pSWoxOh6DW+6c/bggNd2ib4da5YvNoymK+6X4x34Qf9RpXtufwPtWpOlL305eomobLFukBWWZCM8J6oV3SfcHaoJJKoAoBQEXqaysags0i3SVFAcAKHEjdtYOUrHuCAalNp+DzKKktMpz/MRJr1sujYZuMf9Y32cT2cR6pP5dDuK6HGyY3R+5xfUMCWNNtfT7ETIeTy5N0eWjLb3wsEOfKhWeFTmPXOfuCa5rqmTO7J9FcLk7XoODXi4X5iS/ykZouAbt/IsMeRLUk8AfCMo4s+ZXErHEep271pXVuzuuaX2Ok9bVfbQm/ucpcF1+Hz22XmpsZJUy88tJW53KVYJ2Pp27dKy4uXLGuUovx/RXzcCGZjuuyPnXPufftBj4FmXuUvJSW0JGVLUrolI7k9MV3rvjGv1JHyeOHZK90xXlMs7w8049Z4Ts65ICbnPwp1GQeQ2Plbz7ZJPuTWgutds+5naYuPHHqVcST1npuNqqwSLXJIQpQ4mXsZLTg+VQ/ofUE1iLJCeF2jbjo6DNYuFxbkiS6FoZZB4GyBgkE9zt27CgN3qAKAUAoCD1Tpa2anipauLakut5LEllXC6yfVKv7HapTae0eZRUlpooG/6Ju0O+S7OZyXhGVzWC+Snmtr34wBkdcg+4+6q998KX3SXPuW8fGlfHsi/C9jOtsW+MsN25dxRHdaRltAYSpK2wQNl+u+OnfvWttnjyfqKO0za015MV6blpr7f9MxSFwmHpcgy/ikeVKFqS6HVHZKUHh7nbAwa8x/UkoQS0zJL9KDsm3tfPn+Cy9CaBh6ejRJM5Spl0baCQ47jhj7bpbT0HpxdT69q30pyaSb4OZjXCLbiuTda8GQUAoBQCgFAKAUBCam0xA1Ey0JXMZksEmPLYPC60T1we4PcHINRKKktSXg9RnKD3FmoOaB1ChzDN2tjyRsl16KtKwPcJVg/hiqT6dU+GzYLqlyXlLZNad0Izbpzdyu0xVynNbs5QEMsH1Qjfze5JPpirNVEKVqCKd+RZe9zZuFZTCKAUAoBQCgFAKAUAoBUgUAqAKAUAoBQH//2Q=="


st.markdown(f"""
    <a href="https://kevinjwilkinson.openum.ca">
        <img src="{logo_url}" alt="Groupe Prof. Kevin J. Wilkinson Biophysicochimie de l'environnement - Département de chimie - Université de Montréal - Logo" style="height:100px; display: block; margin-top: 20px;">
    </a>
    """, unsafe_allow_html=True)


# Page title
st.title(""":atom_symbol: TOF-ICP-MS Analyse des données""")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
fl = st.file_uploader(':file_folder: Upload a file', type=['csv', 'txt', 'xlsx', 'xls'])

#  bottom bar
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
                  Groupe Prof. Kevin J. Wilkinson Biophysicochimie de l'environnement - Département de chimie - Université de Montréal
</div>
""",
unsafe_allow_html=True
)

### Logic


def find_last_value_after_keyword(data, keyword, column_index=1):
    keyword_found = False
    last_value = None
    for value in data.iloc[:, column_index]:
        if keyword_found:
            if pd.notna(value):
                last_value = value
        elif keyword.lower() in str(value).lower():
            keyword_found = True
    return last_value


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

dilution_factor = st.number_input('Enter Dilution Factor:', format="%f")
acquisition_time = st.number_input('Enter Total Acquisition Time (in seconds):', format="%f")



def find_start_index(df, keyword, column_index=0):
    for i, value in enumerate(df.iloc[:, column_index]):
        if keyword.lower() in str(value).lower():
            return i
    return None



def process_data(df, keyword='event number'):
    start_index = find_start_index(df, keyword)
    if start_index is not None:
        
        new_header = df.iloc[start_index]
        data = df.iloc[start_index+1:].reset_index(drop=True)
        data.columns = [str(col) for col in new_header]  

       
        mass_data_cols = [col for col in data.columns if 'mass' in col and 'total' not in col and not col.endswith('mass %')]
        mass_percent_data_cols = [col for col in data.columns if col.endswith('mass %')]
        mole_data_cols = [col for col in data.columns if 'mole' in col and 'total' not in col and not col.endswith('mole %')]
        mole_percent_data_cols = [col for col in data.columns if col.endswith('mole %')]

        # DataFrame 
        mass_data = data[mass_data_cols].rename(columns=lambda x: x.split(' ')[0]) 
        mass_percent_data = data[mass_percent_data_cols].rename(columns=lambda x: x.split(' ')[0])
        mole_data = data[mole_data_cols].rename(columns=lambda x: x.split(' ')[0])
        mole_percent_data = data[mole_percent_data_cols].rename(columns=lambda x: x.split(' ')[0])

        return mass_data, mass_percent_data, mole_data, mole_percent_data
    
    else:
        return None, None, None, None
    


if fl is not None:
    filename = fl.name
    st.write('Uploaded File: ', filename)
    if 'csv' in filename:
        df = pd.read_csv(fl)
    elif 'xlsx' in filename or 'xls' in filename:
        df = pd.read_excel(fl)
    else:
        st.error('File format not supported. Please upload a CSV or Excel file.')
        df = None

    if df is not None:
        
        event_number_cell = find_last_value_after_keyword(df, 'event number', column_index=0)
        if event_number_cell is not None:
            st.write(f'Total Particles Count: {event_number_cell} Particles')
        else:
            st.write('Event number not found or no valid count available.')

        transport_rate_cell = find_value_at_keyword(df, 'calibrated transport rate', column_index=1)
        transport_rate = extract_numeric_value_from_string(str(transport_rate_cell))
        if transport_rate is not None:
            st.write(f'Calibrated Transport Rate: {transport_rate} µL/s')
        else:
            st.write('Calibrated transport rate not found or no valid rate available.')
            
        if event_number_cell is not None and transport_rate_cell is not None:
            # Calculating particles per ml
            particles_per_ml = calculate_particles_per_ml(event_number_cell, transport_rate_cell, acquisition_time, dilution_factor)
            if particles_per_ml is not None:
                st.write(f'Particles per ml: {particles_per_ml} Particles/mL')
            else:
                st.write('Error in calculation. Please check input values.')
        else:
            st.write('Required data not found in file.')



def plot_pie_chart_with_table(data, threshold, color_choices, title):
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
    data_totals = data.sum()

    if data_totals.sum() == 0:
        st.write("No data available for plotting.")
        return

    data_percent = (data_totals / data_totals.sum()) * 100
    filtered_data = data_percent[data_percent > threshold].copy()
    others_details = data_percent[data_percent <= threshold].copy() if filtered_data.sum() < 100 else pd.Series()
    others_details.sort_values(inplace=True, ascending=False)
    if filtered_data.sum() < 100:
        filtered_data['Autres'] = 100 - filtered_data.sum()

    plot_data = pd.DataFrame({'Element': filtered_data.index, 'Percentage': filtered_data.values})

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'table'}]],column_widths=[0.9, 0.3])
    fig.add_trace(go.Pie(labels=plot_data['Element'], values=plot_data['Percentage'], name=title,
                         marker_colors=[color_choices.get(element, '#000000') for element in plot_data['Element']],
                         textinfo='label+percent',  
                         insidetextorientation='radial'  
                         ),
                  1, 1)

    # Add table for 'Others' 
    if not others_details.empty:
        others_data = pd.DataFrame({'Element': others_details.index, 'Percentage': others_details.values})
        fig.add_trace(go.Table(header=dict(values=["Élément", "Pour.(%)"]),
                               cells=dict(values=[others_data.Element, others_data.Percentage.round(2)])),
                      1, 2)
        fig.update_traces(domain=dict(x=[0.1, 0.1]), selector=dict(type='pie'))
        fig.add_annotation(text=f"Élément inférieur à {threshold}% :",
                       xref="paper", yref="paper",
                       x=0.94, y=1.05, showarrow=False,
                       font=dict(size=14, color="black"),
                       align="right")

    fig.update_layout(title_text=title, showlegend=True)
    pio.write_image(fig, 'figure_high_res.png', width=1920, height=1080, scale=2)
    st.plotly_chart(fig)



def plot_histogram_for_element(data, element, color, bin_size, x_max, title):
    # Convert the selected column to numeric type
    data[element] = pd.to_numeric(data[element], errors='coerce')
    
    # Filtering out zero and NaN values
    filtered_data = data.dropna(subset=[element]).loc[data[element] > 0]
    
    # Creating the histogram with the specified settings
    fig = px.histogram(filtered_data, x=element, title=f"{title}: {element}",
                       nbins=int(x_max/bin_size),  # Calculate number of bins based on the range and bin size
                       color_discrete_sequence=[color],
                       labels={'x': 'Mass (fg)', 'y': 'Fréquence'},  # Custom axis labels
                       range_x=[0, x_max])
    
    # Update layout to adjust axis titles
    fig.update_layout(xaxis_title="Mass (fg)", yaxis_title="Fréquence")
    pio.write_image(fig, 'figure_high_res.png', width=1920, height=1080, scale=2)
    # Showing the histogram in the Streamlit app
    st.plotly_chart(fig)

# Assuming data_type, mass_data, and other variables are defined as in your code snippet




def get_combinations_and_related_data(mole_percent_data, mass_data, mass_percent_data, mole_data):
    start_time = time.time()
    combinations = {}
    raw_combination_data = {}
    related_data = {
        'mass_data': {},
        'mass_percent_data': {},
        'mole_data': {},
        'mole_percent_data': {}  
    }
    
    mole_percent_data = mole_percent_data.apply(pd.to_numeric, errors='coerce')
    
    
    for index, row in mole_percent_data.iterrows():
        elements = [element for element in row.index if row[element] > 0]
        combination_key = ', '.join(sorted(elements))
    
        if combination_key not in raw_combination_data:
            raw_combination_data[combination_key] = [row[elements]]
        else:
            raw_combination_data[combination_key].append(row[elements])
        
        if combination_key not in combinations:
            combinations[combination_key] = {
                'sums': row[elements],
                'counts': 1,
                'average': row[elements],
                'squared_diffs': (row[elements] - row[elements]) ** 2 
            }
        else:
            combinations[combination_key]['sums'] += row[elements]
            combinations[combination_key]['counts'] += 1
            combinations[combination_key]['average'] = combinations[combination_key]['sums'] / combinations[combination_key]['counts']
            combinations[combination_key]['squared_diffs'] += (row[elements] - combinations[combination_key]['average']) ** 2
        
        for data_type, data in [('mass_data', mass_data), ('mass_percent_data', mass_percent_data), ('mole_data', mole_data), ('mole_percent_data', mole_percent_data)]:
            selected_data = data.loc[[index]]
            if combination_key not in related_data[data_type]:
                related_data[data_type][combination_key] = [selected_data] 
            else:
                related_data[data_type][combination_key].append(selected_data)

    # Calculate standard deviation for each combination
    for key, value in combinations.items():
        value['sd'] = np.sqrt(value['squared_diffs'] / value['counts'])

    # Prepare the data for return dataframe
    sd_data = {key: value['sd'] for key, value in combinations.items()}
    sd_df = pd.DataFrame(sd_data).transpose()

    
    # Structure each type of data after combination dictionary
    mass_data_after_combination = {key: pd.concat(value) for key, value in related_data['mass_data'].items()}
    mass_percent_data_after_combination = {key: pd.concat(value) for key, value in related_data['mass_percent_data'].items()}
    mole_data_after_combination = {key: pd.concat(value) for key, value in related_data['mole_data'].items()}
    mole_percent_data_after_combination = {key: pd.concat(value) for key, value in related_data['mole_percent_data'].items()}

    averages_df = pd.DataFrame({k: v['average'] for k, v in combinations.items()})
    
    elapsed_time = time.time() - start_time
    st.write(f"Time taken for  {elapsed_time} seconds") 
    
    return combinations, averages_df.transpose(), mole_percent_data_after_combination, mass_data_after_combination, mass_percent_data_after_combination, mole_data_after_combination, sd_df



def prepare_heatmap_data(mole_percent_data_after_combination, combinations, start=1, end=29):
    heatmap_df = pd.DataFrame()
    combo_counts = {combo: info['counts'] for combo, info in combinations.items()}
    
    for combo, df in mole_percent_data_after_combination.items():
        avg_mole_percents = df.mean().to_frame().T
        combo_with_count = f"{combo} ({combo_counts[combo]})"
        avg_mole_percents.index = [combo_with_count]
        heatmap_df = pd.concat([heatmap_df, avg_mole_percents])
    
    heatmap_df['Counts'] = heatmap_df.index.map(lambda x: combo_counts[x.split(' (')[0]])
    heatmap_df = heatmap_df.sort_values(by='Counts', ascending=False)
    
    
    heatmap_df = heatmap_df.iloc[start-1:end] 
    heatmap_df.drop(columns=['Counts'], inplace=True)
    
    return heatmap_df


def text_color_based_on_background(avg_value, min_val, max_val):
  
    norm_value = (avg_value - min_val) / (max_val - min_val)
    if norm_value < 0.5: # change from < to > if you want to change the text color 
        return "black"  # Light background
    else:
        return "white"  # Dark background




def plot_heatmap(heatmap_df, sd_df, text_size=10, selected_colorscale='ylGnBu'):
    elements = heatmap_df.columns.tolist()
    combinations_with_counts = heatmap_df.index.tolist()

    # color
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=elements,
        y=combinations_with_counts,
        colorscale=selected_colorscale,
        colorbar=dict(title='Mole %'),
        hoverinfo="skip" 
    ))

    
    min_val = heatmap_df.values.min()
    max_val = heatmap_df.values.max()

    # cell background
    for y, comb_with_count in enumerate(combinations_with_counts):
        comb = comb_with_count.split(' (')[0]
        for x, elem in enumerate(elements):
            avg_value = heatmap_df.loc[comb_with_count, elem]
            sd_value = sd_df.loc[comb, elem] if elem in sd_df.columns else np.nan
            
            color = text_color_based_on_background(avg_value, min_val, max_val)
            
           
            if avg_value != 0:
                annotation = f"{avg_value:.2f}\n±{sd_value:.2f}" if not np.isnan(sd_value) else f"{avg_value:.2f}"
                fig.add_annotation(x=x, y=y, text=annotation, showarrow=False,
                                   font=dict(size=text_size, color=color))
                
    fig.update_layout(
        title='Pourcentage Molaire Après Traitement',
        xaxis=dict(title='Éléments', tickangle=-45),
        yaxis=dict(title='Particules (Fréquance)', autorange='reversed'),
        height=max(600, 30 * len(combinations_with_counts))
    )
    
    pio.write_image(fig, 'figure_high_res.png', width=1920, height=1080, scale=2)

    st.plotly_chart(fig, use_container_width=True)
    


def display_aggregated_data(aggregated_data, data_type):
    st.header(f"All {data_type.replace('_', ' ').title()}")
    st.dataframe(aggregated_data)



def aggregate_combination_data(data_dict):
    aggregated_df = pd.concat(data_dict.values(), keys=data_dict.keys())
    aggregated_df.reset_index(level=0, inplace=True)
    aggregated_df.rename(columns={'level_0': 'Combination'}, inplace=True)
    return aggregated_df
        

def plot_combination_distribution_by_counts(combinations, element_to_analyze, count_threshold):
    filtered_combinations = {
        combo: info for combo, info in combinations.items()
        if info['counts'] >= count_threshold and element_to_analyze in combo.split(', ')
    }

    other_counts = sum(info['counts'] for combo, info in combinations.items() if combo not in filtered_combinations and element_to_analyze in combo.split(', '))
    
    if other_counts > 0:
        filtered_combinations['Autres'] = {'counts': other_counts}
    
    if not filtered_combinations:
        st.write(f"No combinations containing {element_to_analyze} meet the specified count threshold.")
        return

    labels = [f"{combo} ({info['counts']})" for combo, info in filtered_combinations.items()]
    values = [info['counts'] for info in filtered_combinations.values()]
    total_counts = sum(values)

    # Combine the combination and counts for display directly on the pie chart slices
    texts = [f"{combo}: {info['counts']} ({info['counts']/total_counts*100:.2f}%)" for combo, info in filtered_combinations.items()]
    if other_counts > 0:
        texts.append(f"Autres: {other_counts} ({other_counts/total_counts*100:.2f}%)")

    # Hover texts to provide detailed information on hover
    hover_texts = [f"{combo}: {info['counts']}<br>({(info['counts']/total_counts*100):.2f}%)" for combo, info in filtered_combinations.items()]
    if other_counts > 0:
        hover_texts.append(f"Autres: {other_counts}<br>({other_counts/total_counts*100:.2f}%)")
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, text=texts, hoverinfo="label+percent+value", hovertext=hover_texts, hovertemplate='%{hovertext}')])
    fig.update_traces(marker=dict(colors=px.colors.sequential.Sunsetdark), textinfo='label+percent')

    fig.update_layout(
        legend={"x": 0.8, "y": 1, "title": {"text": "Combinations"}},
        title=f"Distribution of particles containing {element_to_analyze}",
        title_font_size=20,
        title_x=0.5, 
        title_y=0.95,
        title_xanchor='center', 
        title_yanchor='top'
    )
    pio.write_image(fig, 'figure_high_res.png', width=1920, height=1080, scale=2)

    st.plotly_chart(fig, use_container_width=True)
    st.write(f"The total number of particles: {total_counts}")


if 'df' in globals():
    
    mass_data, mass_percent_data, mole_data, mole_percent_data = process_data(df)

    data_type = st.sidebar.selectbox(
        'Select data type to view',
        ('Mass Data', 'Mass Percent Data', 'Mole Data', 'Mole Percent Data'),
        index=0
    )
    threshold = st.sidebar.slider('Minimum percentage to display:', 0.0, 100.0, 1.0, 0.01)
    
    default_colors = [
    '#191970', '#FF7F50', '#2E8B57', '#8B0000', '#FFD700', 
    '#87CEEB', '#FF69B4', '#CD5C5C', '#FFA07A', '#20B2AA', 
    '#8470FF', '#778899', '#B0C4DE', '#FFFFE0', '#00FA9A', 
    '#1E90FF', '#FF6347', '#7B68EE', '#00FF7F', '#4682B4', 
    '#9ACD32', '#40E0D0', '#EE82EE', '#D2691E', '#6495ED', 
    '#FFDAB9', '#FA8072', '#6B8E23', '#FF4500', '#DA70D6',
]

    # color choices
    color_choices = {}
    data_to_plot = None
    title = ''

    if data_type == 'Mass Data':
        data_to_plot = mass_data
        title = 'Masses'
    elif data_type == 'Mass Percent Data':
        data_to_plot = mass_percent_data
        title = 'Mass Percentages'
    elif data_type == 'Mole Data':
        data_to_plot = mole_data
        title = 'Molaire' 
    elif data_type == 'Mole Percent Data':
        data_to_plot = mole_percent_data
        title = 'Mole Percentages'

    if data_to_plot is not None:
        for i, element in enumerate(data_to_plot.columns):
            # Use the predefined list for default colors, cycling through if more elements than colors
            default_color = default_colors[i % len(default_colors)]
            color = st.sidebar.color_picker(f"Color for {element}", value=default_color)
            color_choices[element] = color
        
        color_choices['Autres'] = st.sidebar.color_picker(f"Color for Others", '#777777')  # Default color for 'Others'

        st.write(data_to_plot)  # Show the table
        plot_pie_chart_with_table(data_to_plot, threshold, color_choices, title)
    else:
        st.write('No data available or the data type was not correctly processed.')

    mass_data, mass_percent_data, mole_data, mole_percent_data = process_data(df)

    # Let the user select which element to display the histogram for
    element_to_plot = st.sidebar.selectbox('Select an element to view the histogram:', mass_data.columns)
    
    # Allow the user to pick a color for the histogram
    histogram_color = st.sidebar.color_picker('Pick a color for the histogram', '#00f900')  # Default color is green

    # Sidebar controls for histogram customization
    bin_size = st.sidebar.slider('Select bin size:', min_value=0.001, max_value=10.0, value=0.1, step=0.01)
    x_max = st.sidebar.slider('Select max value for x-axis (Masse (fg)):', min_value=0, max_value=1000, value=50, step=1)

    # Call the function to plot the histogram with the chosen settings
    plot_histogram_for_element(mass_data, element_to_plot, histogram_color, bin_size, x_max, 'Distribution de masse')
    
    combinations, _, mole_percent_combinations, _, _, _, sd_df = get_combinations_and_related_data(
        mole_percent_data, mass_data, mass_percent_data, mole_data)
    
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

    selected_colorscale = st.sidebar.selectbox('Select a colorscale:', colorscale_options, index=0)  
    
    start = st.sidebar.number_input('Start from combination:', min_value=1, max_value=len(combinations)-1, value=1)
    end = st.sidebar.number_input('End at combination:', min_value=2, max_value=len(combinations), value=2)
    end = max(end, start + 1) 
    
    heatmap_df = prepare_heatmap_data(mole_percent_combinations, combinations, start, end)
    
    plot_heatmap(heatmap_df, sd_df, text_size=10.5, selected_colorscale=selected_colorscale)

    
    _, _, mole_percent_data_after_combination, mass_data_after_combination, mass_percent_data_after_combination, mole_data_after_combination, sd_df = get_combinations_and_related_data(
        mole_percent_data, mass_data, mass_percent_data, mole_data)
    
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
    
    
    
    
    combinations, _, mole_percent_combinations, _, _, _, sd_df = get_combinations_and_related_data(
        mole_percent_data, mass_data, mass_percent_data, mole_data)
    
    
    element_to_analyze = st.sidebar.selectbox('Select an element to analyze:', heatmap_df.columns.tolist())
    count_threshold = st.sidebar.number_input('Set a count threshold for display:', min_value=0, value=10, step=1)
    plot_combination_distribution_by_counts(combinations, element_to_analyze, count_threshold)

