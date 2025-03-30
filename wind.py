import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
import seaborn as sns
import streamlit as st
from streamlit_option_menu import option_menu
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def s_wind_speed(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Month',y='WS', hue='Year',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Months of the season',fontsize=10)
    plt.ylabel(ylabel ='Wind speed in m/sec',fontsize=10)
    plt.title('Variation of wind speed during the Season',fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)
    
@st.cache_data
def s_wind_direction(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Month',y='WD', hue='Year',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Months of the season',fontsize=18)
    plt.ylabel(ylabel ='Wind direction in degrees',fontsize=18)
    plt.title('Variation of wind direction during the Season',fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)

@st.cache_data
def s_temp(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Month',y='Temp', hue='Year',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Months of the season',fontsize=18)
    plt.ylabel(ylabel ='Temperature in degrees celsius',fontsize=18)
    plt.title('Variation of temperature during the Season',fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)

@st.cache_data
def s_solar(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Month',y='IR', hue='Year',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Months of the season',fontsize=18)
    plt.ylabel(ylabel ='solar irradiation in Wh/m2',fontsize=18)
    plt.title('Variation of solar irradiation during the Season',fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)

@st.cache_data
def s_rel_hum(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Month',y='RH', hue='Year',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Months of the season',fontsize=18)
    plt.ylabel(ylabel ='Relative humidity in %',fontsize=18)
    plt.title('Variation of Relative humidity during the Season',fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)

@st.cache_data
def s_pre(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Month',y='PR', hue='Year',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Months of the season',fontsize=18)
    plt.ylabel(ylabel ='Pressure in hPa',fontsize=18)
    plt.title('Variation of Pressure during the Season',fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)

@st.cache_data
def s_ws_cat(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_ws = pd.value_counts(df['WS'])
    df_ws = df_ws.reset_index(drop=False)
    df_ws = df_ws.rename(columns={'index':'WS','WS':'Count'})
    df_ws = df_ws.sort_values(by='WS',ascending=True)
    sns.barplot(x='WS',y='Count',data=df_ws,palette='viridis')
    plt.xticks(rotation=90)
    plt.xlabel(xlabel ='Wind speed in m/sec',fontsize=18)
    plt.ylabel(ylabel ='Count',fontsize=18)
    plt.title('Occurance of various wind speeds through out the season',fontsize=18)
    st.pyplot(fig)

@st.cache_data
def s_ws_cat_2(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_ws = pd.value_counts(df['WS_cat'])
    df_ws = df_ws.reset_index(drop=False)
    df_ws = df_ws.rename(columns={'index':'WS_cat','WS_cat':'Count'})
    df_ws = df_ws.sort_values(by='WS_cat',ascending=True)
    sns.barplot(x='WS_cat',y='Count',data=df_ws,palette='copper_r')
    plt.xlabel(xlabel ='Wind speed categories',fontsize=18)
    plt.ylabel(ylabel ='Count',fontsize=18)
    plt.title('Occurance of various wind speed categories through out the season',fontsize=18)
    st.pyplot(fig)

@st.cache_data
def s_ws_cat_3(df):
    df_ws = pd.DataFrame(pd.value_counts(df['WS_cat']))
    df_ws =df_ws.reset_index(drop=False)
    df_ws = df_ws.rename(columns={'index':'Wind speed category','WS_cat':'Count'})
    total = df_ws['Count'].sum()
    df_ws['Percentage'] = round(df_ws['Count']/total * 100,2)
    df_ws = df_ws.sort_values(by='Wind speed category', ascending = True)
    df_ws = df_ws.reset_index(drop=True)
    df_ws_2 = df.groupby(['WS_cat'])['WS'].mean()
    df_ws_2 = df_ws_2.reset_index(drop=False)
    df_ws_2.drop(columns=['WS_cat'],inplace=True)
    df_new = pd.concat([df_ws,df_ws_2],axis=1)
    df_new = df_new.rename(columns={'WS':'Average wind speed'})
    return df_new

@st.cache_data
def s_wd_cat(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_wdir = pd.value_counts(df['WD2'])
    df_wdir = df_wdir.reset_index(drop=False)
    df_wdir = df_wdir.rename(columns={'index':'WD2','WD2':'Count'})
    df_wdir = df_wdir.sort_values(by='WD2',ascending=True)
    sns.barplot(x='WD2',y='Count',data=df_wdir,palette='viridis_r')
    plt.xlabel('Wind direction categories')
    plt.ylabel('Count')
    plt.title('Occurance of wind direction categories throught out the season')
    st.pyplot(fig)

@st.cache_data
def s_wd_cat_2(df):
    df_wd = pd.DataFrame(pd.value_counts(df['WD2']))
    df_wd =df_wd.reset_index(drop=False)
    df_wd = df_wd.rename(columns={'index':'Wind direction category','WD2':'Count'})
    total = df_wd['Count'].sum()
    df_wd['Percentage'] = round(df_wd['Count']/total * 100,2)
    df_wd = df_wd.sort_values(by='Wind direction category', ascending = True)
    df_wd = df_wd.reset_index(drop=True)
    df_wd_2 = df.groupby(['WD2'])['WD'].mean()
    df_wd_2 = df_wd_2.reset_index(drop=False)
    df_wd_2.drop(columns=['WD2'],inplace=True)
    df_new = pd.concat([df_wd,df_wd_2],axis=1)
    df_new = df_new.rename(columns={'WD':'Average wind direction'})
    df_new['Average wind direction'] = df_new['Average wind direction'].astype(int)
    return df_new
    
@st.cache_data
def s_temp_cat(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_temp_1 = pd.value_counts(df['Temp_cat'])
    df_temp_1 = df_temp_1.reset_index(drop=False)
    df_temp_1 = df_temp_1.rename(columns={'index':'Temp_cat','Temp_cat':'Count'})
    df_temp_1 = df_temp_1.sort_values(by='Temp_cat',ascending=True)
    sns.barplot(x='Temp_cat',y='Count',data=df_temp_1,palette='viridis')
    plt.xlabel('Temperature categories')
    plt.ylabel('Count')
    plt.title('Occurance of temperature categories throught out the season')
    st.pyplot(fig)

@st.cache_data
def s_temp_2(df):
    df_temp = pd.DataFrame(pd.value_counts(df['Temp_cat']))
    df_temp =df_temp.reset_index(drop=False)
    df_temp = df_temp.rename(columns={'index':'Temperature category','Temp_cat':'Count'})
    total = df_temp['Count'].sum()
    df_temp['Percentage'] = round(df_temp['Count']/total * 100,2)
    df_temp = df_temp.sort_values(by='Temperature category', ascending = True)
    df_temp = df_temp.reset_index(drop=True)
    df_temp_2 = df.groupby(['Temp_cat'])['Temp'].mean()
    df_temp_2 = df_temp_2.reset_index(drop=False)
    df_temp_2.drop(columns=['Temp_cat'],inplace=True)
    df_new = pd.concat([df_temp,df_temp_2],axis=1)
    df_new = df_new.rename(columns={'Temp':'Average temperature'})
    df_new['Average temperature'] = round(df_new['Average temperature'],2)
    return df_new

@st.cache_data
def s_ir(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_ir = pd.value_counts(df['IR_cat'])
    df_ir = df_ir.reset_index(drop=False)
    df_ir = df_ir.rename(columns={'index':'IR_cat','IR_cat':'Count'})
    df_ir = df_ir.sort_values(by='IR_cat',ascending=True)
    sns.barplot(x='IR_cat',y='Count',data=df_ir,palette='viridis')
    plt.xlabel('Solar Irradiation categories')
    plt.ylabel('Count')
    plt.title('Occurance of Solar irradiation categories throught out the season')
    st.pyplot(fig)

@st.cache_data
def s_ir_2(df):
    df_ir_2 = pd.DataFrame(pd.value_counts(df['IR_cat']))
    df_ir_2 =df_ir_2.reset_index(drop=False)
    df_ir_2 = df_ir_2.rename(columns={'index':'Solar irradiation category','IR_cat':'Count'})
    total = df_ir_2['Count'].sum()
    df_ir_2['Percentage'] = round(df_ir_2['Count']/total * 100,2)
    df_ir_2 = df_ir_2.sort_values(by='Solar irradiation category', ascending = True)
    df_ir_2 = df_ir_2.reset_index(drop=True)
    df_ir_3 = df.groupby(['IR_cat'])['IR'].mean()
    df_ir_3 = df_ir_3.reset_index(drop=False)
    df_ir_3.drop(columns=['IR_cat'],inplace=True)
    df_new = pd.concat([df_ir_2,df_ir_3],axis=1)
    df_new = df_new.rename(columns={'IR':'Average solar irradiation'})
    df_new['Average solar irradiation'] = round(df_new['Average solar irradiation'],2)
    return df_new

@st.cache_data
def s_rh(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_rh = pd.value_counts(df['RH_cat'])
    df_rh = df_rh.reset_index(drop=False)
    df_rh = df_rh.rename(columns={'index':'RH_cat','RH_cat':'Count'})
    df_rh = df_rh.sort_values(by='RH_cat',ascending=True)
    sns.barplot(x='RH_cat',y='Count',data=df_rh,palette='viridis')
    plt.xlabel('Relative humidity categories')
    plt.ylabel('Count')
    plt.title('Occurance of Relative humidity categories throught out the season')
    st.pyplot(fig)

@st.cache_data
def s_rh_2(df):
    df_rh_2 = pd.DataFrame(pd.value_counts(df['RH_cat']))
    df_rh_2 =df_rh_2.reset_index(drop=False)
    df_rh_2 = df_rh_2.rename(columns={'index':'Relative humidity category','RH_cat':'Count'})
    total = df_rh_2['Count'].sum()
    df_rh_2['Percentage'] = round(df_rh_2['Count']/total * 100,2)
    df_rh_2 = df_rh_2.sort_values(by='Relative humidity category', ascending = True)
    df_rh_2 = df_rh_2.reset_index(drop=True)
    df_rh_3 = df.groupby(['RH_cat'])['RH'].mean()
    df_rh_3 = df_rh_3.reset_index(drop=False)
    df_rh_3.drop(columns=['RH_cat'],inplace=True)
    df_new = pd.concat([df_rh_2,df_rh_3],axis=1)
    df_new = df_new.rename(columns={'RH':'Average relative humidity'})
    df_new['Average relative humidity'] = round(df_new['Average relative humidity'],2)
    return df_new

@st.cache_data
def s_pr(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_pr = pd.value_counts(df['PR_cat'])
    df_pr = df_pr.reset_index(drop=False)
    df_pr = df_pr.rename(columns={'index':'PR_cat','PR_cat':'Count'})
    df_pr = df_pr.sort_values(by='PR_cat',ascending=True)
    sns.barplot(x='PR_cat',y='Count',data=df_pr,palette='viridis')
    plt.xlabel('Pressure categories')
    plt.ylabel('Count')
    plt.title('Occurance of pressure categories throught out the season')
    st.pyplot(fig)

@st.cache_data
def s_pr_2(df):
    df_pr_2 = pd.DataFrame(pd.value_counts(df['PR_cat']))
    df_pr_2 =df_pr_2.reset_index(drop=False)
    df_pr_2 = df_pr_2.rename(columns={'index':'Pressure category','PR_cat':'Count'})
    total = df_pr_2['Count'].sum()
    df_pr_2['Percentage'] = round(df_pr_2['Count']/total * 100,2)
    df_pr_2 = df_pr_2.sort_values(by='Pressure category', ascending = True)
    df_pr_2 = df_pr_2.reset_index(drop=True)
    df_pr_3 = df.groupby(['PR_cat'])['PR'].mean()
    df_pr_3 = df_pr_3.reset_index(drop=False)
    df_pr_3.drop(columns=['PR_cat'],inplace=True)
    df_new = pd.concat([df_pr_2,df_pr_3],axis=1)
    df_new = df_new.rename(columns={'PR':'Average Pressure'})
    df_new['Average Pressure'] = round(df_new['Average Pressure'],2)
    return df_new

@st.cache_data
def s_ws_wd(df):
    fig, ax = plt.subplots(figsize=(8,4))
    legend_labels = {1:'0-3', 2:'3-12', 3:'12-25', 4:'>25'}
    sns.lineplot(data=df,x='Month',y='WD', hue='WS_cat',legend='full',palette='icefire')
    plt.xlabel(xlabel ='Month of the year',fontsize=18)
    plt.ylabel(ylabel ='Wind direction in degrees',fontsize=18)
    plt.title('Effect of wind direction on wind speed',fontsize=18)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=2, labels=[legend_labels[label] for label in np.sort(df['WS_cat'].unique())],title = 'wind speed in m/sec')
    st.pyplot(fig)
    st.markdown('1: 0-3 m/sec,\t 2: 3-12 m/sec,\t 3: 12-25 m/sec,\t 4: >25 m/sec')

@st.cache_data
def s_ws_temp(df):
    fig, ax = plt.subplots(figsize=(8,4))
    legend_labels = {1:'0-3', 2:'3-12', 3:'12-25', 4:'>25'}
    sns.lineplot(data=df,x='Month',y='Temp', hue='WS_cat',legend='full',palette='icefire')
    plt.xlabel(xlabel ='Month of the year',fontsize=18)
    plt.ylabel(ylabel ='Temperature in degrees celsius',fontsize=18)
    plt.title('Effect of temperature on wind speed',fontsize=18)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=2, labels=[legend_labels[label] for label in np.sort(df['WS_cat'].unique())],title = 'wind speed in m/sec')
    st.pyplot(fig)
    st.markdown('1: 0-3 m/sec,\t 2: 3-12 m/sec,\t 3: 12-25 m/sec,\t 4: >25 m/sec')

@st.cache_data
def s_ws_ir(df):
    fig, ax = plt.subplots(figsize=(8,4))
    legend_labels = {1:'0-3', 2:'3-12', 3:'12-25', 4:'>25'}
    sns.lineplot(data=df,x='Month',y='IR', hue='WS_cat',legend='full',palette='icefire')
    plt.xlabel(xlabel ='Month of the year',fontsize=18)
    plt.ylabel(ylabel ='Solar Irradiation in Wh/m2',fontsize=18)
    plt.title('Effect of Solar Irradiation on wind speed',fontsize=18)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=2, labels=[legend_labels[label] for label in np.sort(df['WS_cat'].unique())],title = 'wind speed in m/sec')
    st.pyplot(fig)
    st.markdown('1: 0-3 m/sec,\t 2: 3-12 m/sec,\t 3: 12-25 m/sec,\t 4: >25 m/sec')

@st.cache_data
def s_ws_rh(df):
    fig, ax = plt.subplots(figsize=(8,4))
    legend_labels = {1:'0-3', 2:'3-12', 3:'12-25', 4:'>25'}
    sns.lineplot(data=df,x='Month',y='RH', hue='WS_cat',legend='full',palette='icefire')
    plt.xlabel(xlabel ='Month of the year',fontsize=18)
    plt.ylabel(ylabel ='Relative humidity in %',fontsize=18)
    plt.title('Effect of Relative humidity on wind speed',fontsize=18)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=2, labels=[legend_labels[label] for label in np.sort(df['WS_cat'].unique())],title = 'wind speed in m/sec')
    st.pyplot(fig)
    st.markdown('1: 0-3 m/sec,\t 2: 3-12 m/sec,\t 3: 12-25 m/sec,\t 4: >25 m/sec')

@st.cache_data
def s_ws_pr(df):
    fig, ax = plt.subplots(figsize=(8,4))
    legend_labels = {1:'0-3', 2:'3-12', 3:'12-25', 4:'>25'}
    legend_labels = {1: '0-3', 2: '3-12', 3: '12-25', 4:'> 25'}
    sns.lineplot(data=df,x='Month',y='PR', hue='WS_cat',legend='full',palette='icefire')
    plt.xlabel(xlabel ='Month of the year',fontsize=18)
    plt.ylabel(ylabel ='Pressure in hPa',fontsize=18)
    plt.title('Effect of pressure on wind speed',fontsize=18)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=2, labels=[legend_labels[label] for label in np.sort(df['WS_cat'].unique())],title = 'wind speed in m/sec')
    st.pyplot(fig)
    st.markdown('1: 0-3 m/sec,\t 2: 3-12 m/sec,\t 3: 12-25 m/sec,\t 4: >25 m/sec')

@st.cache_data
def m_wind_speed(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Day',y='WS', hue='Hour',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Days of the month',fontsize=10)
    plt.ylabel(ylabel ='Wind speed in m/sec',fontsize=10)
    plt.title('Variation of wind speed during the Month',fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)
    
@st.cache_data
def m_wind_direction(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Day',y='WD', hue='Hour',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Days of the month',fontsize=18)
    plt.ylabel(ylabel ='Wind direction in degrees',fontsize=18)
    plt.title('Variation of wind direction during the Month',fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)

@st.cache_data
def m_temp(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Day',y='Temp', hue='Hour',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Days of the month',fontsize=18)
    plt.ylabel(ylabel ='Temperature in degrees celsius',fontsize=18)
    plt.title('Variation of temperature during the Month',fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)

@st.cache_data
def m_solar(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Day',y='IR', hue='Hour',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Days of the month',fontsize=18)
    plt.ylabel(ylabel ='solar irradiation in Wh/m2',fontsize=18)
    plt.title('Variation of solar irradiation during the Month',fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)

@st.cache_data
def m_rel_hum(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Day',y='RH', hue='Hour',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Days of the month',fontsize=18)
    plt.ylabel(ylabel ='Relative humidity in %',fontsize=18)
    plt.title('Variation of Relative humidity during the Month',fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)

@st.cache_data
def m_pre(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Day',y='PR', hue='Hour',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Days of the month',fontsize=18)
    plt.ylabel(ylabel ='Pressure in hPa',fontsize=18)
    plt.title('Variation of Pressure during the Month',fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)

@st.cache_data
def m_ws_cat(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_ws = pd.value_counts(df['WS'])
    df_ws = df_ws.reset_index(drop=False)
    df_ws = df_ws.rename(columns={'index':'WS','WS':'Count'})
    df_ws = df_ws.sort_values(by='WS',ascending=True)
    sns.barplot(x='WS',y='Count',data=df_ws,palette='viridis')
    plt.xticks(rotation=90)
    plt.xlabel(xlabel ='Wind speed in m/sec',fontsize=18)
    plt.ylabel(ylabel ='Count',fontsize=18)
    plt.title('Occurance of various wind speeds through out the month',fontsize=18)
    st.pyplot(fig)

@st.cache_data
def m_ws_cat_2(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_ws = pd.value_counts(df['WS_cat'])
    df_ws = df_ws.reset_index(drop=False)
    df_ws = df_ws.rename(columns={'index':'WS_cat','WS_cat':'Count'})
    df_ws = df_ws.sort_values(by='WS_cat',ascending=True)
    sns.barplot(x='WS_cat',y='Count',data=df_ws,palette='copper_r')
    plt.xlabel(xlabel ='Wind speed categories',fontsize=18)
    plt.ylabel(ylabel ='Count',fontsize=18)
    plt.title('Occurance of various wind speed categories through out the month',fontsize=18)
    st.pyplot(fig)

@st.cache_data
def m_ws_cat_3(df):
    df_ws = pd.DataFrame(pd.value_counts(df['WS_cat']))
    df_ws =df_ws.reset_index(drop=False)
    df_ws = df_ws.rename(columns={'index':'Wind speed category','WS_cat':'Count'})
    total = df_ws['Count'].sum()
    df_ws['Percentage'] = round(df_ws['Count']/total * 100,2)
    df_ws = df_ws.sort_values(by='Wind speed category', ascending = True)
    df_ws = df_ws.reset_index(drop=True)
    df_ws_2 = df.groupby(['WS_cat'])['WS'].mean()
    df_ws_2 = df_ws_2.reset_index(drop=False)
    df_ws_2.drop(columns=['WS_cat'],inplace=True)
    df_new = pd.concat([df_ws,df_ws_2],axis=1)
    df_new = df_new.rename(columns={'WS':'Average wind speed'})
    return df_new

@st.cache_data
def m_wd_cat(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_wdir = pd.value_counts(df['WD2'])
    df_wdir = df_wdir.reset_index(drop=False)
    df_wdir = df_wdir.rename(columns={'index':'WD2','WD2':'Count'})
    df_wdir = df_wdir.sort_values(by='WD2',ascending=True)
    sns.barplot(x='WD2',y='Count',data=df_wdir,palette='viridis_r')
    plt.xlabel('Wind direction categories')
    plt.ylabel('Count')
    plt.title('Occurance of wind direction categories throught out the month')
    st.pyplot(fig)

@st.cache_data
def m_wd_cat_2(df):
    df_wd = pd.DataFrame(pd.value_counts(df['WD2']))
    df_wd =df_wd.reset_index(drop=False)
    df_wd = df_wd.rename(columns={'index':'Wind direction category','WD2':'Count'})
    total = df_wd['Count'].sum()
    df_wd['Percentage'] = round(df_wd['Count']/total * 100,2)
    df_wd = df_wd.sort_values(by='Wind direction category', ascending = True)
    df_wd = df_wd.reset_index(drop=True)
    df_wd_2 = df.groupby(['WD2'])['WD'].mean()
    df_wd_2 = df_wd_2.reset_index(drop=False)
    df_wd_2.drop(columns=['WD2'],inplace=True)
    df_new = pd.concat([df_wd,df_wd_2],axis=1)
    df_new = df_new.rename(columns={'WD':'Average wind direction'})
    df_new['Average wind direction'] = df_new['Average wind direction'].astype(int)
    return df_new
    
@st.cache_data
def m_temp_cat(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_temp_1 = pd.value_counts(df['Temp_cat'])
    df_temp_1 = df_temp_1.reset_index(drop=False)
    df_temp_1 = df_temp_1.rename(columns={'index':'Temp_cat','Temp_cat':'Count'})
    df_temp_1 = df_temp_1.sort_values(by='Temp_cat',ascending=True)
    sns.barplot(x='Temp_cat',y='Count',data=df_temp_1,palette='viridis')
    plt.xlabel('Temperature categories')
    plt.ylabel('Count')
    plt.title('Occurance of temperature categories throught out the month')
    st.pyplot(fig)

@st.cache_data
def m_temp_2(df):
    df_temp = pd.DataFrame(pd.value_counts(df['Temp_cat']))
    df_temp =df_temp.reset_index(drop=False)
    df_temp = df_temp.rename(columns={'index':'Temperature category','Temp_cat':'Count'})
    total = df_temp['Count'].sum()
    df_temp['Percentage'] = round(df_temp['Count']/total * 100,2)
    df_temp = df_temp.sort_values(by='Temperature category', ascending = True)
    df_temp = df_temp.reset_index(drop=True)
    df_temp_2 = df.groupby(['Temp_cat'])['Temp'].mean()
    df_temp_2 = df_temp_2.reset_index(drop=False)
    df_temp_2.drop(columns=['Temp_cat'],inplace=True)
    df_new = pd.concat([df_temp,df_temp_2],axis=1)
    df_new = df_new.rename(columns={'Temp':'Average temperature'})
    df_new['Average temperature'] = round(df_new['Average temperature'],2)
    return df_new

@st.cache_data
def m_ir(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_ir = pd.value_counts(df['IR_cat'])
    df_ir = df_ir.reset_index(drop=False)
    df_ir = df_ir.rename(columns={'index':'IR_cat','IR_cat':'Count'})
    df_ir = df_ir.sort_values(by='IR_cat',ascending=True)
    sns.barplot(x='IR_cat',y='Count',data=df_ir,palette='viridis')
    plt.xlabel('Solar Irradiation categories')
    plt.ylabel('Count')
    plt.title('Occurance of Solar irradiation categories throught out the month')
    st.pyplot(fig)

@st.cache_data
def m_ir_2(df):
    df_ir_2 = pd.DataFrame(pd.value_counts(df['IR_cat']))
    df_ir_2 =df_ir_2.reset_index(drop=False)
    df_ir_2 = df_ir_2.rename(columns={'index':'Solar irradiation category','IR_cat':'Count'})
    total = df_ir_2['Count'].sum()
    df_ir_2['Percentage'] = round(df_ir_2['Count']/total * 100,2)
    df_ir_2 = df_ir_2.sort_values(by='Solar irradiation category', ascending = True)
    df_ir_2 = df_ir_2.reset_index(drop=True)
    df_ir_3 = df.groupby(['IR_cat'])['IR'].mean()
    df_ir_3 = df_ir_3.reset_index(drop=False)
    df_ir_3.drop(columns=['IR_cat'],inplace=True)
    df_new = pd.concat([df_ir_2,df_ir_3],axis=1)
    df_new = df_new.rename(columns={'IR':'Average solar irradiation'})
    df_new['Average solar irradiation'] = round(df_new['Average solar irradiation'],2)
    return df_new

@st.cache_data
def m_rh(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_rh = pd.value_counts(df['RH_cat'])
    df_rh = df_rh.reset_index(drop=False)
    df_rh = df_rh.rename(columns={'index':'RH_cat','RH_cat':'Count'})
    df_rh = df_rh.sort_values(by='RH_cat',ascending=True)
    sns.barplot(x='RH_cat',y='Count',data=df_rh,palette='viridis')
    plt.xlabel('Relative humidity categories')
    plt.ylabel('Count')
    plt.title('Occurance of Relative humidity categories throught out the month')
    st.pyplot(fig)

@st.cache_data
def m_rh_2(df):
    df_rh_2 = pd.DataFrame(pd.value_counts(df['RH_cat']))
    df_rh_2 =df_rh_2.reset_index(drop=False)
    df_rh_2 = df_rh_2.rename(columns={'index':'Relative humidity category','RH_cat':'Count'})
    total = df_rh_2['Count'].sum()
    df_rh_2['Percentage'] = round(df_rh_2['Count']/total * 100,2)
    df_rh_2 = df_rh_2.sort_values(by='Relative humidity category', ascending = True)
    df_rh_2 = df_rh_2.reset_index(drop=True)
    df_rh_3 = df.groupby(['RH_cat'])['RH'].mean()
    df_rh_3 = df_rh_3.reset_index(drop=False)
    df_rh_3.drop(columns=['RH_cat'],inplace=True)
    df_new = pd.concat([df_rh_2,df_rh_3],axis=1)
    df_new = df_new.rename(columns={'RH':'Average relative humidity'})
    df_new['Average relative humidity'] = round(df_new['Average relative humidity'],2)
    return df_new

@st.cache_data
def m_pr(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_pr = pd.value_counts(df['PR_cat'])
    df_pr = df_pr.reset_index(drop=False)
    df_pr = df_pr.rename(columns={'index':'PR_cat','PR_cat':'Count'})
    df_pr = df_pr.sort_values(by='PR_cat',ascending=True)
    sns.barplot(x='PR_cat',y='Count',data=df_pr,palette='viridis')
    plt.xlabel('Pressure categories')
    plt.ylabel('Count')
    plt.title('Occurance of pressure categories throught out the month')
    st.pyplot(fig)

@st.cache_data
def m_pr_2(df):
    df_pr_2 = pd.DataFrame(pd.value_counts(df['PR_cat']))
    df_pr_2 =df_pr_2.reset_index(drop=False)
    df_pr_2 = df_pr_2.rename(columns={'index':'Pressure category','PR_cat':'Count'})
    total = df_pr_2['Count'].sum()
    df_pr_2['Percentage'] = round(df_pr_2['Count']/total * 100,2)
    df_pr_2 = df_pr_2.sort_values(by='Pressure category', ascending = True)
    df_pr_2 = df_pr_2.reset_index(drop=True)
    df_pr_3 = df.groupby(['PR_cat'])['PR'].mean()
    df_pr_3 = df_pr_3.reset_index(drop=False)
    df_pr_3.drop(columns=['PR_cat'],inplace=True)
    df_new = pd.concat([df_pr_2,df_pr_3],axis=1)
    df_new = df_new.rename(columns={'PR':'Average Pressure'})
    df_new['Average Pressure'] = round(df_new['Average Pressure'],2)
    return df_new

@st.cache_data
def m_ws_wd(df):
    fig, ax = plt.subplots(figsize=(8,4))
    legend_labels = {1:'0-3', 2:'3-12', 3:'12-25', 4:'>25'}
    sns.lineplot(data=df,x='Day',y='WD', hue='WS_cat',legend='full',palette='icefire')
    plt.xlabel(xlabel ='Day of the month',fontsize=18)
    plt.ylabel(ylabel ='Wind direction in degrees',fontsize=18)
    plt.title('Effect of wind direction on wind speed',fontsize=18)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=2, labels=[legend_labels[label] for label in np.sort(df['WS_cat'].unique())],title = 'wind speed in m/sec')
    st.pyplot(fig)
    st.markdown('1: 0-3 m/sec,\t 2: 3-12 m/sec,\t 3: 12-25 m/sec,\t 4: >25 m/sec')

@st.cache_data
def m_ws_temp(df):
    fig, ax = plt.subplots(figsize=(8,4))
    legend_labels = {1:'0-3', 2:'3-12', 3:'12-25', 4:'>25'}
    sns.lineplot(data=df,x='Day',y='Temp', hue='WS_cat',legend='full',palette='icefire')
    plt.xlabel(xlabel ='Day of the month',fontsize=18)
    plt.ylabel(ylabel ='Temperature in degrees celsius',fontsize=18)
    plt.title('Effect of temperature on wind speed',fontsize=18)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=2, labels=[legend_labels[label] for label in np.sort(df['WS_cat'].unique())],title = 'wind speed in m/sec')
    st.pyplot(fig)
    st.markdown('1: 0-3 m/sec,\t 2: 3-12 m/sec,\t 3: 12-25 m/sec,\t 4: >25 m/sec')

@st.cache_data
def m_ws_ir(df):
    fig, ax = plt.subplots(figsize=(8,4))
    legend_labels = {1:'0-3', 2:'3-12', 3:'12-25', 4:'>25'}
    sns.lineplot(data=df,x='Day',y='IR', hue='WS_cat',legend='full',palette='icefire')
    plt.xlabel(xlabel ='Day of the month',fontsize=18)
    plt.ylabel(ylabel ='Solar Irradiation in Wh/m2',fontsize=18)
    plt.title('Effect of Solar Irradiation on wind speed',fontsize=18)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=2, labels=[legend_labels[label] for label in np.sort(df['WS_cat'].unique())],title = 'wind speed in m/sec')
    st.pyplot(fig)
    st.markdown('1: 0-3 m/sec,\t 2: 3-12 m/sec,\t 3: 12-25 m/sec,\t 4: >25 m/sec')

@st.cache_data
def m_ws_rh(df):
    fig, ax = plt.subplots(figsize=(8,4))
    legend_labels = {1:'0-3', 2:'3-12', 3:'12-25', 4:'>25'}
    sns.lineplot(data=df,x='Day',y='RH', hue='WS_cat',legend='full',palette='icefire')
    plt.xlabel(xlabel ='Day of the month',fontsize=18)
    plt.ylabel(ylabel ='Relative humidity in %',fontsize=18)
    plt.title('Effect of Relative humidity on wind speed',fontsize=18)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=2, labels=[legend_labels[label] for label in np.sort(df['WS_cat'].unique())],title = 'wind speed in m/sec')
    st.pyplot(fig)
    st.markdown('1: 0-3 m/sec,\t 2: 3-12 m/sec,\t 3: 12-25 m/sec,\t 4: >25 m/sec')

@st.cache_data
def m_ws_pr(df):
    fig, ax = plt.subplots(figsize=(8,4))
    legend_labels = {1:'0-3', 2:'3-12', 3:'12-25', 4:'>25'}
    legend_labels = {1: '0-3', 2: '3-12', 3: '12-25', 4:'> 25'}
    sns.lineplot(data=df,x='Day',y='PR', hue='WS_cat',legend='full',palette='icefire')
    plt.xlabel(xlabel ='Day of the month',fontsize=18)
    plt.ylabel(ylabel ='Pressure in hPa',fontsize=18)
    plt.title('Effect of pressure on wind speed',fontsize=18)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=2, labels=[legend_labels[label] for label in np.sort(df['WS_cat'].unique())],title = 'wind speed in m/sec')
    st.pyplot(fig)
    st.markdown('1: 0-3 m/sec,\t 2: 3-12 m/sec,\t 3: 12-25 m/sec,\t 4: >25 m/sec')

@st.cache_data
def d_wind_speed(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Hour',y='WS', hue='Year',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Hour of the day',fontsize=10)
    plt.ylabel(ylabel ='Wind speed in m/sec',fontsize=10)
    plt.title('Variation of wind speed during the Day',fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)
    
@st.cache_data
def d_wind_direction(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Hour',y='WD', hue='Year',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Hour of the day',fontsize=18)
    plt.ylabel(ylabel ='Wind direction in degrees',fontsize=18)
    plt.title('Variation of wind direction during the Day',fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)

@st.cache_data
def d_temp(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Hour',y='Temp', hue='Year',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Hour of the day',fontsize=18)
    plt.ylabel(ylabel ='Temperature in degrees celsius',fontsize=18)
    plt.title('Variation of temperature during the Day',fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)

@st.cache_data
def d_solar(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Hour',y='IR', hue='Year',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Hour of the day',fontsize=18)
    plt.ylabel(ylabel ='solar irradiation in Wh/m2',fontsize=18)
    plt.title('Variation of solar irradiation during the Day',fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)

@st.cache_data
def d_rel_hum(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Hour',y='RH', hue='Year',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Hour of the day',fontsize=18)
    plt.ylabel(ylabel ='Relative humidity in %',fontsize=18)
    plt.title('Variation of Relative humidity during the Day',fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)

@st.cache_data
def d_pre(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Hour',y='PR', hue='Year',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Hour of the day',fontsize=18)
    plt.ylabel(ylabel ='Pressure in hPa',fontsize=18)
    plt.title('Variation of Pressure during the Day',fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)

@st.cache_data
def d_ws_cat(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_ws = pd.value_counts(df['WS'])
    df_ws = df_ws.reset_index(drop=False)
    df_ws = df_ws.rename(columns={'index':'WS','WS':'Count'})
    df_ws = df_ws.sort_values(by='WS',ascending=True)
    sns.barplot(x='WS',y='Count',data=df_ws,palette='viridis')
    plt.xticks(rotation=90)
    plt.xlabel(xlabel ='Wind speed in m/sec',fontsize=18)
    plt.ylabel(ylabel ='Count',fontsize=18)
    plt.title('Occurance of various wind speeds through out the day',fontsize=18)
    st.pyplot(fig)

@st.cache_data
def d_ws_cat_2(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_ws = pd.value_counts(df['WS_cat'])
    df_ws = df_ws.reset_index(drop=False)
    df_ws = df_ws.rename(columns={'index':'WS_cat','WS_cat':'Count'})
    df_ws = df_ws.sort_values(by='WS_cat',ascending=True)
    sns.barplot(x='WS_cat',y='Count',data=df_ws,palette='copper_r')
    plt.xlabel(xlabel ='Wind speed categories',fontsize=18)
    plt.ylabel(ylabel ='Count',fontsize=18)
    plt.title('Occurance of various wind speed categories through out the day',fontsize=18)
    st.pyplot(fig)

@st.cache_data
def d_ws_cat_3(df):
    df_ws = pd.DataFrame(pd.value_counts(df['WS_cat']))
    df_ws =df_ws.reset_index(drop=False)
    df_ws = df_ws.rename(columns={'index':'Wind speed category','WS_cat':'Count'})
    total = df_ws['Count'].sum()
    df_ws['Percentage'] = round(df_ws['Count']/total * 100,2)
    df_ws = df_ws.sort_values(by='Wind speed category', ascending = True)
    df_ws = df_ws.reset_index(drop=True)
    df_ws_2 = df.groupby(['WS_cat'])['WS'].mean()
    df_ws_2 = df_ws_2.reset_index(drop=False)
    df_ws_2.drop(columns=['WS_cat'],inplace=True)
    df_new = pd.concat([df_ws,df_ws_2],axis=1)
    df_new = df_new.rename(columns={'WS':'Average wind speed'})
    return df_new

@st.cache_data
def d_wd_cat(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_wdir = pd.value_counts(df['WD2'])
    df_wdir = df_wdir.reset_index(drop=False)
    df_wdir = df_wdir.rename(columns={'index':'WD2','WD2':'Count'})
    df_wdir = df_wdir.sort_values(by='WD2',ascending=True)
    sns.barplot(x='WD2',y='Count',data=df_wdir,palette='viridis_r')
    plt.xlabel('Wind direction categories')
    plt.ylabel('Count')
    plt.title('Occurance of wind direction categories throught out the day')
    st.pyplot(fig)

@st.cache_data
def d_wd_cat_2(df):
    df_wd = pd.DataFrame(pd.value_counts(df['WD2']))
    df_wd =df_wd.reset_index(drop=False)
    df_wd = df_wd.rename(columns={'index':'Wind direction category','WD2':'Count'})
    total = df_wd['Count'].sum()
    df_wd['Percentage'] = round(df_wd['Count']/total * 100,2)
    df_wd = df_wd.sort_values(by='Wind direction category', ascending = True)
    df_wd = df_wd.reset_index(drop=True)
    df_wd_2 = df.groupby(['WD2'])['WD'].mean()
    df_wd_2 = df_wd_2.reset_index(drop=False)
    df_wd_2.drop(columns=['WD2'],inplace=True)
    df_new = pd.concat([df_wd,df_wd_2],axis=1)
    df_new = df_new.rename(columns={'WD':'Average wind direction'})
    df_new['Average wind direction'] = df_new['Average wind direction'].astype(int)
    return df_new
    
@st.cache_data
def d_temp_cat(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_temp_1 = pd.value_counts(df['Temp_cat'])
    df_temp_1 = df_temp_1.reset_index(drop=False)
    df_temp_1 = df_temp_1.rename(columns={'index':'Temp_cat','Temp_cat':'Count'})
    df_temp_1 = df_temp_1.sort_values(by='Temp_cat',ascending=True)
    sns.barplot(x='Temp_cat',y='Count',data=df_temp_1,palette='viridis')
    plt.xlabel('Temperature categories')
    plt.ylabel('Count')
    plt.title('Occurance of temperature categories throught out the Day')
    st.pyplot(fig)

@st.cache_data
def d_temp_2(df):
    df_temp = pd.DataFrame(pd.value_counts(df['Temp_cat']))
    df_temp =df_temp.reset_index(drop=False)
    df_temp = df_temp.rename(columns={'index':'Temperature category','Temp_cat':'Count'})
    total = df_temp['Count'].sum()
    df_temp['Percentage'] = round(df_temp['Count']/total * 100,2)
    df_temp = df_temp.sort_values(by='Temperature category', ascending = True)
    df_temp = df_temp.reset_index(drop=True)
    df_temp_2 = df.groupby(['Temp_cat'])['Temp'].mean()
    df_temp_2 = df_temp_2.reset_index(drop=False)
    df_temp_2.drop(columns=['Temp_cat'],inplace=True)
    df_new = pd.concat([df_temp,df_temp_2],axis=1)
    df_new = df_new.rename(columns={'Temp':'Average temperature'})
    df_new['Average temperature'] = round(df_new['Average temperature'],2)
    return df_new

@st.cache_data
def d_ir(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_ir = pd.value_counts(df['IR_cat'])
    df_ir = df_ir.reset_index(drop=False)
    df_ir = df_ir.rename(columns={'index':'IR_cat','IR_cat':'Count'})
    df_ir = df_ir.sort_values(by='IR_cat',ascending=True)
    sns.barplot(x='IR_cat',y='Count',data=df_ir,palette='viridis')
    plt.xlabel('Solar Irradiation categories')
    plt.ylabel('Count')
    plt.title('Occurance of Solar irradiation categories throught out the day')
    st.pyplot(fig)

@st.cache_data
def d_ir_2(df):
    df_ir_2 = pd.DataFrame(pd.value_counts(df['IR_cat']))
    df_ir_2 =df_ir_2.reset_index(drop=False)
    df_ir_2 = df_ir_2.rename(columns={'index':'Solar irradiation category','IR_cat':'Count'})
    total = df_ir_2['Count'].sum()
    df_ir_2['Percentage'] = round(df_ir_2['Count']/total * 100,2)
    df_ir_2 = df_ir_2.sort_values(by='Solar irradiation category', ascending = True)
    df_ir_2 = df_ir_2.reset_index(drop=True)
    df_ir_3 = df.groupby(['IR_cat'])['IR'].mean()
    df_ir_3 = df_ir_3.reset_index(drop=False)
    df_ir_3.drop(columns=['IR_cat'],inplace=True)
    df_new = pd.concat([df_ir_2,df_ir_3],axis=1)
    df_new = df_new.rename(columns={'IR':'Average solar irradiation'})
    df_new['Average solar irradiation'] = round(df_new['Average solar irradiation'],2)
    return df_new

@st.cache_data
def d_rh(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_rh = pd.value_counts(df['RH_cat'])
    df_rh = df_rh.reset_index(drop=False)
    df_rh = df_rh.rename(columns={'index':'RH_cat','RH_cat':'Count'})
    df_rh = df_rh.sort_values(by='RH_cat',ascending=True)
    sns.barplot(x='RH_cat',y='Count',data=df_rh,palette='viridis')
    plt.xlabel('Relative humidity categories')
    plt.ylabel('Count')
    plt.title('Occurance of Relative humidity categories throught out the day')
    st.pyplot(fig)

@st.cache_data
def d_rh_2(df):
    df_rh_2 = pd.DataFrame(pd.value_counts(df['RH_cat']))
    df_rh_2 =df_rh_2.reset_index(drop=False)
    df_rh_2 = df_rh_2.rename(columns={'index':'Relative humidity category','RH_cat':'Count'})
    total = df_rh_2['Count'].sum()
    df_rh_2['Percentage'] = round(df_rh_2['Count']/total * 100,2)
    df_rh_2 = df_rh_2.sort_values(by='Relative humidity category', ascending = True)
    df_rh_2 = df_rh_2.reset_index(drop=True)
    df_rh_3 = df.groupby(['RH_cat'])['RH'].mean()
    df_rh_3 = df_rh_3.reset_index(drop=False)
    df_rh_3.drop(columns=['RH_cat'],inplace=True)
    df_new = pd.concat([df_rh_2,df_rh_3],axis=1)
    df_new = df_new.rename(columns={'RH':'Average relative humidity'})
    df_new['Average relative humidity'] = round(df_new['Average relative humidity'],2)
    return df_new

@st.cache_data
def d_pr(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_pr = pd.value_counts(df['PR_cat'])
    df_pr = df_pr.reset_index(drop=False)
    df_pr = df_pr.rename(columns={'index':'PR_cat','PR_cat':'Count'})
    df_pr = df_pr.sort_values(by='PR_cat',ascending=True)
    sns.barplot(x='PR_cat',y='Count',data=df_pr,palette='viridis')
    plt.xlabel('Pressure categories')
    plt.ylabel('Count')
    plt.title('Occurance of pressure categories throught out the day')
    st.pyplot(fig)

@st.cache_data
def d_pr_2(df):
    df_pr_2 = pd.DataFrame(pd.value_counts(df['PR_cat']))
    df_pr_2 =df_pr_2.reset_index(drop=False)
    df_pr_2 = df_pr_2.rename(columns={'index':'Pressure category','PR_cat':'Count'})
    total = df_pr_2['Count'].sum()
    df_pr_2['Percentage'] = round(df_pr_2['Count']/total * 100,2)
    df_pr_2 = df_pr_2.sort_values(by='Pressure category', ascending = True)
    df_pr_2 = df_pr_2.reset_index(drop=True)
    df_pr_3 = df.groupby(['PR_cat'])['PR'].mean()
    df_pr_3 = df_pr_3.reset_index(drop=False)
    df_pr_3.drop(columns=['PR_cat'],inplace=True)
    df_new = pd.concat([df_pr_2,df_pr_3],axis=1)
    df_new = df_new.rename(columns={'PR':'Average Pressure'})
    df_new['Average Pressure'] = round(df_new['Average Pressure'],2)
    return df_new

@st.cache_data
def d_ws_wd(df):
    fig, ax = plt.subplots(figsize=(8,4))
    legend_labels = {1:'0-3', 2:'3-12', 3:'12-25', 4:'>25'}
    sns.lineplot(data=df,x='Hour',y='WD', hue='WS_cat',legend='full',palette='icefire')
    plt.xlabel(xlabel ='Hour of the day',fontsize=18)
    plt.ylabel(ylabel ='Wind direction in degrees',fontsize=18)
    plt.title('Effect of wind direction on wind speed',fontsize=18)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=2, labels=[legend_labels[label] for label in np.sort(df['WS_cat'].unique())],title = 'wind speed in m/sec')
    st.pyplot(fig)
    st.markdown('1: 0-3 m/sec,\t 2: 3-12 m/sec,\t 3: 12-25 m/sec,\t 4: >25 m/sec')

@st.cache_data
def d_ws_temp(df):
    fig, ax = plt.subplots(figsize=(8,4))
    legend_labels = {1:'0-3', 2:'3-12', 3:'12-25', 4:'>25'}
    sns.lineplot(data=df,x='Hour',y='Temp', hue='WS_cat',legend='full',palette='icefire')
    plt.xlabel(xlabel ='Hour of the day',fontsize=18)
    plt.ylabel(ylabel ='Temperature in degrees celsius',fontsize=18)
    plt.title('Effect of temperature on wind speed',fontsize=18)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=2, labels=[legend_labels[label] for label in np.sort(df['WS_cat'].unique())],title = 'wind speed in m/sec')
    st.pyplot(fig)
    st.markdown('1: 0-3 m/sec,\t 2: 3-12 m/sec,\t 3: 12-25 m/sec,\t 4: >25 m/sec')

@st.cache_data
def d_ws_ir(df):
    fig, ax = plt.subplots(figsize=(8,4))
    legend_labels = {1:'0-3', 2:'3-12', 3:'12-25', 4:'>25'}
    sns.lineplot(data=df,x='Hour',y='IR', hue='WS_cat',legend='full',palette='icefire')
    plt.xlabel(xlabel ='Hour of the day',fontsize=18)
    plt.ylabel(ylabel ='Solar Irradiation in Wh/m2',fontsize=18)
    plt.title('Effect of Solar Irradiation on wind speed',fontsize=18)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=2, labels=[legend_labels[label] for label in np.sort(df['WS_cat'].unique())],title = 'wind speed in m/sec')
    st.pyplot(fig)
    st.markdown('1: 0-3 m/sec,\t 2: 3-12 m/sec,\t 3: 12-25 m/sec,\t 4: >25 m/sec')

@st.cache_data
def d_ws_rh(df):
    fig, ax = plt.subplots(figsize=(8,4))
    legend_labels = {1:'0-3', 2:'3-12', 3:'12-25', 4:'>25'}
    sns.lineplot(data=df,x='Hour',y='RH', hue='WS_cat',legend='full',palette='icefire')
    plt.xlabel(xlabel ='Hour of the day',fontsize=18)
    plt.ylabel(ylabel ='Relative humidity in %',fontsize=18)
    plt.title('Effect of Relative humidity on wind speed',fontsize=18)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=2, labels=[legend_labels[label] for label in np.sort(df['WS_cat'].unique())],title = 'wind speed in m/sec')
    st.pyplot(fig)
    st.markdown('1: 0-3 m/sec,\t 2: 3-12 m/sec,\t 3: 12-25 m/sec,\t 4: >25 m/sec')

@st.cache_data
def d_ws_pr(df):
    fig, ax = plt.subplots(figsize=(8,4))
    legend_labels = {1:'0-3', 2:'3-12', 3:'12-25', 4:'>25'}
    legend_labels = {1: '0-3', 2: '3-12', 3: '12-25', 4:'> 25'}
    sns.lineplot(data=df,x='Hour',y='PR', hue='WS_cat',legend='full',palette='icefire')
    plt.xlabel(xlabel ='Hour of the day',fontsize=18)
    plt.ylabel(ylabel ='Pressure in hPa',fontsize=18)
    plt.title('Effect of pressure on wind speed',fontsize=18)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=2, labels=[legend_labels[label] for label in np.sort(df['WS_cat'].unique())],title = 'wind speed in m/sec')
    st.pyplot(fig)
    st.markdown('1: 0-3 m/sec,\t 2: 3-12 m/sec,\t 3: 12-25 m/sec,\t 4: >25 m/sec')

@st.cache_data
def h_wind_speed(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Min',y='WS', hue='Year',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Minutes of the hour',fontsize=10)
    plt.ylabel(ylabel ='Wind speed in m/sec',fontsize=10)
    plt.title('Variation of wind speed during the Hour',fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)
    
@st.cache_data
def h_wind_direction(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Min',y='WD', hue='Year',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Minutes of the hour',fontsize=18)
    plt.ylabel(ylabel ='Wind direction in degrees',fontsize=18)
    plt.title('Variation of wind direction during the Hour',fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)

@st.cache_data
def h_temp(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Min',y='Temp', hue='Year',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Minutes of the hour',fontsize=18)
    plt.ylabel(ylabel ='Temperature in degrees celsius',fontsize=18)
    plt.title('Variation of temperature during the Hour',fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)

@st.cache_data
def h_solar(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Min',y='IR', hue='Year',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Minutes of the hour',fontsize=18)
    plt.ylabel(ylabel ='solar irradiation in Wh/m2',fontsize=18)
    plt.title('Variation of solar irradiation during the Day',fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)

@st.cache_data
def h_rel_hum(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Min',y='RH', hue='Year',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Minutes of the hour',fontsize=18)
    plt.ylabel(ylabel ='Relative humidity in %',fontsize=18)
    plt.title('Variation of Relative humidity during the Day',fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)

@st.cache_data
def h_pre(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df,x='Min',y='PR', hue='Year',legend='full',palette='icefire',ax=ax)
    plt.xlabel(xlabel ='Minutes of the hour',fontsize=18)
    plt.ylabel(ylabel ='Pressure in hPa',fontsize=18)
    plt.title('Variation of Pressure during the Day',fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);
    st.pyplot(fig)

@st.cache_data
def h_ws_cat(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_ws = pd.value_counts(df['WS'])
    df_ws = df_ws.reset_index(drop=False)
    df_ws = df_ws.rename(columns={'index':'WS','WS':'Count'})
    df_ws = df_ws.sort_values(by='WS',ascending=True)
    sns.barplot(x='WS',y='Count',data=df_ws,palette='viridis')
    plt.xticks(rotation=90)
    plt.xlabel(xlabel ='Wind speed in m/sec',fontsize=18)
    plt.ylabel(ylabel ='Count',fontsize=18)
    plt.title('Occurance of various wind speeds through out the hour',fontsize=18)
    st.pyplot(fig)

@st.cache_data
def h_ws_cat_2(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_ws = pd.value_counts(df['WS_cat'])
    df_ws = df_ws.reset_index(drop=False)
    df_ws = df_ws.rename(columns={'index':'WS_cat','WS_cat':'Count'})
    df_ws = df_ws.sort_values(by='WS_cat',ascending=True)
    sns.barplot(x='WS_cat',y='Count',data=df_ws,palette='copper_r')
    plt.xlabel(xlabel ='Wind speed categories',fontsize=18)
    plt.ylabel(ylabel ='Count',fontsize=18)
    plt.title('Occurance of various wind speed categories through out the hour',fontsize=18)
    st.pyplot(fig)

@st.cache_data
def h_ws_cat_3(df):
    df_ws = pd.DataFrame(pd.value_counts(df['WS_cat']))
    df_ws =df_ws.reset_index(drop=False)
    df_ws = df_ws.rename(columns={'index':'Wind speed category','WS_cat':'Count'})
    total = df_ws['Count'].sum()
    df_ws['Percentage'] = round(df_ws['Count']/total * 100,2)
    df_ws = df_ws.sort_values(by='Wind speed category', ascending = True)
    df_ws = df_ws.reset_index(drop=True)
    df_ws_2 = df.groupby(['WS_cat'])['WS'].mean()
    df_ws_2 = df_ws_2.reset_index(drop=False)
    df_ws_2.drop(columns=['WS_cat'],inplace=True)
    df_new = pd.concat([df_ws,df_ws_2],axis=1)
    df_new = df_new.rename(columns={'WS':'Average wind speed'})
    return df_new

@st.cache_data
def h_wd_cat(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_wdir = pd.value_counts(df['WD2'])
    df_wdir = df_wdir.reset_index(drop=False)
    df_wdir = df_wdir.rename(columns={'index':'WD2','WD2':'Count'})
    df_wdir = df_wdir.sort_values(by='WD2',ascending=True)
    sns.barplot(x='WD2',y='Count',data=df_wdir,palette='viridis_r')
    plt.xlabel('Wind direction categories')
    plt.ylabel('Count')
    plt.title('Occurance of wind direction categories throught out the hour')
    st.pyplot(fig)

@st.cache_data
def h_wd_cat_2(df):
    df_wd = pd.DataFrame(pd.value_counts(df['WD2']))
    df_wd =df_wd.reset_index(drop=False)
    df_wd = df_wd.rename(columns={'index':'Wind direction category','WD2':'Count'})
    total = df_wd['Count'].sum()
    df_wd['Percentage'] = round(df_wd['Count']/total * 100,2)
    df_wd = df_wd.sort_values(by='Wind direction category', ascending = True)
    df_wd = df_wd.reset_index(drop=True)
    df_wd_2 = df.groupby(['WD2'])['WD'].mean()
    df_wd_2 = df_wd_2.reset_index(drop=False)
    df_wd_2.drop(columns=['WD2'],inplace=True)
    df_new = pd.concat([df_wd,df_wd_2],axis=1)
    df_new = df_new.rename(columns={'WD':'Average wind direction'})
    df_new['Average wind direction'] = df_new['Average wind direction'].astype(int)
    return df_new
    
@st.cache_data
def h_temp_cat(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_temp_1 = pd.value_counts(df['Temp_cat'])
    df_temp_1 = df_temp_1.reset_index(drop=False)
    df_temp_1 = df_temp_1.rename(columns={'index':'Temp_cat','Temp_cat':'Count'})
    df_temp_1 = df_temp_1.sort_values(by='Temp_cat',ascending=True)
    sns.barplot(x='Temp_cat',y='Count',data=df_temp_1,palette='viridis')
    plt.xlabel('Temperature categories')
    plt.ylabel('Count')
    plt.title('Occurance of temperature categories throught out the hour')
    st.pyplot(fig)

@st.cache_data
def h_temp_2(df):
    df_temp = pd.DataFrame(pd.value_counts(df['Temp_cat']))
    df_temp =df_temp.reset_index(drop=False)
    df_temp = df_temp.rename(columns={'index':'Temperature category','Temp_cat':'Count'})
    total = df_temp['Count'].sum()
    df_temp['Percentage'] = round(df_temp['Count']/total * 100,2)
    df_temp = df_temp.sort_values(by='Temperature category', ascending = True)
    df_temp = df_temp.reset_index(drop=True)
    df_temp_2 = df.groupby(['Temp_cat'])['Temp'].mean()
    df_temp_2 = df_temp_2.reset_index(drop=False)
    df_temp_2.drop(columns=['Temp_cat'],inplace=True)
    df_new = pd.concat([df_temp,df_temp_2],axis=1)
    df_new = df_new.rename(columns={'Temp':'Average temperature'})
    df_new['Average temperature'] = round(df_new['Average temperature'],2)
    return df_new

@st.cache_data
def h_ir(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_ir = pd.value_counts(df['IR_cat'])
    df_ir = df_ir.reset_index(drop=False)
    df_ir = df_ir.rename(columns={'index':'IR_cat','IR_cat':'Count'})
    df_ir = df_ir.sort_values(by='IR_cat',ascending=True)
    sns.barplot(x='IR_cat',y='Count',data=df_ir,palette='viridis')
    plt.xlabel('Solar Irradiation categories')
    plt.ylabel('Count')
    plt.title('Occurance of Solar irradiation categories throught out the hour')
    st.pyplot(fig)

@st.cache_data
def h_ir_2(df):
    df_ir_2 = pd.DataFrame(pd.value_counts(df['IR_cat']))
    df_ir_2 =df_ir_2.reset_index(drop=False)
    df_ir_2 = df_ir_2.rename(columns={'index':'Solar irradiation category','IR_cat':'Count'})
    total = df_ir_2['Count'].sum()
    df_ir_2['Percentage'] = round(df_ir_2['Count']/total * 100,2)
    df_ir_2 = df_ir_2.sort_values(by='Solar irradiation category', ascending = True)
    df_ir_2 = df_ir_2.reset_index(drop=True)
    df_ir_3 = df.groupby(['IR_cat'])['IR'].mean()
    df_ir_3 = df_ir_3.reset_index(drop=False)
    df_ir_3.drop(columns=['IR_cat'],inplace=True)
    df_new = pd.concat([df_ir_2,df_ir_3],axis=1)
    df_new = df_new.rename(columns={'IR':'Average solar irradiation'})
    df_new['Average solar irradiation'] = round(df_new['Average solar irradiation'],2)
    return df_new

@st.cache_data
def h_rh(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_rh = pd.value_counts(df['RH_cat'])
    df_rh = df_rh.reset_index(drop=False)
    df_rh = df_rh.rename(columns={'index':'RH_cat','RH_cat':'Count'})
    df_rh = df_rh.sort_values(by='RH_cat',ascending=True)
    sns.barplot(x='RH_cat',y='Count',data=df_rh,palette='viridis')
    plt.xlabel('Relative humidity categories')
    plt.ylabel('Count')
    plt.title('Occurance of Relative humidity categories throught out the hour')
    st.pyplot(fig)

@st.cache_data
def h_rh_2(df):
    df_rh_2 = pd.DataFrame(pd.value_counts(df['RH_cat']))
    df_rh_2 =df_rh_2.reset_index(drop=False)
    df_rh_2 = df_rh_2.rename(columns={'index':'Relative humidity category','RH_cat':'Count'})
    total = df_rh_2['Count'].sum()
    df_rh_2['Percentage'] = round(df_rh_2['Count']/total * 100,2)
    df_rh_2 = df_rh_2.sort_values(by='Relative humidity category', ascending = True)
    df_rh_2 = df_rh_2.reset_index(drop=True)
    df_rh_3 = df.groupby(['RH_cat'])['RH'].mean()
    df_rh_3 = df_rh_3.reset_index(drop=False)
    df_rh_3.drop(columns=['RH_cat'],inplace=True)
    df_new = pd.concat([df_rh_2,df_rh_3],axis=1)
    df_new = df_new.rename(columns={'RH':'Average relative humidity'})
    df_new['Average relative humidity'] = round(df_new['Average relative humidity'],2)
    return df_new

@st.cache_data
def h_pr(df):
    fig, ax = plt.subplots(figsize=(8,4))
    df_pr = pd.value_counts(df['PR_cat'])
    df_pr = df_pr.reset_index(drop=False)
    df_pr = df_pr.rename(columns={'index':'PR_cat','PR_cat':'Count'})
    df_pr = df_pr.sort_values(by='PR_cat',ascending=True)
    sns.barplot(x='PR_cat',y='Count',data=df_pr,palette='viridis')
    plt.xlabel('Pressure categories')
    plt.ylabel('Count')
    plt.title('Occurance of pressure categories throught out the hour')
    st.pyplot(fig)

@st.cache_data
def h_pr_2(df):
    df_pr_2 = pd.DataFrame(pd.value_counts(df['PR_cat']))
    df_pr_2 =df_pr_2.reset_index(drop=False)
    df_pr_2 = df_pr_2.rename(columns={'index':'Pressure category','PR_cat':'Count'})
    total = df_pr_2['Count'].sum()
    df_pr_2['Percentage'] = round(df_pr_2['Count']/total * 100,2)
    df_pr_2 = df_pr_2.sort_values(by='Pressure category', ascending = True)
    df_pr_2 = df_pr_2.reset_index(drop=True)
    df_pr_3 = df.groupby(['PR_cat'])['PR'].mean()
    df_pr_3 = df_pr_3.reset_index(drop=False)
    df_pr_3.drop(columns=['PR_cat'],inplace=True)
    df_new = pd.concat([df_pr_2,df_pr_3],axis=1)
    df_new = df_new.rename(columns={'PR':'Average Pressure'})
    df_new['Average Pressure'] = round(df_new['Average Pressure'],2)
    return df_new

@st.cache_data
def h_ws_wd(df):
    fig, ax = plt.subplots(figsize=(8,4))
    legend_labels = {1:'0-3', 2:'3-12', 3:'12-25', 4:'>25'}
    sns.lineplot(data=df,x='Min',y='WD', hue='WS_cat',legend='full',palette='icefire')
    plt.xlabel(xlabel ='Minutes of the hour',fontsize=18)
    plt.ylabel(ylabel ='Wind direction in degrees',fontsize=18)
    plt.title('Effect of wind direction on wind speed',fontsize=18)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=2, labels=[legend_labels[label] for label in np.sort(df['WS_cat'].unique())],title = 'wind speed in m/sec')
    st.pyplot(fig)
    st.markdown('1: 0-3 m/sec,\t 2: 3-12 m/sec,\t 3: 12-25 m/sec,\t 4: >25 m/sec')

@st.cache_data
def h_ws_temp(df):
    fig, ax = plt.subplots(figsize=(8,4))
    legend_labels = {1:'0-3', 2:'3-12', 3:'12-25', 4:'>25'}
    sns.lineplot(data=df,x='Min',y='Temp', hue='WS_cat',legend='full',palette='icefire')
    plt.xlabel(xlabel ='Minutes of the hour',fontsize=18)
    plt.ylabel(ylabel ='Temperature in degrees celsius',fontsize=18)
    plt.title('Effect of temperature on wind speed',fontsize=18)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=2, labels=[legend_labels[label] for label in np.sort(df['WS_cat'].unique())],title = 'wind speed in m/sec')
    st.pyplot(fig)
    st.markdown('1: 0-3 m/sec,\t 2: 3-12 m/sec,\t 3: 12-25 m/sec,\t 4: >25 m/sec')

@st.cache_data
def h_ws_ir(df):
    fig, ax = plt.subplots(figsize=(8,4))
    legend_labels = {1:'0-3', 2:'3-12', 3:'12-25', 4:'>25'}
    sns.lineplot(data=df,x='Min',y='IR', hue='WS_cat',legend='full',palette='icefire')
    plt.xlabel(xlabel ='Minutes of the hour',fontsize=18)
    plt.ylabel(ylabel ='Solar Irradiation in Wh/m2',fontsize=18)
    plt.title('Effect of Solar Irradiation on wind speed',fontsize=18)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=2, labels=[legend_labels[label] for label in np.sort(df['WS_cat'].unique())],title = 'wind speed in m/sec')
    st.pyplot(fig)
    st.markdown('1: 0-3 m/sec,\t 2: 3-12 m/sec,\t 3: 12-25 m/sec,\t 4: >25 m/sec')

@st.cache_data
def h_ws_rh(df):
    fig, ax = plt.subplots(figsize=(8,4))
    legend_labels = {1:'0-3', 2:'3-12', 3:'12-25', 4:'>25'}
    sns.lineplot(data=df,x='Min',y='RH', hue='WS_cat',legend='full',palette='icefire')
    plt.xlabel(xlabel ='Minutes of the hour',fontsize=18)
    plt.ylabel(ylabel ='Relative humidity in %',fontsize=18)
    plt.title('Effect of Relative humidity on wind speed',fontsize=18)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=2, labels=[legend_labels[label] for label in np.sort(df['WS_cat'].unique())],title = 'wind speed in m/sec')
    st.pyplot(fig)
    st.markdown('1: 0-3 m/sec,\t 2: 3-12 m/sec,\t 3: 12-25 m/sec,\t 4: >25 m/sec')

@st.cache_data
def h_ws_pr(df):
    fig, ax = plt.subplots(figsize=(8,4))
    legend_labels = {1:'0-3', 2:'3-12', 3:'12-25', 4:'>25'}
    legend_labels = {1: '0-3', 2: '3-12', 3: '12-25', 4:'> 25'}
    sns.lineplot(data=df,x='Min',y='PR', hue='WS_cat',legend='full',palette='icefire')
    plt.xlabel(xlabel ='Minutes of the hour',fontsize=18)
    plt.ylabel(ylabel ='Pressure in hPa',fontsize=18)
    plt.title('Effect of pressure on wind speed',fontsize=18)
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05,1), loc=2, labels=[legend_labels[label] for label in np.sort(df['WS_cat'].unique())],title = 'wind speed in m/sec')
    st.pyplot(fig)
    st.markdown('1: 0-3 m/sec,\t 2: 3-12 m/sec,\t 3: 12-25 m/sec,\t 4: >25 m/sec')

@st.cache_data
def sea_ws_wd_1(df_s1,s1):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(x='WD',y='WS',color='red',data=df_s1,label=s1)
    plt.xlabel('Wind direction in degrees')
    plt.ylabel('Wind speed in m/sec')
    st.pyplot(fig)

@st.cache_data
def sea_ws_wd_2(df_s2,s2):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(x='WD',y='WS',color='green',data=df_s2,label=s2)
    plt.xlabel('Wind direction in degrees')
    plt.ylabel('Wind speed in m/sec')
    st.pyplot(fig)

@st.cache_data
def sea_ws_temp_1(df_s1,s1):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(x='Temp',y='WS',data=df_s1,color='red',label=s1)
    plt.xlabel('Temperature in degree celsius')
    plt.ylabel('Wind speed in m/sec')
    st.pyplot(fig)

@st.cache_data
def sea_ws_temp_2(df_s2,s2):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(x='Temp',y='WS',data=df_s2,color='green',label=s2)
    plt.xlabel('Temperature in degree celsius')
    plt.ylabel('Wind speed in m/sec')
    st.pyplot(fig)

@st.cache_data
def sea_ws_ir_1(df_s1,s1):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(x='IR',y='WS',data=df_s1,color='red',label=s1)
    plt.xlabel('Solar irradiation in Wh/m2')
    plt.ylabel('Wind speed in m/sec')
    st.pyplot(fig)

@st.cache_data
def sea_ws_ir_2(df_s2,s2):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(x='IR',y='WS',data=df_s2,color='green',label=s2)
    plt.xlabel('Solar irradiation in Wh/m2')
    plt.ylabel('Wind speed in m/sec')
    st.pyplot(fig)

@st.cache_data
def sea_ws_rh_1(df_s1,s1):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(x='RH',y='WS',data=df_s1,color='red',label=s1)
    plt.xlabel('Relative humidity in %')
    plt.ylabel('Wind speed in m/sec')
    st.pyplot(fig)

@st.cache_data
def sea_ws_rh_2(df_s2,s2):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(x='RH',y='WS',data=df_s2,color='green',label=s2)
    plt.xlabel('Relative humidity in %')
    plt.ylabel('Wind speed in m/sec')
    st.pyplot(fig)

@st.cache_data
def sea_ws_pr_1(df_s1,s1):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(x='PR',y='WS',data=df_s1,color='red',label=s1)
    plt.xlabel('Pressure in hPa')
    plt.ylabel('Wind speed in m/sec')
    st.pyplot(fig)

@st.cache_data
def sea_ws_pr_2(df_s2,s2):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(x='PR',y='WS',data=df_s2,color='green',label=s2)
    plt.xlabel('Pressure in hPa')
    plt.ylabel('Wind speed in m/sec')
    st.pyplot(fig)

@st.cache_data
def mon_ws_wd_1(df_m1,m1):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(x='WD',y='WS',color='red',data=df_m1,label=m1)
    plt.xlabel('Wind direction in degrees')
    plt.ylabel('Wind speed in m/sec')
    st.pyplot(fig)

@st.cache_data
def mon_ws_wd_2(df_m2,m2):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(x='WD',y='WS',color='green',data=df_m2,label=m2)
    plt.xlabel('Wind direction in degrees')
    plt.ylabel('Wind speed in m/sec')
    st.pyplot(fig)

@st.cache_data
def mon_ws_temp_1(df_m1,m1):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(x='Temp',y='WS',data=df_m1,color='red',label=m1)
    plt.xlabel('Temperature in degree celsius')
    plt.ylabel('Wind speed in m/sec')
    st.pyplot(fig)

@st.cache_data
def mon_ws_temp_2(df_m2,m2):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(x='Temp',y='WS',data=df_m2,color='green',label=m2)
    plt.xlabel('Temperature in degree celsius')
    plt.ylabel('Wind speed in m/sec')
    st.pyplot(fig)

@st.cache_data
def mon_ws_ir_1(df_m1,m1):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(x='IR',y='WS',data=df_m1,color='red',label=m1)
    plt.xlabel('Solar irradiation in Wh/m2')
    plt.ylabel('Wind speed in m/sec')
    st.pyplot(fig)

@st.cache_data
def mon_ws_ir_2(df_m2,m2):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(x='IR',y='WS',data=df_m2,color='green',label=m2)
    plt.xlabel('Solar irradiation in Wh/m2')
    plt.ylabel('Wind speed in m/sec')
    st.pyplot(fig)

@st.cache_data
def mon_ws_rh_1(df_m1,m1):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(x='RH',y='WS',data=df_m1,color='red',label=m1)
    plt.xlabel('Relative humidity in %')
    plt.ylabel('Wind speed in m/sec')
    st.pyplot(fig)

@st.cache_data
def mon_ws_rh_2(df_m2,m2):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(x='RH',y='WS',data=df_m2,color='green',label=m2)
    plt.xlabel('Relative humidity in %')
    plt.ylabel('Wind speed in m/sec')
    st.pyplot(fig)

@st.cache_data
def mon_ws_pr_1(df_m1,m1):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(x='PR',y='WS',data=df_m1,color='red',label=m1)
    plt.xlabel('Pressure in hPa')
    plt.ylabel('Wind speed in m/sec')
    st.pyplot(fig)

@st.cache_data
def mon_ws_pr_2(df_m2,m2):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(x='PR',y='WS',data=df_m2,color='green',label=m2)
    plt.xlabel('Pressure in hPa')
    plt.ylabel('Wind speed in m/sec')
    st.pyplot(fig)




if __name__=='__main__':
    st.set_page_config(layout="wide")
    plt.style.use('seaborn')
    sns.set_style("whitegrid")
    #plt.rcParams["figure.figsize"] = (16, 8)
    centered_text = "<div style='text-align: center; color: Aquamarine;font-family : Monaco ; font-size: 42px;'>Exploratory Data Analysis on wind data for Chitradurga, India</div>"
    st.markdown(centered_text, unsafe_allow_html=True)
    st.write('')
    st.write('')
    #st.subheader(':orange[Exploratory Data Analysis and wind speed prediction]')
    st.write('')
    st.write('')
    selected = option_menu(None, ["Home","Seasonwise", "Monthwise", "Daywise","Hourwise","Seasonwise comparison","Monthwise comparison","Animated charts"], 
    icons=["house", "binoculars-fill", "binoculars", "binoculars-fill", "binoculars","bar-chart","bar-chart-fill","fast-forward-btn-fill"], 
    menu_icon="cast", default_index=0, orientation="horizontal")
    if selected == 'Home':
        st.subheader(":orange[🔖 **About the project** ]")
        st.write('')
        st.write('')
        st.markdown('<div style="text-align: justify"> Renewable energy plays a crucial role in addressing the urgent challenges of climate change, energy security, and sustainable development. Its importance stems from several key factors. Firstly, renewable energy sources such as solar, wind, hydro, and geothermal power generate electricity without relying on finite fossil fuel resources, reducing greenhouse gas emissions and mitigating the harmful effects of climate change. This transition to renewables is essential for achieving global climate goals and limiting global temperature rise. Secondly, renewable energy enhances energy security by diversifying the energy mix and reducing dependence on imported fuels, fostering self-sufficiency and resilience. Additionally, the decentralized nature of renewable energy systems empowers communities and promotes local economic development, creating jobs and stimulating innovation. Furthermore, renewable energy helps address social and environmental concerns by improving air quality, reducing pollution-related health risks, and protecting ecosystems. By transitioning to renewable energy, we can foster a sustainable, low-carbon future that ensures a cleaner, healthier, and more prosperous world for current and future generations.  </div>', unsafe_allow_html=True)
        st.write('')
        st.markdown('<div style="text-align: justify"> Wind energy holds significant significance as a renewable energy source due to its abundance, scalability, and environmental benefits. Harnessing the power of wind through wind turbines allows for the generation of clean and sustainable electricity without producing greenhouse gas emissions or air pollutants. The scalability of wind power enables large-scale installations, both onshore and offshore, contributing to the global energy transition. Wind energy offers a reliable and predictable power source, reducing dependence on fossil fuels and enhancing energy security. Furthermore, wind farms can stimulate local economies by creating jobs, attracting investments, and providing lease income to landowners. As a renewable resource, wind energy plays a vital role in reducing carbon emissions, mitigating climate change, and fostering a sustainable future for generations to come. </div>', unsafe_allow_html=True)
        st.write('')
        st.markdown('<div style="text-align: justify"> Wind energy is of great importance in India, with an installed capacity of 42633 MW (March 2023) of Wind Energy, Renewable Energy Sources (excluding large Hydro) currently accounts for 30.08% (125160 MW) of India\’s overall installed power capacity of 416059 MW (31.03.2023). Wind Energy holds the major portion of 34.06% of total RE capacity among renewable and continued as the major supplier of clean energy. The Government of India has fixed a target of 500 GW of Renewable Energy by 2030 out of which 140 GW will be from Wind. The Wind Potential in India was first estimated by National Institute of Wind Energy (NIWE) at 50m hub-height i.e. 49 GW but according to the survey at 80m hub height, the potential grows as much as 102 GW and 302GW at 100 Meter hub height. Further a new study by NIWE at 120m height has estimated a potential 695GW. One of the major advantages of wind energy is its inherent strength to support rural employment and uplift of rural economy. Further, unlike all other sources of power, wind energy does not consume any water- which in itself will become a scarce commodity. Overall the future of Wind Energy in India is bright as energy security and self-sufficiency is identified as the major driver. The biggest advantage with wind energy is that the fuel is free, and also it doesn’t produce CO2 emission. Wind farm can be built reasonably fast, the wind farm land can be used for farming as well thus serving dual purpose, and it is cost-effective as compare to other forms of renewable energy. </div>', unsafe_allow_html=True)
        st.write('')
        st.markdown('<div style="text-align: justify"> EDA stands for Exploratory Data Analysis. It is a process of analyzing and summarizing data to gain insights, identify patterns, and uncover relationships in a concise and intuitive manner. EDA involves various statistical and visual techniques to explore data sets, understand their underlying structure, and make informed decisions. By using EDA, researchers, analysts, and data scientists can uncover hidden patterns, detect anomalies, validate assumptions, and guide the development of further analysis or modelling. It is an essential step in the data analysis pipeline, providing a foundation for understanding data before diving into more complex analyses or machine learning algorithms. </div>', unsafe_allow_html=True)
        st.write('')
        st.markdown('<div style="text-align: justify"> The main objective of this project is to ascertain the effect of pertaining weather conditions on wind speed. The effects vary with time and hence call for seasonal, monthly, daily and hourly analysis which is exactly what I have tried to achieve here. The dataset comprises of wind data (Datetime, Wind direction, Wind speed, Temperature, Solar irradiation, Relative humidity, Pressure ) for the location of Chitradurga, India. The dataset is taken from SODA MERRA-2 website, in which one datapoint is recorded for every 10 minute from 2002-2022 accounting for a total of 1048551 data points for the entire period. </div>', unsafe_allow_html=True)
        st.write('')
        st.markdown('<div style="text-align: justify"> Karnataka has the following four seasons in the year: The winter season from January to February, The summer season from March to May, The monsoon season from June to September, The post-monsoon season from October to December. Hence, created a separate dataset for each season. For monthly analysis, created separate datasets for all 12 months for 20 years and for daily and hourly analysis used filtering on monthly datasets. </div>', unsafe_allow_html=True)
        st.write('')
        st.markdown('<div style="text-align: justify"> I intentionally didnt try to provide any description for the plots as the plots speak for themselves.  </div>', unsafe_allow_html=True)
        st.write('')

        st.markdown('<div style="text-align: justify"> Finally,i would like to thank https://www.soda-pro.com/ for providing access to their dataset which made this research activity possible. </div>', unsafe_allow_html=True)
        st.write('')
        
        # st.divider()    
        # st.subheader(':orange[✒️ About the developer]')
        # st.write('')
        # st.markdown('<div style="text-align: justify">Gururaj H C is passionate about Machine Learning and fascinated by its myriad real world applications. Possesses work experience with both IT industry and academia. Currently pursuing “IIT-Madras Certified Advanced Programmer with Data Science Mastery Program” course as a part of his learning journey.  </div>', unsafe_allow_html=True)
        # st.divider()
        col1001, col1002, col1003 = st.columns([10,10,10])
        with col1002:
            st.markdown("""
                    <style>
                    @import url('https://fonts.googleapis.com/css2?family=Agdasima');
                    .custom-text { font-family: 'Agdasima', sans-serif; font-size: 28px;color:Aquamarine }
                    </style>
                    <p class="custom-text">An Effort by : MAVERICK_GR</p>
                    """, unsafe_allow_html=True)
            st.markdown(':red[**DEVELOPER CONTACT DETAILS**]')
        col2001, col2002 = st.columns([10,10])
        with col2001:
            st.markdown(":orange[email id:] gururaj008@gmail.com")
            st.markdown(":orange[Personal webpage hosting other Datascience projects: ]") 
            st.markdown(':green[https://gururaj008-personal-webpage.streamlit.app/]')
            st.markdown(':green[http://gururaj008.pythonanywhere.com/]') 
        with col2002:
            st.markdown(":orange[LinkedIn profile :] https://www.linkedin.com/in/gururaj-hc-machine-learning-enthusiast/")
            st.markdown(":orange[Github link:] https://github.com/Gururaj008 ")






    if selected == 'Seasonwise':
        season = st.selectbox('Please select the season from the dropdown menu',('Mansoon', 'Post_mansoon', 'Winter','Summer'))
        if st.button(f'🟢 Get the data for {season} 🟢',use_container_width=True):
            df = pd.read_csv(f'wind_{season}.csv')
            #df = df.drop(columns=['Unnamed: 0'],axis=1)
            r,c = df.shape
            st.success(f'{season} dataset loaded and has {r} instances with {c} features', icon="✅")
            st.write('')
        st.write('')
        st.markdown("<p style='font-size: 23px;color: orange;'>🕛 Use this section to visualize the variation of weather elements with time and thier impact on wind speed 🕕</p>", unsafe_allow_html=True)
        st.write('')
        col4, col5, col6 = st.columns([5,18,5])
        with col5:
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of wind speed during the Season ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=1):
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                s_wind_speed(df)
            st.write('')
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of wind direction during the Season ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=2):
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                s_wind_direction(df)
            st.write('')
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of temperature during the Season ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=3):
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                s_temp(df)
            st.write('')
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of Solar irradation during the Season ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=4):
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                s_solar(df)

            st.write('')
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of Relative humidity during the Season ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=5):
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                s_rel_hum(df)
            
            st.write('')
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of Pressure during the Season ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=6):
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                s_pre(df)

        st.divider()
        st.markdown("<p style='font-size: 23px;color: cyan;'>🕛 Use this section to visualize the weather elements categorized and thier percentage occurance 🕕</p>", unsafe_allow_html=True)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Wind speed - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        st.write('')
        if st.button('📈  Display the plot 📈',use_container_width=True,key=7):
            col12,col13 = st.columns([15,15])
            with col12:
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                s_ws_cat(df)
            with col13:
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                s_ws_cat_2(df)
            st.write('')
            st.write('')
            with col12:
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                res = s_ws_cat_3(df)
                st.table(res)
            with col13:
                dict1 = {'Wind speed category':[1,2,3,4],'Wind speed range':['0-3','3-12','12-25','>25'],
         'Description':['Below cut-in speed',
                       'Speed above cut-in but below rated speed',
                       'Constant rated power output speed range',
                       'Beyond cut-out wind speed']}
                res2 = pd.DataFrame(dict1)
                st.table(res2)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Wind direction - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        if st.button('📈  Display the plot 📈',use_container_width=True,key=8):
            col14,col15 = st.columns([15,15])
            with col14:
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                s_wd_cat(df)        
            with col15:
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                res2 = s_wd_cat_2(df)
                st.table(res2)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Temperature - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        if st.button('📈  Display the plot 📈',use_container_width=True,key=9): 
            col18,col19,col20 = st.columns([5,12,5])        
            with col19:
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                s_temp_cat(df)  
            col16,col17 = st.columns([15,15])      
            with col16:
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                res3 = s_temp_2(df)
                st.table(res3)
            with col17:
                dict_temp = {'Temperature category':[1,2,3,4,5,6,7],'Temperature range in degrees':['0 - 10','10 - 20','20 - 25','25 - 30','30 - 35','35 - 40','> 40'],}
                df_temp = pd.DataFrame(dict_temp)
                st.table(df_temp)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Solar irradiation - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        if st.button('📈  Display the plot 📈',use_container_width=True,key=10): 
            col21,col22,col23 = st.columns([5,12,5])        
            with col22:
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                s_ir(df)  
            col18,col19 = st.columns([15,15])      
            with col18:
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                res4 = s_ir_2(df)
                st.table(res4)
            with col19:
                dict_ir = {'Solar irradiation category':[1,2,3,4,5,6,7],'Solar irradiation in Wh/m2':['0 - 30','30 - 60','60 - 90','90 - 120','120 - 150','150 - 180','> 180'],}
                df_ir = pd.DataFrame(dict_ir)
                st.table(df_ir)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Relative humidity - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        if st.button('📈  Display the plot 📈',use_container_width=True,key=11): 
            col24,col25,col26 = st.columns([5,12,5])        
            with col25:
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                s_rh(df)  
            col27,col28 = st.columns([15,15])      
            with col27:
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                res5 = s_rh_2(df)
                st.table(res5)
            with col28:
                dict_rh = {'Relative humidity category':[1,2,3,4,5],'Relative humidity in %':['0 - 20','20 - 40','40 - 60','60 - 80','80 - 100']}
                df_rh = pd.DataFrame(dict_rh)
                st.table(df_rh)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Pressure - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        if st.button('📈  Display the plot 📈',use_container_width=True,key=12): 
            col29,col30,col31 = st.columns([5,12,5])        
            with col30:
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                s_pr(df)  
            col31,col32 = st.columns([15,15])      
            with col31:
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                res5 = s_pr_2(df)
                st.table(res5)
            with col32:
                dict_pr = {'Pressure category':[1,2,3],'Pressure in hPa':['920 - 930','930 - 940','> 930']}
                df_pr = pd.DataFrame(dict_pr)
                st.table(df_pr)

        st.divider()
        st.markdown("<p style='text-align: center; color: yellow; font-size: 25px;'>🕛 Use this section to visualize the effect weather elements on wind speed 🕕</p>", unsafe_allow_html=True)
        st.write('')
        col33, col34, col35 = st.columns([5,18,5])
        with col34:
            st.write('')
            centered_text = "<div style='text-align: center; color: yellow; font-size: 25px;'>💹 Effect of wind direction on wind speed 💹</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=13):
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                s_ws_wd(df)
        st.write('')
        st.write('')
        col36, col37, col38 = st.columns([5,18,5])
        with col37:
            st.write('')
            centered_text = "<div style='text-align: center; color: yellow; font-size: 25px;'>💹 Effect of temperature on wind speed 💹</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=14):
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                s_ws_temp(df)
            st.write('')
            st.write('')
        col39, col40, col41 = st.columns([5,18,5])
        with col40:
            st.write('')
            centered_text = "<div style='text-align: center; color: yellow; font-size: 25px;'>💹 Effect of solar irradiation on wind speed 💹</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=15):
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                s_ws_ir(df)
            st.write('')
            st.write('')
        col42, col43, col44 = st.columns([5,18,5])
        with col43:
            st.write('')
            centered_text = "<div style='text-align: center; color: yellow; font-size: 25px;'>💹 Effect of relative humidity on wind speed 💹</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=16):
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                s_ws_rh(df)
            st.write('')
            st.write('')
        col45, col46, col47 = st.columns([5,18,5])
        with col46:
            st.write('')
            centered_text = "<div style='text-align: center; color: yellow; font-size: 25px;'>💹 Effect of pressure on wind speed 💹</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=17):
                df = pd.read_csv(f'wind_{season}.csv')
                #df = df.drop(columns=['Unnamed: 0'],axis=1)
                s_ws_pr(df)
            st.write('')
            st.write('')
            st.divider()
            col1001, col1002, col1003,col1004, col1005 = st.columns([10,10,10,10,15])
            with col1005:
                st.markdown("""
                        <style>
                        @import url('https://fonts.googleapis.com/css2?family=Agdasima');
                        .custom-text { font-family: 'Agdasima', sans-serif; font-size: 28px;color:cyan }
                        </style>
                        <p class="custom-text">An Effort by : MAVERICK_GR</p>
                        """, unsafe_allow_html=True)
    
    if selected == 'Monthwise':
        month = st.selectbox('Please select the month from the dropdown menu',('January','February','March','April','May','June','July','August','September','November','December'))
        if st.button(f'🟢 Get the data for {month} 🟢',use_container_width=True):
            df = pd.read_csv(f'wind_{month}.csv')
            df = df.drop(columns=['Unnamed: 0'],axis=1)
            r,c = df.shape
            st.success(f'{month} dataset loaded and has {r} instances with {c} features', icon="✅")
        st.write('')
        st.divider()
        st.markdown("<p style='font-size: 23px;color: orange;'>🕛 Use this section to visualize the variation of weather elements with time and thier impact on wind speed 🕕</p>", unsafe_allow_html=True)
        st.write('')
        col104, col105, col106 = st.columns([5,18,5])
        with col105:
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of wind speed during the Month ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=101):
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                m_wind_speed(df)
            st.write('')
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of wind direction during the Month ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=102):
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                m_wind_direction(df)
            st.write('')
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of temperature during the Month ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=103):
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                m_temp(df)
            st.write('')
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of Solar irradation during the Month ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=104):
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                m_solar(df)

            st.write('')
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of Relative humidity during the Month ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=5):
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                m_rel_hum(df)
            
            st.write('')
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of Pressure during the Month ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=106):
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                m_pre(df)
        
        st.divider()
        st.markdown("<p style='font-size: 23px;color: cyan;'>🕛 Use this section to visualize the weather elements categorized and thier percentage occurance 🕕</p>", unsafe_allow_html=True)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Wind speed - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        st.write('')
        if st.button('📈  Display the plot 📈',use_container_width=True,key=107):
            col112,col113 = st.columns([15,15])
            with col112:
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                m_ws_cat(df)
            with col113:
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                m_ws_cat_2(df)
            st.write('')
            st.write('')
            with col112:
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                res = m_ws_cat_3(df)
                st.table(res)
            with col113:
                dict1 = {'Wind speed category':[1,2,3,4],'Wind speed range':['0-3','3-12','12-25','>25'],
         'Description':['Below cut-in speed',
                       'Speed above cut-in but below rated speed',
                       'Constant rated power output speed range',
                       'Beyond cut-out wind speed']}
                res2 = pd.DataFrame(dict1)
                st.table(res2)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Wind direction - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        if st.button('📈  Display the plot 📈',use_container_width=True,key=108):
            col114,col115 = st.columns([15,15])
            with col114:
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                m_wd_cat(df)        
            with col115:
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                res2 = m_wd_cat_2(df)
                st.table(res2)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Temperature - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        if st.button('📈  Display the plot 📈',use_container_width=True,key=109): 
            col118,col119,col120 = st.columns([5,12,5])        
            with col119:
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                m_temp_cat(df)  
            col116,col117 = st.columns([15,15])      
            with col116:
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                res3 = m_temp_2(df)
                st.table(res3)
            with col117:
                dict_temp = {'Temperature category':[1,2,3,4,5,6,7],'Temperature range in degrees':['0 - 10','10 - 20','20 - 25','25 - 30','30 - 35','35 - 40','> 40'],}
                df_temp = pd.DataFrame(dict_temp)
                st.table(df_temp)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Solar irradiation - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        if st.button('📈  Display the plot 📈',use_container_width=True,key=10): 
            col121,col122,col123 = st.columns([5,12,5])        
            with col122:
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                m_ir(df)  
            col118,col119 = st.columns([15,15])      
            with col118:
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                res4 = m_ir_2(df)
                st.table(res4)
            with col119:
                dict_ir = {'Solar irradiation category':[1,2,3,4,5,6,7],'Solar irradiation in Wh/m2':['0 - 30','30 - 60','60 - 90','90 - 120','120 - 150','150 - 180','> 180'],}
                df_ir = pd.DataFrame(dict_ir)
                st.table(df_ir)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Relative humidity - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        if st.button('📈  Display the plot 📈',use_container_width=True,key=11): 
            col124,col125,col126 = st.columns([5,12,5])        
            with col125:
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                m_rh(df)  
            col127,col128 = st.columns([15,15])      
            with col127:
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                res5 = m_rh_2(df)
                st.table(res5)
            with col128:
                dict_rh = {'Relative humidity category':[1,2,3,4,5],'Relative humidity in %':['0 - 20','20 - 40','40 - 60','60 - 80','80 - 100']}
                df_rh = pd.DataFrame(dict_rh)
                st.table(df_rh)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Pressure - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        if st.button('📈  Display the plot 📈',use_container_width=True,key=12): 
            col129,col130,col131 = st.columns([5,12,5])        
            with col130:
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                m_pr(df)  
            col131,col132 = st.columns([15,15])      
            with col131:
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                res5 = m_pr_2(df)
                st.table(res5)
            with col132:
                dict_pr = {'Pressure category':[1,2,3],'Pressure in hPa':['920 - 930','930 - 940','> 930']}
                df_pr = pd.DataFrame(dict_pr)
                st.table(df_pr)

        st.divider()
        st.markdown("<p style='text-align: center; color: yellow; font-size: 25px;'>🕛 Use this section to visualize the effect weather elements on wind speed 🕕</p>", unsafe_allow_html=True)
        st.write('')
        col133, col134, col135 = st.columns([5,18,5])
        with col134:
            st.write('')
            centered_text = "<div style='text-align: center; color: yellow; font-size: 25px;'>💹 Effect of wind direction on wind speed 💹</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=113):
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                m_ws_wd(df)
        st.write('')
        st.write('')
        col136, col137, col138 = st.columns([5,18,5])
        with col137:
            st.write('')
            centered_text = "<div style='text-align: center; color: yellow; font-size: 25px;'>💹 Effect of temperature on wind speed 💹</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=114):
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                m_ws_temp(df)
            st.write('')
            st.write('')
        col139, col140, col141 = st.columns([5,18,5])
        with col140:
            st.write('')
            centered_text = "<div style='text-align: center; color: yellow; font-size: 25px;'>💹 Effect of solar irradiation on wind speed 💹</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=115):
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                m_ws_ir(df)
            st.write('')
            st.write('')
        col142, col143, col144 = st.columns([5,18,5])
        with col143:
            st.write('')
            centered_text = "<div style='text-align: center; color: yellow; font-size: 25px;'>💹 Effect of relative humidity on wind speed 💹</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=116):
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                m_ws_rh(df)
            st.write('')
            st.write('')
        col145, col146, col147 = st.columns([5,18,5])
        with col146:
            st.write('')
            centered_text = "<div style='text-align: center; color: yellow; font-size: 25px;'>💹 Effect of pressure on wind speed 💹</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=117):
                df = pd.read_csv(f'wind_{month}.csv')
                df = df.drop(columns=['Unnamed: 0'],axis=1)
                m_ws_pr(df)
            st.write('')
            st.write('')
        st.divider()
        col1001, col1002, col1003,col1004, col1005 = st.columns([10,10,10,10,15])
        with col1005:
            st.markdown("""
                    <style>
                    @import url('https://fonts.googleapis.com/css2?family=Agdasima');
                    .custom-text { font-family: 'Agdasima', sans-serif; font-size: 28px;color:aquamarine }
                    </style>
                    <p class="custom-text">An Effort by : MAVERICK_GR</p>
                    """, unsafe_allow_html=True)



    if selected == 'Daywise':
        col171,col172 = st.columns([10,10])
        with col171:
            month = st.selectbox('Please select the month from the dropdown menu',('January','February','March','April','May','June','July','August','September','November','December'))
        with col172:
            date = [i for i in range(1,32)]
            day = st.selectbox('Please select the date from dropdown menu',(list(date)))
         
        if st.button(f'🟢 Get the data for {day} - {month} 🟢',use_container_width=True):
            df_1 = pd.read_csv(f'wind_{month}.csv')
            df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
            df = df_1[df_1['Day'] == day]
            if len(df) == 0:
                st.error('Kindly choose the right combination of date and month', icon="🚨")
            else:
                r,c = df.shape
                st.success(f'{day} of {month} dataset loaded and has {r} instances with {c} features', icon="✅")
        st.write('')
        st.divider()

        st.markdown("<p style='font-size: 23px;color: orange;'>🕛 Use this section to visualize the variation of weather elements with time and thier impact on wind speed 🕕</p>", unsafe_allow_html=True)
        st.write('')
        col304, col305, col306 = st.columns([5,18,5])
        with col305:
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of wind speed during the Day ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=301):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                d_wind_speed(df)
            st.write('')
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of wind direction during the Day ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=302):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                d_wind_direction(df)
            st.write('')
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of temperature during the Day ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=303):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                d_temp(df)
            st.write('')
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of Solar irradation during the Day ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=304):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                d_solar(df)

            st.write('')
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of Relative humidity during the Day ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=305):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                d_rel_hum(df)
            
            st.write('')
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of Pressure during the Day ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=306):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                d_pre(df)

        st.divider()
        st.markdown("<p style='font-size: 23px;color: cyan;'>🕛 Use this section to visualize the weather elements categorized and thier percentage occurance 🕕</p>", unsafe_allow_html=True)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Wind speed - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        st.write('')
        if st.button('📈  Display the plot 📈',use_container_width=True,key=307):
            col312,col313 = st.columns([15,15])
            with col312:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                d_ws_cat(df)
            with col313:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                d_ws_cat_2(df)
            st.write('')
            st.write('')
            with col312:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                res = d_ws_cat_3(df)
                st.table(res)
            with col313:
                dict1 = {'Wind speed category':[1,2,3,4],'Wind speed range':['0-3','3-12','12-25','>25'],
         'Description':['Below cut-in speed',
                       'Speed above cut-in but below rated speed',
                       'Constant rated power output speed range',
                       'Beyond cut-out wind speed']}
                res2 = pd.DataFrame(dict1)
                st.table(res2)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Wind direction - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        if st.button('📈  Display the plot 📈',use_container_width=True,key=308):
            col314,col315 = st.columns([15,15])
            with col314:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                d_wd_cat(df)        
            with col315:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                res2 = d_wd_cat_2(df)
                st.table(res2)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Temperature - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        if st.button('📈  Display the plot 📈',use_container_width=True,key=309): 
            col318,col319,col320 = st.columns([5,12,5])        
            with col319:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                d_temp_cat(df)  
            col316,col317 = st.columns([15,15])      
            with col316:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                res3 = d_temp_2(df)
                st.table(res3)
            with col317:
                dict_temp = {'Temperature category':[1,2,3,4,5,6,7],'Temperature range in degrees':['0 - 10','10 - 20','20 - 25','25 - 30','30 - 35','35 - 40','> 40'],}
                df_temp = pd.DataFrame(dict_temp)
                st.table(df_temp)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Solar irradiation - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        if st.button('📈  Display the plot 📈',use_container_width=True,key=310): 
            col321,col322,co323 = st.columns([5,12,5])        
            with col322:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                d_ir(df)  
            col318,col319 = st.columns([15,15])      
            with col318:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                res4 = d_ir_2(df)
                st.table(res4)
            with col319:
                dict_ir = {'Solar irradiation category':[1,2,3,4,5,6,7],'Solar irradiation in Wh/m2':['0 - 30','30 - 60','60 - 90','90 - 120','120 - 150','150 - 180','> 180'],}
                df_ir = pd.DataFrame(dict_ir)
                st.table(df_ir)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Relative humidity - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        if st.button('📈  Display the plot 📈',use_container_width=True,key=311): 
            col324,col325,col326 = st.columns([5,12,5])        
            with col325:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                d_rh(df)  
            col327,col328 = st.columns([15,15])      
            with col327:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                res5 = d_rh_2(df)
                st.table(res5)
            with col328:
                dict_rh = {'Relative humidity category':[1,2,3,4,5],'Relative humidity in %':['0 - 20','20 - 40','40 - 60','60 - 80','80 - 100']}
                df_rh = pd.DataFrame(dict_rh)
                st.table(df_rh)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Pressure - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        if st.button('📈  Display the plot 📈',use_container_width=True,key=12): 
            col329,col330,col331 = st.columns([5,12,5])        
            with col330:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                d_pr(df)  
            col331,col332 = st.columns([15,15])      
            with col331:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                res5 = d_pr_2(df)
                st.table(res5)
            with col332:
                dict_pr = {'Pressure category':[1,2,3],'Pressure in hPa':['920 - 930','930 - 940','> 930']}
                df_pr = pd.DataFrame(dict_pr)
                st.table(df_pr)

        st.divider()
        st.markdown("<p style='text-align: center; color: yellow; font-size: 25px;'>🕛 Use this section to visualize the effect weather elements on wind speed 🕕</p>", unsafe_allow_html=True)
        st.write('')
        col333, col334, col335 = st.columns([5,18,5])
        with col334:
            st.write('')
            centered_text = "<div style='text-align: center; color: yellow; font-size: 25px;'>💹 Effect of wind direction on wind speed 💹</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=313):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                d_ws_wd(df)
        st.write('')
        st.write('')
        col336, col337, col338 = st.columns([5,18,5])
        with col337:
            st.write('')
            centered_text = "<div style='text-align: center; color: yellow; font-size: 25px;'>💹 Effect of temperature on wind speed 💹</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=314):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                d_ws_temp(df)
            st.write('')
            st.write('')
        col339, col340, col341 = st.columns([5,18,5])
        with col340:
            st.write('')
            centered_text = "<div style='text-align: center; color: yellow; font-size: 25px;'>💹 Effect of solar irradiation on wind speed 💹</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=315):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                d_ws_ir(df)
            st.write('')
            st.write('')
        col342, col343, col344 = st.columns([5,18,5])
        with col343:
            st.write('')
            centered_text = "<div style='text-align: center; color: yellow; font-size: 25px;'>💹 Effect of relative humidity on wind speed 💹</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=316):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                d_ws_rh(df)
            st.write('')
            st.write('')
        col345, col346, col347 = st.columns([5,18,5])
        with col346:
            st.write('')
            centered_text = "<div style='text-align: center; color: yellow; font-size: 25px;'>💹 Effect of pressure on wind speed 💹</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=117):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df = df_1[df_1['Day'] == day]
                d_ws_pr(df)
            st.write('')
            st.write('')
        st.divider()
        col1001, col1002, col1003,col1004, col1005 = st.columns([10,10,10,10,15])
        with col1005:
            st.markdown("""
                    <style>
                    @import url('https://fonts.googleapis.com/css2?family=Agdasima');
                    .custom-text { font-family: 'Agdasima', sans-serif; font-size: 28px;color:orange }
                    </style>
                    <p class="custom-text">An Effort by : MAVERICK_GR</p>
                    """, unsafe_allow_html=True)
    
    
    
    if selected == 'Hourwise':
        col181, col182, col183 = st.columns([10,10,10])
        with col181:
            month = st.selectbox('Please select the month from the dropdown menu',('January','February','March','April','May','June','July','August','September','November','December'))
        with col182:
            date = [i for i in range(1,32)]
            day = st.selectbox('Please select the date from dropdown menu',(list(date)))
        with col183:
            hours = [i for i in range(0,24)]
            hour = st.selectbox('Please select the hour from dropdown menu',(list(hours)))
        
        if st.button(f'🟢 Get the data for {hour} o\' clock of {day} - {month} 🟢',use_container_width=True):
            df_1 = pd.read_csv(f'wind_{month}.csv')
            df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
            df_2 = df_1[df_1['Day'] == day]
            df = df_2[df_2['Hour'] == hour]
            if len(df) == 0:
                st.error('Kindly choose the right combination of date and month', icon="🚨")
            else:
                r,c = df.shape
                st.success(f'{hour} o\' clock {day} of {month} dataset loaded and has {r} instances with {c} features', icon="✅")
        st.write('')
        st.divider()

        st.markdown("<p style='font-size: 23px;color: orange;'>🕛 Use this section to visualize the variation of weather elements with time and thier impact on wind speed 🕕</p>", unsafe_allow_html=True)
        st.write('')
        col404, col405, col406 = st.columns([5,18,5])
        with col405:
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of wind speed during the Hour ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=401):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                h_wind_speed(df)
            st.write('')
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of wind direction during the Hour ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=402):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                h_wind_direction(df)
            st.write('')
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of temperature during the Hour ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=403):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                h_temp(df)
            st.write('')
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of Solar irradation during the Hour ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=404):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                h_solar(df)

            st.write('')
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of Relative humidity during the Hour ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=405):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                h_rel_hum(df)
            
            st.write('')
            st.write('')
            centered_text = "<div style='text-align: center; color: orange; font-size: 25px;'>⏪ Variation of Pressure during the Hour ⏩</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=406):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                h_pre(df)

        st.divider()
        st.markdown("<p style='font-size: 23px;color: cyan;'>🕛 Use this section to visualize the weather elements categorized and thier percentage occurance 🕕</p>", unsafe_allow_html=True)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Wind speed - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        st.write('')
        if st.button('📈  Display the plot 📈',use_container_width=True,key=407):
            col412,col413 = st.columns([15,15])
            with col412:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                h_ws_cat(df)
            with col413:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                h_ws_cat_2(df)
            st.write('')
            st.write('')
            with col412:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                res = h_ws_cat_3(df)
                st.table(res)
            with col413:
                dict1 = {'Wind speed category':[1,2,3,4],'Wind speed range':['0-3','3-12','12-25','>25'],
         'Description':['Below cut-in speed',
                       'Speed above cut-in but below rated speed',
                       'Constant rated power output speed range',
                       'Beyond cut-out wind speed']}
                res2 = pd.DataFrame(dict1)
                st.table(res2)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Wind direction - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        if st.button('📈  Display the plot 📈',use_container_width=True,key=408):
            col414,col415 = st.columns([15,15])
            with col414:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                h_wd_cat(df)        
            with col415:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                res2 = h_wd_cat_2(df)
                st.table(res2)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Temperature - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        if st.button('📈  Display the plot 📈',use_container_width=True,key=409): 
            col4018,col4019,col4020 = st.columns([5,12,5])        
            with col4019:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                h_temp_cat(df)  
            col416,col417 = st.columns([15,15])      
            with col416:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                res3 = h_temp_2(df)
                st.table(res3)
            with col417:
                dict_temp = {'Temperature category':[1,2,3,4,5,6,7],'Temperature range in degrees':['0 - 10','10 - 20','20 - 25','25 - 30','30 - 35','35 - 40','> 40'],}
                df_temp = pd.DataFrame(dict_temp)
                st.table(df_temp)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Solar irradiation - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        if st.button('📈  Display the plot 📈',use_container_width=True,key=410): 
            col421,col422,col423 = st.columns([5,12,5])        
            with col422:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                h_ir(df)  
            col418,col419 = st.columns([15,15])      
            with col418:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                res4 = s_ir_2(df)
                st.table(res4)
            with col419:
                dict_ir = {'Solar irradiation category':[1,2,3,4,5,6,7],'Solar irradiation in Wh/m2':['0 - 30','30 - 60','60 - 90','90 - 120','120 - 150','150 - 180','> 180'],}
                df_ir = pd.DataFrame(dict_ir)
                st.table(df_ir)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Relative humidity - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        if st.button('📈  Display the plot 📈',use_container_width=True,key=411): 
            col424,col425,col426 = st.columns([5,12,5])        
            with col425:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                h_rh(df)  
            col427,col428 = st.columns([15,15])      
            with col427:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                res5 = h_rh_2(df)
                st.table(res5)
            with col428:
                dict_rh = {'Relative humidity category':[1,2,3,4,5],'Relative humidity in %':['0 - 20','20 - 40','40 - 60','60 - 80','80 - 100']}
                df_rh = pd.DataFrame(dict_rh)
                st.table(df_rh)
        st.write('')
        st.write('')
        centered_text = "<div style='text-align: center; color: cyan; font-size: 25px;'>🌏 Pressure - Most and least occuring categories 🌏</div>"
        st.markdown(centered_text, unsafe_allow_html=True)
        if st.button('📈  Display the plot 📈',use_container_width=True,key=412): 
            col429,col430,col431 = st.columns([5,12,5])        
            with col430:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                h_pr(df)  
            col431,col432 = st.columns([15,15])      
            with col431:
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                res5 = h_pr_2(df)
                st.table(res5)
            with col432:
                dict_pr = {'Pressure category':[1,2,3],'Pressure in hPa':['920 - 930','930 - 940','> 930']}
                df_pr = pd.DataFrame(dict_pr)
                st.table(df_pr)

        st.divider()
        st.markdown("<p style='text-align: center; color: yellow; font-size: 25px;'>🕛 Use this section to visualize the effect weather elements on wind speed 🕕</p>", unsafe_allow_html=True)
        st.write('')
        col433, col434, col435 = st.columns([5,18,5])
        with col434:
            st.write('')
            centered_text = "<div style='text-align: center; color: yellow; font-size: 25px;'>💹 Effect of wind direction on wind speed 💹</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=413):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                h_ws_wd(df)
        st.write('')
        st.write('')
        col436, col437, col438 = st.columns([5,18,5])
        with col437:
            st.write('')
            centered_text = "<div style='text-align: center; color: yellow; font-size: 25px;'>💹 Effect of temperature on wind speed 💹</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=414):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                h_ws_temp(df)
            st.write('')
            st.write('')
        col439, col440, col441 = st.columns([5,18,5])
        with col440:
            st.write('')
            centered_text = "<div style='text-align: center; color: yellow; font-size: 25px;'>💹 Effect of solar irradiation on wind speed 💹</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=415):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                h_ws_ir(df)
            st.write('')
            st.write('')
        col442, col443, col444 = st.columns([5,18,5])
        with col443:
            st.write('')
            centered_text = "<div style='text-align: center; color: yellow; font-size: 25px;'>💹 Effect of relative humidity on wind speed 💹</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=416):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                h_ws_rh(df)
            st.write('')
            st.write('')
        col445, col446, col447 = st.columns([5,18,5])
        with col446:
            st.write('')
            centered_text = "<div style='text-align: center; color: yellow; font-size: 25px;'>💹 Effect of pressure on wind speed 💹</div>"
            st.markdown(centered_text, unsafe_allow_html=True)
            st.write('')
            if st.button('📈  Display the plot 📈',use_container_width=True,key=117):
                df_1 = pd.read_csv(f'wind_{month}.csv')
                df_1 = df_1.drop(columns=['Unnamed: 0'],axis=1)
                df_2 = df_1[df_1['Day'] == day]
                df = df_2[df_2['Hour'] == hour]
                h_ws_pr(df)
            st.write('')
            st.write('')
        st.divider()
        col1001, col1002, col1003,col1004, col1005 = st.columns([10,10,10,10,15])
        with col1005:
            st.markdown("""
                    <style>
                    @import url('https://fonts.googleapis.com/css2?family=Agdasima');
                    .custom-text { font-family: 'Agdasima', sans-serif; font-size: 28px;color:white }
                    </style>
                    <p class="custom-text">An Effort by : MAVERICK_GR</p>
                    """, unsafe_allow_html=True)
    
    if selected == "Seasonwise comparison":
        col501, col502 = st.columns([10,10])
        with col501:
            s1 = st.selectbox('Please select the season from the dropdown menu',('Mansoon','Post_mansoon','Winter','Summer'))
        with col502:
            s2 = st.selectbox('Please select any other season from the dropdown menu',('Mansoon','Post_mansoon','Winter','Summer'))
        st.write('')
        st.write('')
        if st.button(f'⏫ Load the datasets ⏫',use_container_width=True):
            df_s1 = pd.read_csv(f'wind_{s1}.csv')
            df_s2 = pd.read_csv(f'wind_{s2}.csv')
            r1, c1 = df_s1.shape
            r2, c2 = df_s2.shape
            st.success(f'{s1} season dataset loaded and has {r1} instances with {c1} features', icon="✅")
            st.success(f'{s2} season dataset loaded and has {r2} instances with {c2} features', icon="✅")
        st.divider()
        if st.button('📉 Variation of wind speed with wind direction 📉',use_container_width=True,key=615): 
            col601, col602 = st.columns([10,10])
            st.write('')
            st.write('')
            with col601:
                df_s1 = pd.read_csv(f'wind_{s1}.csv')
                sea_ws_wd_1(df_s1,s1)
            with col602:
                df_s2 = pd.read_csv(f'wind_{s2}.csv')
                sea_ws_wd_2(df_s2,s2)
        st.write('')
        st.write('')
        if st.button('📉 Variation of wind speed with temperature 📉',use_container_width=True,key=616): 
            col601, col602 = st.columns([10,10])
            st.write('')
            st.write('')
            with col601:
                df_s1 = pd.read_csv(f'wind_{s1}.csv')
                sea_ws_temp_1(df_s1,s1)
            with col602:
                df_s2 = pd.read_csv(f'wind_{s2}.csv')
                sea_ws_temp_2(df_s2,s2)
        st.write('')
        st.write('')
        if st.button('📉 Variation of wind speed with solar irradiation 📉',use_container_width=True,key=617): 
            col601, col602 = st.columns([10,10])
            st.write('')
            st.write('')
            with col601:
                df_s1 = pd.read_csv(f'wind_{s1}.csv')
                sea_ws_ir_1(df_s1,s1)
            with col602:
                df_s2 = pd.read_csv(f'wind_{s2}.csv')
                sea_ws_ir_2(df_s2,s2)
        st.write('')
        st.write('')
        if st.button('📉 Variation of wind speed with relative humidity 📉',use_container_width=True,key=618): 
            col601, col602 = st.columns([10,10])
            st.write('')
            st.write('')
            with col601:
                df_s1 = pd.read_csv(f'wind_{s1}.csv')
                sea_ws_rh_1(df_s1,s1)
            with col602:
                df_s2 = pd.read_csv(f'wind_{s2}.csv')
                sea_ws_rh_2(df_s2,s2)
        st.write('')
        st.write('')
        if st.button('📉 Variation of wind speed with pressure 📉',use_container_width=True,key=619): 
            col601, col602 = st.columns([10,10])
            st.write('')
            st.write('')
            with col601:
                df_s1 = pd.read_csv(f'wind_{s1}.csv')
                sea_ws_pr_1(df_s1,s1)
            with col602:
                df_s2 = pd.read_csv(f'wind_{s2}.csv')
                sea_ws_pr_2(df_s2,s2)
        st.divider()
        col1001, col1002, col1003,col1004, col1005 = st.columns([10,10,10,10,15])
        with col1005:
            st.markdown("""
                    <style>
                    @import url('https://fonts.googleapis.com/css2?family=Agdasima');
                    .custom-text { font-family: 'Agdasima', sans-serif; font-size: 28px;color:red }
                    </style>
                    <p class="custom-text">An Effort by : MAVERICK_GR</p>
                    """, unsafe_allow_html=True)           



            
    if selected == "Monthwise comparison":
            col701, col702 = st.columns([10,10])
            with col701:
                m1 = st.selectbox('Please select the month from the dropdown menu',('January','February','March','April','May','June','July','August','September','November','December'))
            with col702:
                m2 = st.selectbox('Please select the another month from the dropdown menu',('January','February','March','April','May','June','July','August','September','November','December'))
            st.write('')
            st.write('')
            if st.button(f'⏫ Load the datasets ⏫',use_container_width=True):
                df_m1 = pd.read_csv(f'wind_{m1}.csv')
                df_m2 = pd.read_csv(f'wind_{m2}.csv')
                r1, c1 = df_m1.shape
                r2, c2 = df_m2.shape
                st.success(f'{m1} month dataset loaded and has {r1} instances with {c1} features', icon="✅")
                st.success(f'{m2} month dataset loaded and has {r2} instances with {c2} features', icon="✅")
            st.divider()
            if st.button('📉 Variation of wind speed with wind direction 📉',use_container_width=True,key=715): 
                col701, col702 = st.columns([10,10])
                st.write('')
                st.write('')
                with col701:
                    df_m1 = pd.read_csv(f'wind_{m1}.csv')
                    mon_ws_wd_1(df_m1,m1)
                with col702:
                    df_m2 = pd.read_csv(f'wind_{m2}.csv')
                    mon_ws_wd_2(df_m2,m2)
            st.write('')
            st.write('')
            if st.button('📉 Variation of wind speed with temperature 📉',use_container_width=True,key=616): 
                col701, col702 = st.columns([10,10])
                st.write('')
                st.write('')
                with col701:
                    df_m1 = pd.read_csv(f'wind_{m1}.csv')
                    mon_ws_temp_1(df_m1,m1)
                with col702:
                    df_m2 = pd.read_csv(f'wind_{m2}.csv')
                    mon_ws_temp_2(df_m2,m2)
            st.write('')
            st.write('')
            if st.button('📉 Variation of wind speed with solar irradiation 📉',use_container_width=True,key=617): 
                col701, col702 = st.columns([10,10])
                st.write('')
                st.write('')
                with col701:
                    df_m1 = pd.read_csv(f'wind_{m1}.csv')
                    mon_ws_ir_1(df_m1,m1)
                with col702:
                    df_m2 = pd.read_csv(f'wind_{m2}.csv')
                    mon_ws_ir_2(df_m2,m2)
            st.write('')
            st.write('')
            if st.button('📉 Variation of wind speed with relative humidity 📉',use_container_width=True,key=618): 
                col701, col702 = st.columns([10,10])
                st.write('')
                st.write('')
                with col701:
                    df_m1 = pd.read_csv(f'wind_{m1}.csv')
                    mon_ws_rh_1(df_m1,m1)
                with col702:
                    df_m2 = pd.read_csv(f'wind_{m2}.csv')
                    mon_ws_rh_2(df_m2,m2)
            st.write('')
            st.write('')
            if st.button('📉 Variation of wind speed with pressure 📉',use_container_width=True,key=619): 
                col701, col702 = st.columns([10,10])
                st.write('')
                st.write('')
                with col701:
                    df_m1 = pd.read_csv(f'wind_{m1}.csv')
                    mon_ws_pr_1(df_m1,m1)
                with col702:
                    df_m2 = pd.read_csv(f'wind_{m2}.csv')
                    mon_ws_pr_2(df_m2,m2)
            st.divider()
            col1001, col1002, col1003,col1004, col1005 = st.columns([10,10,10,10,15])
            with col1005:
                st.markdown("""
                        <style>
                        @import url('https://fonts.googleapis.com/css2?family=Agdasima');
                        .custom-text { font-family: 'Agdasima', sans-serif; font-size: 28px;color:cyan }
                        </style>
                        <p class="custom-text">An Effort by : MAVERICK_GR</p>
                        """, unsafe_allow_html=True)

    if selected == 'Animated charts':
        st.divider()
        
        if st.button('🎆 Show me how wind speed varies with wind direction in different seasons 🎆',use_container_width=True):
            df1 = pd.read_csv('wind_Mansoon.csv')
            #df1 = df1.drop(columns=['Unnamed: 0'],axis=1)
            
            df2 = pd.read_csv('wind_Post_mansoon.csv')
            #df2 = df2.drop(columns=['Unnamed: 0'],axis=1)
            
            df3 = pd.read_csv('wind_Winter.csv')
            #df3 = df3.drop(columns=['Unnamed: 0'],axis=1)
            
            df4 = pd.read_csv('wind_Summer.csv')
            #df4 = df4.drop(columns=['Unnamed: 0'],axis=1)
            
            df_new = df1.drop(columns=['DT','Year','Month','Day','Hour','Min','Temp','RH','PR','RA','IR','WD2','WS_cat','Temp_cat','RH_cat','PR_cat','IR_cat'],axis=1)
            df_new = df_new.groupby('WD').mean()
            df_new = df_new.reset_index(drop=False)
            df_1_new = df2.drop(columns=['DT','Year','Month','Day','Hour','Min','Temp','RH','PR','RA','IR','WD2','WS_cat','Temp_cat','RH_cat','PR_cat','IR_cat'],axis=1)
            df_1_new = df_1_new.groupby('WD').mean()
            df_1_new = df_1_new.reset_index(drop=False)
            df_2_new = df3.drop(columns=['DT','Year','Month','Day','Hour','Min','Temp','RH','PR','RA','IR','WD2','WS_cat','Temp_cat','RH_cat','PR_cat','IR_cat'],axis=1)
            df_2_new = df_2_new.groupby('WD').mean()
            df_2_new = df_2_new.reset_index(drop=False)
            df_3_new = df4.drop(columns=['DT','Year','Month','Day','Hour','Min','Temp','RH','PR','RA','IR','WD2','WS_cat','Temp_cat','RH_cat','PR_cat','IR_cat'],axis=1)
            df_3_new = df_3_new.groupby('WD').mean()
            df_3_new = df_3_new.reset_index(drop=False)

            
            fig,axes = plt.subplots(figsize=(12,5))
            axes.set_xlim(0,365)
            axes.set_ylim(0,14)
            plt.style.use('ggplot')
            plt.xlabel('Wind direction in degrees')
            plt.ylabel('Wind speed in m/sec')

            
            x, y1, y2, y3,y4 = [], [], [], [], []
            xvalues = list(range(0,361))
            l1 = list(df_new['WS'])
            l2 = list(df_1_new['WS'])
            l3 = list(df_2_new['WS'])
            l4 = list(df_3_new['WS'])
            col101, col102 = st.columns([20,3])
            with col101:
                with st.empty():
                    for i in range(0,361,10):
                        x.extend(xvalues[i:i+10])
                        y1.extend(l1[i:i+10])
                        y2.extend(l2[i:i+10])
                        y3.extend(l3[i:i+10])
                        y4.extend(l4[i:i+10])
                        axes.plot(x,y1,color='red',label='Monsoon')
                        axes.plot(x,y2,color='green',label='Post_Monsoon')
                        axes.plot(x,y3,color='orange',label='Winter')
                        axes.plot(x,y4,color='blue',label='Summer')
                        st.pyplot(fig)
            with col102:
                st.image('legend_1.jpg')

        st.divider()
        if st.button('🎆 Show me how wind speed varies with wind direction in different months 🎆',use_container_width=True):
            df1 = pd.read_csv('wind_January.csv')
            df1 = df1.drop(columns=['Unnamed: 0'],axis=1)
            df2 = pd.read_csv('wind_February.csv')
            df2 = df2.drop(columns=['Unnamed: 0'],axis=1)
            df3 = pd.read_csv('wind_March.csv')
            df3 = df3.drop(columns=['Unnamed: 0'],axis=1)
            df4 = pd.read_csv('wind_April.csv')
            df4 = df4.drop(columns=['Unnamed: 0'],axis=1)
            df5 = pd.read_csv('wind_May.csv')
            df5 = df5.drop(columns=['Unnamed: 0'],axis=1)
            df6 = pd.read_csv('wind_June.csv')
            df6 = df6.drop(columns=['Unnamed: 0'],axis=1)
            df7 = pd.read_csv('wind_July.csv')
            df7 = df7.drop(columns=['Unnamed: 0'],axis=1)
            df8 = pd.read_csv('wind_August.csv')
            df8 = df8.drop(columns=['Unnamed: 0'],axis=1)
            df9 = pd.read_csv('wind_September.csv')
            df9 = df9.drop(columns=['Unnamed: 0'],axis=1)
            df10 = pd.read_csv('wind_October.csv')
            df10 = df10.drop(columns=['Unnamed: 0'],axis=1)
            df11 = pd.read_csv('wind_November.csv')
            df11 = df11.drop(columns=['Unnamed: 0'],axis=1)
            df12 = pd.read_csv('wind_December.csv')
            df12 = df12.drop(columns=['Unnamed: 0'],axis=1)

          
            df_new1 = df1.drop(columns=['DT','Year','Month','Day','Hour','Min','Temp','RH','PR','RA','IR','WD2','WS_cat','Temp_cat','RH_cat','PR_cat','IR_cat'],axis=1)
            df_new1 = df_new1.groupby('WD').mean()
            df_new1 = df_new1.reset_index(drop=False)
            df_new2 = df2.drop(columns=['DT','Year','Month','Day','Hour','Min','Temp','RH','PR','RA','IR','WD2','WS_cat','Temp_cat','RH_cat','PR_cat','IR_cat'],axis=1)
            df_new2 = df_new2.groupby('WD').mean()
            df_new2 = df_new2.reset_index(drop=False)
            df_new3 = df3.drop(columns=['DT','Year','Month','Day','Hour','Min','Temp','RH','PR','RA','IR','WD2','WS_cat','Temp_cat','RH_cat','PR_cat','IR_cat'],axis=1)
            df_new3 = df_new3.groupby('WD').mean()
            df_new3 = df_new3.reset_index(drop=False)
            df_new4 = df4.drop(columns=['DT','Year','Month','Day','Hour','Min','Temp','RH','PR','RA','IR','WD2','WS_cat','Temp_cat','RH_cat','PR_cat','IR_cat'],axis=1)
            df_new4 = df_new4.groupby('WD').mean()
            df_new4 = df_new4.reset_index(drop=False)
            df_new5 = df5.drop(columns=['DT','Year','Month','Day','Hour','Min','Temp','RH','PR','RA','IR','WD2','WS_cat','Temp_cat','RH_cat','PR_cat','IR_cat'],axis=1)
            df_new5 = df_new5.groupby('WD').mean()
            df_new5 = df_new5.reset_index(drop=False)
            df_new6 = df6.drop(columns=['DT','Year','Month','Day','Hour','Min','Temp','RH','PR','RA','IR','WD2','WS_cat','Temp_cat','RH_cat','PR_cat','IR_cat'],axis=1)
            df_new6 = df_new6.groupby('WD').mean()
            df_new6 = df_new6.sample(361,replace=True)
            df_new6 = df_new6.reset_index(drop=False)
            df_new7 = df7.drop(columns=['DT','Year','Month','Day','Hour','Min','Temp','RH','PR','RA','IR','WD2','WS_cat','Temp_cat','RH_cat','PR_cat','IR_cat'],axis=1)
            df_new7 = df_new7.groupby('WD').mean()
            df_new7 = df_new7.sample(361,replace=True)
            df_new7 = df_new7.reset_index(drop=False)
            df_new8 = df8.drop(columns=['DT','Year','Month','Day','Hour','Min','Temp','RH','PR','RA','IR','WD2','WS_cat','Temp_cat','RH_cat','PR_cat','IR_cat'],axis=1)
            df_new8 = df_new8.groupby('WD').mean()
            df_new8 = df_new8.sample(361,replace=True)
            df_new8 = df_new8.reset_index(drop=False)
            df_new9 = df9.drop(columns=['DT','Year','Month','Day','Hour','Min','Temp','RH','PR','RA','IR','WD2','WS_cat','Temp_cat','RH_cat','PR_cat','IR_cat'],axis=1)
            df_new9 = df_new9.groupby('WD').mean()
            df_new9 = df_new9.reset_index(drop=False)
            df_new10 = df10.drop(columns=['DT','Year','Month','Day','Hour','Min','Temp','RH','PR','RA','IR','WD2','WS_cat','Temp_cat','RH_cat','PR_cat','IR_cat'],axis=1)
            df_new10 = df_new10.groupby('WD').mean()
            df_new10 = df_new10.reset_index(drop=False)
            df_new11 = df11.drop(columns=['DT','Year','Month','Day','Hour','Min','Temp','RH','PR','RA','IR','WD2','WS_cat','Temp_cat','RH_cat','PR_cat','IR_cat'],axis=1)
            df_new11 = df_new11.groupby('WD').mean()
            df_new11 = df_new11.reset_index(drop=False)
            df_new12 = df12.drop(columns=['DT','Year','Month','Day','Hour','Min','Temp','RH','PR','RA','IR','WD2','WS_cat','Temp_cat','RH_cat','PR_cat','IR_cat'],axis=1)
            df_new12 = df_new12.groupby('WD').mean()
            df_new12 = df_new12.reset_index(drop=False)
                        
          
            fig,axes = plt.subplots(figsize=(12,5))
            axes.set_xlim(0,365)
            axes.set_ylim(0,18)
            plt.style.use('ggplot')
            plt.xlabel('Wind direction in degrees')
            plt.ylabel('Wind speed in m/sec')

            y1, y2, y3,y4, y5, y6, y7, y8, y9, y10, y11, y12 = [], [], [], [], [],[], [], [], [], [],[], []
            x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12 = [], [], [], [], [],[], [], [], [], [],[], []
            xvalues = list(range(0,361))
            l1 = list(df_new1['WS'])
            l2 = list(df_new2['WS'])
            l3 = list(df_new3['WS'])
            l4 = list(df_new4['WS'])
            l5 = list(df_new5['WS'])
            l6 = list(df_new6['WS'])
            l7 = list(df_new7['WS'])
            l8 = list(df_new8['WS'])
            l9 = list(df_new9['WS'])
            l10 = list(df_new10['WS'])
            l11 = list(df_new11['WS'])
            l12 = list(df_new12['WS'])
            st.write('')
            st.write('')
            col101, col102 = st.columns([20,1])
            with col101:
                with st.empty():
                    fig,axes = plt.subplots(figsize=(12,5))
                    axes.set_xlim(0,365)
                    axes.set_ylim(0,10)
                    plt.style.use('ggplot')
                    plt.xlabel('Wind direction in degrees')
                    plt.ylabel('Wind speed in m/sec')
                    plt.title('Winter season\nRed-->January\nGreen-->February')
                    xvalues = list(range(0,361))
                    l1 = list(df_new1['WS'])
                    l2 = list(df_new2['WS'])
                    for i in range(0,361,10):
                        x1.extend(xvalues[i:i+10])
                        y1.extend(l1[i:i+10])
                        y2.extend(l2[i:i+10])
                        axes.plot(x1,y1,color='red',label='January')
                        axes.plot(x1,y2,color='green',label='February')
                        st.pyplot(fig)
                      


            st.write('')
            st.write('')
            
            with col101:
                with st.empty():
                    plt.cla()
                    fig,axes = plt.subplots(figsize=(12,5))
                    axes.set_xlim(0,365)
                    axes.set_ylim(0,10)
                    plt.style.use('ggplot')
                    plt.xlabel('Wind direction in degrees')
                    plt.ylabel('Wind speed in m/sec')
                    plt.title('Summer season\nOrange-->March\nBlue-->April\nRed-->May')
                    x2values = list(range(0,361))
                    l3 = list(df_new3['WS'])
                    l4 = list(df_new4['WS'])
                    l5 = list(df_new5['WS'])
                    for i in range(0,361,10):
                        x2.extend(x2values[i:i+10])
                        y3.extend(l3[i:i+10])
                        y4.extend(l4[i:i+10])
                        y5.extend(l5[i:i+10])
                        axes.plot(x2,y3,color='orange',label='March')
                        axes.plot(x2,y4,color='blue',label='April')
                        axes.plot(x2,y5,color='red',label='May')
                        st.pyplot(fig)
                        
            
            
            st.write('')
            st.write('')
            

            with col101:
                with st.empty():
                    plt.cla()
                    fig,axes = plt.subplots(figsize=(12,5))
                    axes.set_xlim(0,365)
                    axes.set_ylim(0,15)
                    plt.style.use('ggplot')
                    plt.xlabel('Wind direction in degrees')
                    plt.ylabel('Wind speed in m/sec')
                    plt.title('Manson season\nGreen-->June\nOrange-->July\nBlue-->August\nRed-->September')
                    x3values = list(range(0,361))
                    l6 = list(df_new6['WS'])
                    l7 = list(df_new7['WS'])
                    l8 = list(df_new8['WS'])
                    l9 = list(df_new9['WS'])
                    for i in range(0,361,10):
                        x3.extend(x3values[i:i+10])
                        y6.extend(l6[i:i+10])
                        y7.extend(l7[i:i+10])
                        y8.extend(l8[i:i+10])
                        y9.extend(l9[i:i+10])
                        axes.plot(x3,y6,color='green',label='June')
                        axes.plot(x3,y7,color='orange',label='July')
                        axes.plot(x3,y8,color='blue',label='August')
                        axes.plot(x3,y9,color='red',label='September')
                        st.pyplot(fig)
            with col102:
                st.image('legend_1.jpg')

            st.write('')
            st.write('')
            
            with col101:
                with st.empty():
                    plt.cla()
                    fig,axes = plt.subplots(figsize=(12,5))
                    axes.set_xlim(0,365)
                    axes.set_ylim(0,10)
                    plt.style.use('ggplot')
                    plt.xlabel('Wind direction in degrees')
                    plt.ylabel('Wind speed in m/sec')
                    plt.title('Post Mansoon season\nGreen-->October\nOrange-->November\nBlue-->December')
                    x4values = list(range(0,361))
                    l10 = list(df_new10['WS'])
                    l11 = list(df_new11['WS'])
                    l12 = list(df_new12['WS'])
                    for i in range(0,361,10):
                        x4.extend(x4values[i:i+10])
                        y10.extend(l10[i:i+10])
                        y11.extend(l11[i:i+10])
                        y12.extend(l12[i:i+10])
                        axes.plot(x4,y10,color='green',label='October')
                        axes.plot(x4,y11,color='orange',label='Novemeber')
                        axes.plot(x4,y12,color='blue',label='December')
                        st.pyplot(fig)
            

            centered_text = "<div style='text-align: center; color: cyan; font-size: 30px;'>Similarly we can plot variation of wind speed with temperature, solar irradiation, relative humidity and pressure during different seasons and months </div>"
            st.markdown(centered_text, unsafe_allow_html=True)

                
            


        st.divider()
        col1001, col1002, col1003,col1004, col1005 = st.columns([10,10,10,10,15])
        with col1005:
            st.markdown("""
                        <style>
                        @import url('https://fonts.googleapis.com/css2?family=Agdasima');
                        .custom-text { font-family: 'Agdasima', sans-serif; font-size: 28px;color:Cyan }
                        </style>
                        <p class="custom-text">An Effort by : MAVERICK_GR</p>
                        """, unsafe_allow_html=True)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    













































                
            
