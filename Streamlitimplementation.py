# %%writefile app.py
import streamlit as st
#st.write('# hello world')
import streamlit as st
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from Germany import germany
from common import *

# option = st.selectbox(
# ,
# ('Germany', 'Rhinelandpalatinate'))
# # st.write('You selected:', option)
with st.sidebar:
    st.title('Research Internship: AI & COVID 2022')
    st.caption('The Forschungspraktikum is dealing with explainability and decision support through AI methods in pandemic situations and is part of the KI&Covid project here at the university')
    st.caption('Within this practical we planned and carried out a research project related to identifying and evaluating (through prototypical implementation) various methods and tools for data analytics of data related to COVID')
    st.caption('These techniques have helped us to analyse different data sources and be suitable to build a knowledge base to support decision makers')

    option = st.radio(
        'Covid 19 Insights in a Nutshell',
        ('Germany','Rhinelandpalatinate'))


if option == 'Germany':
  germany()
