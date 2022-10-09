import streamlit as st
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import plotly.figure_factory as ff
from datetime import timedelta, date
import pandas as pd
import requests

from pathlib import Path
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import datetime as dt
from datetime import timedelta
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import plotly.graph_objects as go

from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from functools import reduce

def germany():
  csv = pd.read_csv('date_fix.csv',sep = ',',index_col = False)
  data = csv.iloc[:,3:]
  data['Recoveries']= data['Recoveries'].astype(np.int64)
  data['Date'] = pd.to_datetime(data['Date'])
  byDate = data.groupby(['Date']).agg({'Infected':'sum','Deaths':'sum','Recoveries':'sum'})
  byDate = byDate.set_index(pd.to_datetime(byDate.index))
  
  
  st.title('Covid 19 Insights - Germany')
  label=['Infected','Deaths','Recoveries']
  values = [byDate['Infected'].iloc[-1],byDate['Deaths'].iloc[-1],byDate['Recoveries'].iloc[-1]]
    #The plot
  st.header("The Number of cases as of June",(byDate.index[-1]))
  col1, col2, col3 = st.columns(3)
  col1.metric("Infected", values[0])
  col2.metric("Deaths", values[1])
  col3.metric("Recoveries",values[2]) 

  st.dataframe(byDate) 

  st.header("Cases with respect to the country")
        
  fig = go.Figure(
        go.Pie(
        labels = label,
        values = values,
        hoverinfo = "label+percent",
        textinfo = "value"
    ))
  st.plotly_chart(fig)
    
  st.header('Distriutiom of cases across Germay over the period of time ')
  st.area_chart(byDate['Infected'])
  st.area_chart(byDate['Deaths'])
  st.area_chart(byDate['Recoveries'])
  st.header('Predicting and Forecasting 7 days of cases for Germany')

  data['Date'] = pd.to_datetime(data['Date'], format='%Y-%d-%m')
  data.set_index('Date', inplace=True)
  data.sort_values(by='Date',ascending=False)


  data['Days'] = data.index - data.index[0]
  data['Days'] = data['Days'].dt.days

  feeddata = data.head(len(data)-7) #Considering the data with 7 days lesser, in order to compare the predicted and the original values
  
  tscv = TimeSeriesSplit(n_splits = 10)
  for train_index, test_index in tscv.split(feeddata):
    train_data, test_data = feeddata.iloc[train_index], feeddata.iloc[test_index]

    
  accu = []

  def mape(Y_actual,Y_Predicted):
        mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
        return mape

 #Intializing SVR Model
  def sVM_model(ker,feature):
    model_predictions = []
    MAPE_scores = []
    svm=SVR(C=1,degree=7,kernel=ker,epsilon=0.01,shrinking = True)
    svm.fit(np.array(train_data["Days"]).reshape(-1,1),np.array(train_data[feature]).reshape(-1,1))
    prediction =svm.predict(np.array(test_data["Days"]).reshape(-1,1))

    #model_scores.append(np.sqrt(mean_squared_error(test_data[feature],prediction)))
    rmse = np.sqrt(np.mean((test_data[feature]-prediction)**2))
    MAPE_scores.append(mape(test_data[feature],prediction).astype('int64'))
    # print("mape: ",MAPE_scores) #Lower the MAPE(mean absolute percentage error), better fit is the model.
    # print("Root Mean Square Error for: ",ker,rmse.astype('int64'))

    # plt.figure(figsize=(10,4))
    prediction_svm=svm.predict(np.array(feeddata["Days"]).reshape(-1,1))
    # fig=go.Figure()
    # fig.add_trace(go.Scatter(x=feeddata.index, y=feeddata[feature], mode='lines+markers',name="Train Data for number of Cases"))
    # fig.add_trace(go.Scatter(x=feeddata.index, y=np.rint(prediction_svm),
    #                     mode='lines',name="Support Vector Machine Best fit Kernel",
    #                     line=dict(color='black', dash='dot')))
    # fig.update_layout(title="Cases Support Vectore Machine Regressor Prediction",
    #                  xaxis_title="Date",yaxis_title="RLP Cases",legend=dict(x=0,y=1,traceorder="normal"))
    # fig.show()
    
    st.line_chart(feeddata[feature])
    st.line_chart(np.rint(prediction_svm))
    

        

    new_date=[]
    new_prediction_svm=[]
    
    for i in range(1,15):
        new_date.append(feeddata.index[-1]+timedelta(days=i))
        new_prediction_svm.append((svm.predict(np.array(feeddata["Days"].max()+i).reshape(-1,1))[0]).astype('int64'))

    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    model_predictions = (pd.DataFrame(zip(new_date,new_prediction_svm),
                                columns=["Date","SVM Prediction"]))
    
    return model_predictions,MAPE_scores


  inf_polyPred = sVM_model('poly','Infected')
  inf_rbfPred = sVM_model('rbf','Infected')
  inf_SigPred = sVM_model('sigmoid','Infected')

  inf_polyPred[0].columns = inf_polyPred[0].columns.str.replace('SVM Prediction', 'Poly')
  inf_rbfPred[0].columns = inf_rbfPred[0].columns.str.replace('SVM Prediction', 'RBF')
  # inf_linPred[0].columns = inf_linPred[0].columns.str.replace('SVM Prediction', 'Linear')
  inf_SigPred[0].columns = inf_SigPred[0].columns.str.replace('SVM Prediction', 'Sigmoid')
  ActualData = data.tail(7)

  data_frames = [inf_polyPred[0], inf_rbfPred[0],inf_SigPred[0],ActualData['Infected']]
  df_final = reduce(lambda left,right: pd.merge(left,right,on='Date'), data_frames)

  df_MAPE_infected = pd.DataFrame({'poly': inf_polyPred[1],'RBF': inf_rbfPred[1],'Sigmoid': inf_SigPred[1]}, index = ['MAPE for infected cases'])

  plt.figure(figsize = (15,4))

  plt.plot(df_final.Date, df_final.Infected, color ='orange',
          marker ='o', markersize = 8, 
          label ='Actual Infected cases')
    
  plt.plot(inf_polyPred[0].Date, inf_polyPred[0].Poly, color ='seagreen',
          marker ='o', markersize = 8,
          label ='Poly')

  plt.plot(inf_rbfPred[0].Date, inf_rbfPred[0].RBF, color ='Black',
          marker ='+', markersize = 20,
          label ='RBF')

  # plt.plot(inf_linPred[0].Date, inf_linPred[0].Linear, color ='violet',
  #          marker ='o', markersize = 8,
  #          label ='Linear')

  plt.plot(inf_SigPred[0].Date, inf_SigPred[0].Sigmoid, color ='red',
          marker ='o', markersize = 8,
          label ='Sigmoid')
    
  plt.xlabel('Date')
  plt.ylabel('Number of cases')
  plt.xticks(rotation=90)
    
    
  plt.title('Predictions of infected cases with respect to different kernels')
    
  plt.legend()
  st.pyplot(plt)


  # Deceased cases
  dec_polyPred = sVM_model('poly','Deaths')
  dec_rbfPred = sVM_model('rbf','Deaths')
  # dec_linPred = SVM_model('linear','Deaths')
  dec_SigPred = sVM_model('sigmoid','Deaths')


  dec_polyPred[0].columns = dec_polyPred[0].columns.str.replace('SVM Prediction', 'Poly')
  dec_rbfPred[0].columns = dec_rbfPred[0].columns.str.replace('SVM Prediction', 'RBF')
  # dec_linPred[0].columns = dec_linPred[0].columns.str.replace('SVM Prediction', 'Linear')
  dec_SigPred[0].columns = dec_SigPred[0].columns.str.replace('SVM Prediction', 'Sigmoid')
  ActualData = data.tail(7)

  data_frames = [dec_polyPred[0], dec_rbfPred[0],dec_SigPred[0],ActualData['Deaths']]
  df_final = reduce(lambda left,right: pd.merge(left,right,on='Date'), data_frames)

  df_MAPE_Death = pd.DataFrame({'poly': dec_polyPred[1],'RBF': dec_rbfPred[1],'Sigmoid': dec_SigPred[1]}, index = ['MAPE for Deceased cases'])
  st.write(df_final)

  plt.figure(figsize = (20,3))

  plt.plot(df_final.Date, df_final.Deaths, color ='orange',
          marker ='o', markersize = 8, 
          label ='Actual Deaths cases')
    
  plt.plot(dec_polyPred[0].Date, dec_polyPred[0].Poly, color ='seagreen',
          marker ='o', markersize = 8,
          label ='Poly')

  plt.plot(dec_rbfPred[0].Date, dec_rbfPred[0].RBF,color ='Black',
          marker ='+', markersize = 20,
          label ='RBF')

  # plt.plot(dec_linPred[0].Date, dec_linPred[0].Linear, color ='violet',
  #          marker ='o', markersize = 8,
  #          label ='Linear')

  plt.plot(dec_SigPred[0].Date, dec_SigPred[0].Sigmoid, color ='red',
          marker ='o', markersize = 8,
          label ='Sigmoid')
    
  plt.xlabel('Date')
  plt.ylabel('Number of cases')
  plt.xticks(rotation=90)
    
    
  plt.title('Predictions of Death cases with respect to different kernels')
    
  plt.legend()
  st.pyplot(plt)


  rec_polyPred = sVM_model('poly','Recoveries')
  rec_rbfPred = sVM_model('rbf','Recoveries')
  # rec_linPred = SVM_model('linear','Recoveries')
  # SVM_model('sigmoid','RLP_Deceased')
  rec_SigPred = sVM_model('sigmoid','Recoveries')

  rec_polyPred[0].columns = rec_polyPred[0].columns.str.replace('SVM Prediction', 'Poly')
  rec_rbfPred[0].columns = rec_rbfPred[0].columns.str.replace('SVM Prediction', 'RBF')
  # rec_linPred[0].columns = rec_linPred[0].columns.str.replace('SVM Prediction', 'Linear')
  rec_SigPred[0].columns = rec_SigPred[0].columns.str.replace('SVM Prediction', 'Sigmoid')
  ActualData = data.tail(7)

  data_frames = [rec_polyPred[0], rec_rbfPred[0],rec_SigPred[0],ActualData['Recoveries']]
  df_final = reduce(lambda left,right: pd.merge(left,right,on='Date'), data_frames)
  df_MAPE_recovered = pd.DataFrame({'poly': rec_polyPred[1],'RBF': rec_rbfPred[1],'Sigmoid': rec_SigPred[1]}, index = ['MAPE for Recovered cases'])
  st.write(df_final)

  plt.figure(figsize = (10,5))

  plt.plot(df_final.Date, df_final.Recoveries, color ='orange',
          marker ='o', markersize = 8, 
          label ='Actual Recovered cases')
    
  plt.plot(rec_polyPred[0].Date, rec_polyPred[0].Poly, color ='seagreen',
          marker ='o', markersize = 8,
          label ='Poly')

  plt.plot(rec_rbfPred[0].Date, rec_rbfPred[0].RBF, color ='Black',
          marker ='+', markersize = 20,
          label ='RBF')

  # plt.plot(rec_linPred[0].Date, rec_linPred[0].Linear, color ='violet',
  #          marker ='o', markersize = 8,
  #          label ='Linear')

  plt.plot(rec_SigPred[0].Date, rec_SigPred[0].Sigmoid, color ='red',
          marker ='o', markersize = 8,
          label ='Sigmoid')
    
  plt.xlabel('Date')
  plt.ylabel('Number of cases')
  plt.xticks(rotation=90)
    
    
  plt.title('Predictions of Recovered cases with respect to different kernels')
    
  plt.legend()
  st.pyplot(plt)

  df_MAPE_infected = df_MAPE_infected.append(df_MAPE_Death)
  df_MAPE_infected = df_MAPE_infected.append(df_MAPE_recovered)

  st.write(df_MAPE_infected)

 