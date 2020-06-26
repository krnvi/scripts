#!/usr/bin/python

import sys ;import os ; import numpy as np ; import pandas as pd ; import datetime as dt ; from dateutil import tz, rrule ; import pytz
import matplotlib.pyplot as plt; from pylab import savefig 

import tensorflow as tf  
from sklearn.metrics import explained_variance_score, mean_absolute_error,  median_absolute_error ,mean_squared_error
from sklearn.model_selection import train_test_split  


main='/home/vkvalappil/Data/oppModel' ; output=main+'/output/output/stat/' ; inp=main+'/output/output/'
date='2017123106'

date_1=dt.datetime.strptime(date,'%Y%m%d%H')+dt.timedelta(days=0)
date_2=dt.datetime.strptime(date,'%Y%m%d%H')+dt.timedelta(days=179)
date_list=[x.strftime('%Y%m%d%H') for x in rrule.rrule(rrule.DAILY,dtstart=date_1,until=date_2)]

bias_day1 = [] ; rmse_day1 = [] ; bias_day2 = [] ; rmse_day2 = [] ; bias_hour_day1=[] ; bias_hour_day2=[]
mod_hour_day1=[] ; obs_hour_day1=[] ; mod_hour_day2=[] ; obs_hour_day2=[] ;

for dte in date_list[:]:
    
    file_2=inp+'domain_2/surfaceLevel/mr_2016_2017_2018_ad/hourly'+dte+'.csv'
    if (os.path.isfile(file_2)):

        mod_dom_2=pd.read_csv(file_2) ; #mod_dom_2=mod_dom_2.iloc[72:144,:] ; 

        o_date_1=dte ; 
        o_date_2=(dt.datetime.strptime(dte,'%Y%m%d%H')+dt.timedelta(days=1)).strftime('%Y%m%d%H')
        o_date_3=(dt.datetime.strptime(dte,'%Y%m%d%H')+dt.timedelta(days=2)).strftime('%Y%m%d%H')
        o_date_4=(dt.datetime.strptime(dte,'%Y%m%d%H')+dt.timedelta(days=3)).strftime('%Y%m%d%H')
    
        obs_file_1='/home/vkvalappil/Data/masdar_station_data/wyoming/'+o_date_1[0:6]+'/AbuDhabi_surf_mr'+o_date_1[0:8]+'.csv'
        obs_file_2='/home/vkvalappil/Data/masdar_station_data/wyoming/'+o_date_2[0:6]+'/AbuDhabi_surf_mr'+o_date_2[0:8]+'.csv'
        obs_file_3='/home/vkvalappil/Data/masdar_station_data/wyoming/'+o_date_3[0:6]+'/AbuDhabi_surf_mr'+o_date_3[0:8]+'.csv'
        obs_file_4='/home/vkvalappil/Data/masdar_station_data/wyoming/'+o_date_4[0:6]+'/AbuDhabi_surf_mr'+o_date_4[0:8]+'.csv'
        
        mod_dom_2['localTime']=mod_dom_2['localTime'].apply(pd.to_datetime, errors='ignore')    
        mod_dom_2.iloc[:,4:]=mod_dom_2.iloc[:,4:].apply(pd.to_numeric,errors='coerce')
        mod_dom_2_1=mod_dom_2.iloc[:,3:]
        mod_dom_2_1.index=mod_dom_2_1.localTime
        mod_dom_2_1.index=mod_dom_2_1.index.tz_localize(pytz.timezone('Asia/Dubai')).tz_convert(pytz.utc)
        mod_dom_2_1['localTime']=mod_dom_2_1.index
        
        obs_1=pd.read_csv(obs_file_1) ; obs_2=pd.read_csv(obs_file_2) ; obs_3=pd.read_csv(obs_file_3) ; obs_4=pd.read_csv(obs_file_4)
        obs=pd.concat([obs_1,obs_2,obs_3,obs_4],axis=0)
    
        obs['TIME']=obs['TIME'].apply(pd.to_datetime,errors='ignore')
        obs.iloc[:,3:]=obs.iloc[:,3:].apply(pd.to_numeric,errors='coerce')
        obs_1=obs.iloc[:,2:]
        obs_1.index=obs_1.TIME
        obs_1.index=obs_1.index.tz_localize(pytz.utc)
  
        idx = obs_1.index.intersection(mod_dom_2_1.index)
        obs_2=obs_1.loc[idx]
        obs_3=pd.concat([obs_2['TIME'],obs_2['TMP'],obs_2['DEW'],obs_2['RH'],obs_2['mrio'],obs_2['SPD']],axis=1)

        mod_dom_2_2=pd.concat([mod_dom_2_1['localTime'],mod_dom_2_1['TEMP'],mod_dom_2_1['DTEMP'],mod_dom_2_1['RH'],mod_dom_2_1['MXRATIO'],mod_dom_2_1['WDIR']*0.277],axis=1)    

        mod_dom_2_2.columns=obs_3.columns
        
################################### Calculating Daily bias and daily rmse ############################################################################
        mod_dom_2_bias_1=mod_dom_2_2.iloc[7:31,1:].sub(obs_3.iloc[7:31,1:],axis=0)
        
        mod_dom_2_rmse_1=((mod_dom_2_bias_1**2).mean(axis=0))**0.5


        mod_dom_2_bias_2=mod_dom_2_2.iloc[31:55,1:].sub(obs_3.iloc[31:55,1:],axis=0)      

        mod_dom_2_rmse_2=((mod_dom_2_bias_2**2).mean(axis=0))**0.5

##################################################################################################        

        bias_day_1=np.vstack(mod_dom_2_bias_1.mean(axis=0).values).T
        bias_day_1=pd.DataFrame(bias_day_1,columns=['tmp_02','dpt_02','rh_02','mr_02','sknt_02'])
        bias_day_1.insert(0,'Date',dte)

        rmse_day_1=np.vstack(mod_dom_2_rmse_1.values).T
        rmse_day_1=pd.DataFrame(rmse_day_1,columns=['tmp_02','dpt_02','rh_02','mr_02','sknt_02'])
        rmse_day_1.insert(0,'Date',dte)  

        bias_day1.append(bias_day_1) ;     rmse_day1.append(rmse_day_1) ####### daily bias day wise appended day1
       
########################################                
        bias_hour_day1.append(mod_dom_2_bias_1)  ####### hourly bias appended for each day day1
        mod_hour_day1.append(mod_dom_2_2.iloc[7:31,1:]) ;  obs_hour_day1.append(obs_3.iloc[7:31,1:])
####################################     
        bias_day_2=np.vstack(mod_dom_2_bias_2.mean(axis=0).values).T

        bias_day_2=pd.DataFrame(bias_day_2,columns=['tmp_02','dpt_02','rh_02','mr_02','sknt_02'])

        bias_day_2.insert(0,'Date',dte)

        rmse_day_2=np.vstack(mod_dom_2_rmse_2.values).T 
        
        rmse_day_2=pd.DataFrame(rmse_day_2,columns=['tmp_02','dpt_02','rh_02','mr_02','sknt_02'])

        rmse_day_2.insert(0,'Date',dte)

        bias_day2.append(bias_day_2) ; rmse_day2.append(rmse_day_2) ## daily bias day wise appended day2
###################################
        bias_hour_day2.append(mod_dom_2_bias_2)  ## hourly bias appended for each day day2
        mod_hour_day2.append(mod_dom_2_2.iloc[31:55,1:]) ;  obs_hour_day2.append(obs_3.iloc[31:55,1:])
##################################
    else:
       print (dte) 
       print("No Data Exist")

#############################################################################################################################

def plot_bars(C,parm,parm1,metric,ds ):
    
    fig=plt.figure(figsize=(20,12),dpi=50,frameon=True); ax = fig.add_subplot(111,facecolor='white')

    pos_1  = np.arange(0,14*C.shape[0],14) ; wdh=3.5 ; opacity = 1.0 ; #ax.hold(True);

    ax.bar(pos_1+wdh,C['mod'],width=wdh, color='b',alpha=opacity,label='Dom 2 model')
    ax.bar(pos_1+2*wdh ,C['mod_cor'],width=wdh, color='r',alpha=opacity,label='Dom 2 model cor')
    ax.bar(pos_1+3*wdh ,C['obs'],width=wdh, color='g',alpha=opacity,label='Dom 2 model bias added')
    ax.bar(pos_1+4*wdh ,C['mod_cor_ba'],width=wdh, color='k',alpha=opacity,label='obs')


    ax.axhline(0, color="k")

    plt.tick_params(axis='y',which='both', left='on',  right='off', labelright='off')
    ax.tick_params(axis='y', colors='black',labelsize=16) ; ax.yaxis.set_ticks_position('left') ;

    #ax.set_yticks(np.arange(-10,45,10)) ; 
    ax.set_ylabel(metric,color='black',fontsize=16,fontweight='bold') ;
    ax.set_xlabel('TIME: UTC ',color='black',fontsize=16,fontweight='bold') ;

    ax.set_xticks(np.arange(0,14*C.shape[0],14)+2*wdh) ; 
    xTickMarks=C.index #pd.TimedeltaIndex.to_series(C.index).dt.components.hours 
    xtickNames =ax.set_xticklabels(xTickMarks,rotation=90,fontsize=14,fontweight='bold')

    #plt.tick_params(axis='x',which='both', bottom='off',  top='off', labelbottom='off')
#    for n, row in enumerate(C.iloc[:]):
#        plt.text(4*n, row, (row,3), ha='center', rotation=0, va='bottom',fontsize=16,fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    display = (0,1,2,3,4)

    leg=ax.legend(loc='lower right',fontsize=18)

    plt.tight_layout(pad=3) ;

    plt.title( parm1+' : 2M '+parm,fontsize=16)
    #outFile=output+'stat/corrected/dom2_'+parm1+'_'+metric+'_'+ds+'.png' 
    outFile=output+'/corrected/dom2_'+parm+'_'+parm1+'_'+metric+'_'+ds+'.png' 
    savefig(outFile);
    plt.close(fig) ; fig.clf(fig)

##############################################################################################################################
def plot_bars_bias(C,parm,parm1,metric,ds ):
    
    fig=plt.figure(figsize=(20,12),dpi=50,frameon=True); ax = fig.add_subplot(111,facecolor='white')

    pos_1  = np.arange(0,10*C.shape[0],10) ;   wdh=2.5 ; opacity = 1.0 ; #ax.hold(True);

    ax.bar(pos_1+wdh,C['bias_mod'],width=wdh, color='b',alpha=opacity,label='Dom 2 bias')
    ax.bar(pos_1+2*wdh ,C['bias_cor'],width=wdh, color='r',alpha=opacity,label='Dom 2 bias cor')
    ax.bar(pos_1+3*wdh ,C['bias_ba'],width=wdh, color='g',alpha=opacity,label='Dom 2 bias cor ba')


    ax.axhline(0, color="k")

    plt.tick_params(axis='y',which='both', left='on',  right='off', labelright='off')
    ax.tick_params(axis='y', colors='black',labelsize=16) ; ax.yaxis.set_ticks_position('left') ;

    #ax.set_yticks(np.arange(-10,45,10)) ; 
    ax.set_ylabel(metric,color='black',fontsize=16,fontweight='bold') ;
    ax.set_xlabel('TIME: UTC ',color='black',fontsize=16,fontweight='bold') ;

    ax.set_xticks(np.arange(0,10*C.shape[0],10)+2*wdh) ; 
    xTickMarks=C.index #pd.TimedeltaIndex.to_series(C.index).dt.components.hours 
    xtickNames =ax.set_xticklabels(xTickMarks,rotation=90,fontsize=14,fontweight='bold')

    #plt.tick_params(axis='x',which='both', bottom='off',  top='off', labelbottom='off')
#    for n, row in enumerate(C.iloc[:]):
#        plt.text(4*n, row, (row,3), ha='center', rotation=0, va='bottom',fontsize=16,fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    display = (0,1,2,3,4)

    leg=ax.legend(loc='lower right',fontsize=18)

    plt.tight_layout(pad=3) ;

    plt.title( parm1+' : 2M '+parm,fontsize=16)
    #outFile=output+'stat/corrected/dom2_'+parm1+'_'+metric+'_'+ds+'.png' 
    outFile=output+'/corrected/dom2_'+parm+'_'+parm1+'_'+metric+'_'+ds+'.png' 
    savefig(outFile);
    plt.close(fig) ; fig.clf(fig)
 ################################################################################################################
def plot_bars_rmse(C,parm,parm1,metric,ds ):
    
    fig=plt.figure(figsize=(20,12),dpi=50,frameon=True); ax = fig.add_subplot(111,facecolor='white')

    pos_1  = np.arange(0,10*C.shape[0],10) ; wdh=2.5 ; opacity = 1.0 ; #ax.hold(True);

    ax.bar(pos_1+wdh,C['mod'],width=wdh, color='b',alpha=opacity,label='Dom 2 rmse')
    ax.bar(pos_1+2*wdh ,C['rmse_cor'],width=wdh, color='r',alpha=opacity,label='Dom 2 rmse cor')
    #ax.bar(pos_1+3*wdh ,C['rmse_cor_ba'],width=wdh, color='g',alpha=opacity,label='Dom 2 rmse cor ba')


    #ax.axhline(1, color="k")

    plt.tick_params(axis='y',which='both', left='on',  right='off', labelright='off')
    ax.tick_params(axis='y', colors='black',labelsize=16) ; ax.yaxis.set_ticks_position('left') ;

    #ax.set_yticks(np.arange(-10,45,10)) ; 
    ax.set_ylabel(metric,color='black',fontsize=16,fontweight='bold') ;
    ax.set_xlabel('TIME: UTC ',color='black',fontsize=16,fontweight='bold') ;

    ax.set_xticks(np.arange(0,10*C.shape[0],10)+2*wdh) ; 
    xTickMarks=C.index.strftime('%Y%m%d') #pd.TimedeltaIndex.to_series(C.index).dt.components.hours 
    xtickNames =ax.set_xticklabels(xTickMarks,rotation=90,fontsize=14,fontweight='bold')

    #plt.tick_params(axis='x',which='both', bottom='off',  top='off', labelbottom='off')
#    for n, row in enumerate(C.iloc[:]):
#        plt.text(4*n, row, (row,3), ha='center', rotation=0, va='bottom',fontsize=16,fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    display = (0,1,2,3,4)

    leg=ax.legend(loc='lower right',fontsize=18)

    plt.tight_layout(pad=3) ;

    plt.title( parm1+' : 2M '+parm,fontsize=16)
    outFile=output+'/corrected/dom2_'+parm+'_'+parm1+'_'+metric+'_'+ds+'.png' 
    savefig(outFile);
    plt.close(fig) ; fig.clf(fig)    
############################################################################################################################    
def plot_bars_rmse_hw(C,parm,parm1,metric,ds ):

    fig=plt.figure(figsize=(20,12),dpi=50,frameon=True); ax = fig.add_subplot(111,facecolor='white')

    pos_1  = np.arange(0,10*C.shape[0],10) ; wdh=2.5 ; opacity = 1.0 ; #ax.hold(True);

    ax.bar(pos_1+wdh,C['uc_rmse'],width=wdh, color='b',alpha=opacity,label='Dom 2 rmse')
    ax.bar(pos_1+2*wdh ,C['cor_rmse'],width=wdh, color='r',alpha=opacity,label='Dom 2 rmse cor')
    #ax.bar(pos_1+3*wdh ,C['rmse_cor_ba'],width=wdh, color='g',alpha=opacity,label='Dom 2 rmse cor ba')


    #ax.axhline(1, color="k")

    plt.tick_params(axis='y',which='both', left='on',  right='off', labelright='off')
    ax.tick_params(axis='y', colors='black',labelsize=16) ; ax.yaxis.set_ticks_position('left') ;

    #ax.set_yticks(np.arange(-10,45,10)) ; 
    ax.set_ylabel(metric,color='black',fontsize=16,fontweight='bold') ;
    ax.set_xlabel('TIME: UTC ',color='black',fontsize=16,fontweight='bold') ;

    ax.set_xticks(np.arange(0,10*C.shape[0],10)+2*wdh) ;
    xTickMarks=C.index.strftime('%H:%S') #pd.TimedeltaIndex.to_series(C.index).dt.components.hours 
    xtickNames =ax.set_xticklabels(xTickMarks,rotation=90,fontsize=14,fontweight='bold')

    #plt.tick_params(axis='x',which='both', bottom='off',  top='off', labelbottom='off')
#    for n, row in enumerate(C.iloc[:]):
#        plt.text(4*n, row, (row,3), ha='center', rotation=0, va='bottom',fontsize=16,fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    display = (0,1,2,3,4)

    leg=ax.legend(loc='lower right',fontsize=18)

    plt.tight_layout(pad=3) ;

    plt.title( parm1+' : 2M '+parm,fontsize=16)
    outFile=output+'/corrected/dom2_'+parm+'_'+parm1+'_'+metric+'_'+ds+'.png'
    savefig(outFile);
    plt.close(fig) ; fig.clf(fig)
##################################################################################################################################


def plot_line(C,parm,parm1,ds ):
    
    fig=plt.figure(figsize=(20,12),dpi=50,frameon=True); ax1 = fig.add_subplot(111,facecolor='white')

    pos=np.arange(0,C.shape[0],1) ; opacity = 1.0 ; #ax.hold(True);

    ax1.plot(pos,C['mod'],color='b',alpha=opacity,lw=4,linestyle='-',label='Dom 2 model')
    ax1.plot(pos,C['mod_cor'],color='r',alpha=opacity,lw=4,linestyle='-',label='Dom 2 model cor')
    ax1.plot(pos,C['obs'],color='k',alpha=opacity,lw=4,linestyle='-',label='obs')

    plt.tick_params(axis='y',which='both', left='on',  right='off', labelright='off')
    ax1.tick_params(axis='y', colors='black',labelsize=16) ; ax1.yaxis.set_ticks_position('left') ;

    #ax.set_yticks(np.arange(-10,45,10)) ; 
    ax1.set_ylabel(parm,color='black',fontsize=16,fontweight='bold') ;
    ax1.set_xlabel('TIME: UTC ',color='black',fontsize=16,fontweight='bold') ;

    ax1.set_xticks(np.arange(0,C.shape[0],1)) ; 
    xTickMarks=C.index #pd.TimedeltaIndex.to_series(C.index).dt.components.hours 
    xtickNames =ax1.set_xticklabels(xTickMarks,rotation=90,fontsize=14,fontweight='bold')

    #plt.tick_params(axis='x',which='both', bottom='off',  top='off', labelbottom='off')
#    for n, row in enumerate(C.iloc[:]):
#        plt.text(4*n, row, (row,3), ha='center', rotation=0, va='bottom',fontsize=16,fontweight='bold')

    handles, labels = ax1.get_legend_handles_labels()
    display = (0,1,2,3,4)

    leg=ax1.legend(loc='lower right',fontsize=18)

    plt.tight_layout(pad=3) ;

    plt.title( parm1+' : 2M '+parm,fontsize=16)
    #outFile=output+'stat/corrected/dom2_'+parm1+'_'+metric+'_'+ds+'.png' 
    outFile=output+'/corrected/dom2_'+parm1+'_'+ds+'.png' 
    savefig(outFile);
    plt.close(fig) ; fig.clf(fig)    
################################### hourly Analysis ######################################################################### 
bias_hour_day_1=pd.concat(bias_hour_day1,axis=0) ; bias_hour_day_2=pd.concat(bias_hour_day2,axis=0)
mod_hour_day_1=pd.concat(mod_hour_day1,axis=0)   ; mod_hour_day_2=pd.concat(mod_hour_day2,axis=0)   ;
obs_hour_day_1=pd.concat(obs_hour_day1,axis=0)   ; mod_hour_day_2=pd.concat(mod_hour_day2,axis=0)   ;
##############################################################################################################################
#bias_hour_day_1.insert(0,'Date',bias_hour_day_1.index)

d_t_1='2017-12-31' ; d_t_2='2018-06-30' ; 

A_bias_hour_day1_1=bias_hour_day_1[d_t_1:d_t_2] ;  B_bias_hour_day2_1=bias_hour_day_2[d_t_1:d_t_2] ;   

hourt = pd.to_timedelta(A_bias_hour_day1_1.index.hour,  unit='H')
bias_hour_avg_day1=A_bias_hour_day1_1.groupby(hourt).mean()           ###### bias for that particular hour , 01 means 01 UTC foracast , not forecast + 01 UTC

rmse_hour_avg_day1=(((A_bias_hour_day1_1**2).groupby(hourt)).mean())**0.5  
rmse_hour_avg_day1=pd.concat([rmse_hour_avg_day1.iloc[13:],rmse_hour_avg_day1[0:13]],axis=0)
mf=int(mod_hour_day_1.shape[0]/24.0)
rmse_hour_avg_day1=pd.concat([rmse_hour_avg_day1]*mf)

################### 
hourt = pd.to_timedelta(B_bias_hour_day2_1.index.hour,  unit='H')
bias_hour_avg_day2=B_bias_hour_day2_1.groupby(hourt).mean()           ###### bias for that particular hour , 01 means 01 UTC foracast , not forecast + 01 UTC
rmse_hour_avg_day2=(((B_bias_hour_day2_1**2).groupby(hourt)).mean())**0.5  

################################################### add avg hourly bias with forecast ############################################
####RH
rmse_hour_avg_day1.index=mod_hour_day_1.index
rh_input=pd.concat([mod_hour_day_1['RH'],obs_hour_day_1['RH'],bias_hour_day_1['RH'],rmse_hour_avg_day1['RH']],axis=1)
rh_input.columns=['mod','obs','bias','rmse']
rh_input['hour']=mod_hour_day_1.index.hour
rh_input.insert(0,'Date',(mod_hour_day_1.index.strftime('%m')).astype(int)) 

rh_input=rh_input[d_t_1:d_t_2]
rh_input=rh_input.dropna(how='any')
######################################################################################################

date_index=rh_input.index[::24] ; rmse_cor=[] ; rh_data_new_hour=[]
date_index_req=date_index[1:]
for ii in range(0,len(date_index_req)):

    #dte1=(dt.datetime.strptime(dte,'%Y%m%d%H')+dt.timedelta(days=0)).strftime('%Y-%m-%d %H')
    #prev_dt=(dt.datetime.strptime(dte,'%Y-%m-%d %H')+dt.timedelta(days=-1)).strftime('%Y-%m-%d %H')
    
    dte=date_index_req[ii] ; #dte1=date_index[ii+2]

    dte1=(dt.datetime.strptime(dte.strftime('%Y%m%d %H:%M:%S'),'%Y%m%d %H:%M:%S')+dt.timedelta(days=-1)).strftime('%Y-%m-%d %H:%M:%S')
    dte2=(dt.datetime.strptime(dte.strftime('%Y%m%d %H:%M:%S'),'%Y%m%d %H:%M:%S')+dt.timedelta(days=+1)).strftime('%Y-%m-%d %H:%M:%S')
    
    dt_rh_input_1=rh_input[dte1:dte2] ;  
   
    if dt_rh_input_1.shape >= (48,6) :
        dt_rh_input=dt_rh_input_1 ;
    else:
        dte_t=(dt.datetime.strptime(dte.strftime('%Y%m%d %H:%M:%S'),'%Y%m%d %H:%M:%S')+dt.timedelta(days=-2)).strftime('%Y-%m-%d %H:%M:%S')
        dt_rh_input=rh_input[dte_t:dte2]


    rh_input_1=pd.concat([(dt_rh_input.iloc[0:24]).reset_index(drop=True),dt_rh_input['mod'].iloc[24:48].reset_index(drop=True)],axis=1)
    rh_target=dt_rh_input['obs'].iloc[24:48].reset_index(drop=True) 

    rh_input_1.columns=['Date','mod.p','obs','bias','rmse','hour','mod'] 


    rh_input_1_m=pd.concat([rh_input_1['Date'],rh_input_1['mod.p'],rh_input_1['obs'],rh_input_1['bias'], \
                        rh_input_1['rmse'],rh_input_1['hour'],rh_input_1['mod']],axis=1)
                        
#    cols=pd.Series(tmp_input_1.columns)
#    for dup in tmp_input_1.columns.get_duplicates(): 
#        cols[tmp_input_1.columns.get_loc(dup)]=[dup+'.'+str(d_idx) if d_idx!=0 else dup for d_idx in range(tmp_input_1.columns.get_loc(dup).sum())]
#    tmp_input_1.columns=cols
  

    ##################### new Forecast Bias added ######################################################################
    #hour_avg_bias_1=(pd.concat([hour_avg_bias['rhio'][13:],hour_avg_bias['rhio'][1:13]],axis=0)).reset_index(drop=True)
    #b_add_tmp=rh_input_1['mod']- hour_avg_bias_1
     
##############################################################################################################
    X=rh_input_1_m.dropna(how='any').reset_index(drop=True)  ; Y=rh_target.dropna(how='any').reset_index(drop=True)  ; 

############## Normalise input and target ##############################################################################################
#    min_X = np.min(X,axis=0) ; max_X = np.max(X,axis=0) ; 
#    norm_X = (X - min_X) / (max_X - min_X)
#
#    min_mnth=1.0 ; max_mnth=12.0
#    norm_mnth=((X['Date']- min_mnth) /(max_mnth - min_mnth ))/100
#    norm_X['Date']=norm_mnth
#    X['Date']=norm_mnth
#
#    min_hour=0.0 ; max_hour=23.0
#    norm_hour=((X['hour']- min_hour) /(max_hour - min_hour ))/100
#    #norm_X['hour']=norm_hour
#    X['hour']=norm_hour
#
#    min_Y = np.min(Y,axis=0) ; max_Y = np.max(Y,axis=0) ; 
#    norm_Y = (Y - min_Y) / (max_Y - min_Y)

##########################################################################################################
    def wx_input_fn(X,y=None, num_epochs=None, shuffle=True, batch_size=400):  
        
        return tf.estimator.inputs.pandas_input_fn(x=X, shuffle=shuffle)
        
#    def serving_input_receiver_fn():
#        inputs = {"x": tf.placeholder(shape=[None, 6], dtype=tf.float32)}
#        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]  
  
############### loading and predict DNN regressor  
#    reg_dnn = tf.estimator.DNNRegressor(feature_columns=feature_cols, hidden_units=[10,10,1],model_dir='dnn_bias-cor')
#    predict=reg_dnn.predict(input_fn=wx_input_fn(norm_X, num_epochs=1,  shuffle=False ))  
#    y_new=[p['predictions'] for p in predict]
#    y_new =np.array(y_new) * (max_Y - min_Y) + min_Y

############### loading and predict Linear DNN combined regressor
    epochs=2000 ;
    h_size1=30 ; h_size2=30 ; itrn=100

    m_dir='rh_linear_dnn_bias-cor_full_'+str(itrn)+'.bsz_ep.'+str(epochs)+'_hu.'+str(h_size1)+'.'+str(h_size1)+'.'+str(h_size2)+'.1_Mn_2016_2017'

    reg_linear_dnn = tf.estimator.DNNLinearCombinedRegressor(linear_feature_columns=feature_cols,dnn_feature_columns=feature_cols,\
                     dnn_hidden_units=[h_size1,h_size1,h_size2,1],model_dir=m_dir) #linear_dnn_bias-cor
                     
    predict_l_dnn=reg_linear_dnn.predict(input_fn=wx_input_fn(X, num_epochs=1,  shuffle=False ))  
    y_new_l_dnn=[p['predictions'] for p in predict_l_dnn]
    y_new_l_dnn =np.array(y_new_l_dnn) #* (max_Y - min_Y) + min_Y
    y_uc=X['mod'] 
##############################Loading and Predict Linear regressor #############################################################
#    reg_linear=tf.estimator.LinearRegressor(feature_columns=feature_cols, model_dir='linear_bias-cor')
#
#    predict_l=reg_linear.predict(input_fn=wx_input_fn(norm_X, num_epochs=1,  shuffle=False ))  
#    y_new_l=[p['predictions'] for p in predict_l]
#    y_new_l=np.array(y_new_l) * (max_Y - min_Y) + min_Y
#    y_new_l_dnn=y_new_l
#    print dte
#    print("The Mean Absolute Error: %.2f degrees Celcius" % mean_absolute_error(Y,(y_new_l_dnn)))  
#    print("The Mean square Error: %.2f degrees Celcius" % np.sqrt(mean_squared_error(Y,(y_new_l_dnn))) )

#    print("The Mean Absolute Error: %.2f degrees Celcius" % mean_absolute_error(Y,tmp_input_1['mod']))  
#    print("The Mean square Error: %.2f degrees Celcius" % np.sqrt(mean_squared_error(Y,tmp_input_1['mod']))) 
#    print("#################################### ##############################################")
   
################################################################################################################################# 
    bias_new=pd.DataFrame((np.vstack(y_new_l_dnn))-np.vstack(Y.as_matrix()))
    rmse_new=(np.sqrt(mean_squared_error(Y,(y_new_l_dnn))))
    rmse_mod=(np.sqrt(mean_squared_error(Y,y_uc)))
 
    #bias_ba=pd.DataFrame((np.vstack(b_add_tmp))-np.vstack(Y.as_matrix()))
    #rmse_ba=np.sqrt(np.nanmean((Y-(b_add_tmp))**2))
      
    rmse_cor.append([rmse_mod,rmse_new])

    rh_data_new=pd.concat([rh_input_1['mod'],pd.DataFrame((y_new_l_dnn)),Y,dt_rh_input['bias'].iloc[24:48].reset_index(drop=True),\
                 bias_new],axis=1)
    rh_data_new.columns=['mod','mod_cor','obs','bias_mod','bias_cor']
    rh_data_new.index=dt_rh_input.index[24:48]
    rh_data_new.insert(0,'Date',rh_data_new.index)
    rh_data_new_hour.append(rh_data_new)
    #plot_bars(tmp_data_new,'tmperature','tmp_mod','day_1',tmp_data_new.index.strftime('%Y%m%d')[0] )
    #plot_line(tmp_data_new,'Temperature','Temperature',tmp_data_new.index.strftime('%Y%m%d')[0])
    #plot_bars_bias(tmp_data_new,'tmperature','bias','day_1',tmp_data_new.index.strftime('%Y%m%d')[0])
#####################################################################################################################################
rh_data_new_hour=pd.concat(rh_data_new_hour,axis=0)     


rmse_cor=pd.DataFrame(rmse_cor,columns=['mod','rmse_cor']) 
rmse_cor.index=date_index_req ; rmse_cor.insert(0,'Date',rmse_cor.index)


#plot_bars_rmse(rmse_cor,'tmperature','day_1','rmse',rmse_cor.index.strftime('%Y%m')[0])
rmse_cor_j=rmse_cor['2018-01-01':'2018-01-31']
rmse_cor_f=rmse_cor['2018-02-01':'2018-02-28']
rmse_cor_m=rmse_cor['2018-03-01':'2018-03-31']
rmse_cor_apr=rmse_cor['2018-04-01':'2018-04-30']
rmse_cor_may=rmse_cor['2018-05-01':'2018-05-31']
rmse_cor_jun=rmse_cor['2018-06-01':'2018-06-30']


cnt=(rmse_cor_j['rmse_cor'] <rmse_cor_j['mod']).sum()
#print(rmse_cor_j)
print ('Jan : ' + str(cnt) +':::'+ str(cnt/31.0*100))


cnt=(rmse_cor_f['rmse_cor'] <rmse_cor_f['mod']).sum()
#print(rmse_cor_f)
print ('Feb : ' + str(cnt) +':::'+ str(cnt/28.0*100))


cnt=(rmse_cor_m['rmse_cor'] <rmse_cor_m['mod']).sum()
#print(rmse_cor_m)
print ('Mar : ' + str(cnt) +':::'+ str(cnt/31.0*100))

cnt=(rmse_cor_apr['rmse_cor'] <rmse_cor_apr['mod']).sum()
#print(rmse_cor_m)
print ('Apr : ' + str(cnt) +':::'+ str(cnt/30.0*100))

cnt=(rmse_cor_may['rmse_cor'] <rmse_cor_may['mod']).sum()
#print(rmse_cor_m)
print ('May : ' + str(cnt) +':::'+ str(cnt/31.0*100))

cnt=(rmse_cor_jun['rmse_cor'] <rmse_cor_jun['mod']).sum()
#print(rmse_cor_m)
print ('Jun : ' + str(cnt) +':::'+ str(cnt/30.0*100))

cnt=(rmse_cor['rmse_cor'] <rmse_cor['mod']).sum()
print ('All : ' + str(cnt) +':::'+ str(cnt/180.0*100))

######################################################################################################################################


rh_data_new_hour_jan=rh_data_new_hour['2017-12-31 13':'2018-02-01 12']
hourt = pd.to_timedelta(rh_data_new_hour_jan.index.hour,  unit='H')
mf=int(rh_data_new_hour_jan.shape[0]/24.0)

uc_rh_rmse_hour_avg_day1_jan=(((rh_data_new_hour_jan['bias_mod']**2).groupby(hourt)).mean())**0.5
uc_rh_rmse_hour_avg_day1_jan=pd.concat([uc_rh_rmse_hour_avg_day1_jan[13:24],uc_rh_rmse_hour_avg_day1_jan[0:13]])
uc_rh_rmse_hour_avg_day1_jan=pd.concat([uc_rh_rmse_hour_avg_day1_jan]*mf)
uc_rh_rmse_hour_avg_day1_jan.index=rh_data_new_hour_jan.index

cor_rh_rmse_hour_avg_day1_jan=(((rh_data_new_hour_jan['bias_cor']**2).groupby(hourt)).mean())**0.5
cor_rh_rmse_hour_avg_day1_jan=pd.concat([cor_rh_rmse_hour_avg_day1_jan[13:24],cor_rh_rmse_hour_avg_day1_jan[0:13]])
cor_rh_rmse_hour_avg_day1_jan=pd.concat([cor_rh_rmse_hour_avg_day1_jan]*mf)
cor_rh_rmse_hour_avg_day1_jan.index=rh_data_new_hour_jan.index 

rh_data_new_hour_jan.insert(6,'uc_rmse',uc_rh_rmse_hour_avg_day1_jan)
rh_data_new_hour_jan.insert(7,'cor_rmse',cor_rh_rmse_hour_avg_day1_jan)

plot_bars_rmse_hw(rh_data_new_hour_jan.iloc[11:35,:],'rh','Hour_wise','RMSE','201801' )


cnt=(rh_data_new_hour_jan['cor_rmse'] <rh_data_new_hour_jan['uc_rmse']).sum()
print ('Jan : ' + str(cnt) +':::'+ str(cnt/float(rh_data_new_hour_jan.shape[0])*100))


rmse_cor_jan=rmse_cor['2018-01-01':'2018-01-31']
plot_bars_rmse(rmse_cor_jan,'rh','day_1','rmse',rmse_cor_jan.index.strftime('%Y%m')[0])
###############################################################################################################################################
rh_data_new_hour_feb=rh_data_new_hour['2018-01-31 13':'2018-03-01 12']
hourt = pd.to_timedelta(rh_data_new_hour_feb.index.hour,  unit='H')
mf=int(rh_data_new_hour_feb.shape[0]/24.0)

uc_rh_rmse_hour_avg_day1_feb=(((rh_data_new_hour_feb['bias_mod']**2).groupby(hourt)).mean())**0.5
uc_rh_rmse_hour_avg_day1_feb=pd.concat([uc_rh_rmse_hour_avg_day1_feb[13:24],uc_rh_rmse_hour_avg_day1_feb[0:13]])
uc_rh_rmse_hour_avg_day1_feb=pd.concat([uc_rh_rmse_hour_avg_day1_feb]*mf)
uc_rh_rmse_hour_avg_day1_feb.index=rh_data_new_hour_feb.index

cor_rh_rmse_hour_avg_day1_feb=(((rh_data_new_hour_feb['bias_cor']**2).groupby(hourt)).mean())**0.5
cor_rh_rmse_hour_avg_day1_feb=pd.concat([cor_rh_rmse_hour_avg_day1_feb[13:24],cor_rh_rmse_hour_avg_day1_feb[0:13]])
cor_rh_rmse_hour_avg_day1_feb=pd.concat([cor_rh_rmse_hour_avg_day1_feb]*mf)
cor_rh_rmse_hour_avg_day1_feb.index=rh_data_new_hour_feb.index 

rh_data_new_hour_feb.insert(6,'uc_rmse',uc_rh_rmse_hour_avg_day1_feb)
rh_data_new_hour_feb.insert(7,'cor_rmse',cor_rh_rmse_hour_avg_day1_feb)

plot_bars_rmse_hw(rh_data_new_hour_feb.iloc[11:35,:],'rh','Hour_wise','RMSE','201802' )

cnt=(rh_data_new_hour_feb['cor_rmse'] <rh_data_new_hour_feb['uc_rmse']).sum()
print ('Feb : ' + str(cnt) +':::'+ str(cnt/float(rh_data_new_hour_feb.shape[0])*100))

rmse_cor_feb=rmse_cor['2018-02-01':'2018-02-28']
plot_bars_rmse(rmse_cor_feb,'rh','day_1','rmse',rmse_cor_feb.index.strftime('%Y%m')[0])

##################################################################################################################
rh_data_new_hour_mar=rh_data_new_hour['2018-02-28 13':'2018-04-01 12']
hourt = pd.to_timedelta(rh_data_new_hour_mar.index.hour,  unit='H')
mf=int(rh_data_new_hour_mar.shape[0]/24.0)

uc_rh_rmse_hour_avg_day1_mar=(((rh_data_new_hour_mar['bias_mod']**2).groupby(hourt)).mean())**0.5
uc_rh_rmse_hour_avg_day1_mar=pd.concat([uc_rh_rmse_hour_avg_day1_mar[13:24],uc_rh_rmse_hour_avg_day1_mar[0:13]])
uc_rh_rmse_hour_avg_day1_mar=pd.concat([uc_rh_rmse_hour_avg_day1_mar]*mf)
uc_rh_rmse_hour_avg_day1_mar.index=rh_data_new_hour_mar.index

cor_rh_rmse_hour_avg_day1_mar=(((rh_data_new_hour_mar['bias_cor']**2).groupby(hourt)).mean())**0.5
cor_rh_rmse_hour_avg_day1_mar=pd.concat([cor_rh_rmse_hour_avg_day1_mar[13:24],cor_rh_rmse_hour_avg_day1_mar[0:13]])
cor_rh_rmse_hour_avg_day1_mar=pd.concat([cor_rh_rmse_hour_avg_day1_mar]*mf)
cor_rh_rmse_hour_avg_day1_mar.index=rh_data_new_hour_mar.index 

rh_data_new_hour_mar.insert(6,'uc_rmse',uc_rh_rmse_hour_avg_day1_mar)
rh_data_new_hour_mar.insert(7,'cor_rmse',cor_rh_rmse_hour_avg_day1_mar)

plot_bars_rmse_hw(rh_data_new_hour_mar.iloc[11:35,:],'rh','Hour_wise','RMSE','201803' )

cnt=(rh_data_new_hour_mar['cor_rmse'] <rh_data_new_hour_mar['uc_rmse']).sum()
print ('mar : ' + str(cnt) +':::'+ str(cnt/float(rh_data_new_hour_mar.shape[0])*100))

rmse_cor_mar=rmse_cor['2018-03-01':'2018-03-31']
plot_bars_rmse(rmse_cor_mar,'rh','day_1','rmse',rmse_cor_mar.index.strftime('%Y%m')[0])
############################################################################################################################
rh_data_new_hour_apr=rh_data_new_hour['2018-03-31 13':'2018-05-01 12']
hourt = pd.to_timedelta(rh_data_new_hour_apr.index.hour,  unit='H')
mf=int(rh_data_new_hour_apr.shape[0]/24.0)

uc_rh_rmse_hour_avg_day1_apr=(((rh_data_new_hour_apr['bias_mod']**2).groupby(hourt)).mean())**0.5
uc_rh_rmse_hour_avg_day1_apr=pd.concat([uc_rh_rmse_hour_avg_day1_apr[13:24],uc_rh_rmse_hour_avg_day1_apr[0:13]])
uc_rh_rmse_hour_avg_day1_apr=pd.concat([uc_rh_rmse_hour_avg_day1_apr]*mf)
uc_rh_rmse_hour_avg_day1_apr.index=rh_data_new_hour_apr.index

cor_rh_rmse_hour_avg_day1_apr=(((rh_data_new_hour_apr['bias_cor']**2).groupby(hourt)).mean())**0.5
cor_rh_rmse_hour_avg_day1_apr=pd.concat([cor_rh_rmse_hour_avg_day1_apr[13:24],cor_rh_rmse_hour_avg_day1_apr[0:13]])
cor_rh_rmse_hour_avg_day1_apr=pd.concat([cor_rh_rmse_hour_avg_day1_apr]*mf)
cor_rh_rmse_hour_avg_day1_apr.index=rh_data_new_hour_apr.index 

rh_data_new_hour_apr.insert(6,'uc_rmse',uc_rh_rmse_hour_avg_day1_apr)
rh_data_new_hour_apr.insert(7,'cor_rmse',cor_rh_rmse_hour_avg_day1_apr)

plot_bars_rmse_hw(rh_data_new_hour_apr.iloc[11:35,:],'rh','Hour_wise','RMSE','201804' )

cnt=(rh_data_new_hour_apr['cor_rmse'] <rh_data_new_hour_apr['uc_rmse']).sum()
print ('Apr : ' + str(cnt) +':::'+ str(cnt/float(rh_data_new_hour_apr.shape[0])*100))


rmse_cor_apr=rmse_cor['2018-04-01':'2018-04-30']
plot_bars_rmse(rmse_cor_apr,'rh','day_1','rmse',rmse_cor_apr.index.strftime('%Y%m')[0])

####################################################################################################################################
rh_data_new_hour_may=rh_data_new_hour['2018-04-30 13':'2018-06-01 12']
hourt = pd.to_timedelta(rh_data_new_hour_may.index.hour,  unit='H')
mf=int(rh_data_new_hour_may.shape[0]/24.0)

uc_rh_rmse_hour_avg_day1_may=(((rh_data_new_hour_may['bias_mod']**2).groupby(hourt)).mean())**0.5
uc_rh_rmse_hour_avg_day1_may=pd.concat([uc_rh_rmse_hour_avg_day1_may[13:24],uc_rh_rmse_hour_avg_day1_may[0:13]])
uc_rh_rmse_hour_avg_day1_may=pd.concat([uc_rh_rmse_hour_avg_day1_may]*mf)
uc_rh_rmse_hour_avg_day1_may.index=rh_data_new_hour_may.index

cor_rh_rmse_hour_avg_day1_may=(((rh_data_new_hour_may['bias_cor']**2).groupby(hourt)).mean())**0.5
cor_rh_rmse_hour_avg_day1_may=pd.concat([cor_rh_rmse_hour_avg_day1_may[13:24],cor_rh_rmse_hour_avg_day1_may[0:13]])
cor_rh_rmse_hour_avg_day1_may=pd.concat([cor_rh_rmse_hour_avg_day1_may]*mf)
cor_rh_rmse_hour_avg_day1_may.index=rh_data_new_hour_may.index 

rh_data_new_hour_may.insert(6,'uc_rmse',uc_rh_rmse_hour_avg_day1_may)
rh_data_new_hour_may.insert(7,'cor_rmse',cor_rh_rmse_hour_avg_day1_may)

plot_bars_rmse_hw(rh_data_new_hour_may.iloc[11:35,:],'rh','Hour_wise','RMSE','201805' )

cnt=(rh_data_new_hour_may['cor_rmse'] <rh_data_new_hour_may['uc_rmse']).sum()
print ('May : ' + str(cnt) +':::'+ str(cnt/float(rh_data_new_hour_may.shape[0])*100))


rmse_cor_may=rmse_cor['2018-05-01':'2018-05-31']
plot_bars_rmse(rmse_cor_may,'rh','day_1','rmse',rmse_cor_may.index.strftime('%Y%m')[0])

##################################################################################################################################

rh_data_new_hour_jun=rh_data_new_hour['2018-05-31 13':'2018-06-29 12']
hourt = pd.to_timedelta(rh_data_new_hour_jun.index.hour,  unit='H')
mf=int(rh_data_new_hour_jun.shape[0]/24.0)

uc_rh_rmse_hour_avg_day1_jun=(((rh_data_new_hour_jun['bias_mod']**2).groupby(hourt)).mean())**0.5
uc_rh_rmse_hour_avg_day1_jun=pd.concat([uc_rh_rmse_hour_avg_day1_jun[13:24],uc_rh_rmse_hour_avg_day1_jun[0:13]])
uc_rh_rmse_hour_avg_day1_jun=pd.concat([uc_rh_rmse_hour_avg_day1_jun]*mf)
uc_rh_rmse_hour_avg_day1_jun.index=rh_data_new_hour_jun.index

cor_rh_rmse_hour_avg_day1_jun=(((rh_data_new_hour_jun['bias_cor']**2).groupby(hourt)).mean())**0.5
cor_rh_rmse_hour_avg_day1_jun=pd.concat([cor_rh_rmse_hour_avg_day1_jun[13:24],cor_rh_rmse_hour_avg_day1_jun[0:13]])
cor_rh_rmse_hour_avg_day1_jun=pd.concat([cor_rh_rmse_hour_avg_day1_jun]*mf)
cor_rh_rmse_hour_avg_day1_jun.index=rh_data_new_hour_jun.index 

rh_data_new_hour_jun.insert(6,'uc_rmse',uc_rh_rmse_hour_avg_day1_jun)
rh_data_new_hour_jun.insert(7,'cor_rmse',cor_rh_rmse_hour_avg_day1_jun)

plot_bars_rmse_hw(rh_data_new_hour_jun.iloc[11:35,:],'rh','Hour_wise','RMSE','201806' )

cnt=(rh_data_new_hour_jun['cor_rmse'] <rh_data_new_hour_jun['uc_rmse']).sum()
print ('Jun : ' + str(cnt) +':::'+ str(cnt/float(rh_data_new_hour_jun.shape[0])*100))

rmse_cor_jun=rmse_cor['2018-06-01':'2018-06-29']
plot_bars_rmse(rmse_cor_jun,'rh','day_1','rmse',rmse_cor_jun.index.strftime('%Y%m')[0])

##########################################################################################################################################


#hourt = pd.to_timedelta(tmp_data_new_hour.index.hour,  unit='H')
#mf=int(tmp_data_new_hour.shape[0]/24.0)
#
#uc_tmp_rmse_hour_avg_day1=(((tmp_data_new_hour['bias_mod']**2).groupby(hourt)).mean())**0.5
#uc_tmp_rmse_hour_avg_day1=pd.concat([uc_tmp_rmse_hour_avg_day1[13:24],uc_tmp_rmse_hour_avg_day1[0:13]])
#uc_tmp_rmse_hour_avg_day1=pd.concat([uc_tmp_rmse_hour_avg_day1]*mf)
#uc_tmp_rmse_hour_avg_day1.index=tmp_data_new_hour.index
#
#cor_tmp_rmse_hour_avg_day1=(((tmp_data_new_hour['bias_cor']**2).groupby(hourt)).mean())**0.5
#cor_tmp_rmse_hour_avg_day1=pd.concat([cor_tmp_rmse_hour_avg_day1[13:24],cor_tmp_rmse_hour_avg_day1[0:13]])
#cor_tmp_rmse_hour_avg_day1=pd.concat([cor_tmp_rmse_hour_avg_day1]*mf)
#cor_tmp_rmse_hour_avg_day1.index=tmp_data_new_hour.index 
#
#tmp_data_new_hour.insert(8,'uc_rmse',uc_tmp_rmse_hour_avg_day1)
#tmp_data_new_hour.insert(9,'cor_rmse',cor_tmp_rmse_hour_avg_day1)
#
#plot_bars_rmse_hw(tmp_data_new_hour.iloc[11:34,:],'Temperature','Hour_wise','RMSE',tmp_data_new_hour.index.strftime('%Y%m')[0] )

#############################################################################################################################################



























