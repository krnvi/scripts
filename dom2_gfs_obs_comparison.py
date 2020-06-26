#!/usr/bin/python

import sys ;import os ; import numpy as np ; import pandas as pd ; import datetime as dt ; from dateutil import tz, rrule ; import pytz
import matplotlib.pyplot as plt; from pylab import savefig 


############################################################################################################################################

main='/home/vkvalappil/Data/oppModel' ; output=main+'/output/output/stat/' ; inp=output=main+'/output/output/'
date='2017010106'

date_1=dt.datetime.strptime(date,'%Y%m%d%H')+dt.timedelta(days=0)
date_2=dt.datetime.strptime(date,'%Y%m%d%H')+dt.timedelta(days=545)
date_list=[x.strftime('%Y%m%d%H') for x in rrule.rrule(rrule.DAILY,dtstart=date_1,until=date_2)]

bias_day1 = [] ; rmse_day1 = [] ; bias_day2 = [] ; rmse_day2 = [] ; bias_hour_day1=[] ; bias_hour_day2=[]
mod_hour_day1=[] ; obs_hour_day1=[] ; mod_hour_day2=[] ; obs_hour_day2=[] ;
gfs_ae=[] ; min_tmp_ae=[]

##############################################################################################################################################

def plot_bars_rmse(C,parm,parm1,metric,ds ):
    
    fig=plt.figure(figsize=(20,12),dpi=50,frameon=True); ax = fig.add_subplot(111,facecolor='white')

    pos_1  = np.arange(0,12*C.shape[0],12) ; wdh=3 ; opacity = 1.0 ; #ax.hold(True);

    ax.bar(pos_1+wdh,C['GFS'],width=wdh, color='b',alpha=opacity,label='GFS RSE')
    ax.bar(pos_1+2*wdh ,C['Dom_02'],width=wdh, color='r',alpha=opacity,label='Dom 2 rmse')
    ax.bar(pos_1+3*wdh ,C['min_tmp'],width=wdh, color='g',alpha=opacity,label='min_tmp ae')

    #ax.axhline(1, color="k")

    plt.tick_params(axis='y',which='both', left='on',  right='off', labelright='off')
    ax.tick_params(axis='y', colors='black',labelsize=16) ; ax.yaxis.set_ticks_position('left') ;

    #ax.set_yticks(np.arange(-10,45,10)) ; 
    ax.set_ylabel(metric,color='black',fontsize=16,fontweight='bold') ;
    ax.set_xlabel('TIME: UTC ',color='black',fontsize=16,fontweight='bold') ;

    ax.set_xticks(np.arange(0,10*C.shape[0],10)+2*wdh) ; 
    xTickMarks=C.index.strftime('%Y%m%d') #pd.TimedeltaIndex.to_series(C.index).dt.components.hours 
    xtickNames =ax.set_xticklabels(xTickMarks,rotation=90,fontsize=14,fontweight='bold')

#    plt.tick_params(axis='x',which='both', bottom='off',  top='off', labelbottom='off')
#    for n, row in enumerate(C.iloc[:]):
#        plt.text(4*n, row, np.round(row,3), ha='center', rotation=0, va='bottom',fontsize=16,fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    display = (0,1,2,3,4)

    leg=ax.legend(loc='lower right',fontsize=18)

    plt.tight_layout(pad=3) ;

    plt.title( parm1+' : 2M '+parm,fontsize=16)
    outFile=output+'/gfs_dom2/'+parm+'_'+parm1+'_'+metric+'_'+ds+'.png' 
    savefig(outFile);
    plt.close(fig) ; fig.clf(fig)    

############################################################################################################################
def plot_scater_rmse(C,parm,parm1):
    
    fig=plt.figure(figsize=(20,12),dpi=50,frameon=True); ax = fig.add_subplot(111,facecolor='white')
    ax.scatter(C['GFS'],C['min_tmp'],s=100, c='k', marker='o')
     
    #ax.axhline(1, color="k")

    plt.tick_params(axis='y',which='both', left='on',  right='off', labelright='off')
    ax.tick_params(axis='y', colors='black',labelsize=16) ; ax.yaxis.set_ticks_position('left') ;

    #ax.set_yticks(np.arange(-10,45,10)) ; 
    ax.set_ylabel('Dom2',color='black',fontsize=16,fontweight='bold') ;
    ax.set_xlabel('GFS',color='black',fontsize=16,fontweight='bold') ;
    plt.tight_layout(pad=3) ;

    plt.title( parm1+' : 2M '+parm,fontsize=16)
    outFile=output+'/gfs_dom2/'+parm+'_'+parm1+'_scater.png' 
    savefig(outFile);
    plt.close(fig) ; fig.clf(fig)    



#########################################################################################################################################


for dte in date_list[:]:
    
    file_2=inp+'domain_2/surfaceLevel/hourly'+dte+'.csv' ;     gfs_file=inp+'gfs_output/gfs_hourly'+dte+'.csv'
    if (os.path.isfile(file_2)) & (os.path.isfile(gfs_file)):

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

        
        gfs_data= pd.read_csv(gfs_file) ;  gfs_data=gfs_data.iloc[0:11,:]       
        gfs_data['localTime']=gfs_data['localTime'].apply(pd.to_datetime, errors='ignore')    
        gfs_data.iloc[:,4:]=gfs_data.iloc[:,4:].apply(pd.to_numeric,errors='coerce')
        gfs_data_1=gfs_data.iloc[:,3:]
        gfs_data_1.index=gfs_data_1.localTime
        gfs_data_1.index=gfs_data_1.index.tz_localize(pytz.timezone('Asia/Dubai')).tz_convert(pytz.utc)
        gfs_data_1['localTime']=gfs_data_1.index 

        gfs_data_2=gfs_data_1.iloc[0,:] ; gfs_data_3=gfs_data_2.drop(['DAY_SEQUENCE','CLOUD','SPHUM','SURFPRES','WSPD'],axis=0) ;
        gfs_data_4=pd.DataFrame([gfs_data_3],columns=gfs_data_3.index) ; gfs_data_4.WDIR=gfs_data_4.WDIR*0.2777
        
        obs_4=pd.DataFrame([obs_3.loc[gfs_data_2.localTime]],columns=obs_3.columns)
        gfs_data_4.columns=obs_4.columns
        
        gfs_bias=gfs_data_4.iloc[0,1:].sub(obs_4.iloc[0,1:],axis=0)        
        gfs_rse=(gfs_bias**2)**0.5 

        gfs_rse_1=pd.DataFrame(np.vstack(gfs_rse.values).T,columns=['tmp_02','dpt_02','rh_02','mr_02','sknt_02'])
        gfs_rse_1.insert(0,'Date',dte)

################################### Calculating Daily bias and daily rmse ############################################################################

        min_tmp_obs=(obs_3.iloc[7:31,1:]).TMP.min() ;  min_tmp_obs_tim=(obs_3.TMP.iloc[7:31]).idxmin().strftime('%Y%m%d%H')

        mod_min_tmp_b1=(dt.datetime.strptime(min_tmp_obs_tim,'%Y%m%d%H')).strftime('%Y-%m-%d %H') ;  
        mod_min_tmp_b2=(dt.datetime.strptime(min_tmp_obs_tim,'%Y%m%d%H') +dt.timedelta(hours=-3)).strftime('%Y-%m-%d %H')

        min_tmp_mod=(mod_dom_2_2[mod_min_tmp_b2:mod_min_tmp_b1]).TMP.min()
 
        min_tmp_bias=np.abs(min_tmp_mod-min_tmp_obs)

        min_tmp_ae1=pd.DataFrame(np.vstack([dte,min_tmp_bias]).T,columns=['Date','min_tmp_ae'])

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
        gfs_ae.append(gfs_rse_1) ;  min_tmp_ae.append(min_tmp_ae1)
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
       print dte 
       print("No Data Exist")

#############################################################################################################################
################################### hourly Analysis ######################################################################### 
bias_hour_day_1=pd.concat(bias_hour_day1,axis=0) ; bias_hour_day_2=pd.concat(bias_hour_day2,axis=0)
mod_hour_day_1=pd.concat(mod_hour_day1,axis=0)   ; mod_hour_day_2=pd.concat(mod_hour_day2,axis=0)   ;
obs_hour_day_1=pd.concat(obs_hour_day1,axis=0)   ; mod_hour_day_2=pd.concat(mod_hour_day2,axis=0)   ;

rmse_day1=pd.concat(rmse_day1,axis=0) ; gfs_ae1=pd.concat(gfs_ae,axis=0)

ae_mint_day1=pd.concat(min_tmp_ae,axis=0)

### TEMP

#tmp_rmse=pd.concat([gfs_ae1['Date'],gfs_ae1['tmp_02'],rmse_day1['tmp_02']],axis=1)
#tmp_rmse['Date']=tmp_rmse['Date'].apply(pd.to_datetime, errors='ignore',format='%Y%m%d%H')    
#tmp_rmse.index=tmp_rmse.Date ; tmp_rmse.columns=['Date','GFS','Dom_02']


tmp_rmse=pd.concat([gfs_ae1['Date'],gfs_ae1['tmp_02'],ae_mint_day1['min_tmp_ae'],rmse_day1['tmp_02']],axis=1)
tmp_rmse['Date']=tmp_rmse['Date'].apply(pd.to_datetime, errors='ignore',format='%Y%m%d%H')  
tmp_rmse.iloc[:,1:]=tmp_rmse.iloc[:,1:].apply(pd.to_numeric,errors='coerce')
tmp_rmse.index=tmp_rmse.Date ; tmp_rmse.columns=['Date','GFS','min_tmp','Dom_02']

plot_scater_rmse(tmp_rmse,'tmperature','day_1')
#########################################################################################
rmse_jan=tmp_rmse['2017-01-01':'2017-01-31']
rmse_feb=tmp_rmse['2017-02-01':'2017-02-28']
rmse_mar=tmp_rmse['2017-03-01':'2017-03-31']
rmse_apr=tmp_rmse['2017-04-01':'2017-04-30']
rmse_may=tmp_rmse['2017-05-01':'2017-05-31']
rmse_jun=tmp_rmse['2017-06-01':'2017-06-30']
rmse_jul=tmp_rmse['2017-07-01':'2017-07-31']
rmse_aug=tmp_rmse['2017-08-01':'2017-08-31']
rmse_sep=tmp_rmse['2017-09-01':'2017-09-30']
rmse_oct=tmp_rmse['2017-10-01':'2017-10-31']
rmse_nov=tmp_rmse['2017-11-01':'2017-11-30']
rmse_dec=tmp_rmse['2017-12-01':'2017-12-31']

plot_bars_rmse(rmse_jan,'tmperature','day_1','rmse',rmse_jan.index.strftime('%Y%m')[0])
plot_bars_rmse(rmse_feb,'tmperature','day_1','rmse',rmse_feb.index.strftime('%Y%m')[0])
plot_bars_rmse(rmse_mar,'tmperature','day_1','rmse',rmse_mar.index.strftime('%Y%m')[0])
plot_bars_rmse(rmse_apr,'tmperature','day_1','rmse',rmse_apr.index.strftime('%Y%m')[0])
plot_bars_rmse(rmse_may,'tmperature','day_1','rmse',rmse_may.index.strftime('%Y%m')[0])
plot_bars_rmse(rmse_jun,'tmperature','day_1','rmse',rmse_jun.index.strftime('%Y%m')[0])
plot_bars_rmse(rmse_jul,'tmperature','day_1','rmse',rmse_jul.index.strftime('%Y%m')[0])
plot_bars_rmse(rmse_aug,'tmperature','day_1','rmse',rmse_aug.index.strftime('%Y%m')[0])
plot_bars_rmse(rmse_sep,'tmperature','day_1','rmse',rmse_sep.index.strftime('%Y%m')[0])
plot_bars_rmse(rmse_oct,'tmperature','day_1','rmse',rmse_oct.index.strftime('%Y%m')[0])
plot_bars_rmse(rmse_nov,'tmperature','day_1','rmse',rmse_nov.index.strftime('%Y%m')[0])
plot_bars_rmse(rmse_dec,'tmperature','day_1','rmse',rmse_dec.index.strftime('%Y%m')[0])



#plot_bars_rmse(rmse_jan,'min_tmperature','day_1','ae',rmse_jan.index.strftime('%Y%m')[0])
#plot_bars_rmse(rmse_feb,'min_tmperature','day_1','ae',rmse_feb.index.strftime('%Y%m')[0])
#plot_bars_rmse(rmse_mar,'min_tmperature','day_1','ae',rmse_mar.index.strftime('%Y%m')[0])
#plot_bars_rmse(rmse_apr,'min_tmperature','day_1','ae',rmse_apr.index.strftime('%Y%m')[0])
#plot_bars_rmse(rmse_may,'min_tmperature','day_1','ae',rmse_may.index.strftime('%Y%m')[0])
#plot_bars_rmse(rmse_jun,'min_tmperature','day_1','ae',rmse_jun.index.strftime('%Y%m')[0])
#plot_bars_rmse(rmse_jul,'min_tmperature','day_1','ae',rmse_jul.index.strftime('%Y%m')[0])
#plot_bars_rmse(rmse_aug,'min_tmperature','day_1','ae',rmse_aug.index.strftime('%Y%m')[0])
#plot_bars_rmse(rmse_sep,'min_tmperature','day_1','ae',rmse_sep.index.strftime('%Y%m')[0])
#plot_bars_rmse(rmse_oct,'min_tmperature','day_1','ae',rmse_oct.index.strftime('%Y%m')[0])
#plot_bars_rmse(rmse_nov,'min_tmperature','day_1','ae',rmse_nov.index.strftime('%Y%m')[0])
#plot_bars_rmse(rmse_dec,'min_tmperature','day_1','ae',rmse_dec.index.strftime('%Y%m')[0])

###################################################################################################






















