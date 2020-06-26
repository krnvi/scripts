#!/usr/bin/python

import sys ;import os ; import numpy as np ; import pandas as pd ; import datetime as dt ; from dateutil import tz, rrule ; import pytz
import matplotlib.pyplot as plt; from pylab import savefig 


############################################################################################################################################

main='/home/vkvalappil/Data/modelWRF/nudging/'  ; output=main+'ARW/output/domain_02/stat/' ; inp=main+'ARW/output/'
date='2017122006'
##########################################################################################################################

def plot_line(C,parm,ds ):
    
    fig=plt.figure(figsize=(20,12),dpi=50,frameon=True); ax1 = fig.add_subplot(111,facecolor='white')

    pos=np.arange(0,C.shape[0],1) ; opacity = 1.0 ; #ax.hold(True);

    ax1.plot(pos,C['mod_nudge'],color='b',alpha=opacity,lw=4,linestyle='-',label='Dom 2 nudge model')
    ax1.plot(pos,C['mod_opp'],color='r',alpha=opacity,lw=4,linestyle='-',label='Dom 2 model opp')
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
#        plt.text(4*n, row, np.round(row,3), ha='center', rotation=0, va='bottom',fontsize=16,fontweight='bold')

    handles, labels = ax1.get_legend_handles_labels()
    display = (0,1,2,3,4)

    leg=ax1.legend(loc='lower right',fontsize=18)

    plt.tight_layout(pad=3) ;

    plt.title( '2M '+parm,fontsize=16)
    #outFile=output+'stat/corrected/dom2_'+parm1+'_'+metric+'_'+ds+'.png' 
    outFile=output+'/dom2_'+parm+'_'+ds+'.png' 
    savefig(outFile);
    plt.close(fig) ; fig.clf(fig) 




#########################################################################################################################
date_1=dt.datetime.strptime(date,'%Y%m%d%H')+dt.timedelta(days=0)
date_2=dt.datetime.strptime(date,'%Y%m%d%H')+dt.timedelta(days=7)
date_list=[x.strftime('%Y%m%d%H') for x in rrule.rrule(rrule.DAILY,dtstart=date_1,until=date_2)]

bias_day1 = [] ; rmse_day1 = [] ; bias_day2 = [] ; rmse_day2 = [] ; bias_hour_day1=[] ; bias_hour_day2=[]
mod_hour_day1=[] ; obs_hour_day1=[] ; mod_hour_day2=[] ; obs_hour_day2=[] ;
gfs_ae=[] ; min_tmp_ae=[]

for dte in date_list[:]:
    
    file_1=inp+'domain_02/surfaceLevel/nudge_hourly'+dte+'.csv' ; file_2=inp+'domain_02/surfaceLevel/no_nudge_hourly'+dte+'.csv' ;
    if (os.path.isfile(file_1)) & (os.path.isfile(file_2)):

        mod_dom_2_nu=pd.read_csv(file_1) ; mod_dom_2_nu=mod_dom_2_nu.iloc[72:144,:] ; 
        mod_dom_2_nn=pd.read_csv(file_2) ; mod_dom_2_nn=mod_dom_2_nn.iloc[72:144,:] ; 
        
        o_date_1=dte ; 
        o_date_2=(dt.datetime.strptime(dte,'%Y%m%d%H')+dt.timedelta(days=1)).strftime('%Y%m%d%H')
        o_date_3=(dt.datetime.strptime(dte,'%Y%m%d%H')+dt.timedelta(days=2)).strftime('%Y%m%d%H')
        o_date_4=(dt.datetime.strptime(dte,'%Y%m%d%H')+dt.timedelta(days=3)).strftime('%Y%m%d%H')
    
        obs_file_1='/home/vkvalappil/Data/masdar_station_data/wyoming/'+o_date_1[0:6]+'/AbuDhabi_surf_mr'+o_date_1[0:8]+'.csv'
        obs_file_2='/home/vkvalappil/Data/masdar_station_data/wyoming/'+o_date_2[0:6]+'/AbuDhabi_surf_mr'+o_date_2[0:8]+'.csv'
        obs_file_3='/home/vkvalappil/Data/masdar_station_data/wyoming/'+o_date_3[0:6]+'/AbuDhabi_surf_mr'+o_date_3[0:8]+'.csv'
        obs_file_4='/home/vkvalappil/Data/masdar_station_data/wyoming/'+o_date_4[0:6]+'/AbuDhabi_surf_mr'+o_date_4[0:8]+'.csv'
        
        mod_dom_2_nu['localTime']=mod_dom_2_nu['localTime'].apply(pd.to_datetime, errors='ignore')    
        mod_dom_2_nu.iloc[:,4:]=mod_dom_2_nu.iloc[:,4:].apply(pd.to_numeric,errors='coerce')
        mod_dom_2_1_nu=mod_dom_2_nu.iloc[:,3:]
        mod_dom_2_1_nu.index=mod_dom_2_1_nu.localTime
        mod_dom_2_1_nu.index=mod_dom_2_1_nu.index.tz_localize(pytz.timezone('Asia/Dubai')).tz_convert(pytz.utc)
        mod_dom_2_1_nu['localTime']=mod_dom_2_1_nu.index
        
        mod_dom_2_nn['localTime']=mod_dom_2_nn['localTime'].apply(pd.to_datetime, errors='ignore')    
        mod_dom_2_nn.iloc[:,4:]=mod_dom_2_nn.iloc[:,4:].apply(pd.to_numeric,errors='coerce')
        mod_dom_2_1_nn=mod_dom_2_nn.iloc[:,3:]
        mod_dom_2_1_nn.index=mod_dom_2_1_nn.localTime
        mod_dom_2_1_nn.index=mod_dom_2_1_nn.index.tz_localize(pytz.timezone('Asia/Dubai')).tz_convert(pytz.utc)
        mod_dom_2_1_nn['localTime']=mod_dom_2_1_nn.index        
              
        obs_1=pd.read_csv(obs_file_1) ; obs_2=pd.read_csv(obs_file_2) ; obs_3=pd.read_csv(obs_file_3) ; obs_4=pd.read_csv(obs_file_4)
        obs=pd.concat([obs_1,obs_2,obs_3,obs_4],axis=0)
    
        obs['TIME']=obs['TIME'].apply(pd.to_datetime,errors='ignore')
        obs.iloc[:,3:]=obs.iloc[:,3:].apply(pd.to_numeric,errors='coerce')
        obs_1=obs.iloc[:,2:]
        obs_1.index=obs_1.TIME
        obs_1.index=obs_1.index.tz_localize(pytz.utc)
  
        idx = obs_1.index.intersection(mod_dom_2_1_nu.index)
        obs_2=obs_1.loc[idx]
        obs_3=pd.concat([obs_2['TIME'],obs_2['TMP'],obs_2['DEW'],obs_2['RH'],obs_2['mrio'],obs_2['SPD']],axis=1)

        mod_dom_2_2_nu=pd.concat([mod_dom_2_1_nu['localTime'],mod_dom_2_1_nu['TEMP'],mod_dom_2_1_nu['DTEMP'],mod_dom_2_1_nu['RH'],mod_dom_2_1_nu['MXRATIO'],mod_dom_2_1_nu['WDIR']*0.277],axis=1)    

        mod_dom_2_2_nu.columns=obs_3.columns

        mod_dom_2_2_nn=pd.concat([mod_dom_2_1_nn['localTime'],mod_dom_2_1_nn['TEMP'],mod_dom_2_1_nn['DTEMP'],mod_dom_2_1_nn['RH'],mod_dom_2_1_nn['MXRATIO'],mod_dom_2_1_nn['WDIR']*0.277],axis=1)    

        mod_dom_2_2_nn.columns=obs_3.columns
        
        ########## Temp model
        tmp_=pd.concat([mod_dom_2_2_nu['TIME'],mod_dom_2_2_nu['TMP'],mod_dom_2_2_nn['TMP'],obs_3['TMP']],axis=1)
        tmp_.columns=['Date','mod_nudge','mod_opp','obs']
        
        plot_line(tmp_.iloc[0:31,:],'Temperature', tmp_.index.strftime('%Y%m%d%H')[0])
        

        rh_=pd.concat([mod_dom_2_2_nu['TIME'],mod_dom_2_2_nu['RH'],mod_dom_2_2_nn['RH'],obs_3['RH']],axis=1)
        rh_.columns=['Date','mod_nudge','mod_opp','obs']
        
        plot_line(rh_.iloc[0:31,:],'Relative_humidity', tmp_.index.strftime('%Y%m%d%H')[0])
  

        mr_=pd.concat([mod_dom_2_2_nu['TIME'],mod_dom_2_2_nu['mrio'],mod_dom_2_2_nn['mrio'],obs_3['mrio']],axis=1)
        mr_.columns=['Date','mod_nudge','mod_opp','obs']
        
        plot_line(mr_.iloc[0:31,:],'Mixing_ratio', tmp_.index.strftime('%Y%m%d%H')[0])

        

################################### Calculating Daily bias and daily rmse ############################################################################

        nu_mod_dom_2_bias_1=mod_dom_2_2_nu.iloc[7:31,1:].sub(obs_3.iloc[7:31,1:],axis=0)
        nu_mod_dom_2_rmse_1=((nu_mod_dom_2_bias_1**2).mean(axis=0))**0.5


        #mod_dom_2_bias_2=mod_dom_2_2.iloc[31:55,1:].sub(obs_3.iloc[31:55,1:],axis=0)      
        #mod_dom_2_rmse_2=((mod_dom_2_bias_2**2).mean(axis=0))**0.5

##################################################################################################        

#        bias_day_1=np.vstack(mod_dom_2_bias_1.mean(axis=0).values).T
#        bias_day_1=pd.DataFrame(bias_day_1,columns=['tmp_02','dpt_02','rh_02','mr_02','sknt_02'])
#        bias_day_1.insert(0,'Date',dte)
#
#        rmse_day_1=np.vstack(mod_dom_2_rmse_1.values).T
#        rmse_day_1=pd.DataFrame(rmse_day_1,columns=['tmp_02','dpt_02','rh_02','mr_02','sknt_02'])
#        rmse_day_1.insert(0,'Date',dte)  
#
#        bias_day1.append(bias_day_1) ;     rmse_day1.append(rmse_day_1) ####### daily bias day wise appended day1
#        gfs_ae.append(gfs_rse_1) ;  min_tmp_ae.append(min_tmp_ae1)
#########################################                
#        bias_hour_day1.append(mod_dom_2_bias_1)  ####### hourly bias appended for each day day1
#        mod_hour_day1.append(mod_dom_2_2.iloc[7:31,1:]) ;  obs_hour_day1.append(obs_3.iloc[7:31,1:])
#####################################     
#        bias_day_2=np.vstack(mod_dom_2_bias_2.mean(axis=0).values).T
#
#        bias_day_2=pd.DataFrame(bias_day_2,columns=['tmp_02','dpt_02','rh_02','mr_02','sknt_02'])
#
#        bias_day_2.insert(0,'Date',dte)
#
#        rmse_day_2=np.vstack(mod_dom_2_rmse_2.values).T 
#        
#        rmse_day_2=pd.DataFrame(rmse_day_2,columns=['tmp_02','dpt_02','rh_02','mr_02','sknt_02'])
#
#        rmse_day_2.insert(0,'Date',dte)
#
#        bias_day2.append(bias_day_2) ; rmse_day2.append(rmse_day_2) ## daily bias day wise appended day2
####################################
#        bias_hour_day2.append(mod_dom_2_bias_2)  ## hourly bias appended for each day day2
#        mod_hour_day2.append(mod_dom_2_2.iloc[31:55,1:]) ;  obs_hour_day2.append(obs_3.iloc[31:55,1:])
##################################
    else:
       print dte 
       print("No Data Exist")

#############################################################################################################################
################################### hourly Analysis ######################################################################### 
#bias_hour_day_1=pd.concat(bias_hour_day1,axis=0) ; bias_hour_day_2=pd.concat(bias_hour_day2,axis=0)
#mod_hour_day_1=pd.concat(mod_hour_day1,axis=0)   ; mod_hour_day_2=pd.concat(mod_hour_day2,axis=0)   ;
#obs_hour_day_1=pd.concat(obs_hour_day1,axis=0)   ; mod_hour_day_2=pd.concat(mod_hour_day2,axis=0)   ;

#rmse_day1=pd.concat(rmse_day1,axis=0) ; gfs_ae1=pd.concat(gfs_ae,axis=0)
#ae_mint_day1=pd.concat(min_tmp_ae,axis=0)