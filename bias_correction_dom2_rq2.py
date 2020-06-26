import sys ;import os ; import numpy as np ; import pandas as pd ; import datetime as dt ; from dateutil import rrule ; import pytz
import metpy.calc as mcalc ; from metpy.units import units

main='/home/vkvalappil/Data/oppModel/' ; output=main+'/output/' ; inp=main+'l_dnn/model/'
date=str(sys.argv[1])
########################################################################################################################################

date_tdy=(dt.datetime.strptime(date,'%Y%m%d%H')+dt.timedelta(days=0)).strftime('%Y%m%d%H') ; 

file_tdy=inp+'hourly'+date_tdy+'.csv' ; 

if (os.path.isfile(file_tdy)):
    
        mod_dom_tdy=pd.read_csv(file_tdy) ; mod_dom_tdy=mod_dom_tdy.iloc[7:31,1:]        
        
        mod_dom_tdy['localTime']=mod_dom_tdy['localTime'].apply(pd.to_datetime, errors='ignore')    
        mod_dom_tdy.iloc[:,3:]=mod_dom_tdy.iloc[:,3:].apply(pd.to_numeric,errors='coerce')
        mod_dom_tdy_1=mod_dom_tdy.iloc[:,2:]
        mod_dom_tdy_1.index=mod_dom_tdy_1.localTime
        mod_dom_tdy_1.index=mod_dom_tdy_1.index.tz_localize(pytz.timezone('Asia/Dubai')).tz_convert(pytz.utc)
        mod_dom_tdy_1['localTime']=mod_dom_tdy_1.index

#######################################################################################################################################
        no_lag_days=4 ;        
        dte_1=dt.datetime.strptime(date_tdy,'%Y%m%d%H')+dt.timedelta(days=-no_lag_days) ; 
        dte_2=dt.datetime.strptime(date_tdy,'%Y%m%d%H')+dt.timedelta(days=-1) ;     
        
        date_req_list=[x.strftime('%Y%m%d%H') for x in rrule.rrule(rrule.DAILY,dtstart=dte_1,until=dte_2)]
        
        obs_mod=[] ; bias_day1 = [] ; rmse_day1 = [] ;
        
        for dte1 in date_req_list[:]:

            file_2=inp+'/hourly'+dte1+'.csv'

            if (os.path.isfile(file_2)):
                
                mod_dom_2=pd.read_csv(file_2) ; mod_dom_2=mod_dom_2.iloc[6:30,:] 
                mod_dom_2['localTime']=mod_dom_2['localTime'].apply(pd.to_datetime, errors='ignore')    
                mod_dom_2.iloc[:,4:]=mod_dom_2.iloc[:,4:].apply(pd.to_numeric,errors='coerce')
                mod_dom_2_1=mod_dom_2.iloc[:,3:]
                mod_dom_2_1.index=mod_dom_2_1.localTime
                mod_dom_2_1.index=mod_dom_2_1.index.tz_localize(pytz.timezone('Asia/Dubai')).tz_convert(pytz.utc)
                mod_dom_2_1['localTime']=mod_dom_2_1.index
 
                o_date_1=(dt.datetime.strptime(dte1,'%Y%m%d%H')+dt.timedelta(days=1)).strftime('%Y%m%d%H')               
                date_1=dt.datetime.strptime(dte1,'%Y%m%d%H')+dt.timedelta(hours=6) ; date_2=dt.datetime.strptime(o_date_1,'%Y%m%d%H')+dt.timedelta(hours=05)
                date_list=[x.strftime('%Y-%m-%d %H:%M') for x in rrule.rrule(rrule.HOURLY,dtstart=date_1,until=date_2)]

                obs_file_1=main+'l_dnn/obs/AbuDhabi_surf_'+o_date_1[0:8]+'.csv'   ;        
                metar_data=pd.read_csv(obs_file_1) ; 
                metar_data=metar_data[['STN','TIME','ALTM','TMP','DEW','RH','DIR','SPD','VIS']]
                metar_data=metar_data.drop(metar_data.index[0])
                metar_data['TIME']=date_list
                metar_data['TIME']=metar_data['TIME'].apply(pd.to_datetime,errors='ignore')
                metar_data.iloc[:,3:]=metar_data.iloc[:,3:].apply(pd.to_numeric,errors='coerce')

                tmp=np.array((metar_data.iloc[:,3]).apply(pd.to_numeric,errors='coerce')+273.15)*units('K')
                rh=np.array((metar_data.iloc[:,5]).apply(pd.to_numeric,errors='coerce')/100.0)
                press=np.array((metar_data.iloc[:,2]).apply(pd.to_numeric,errors='coerce'))*units('millibar')

                mrio=mcalc.mixing_ratio_from_relative_humidity(rh,tmp,press)
                metar_data['mrio']=mrio
                obs=metar_data
                obs_1=obs.iloc[:,1:]
                obs_1.index=obs_1.TIME
                obs_1.index=obs_1.index.tz_localize(pytz.utc)

                idx = obs_1.index.intersection(mod_dom_2_1.index)
                obs_2=obs_1.loc[idx]
                obs_3=pd.concat([obs_2['TIME'],obs_2['TMP'],obs_2['DEW'],obs_2['RH'],obs_2['mrio'],obs_2['SPD']],axis=1)
                obs_3.columns=['TIME','OTMP','ODEW','ORH','OMRIO','OSPD']
                
                mod_dom_2_2=pd.concat([mod_dom_2_1['localTime'],mod_dom_2_1['TEMP'],mod_dom_2_1['DTEMP'],mod_dom_2_1['RH'],mod_dom_2_1['MXRATIO'],mod_dom_2_1['WDIR']*0.277],axis=1)    
                mod_dom_2_2.columns=['localTime','MTMP','MDEW','MRH','MMRIO','MSPD']
                
                mod_obs_1=pd.concat([mod_dom_2_2,obs_3],axis=1).iloc[0:31,:]    
                
                ##### Day 1
                mod_dom_2_bias_1=mod_dom_2_2.iloc[:,1:].sub(obs_3.iloc[:,1:].values,axis=0)  
                mean_bias_day1=mod_dom_2_bias_1.mean(axis=0)
                bias_day1.append(mean_bias_day1) 
                mod_dom_2_rmse_1=((mod_dom_2_bias_1**2).mean(axis=0))**0.5                               
                rmse_day1.append(mod_dom_2_rmse_1)                
                obs_mod.append(mod_obs_1)
            else :   
               print(dte1)
               print("No Data Exist")
 
        obs_mod=pd.concat(obs_mod,axis=0) ; rmse_day1=pd.concat(rmse_day1,axis=0) ;  bias_day1=pd.concat(bias_day1,axis=0)                   
##################################################################################################################################################################################################        
       ########## Correction for Temperature #######################################################
        rmse_tmp =rmse_day1['MTMP'] ; bias_tmp=bias_day1['MTMP']
            
        if ((np.sign(bias_tmp) < 0 ).all()) | ((np.sign(bias_tmp) > 0 ).all()):
            print("Mean bias for all days have same sign" )
               
            hourt = pd.to_timedelta(obs_mod.index.hour,  unit='H')
            obs_mod_hour_sum_day1=obs_mod.groupby(hourt).sum() 
            a = obs_mod_hour_sum_day1.values ; 
        
            obs_mod_q = a[:,5:]/a[:,0:5].astype(str).astype(float)
            obs_mod_q=pd.DataFrame(obs_mod_q)
            obs_mod_q.columns=obs_mod_hour_sum_day1.columns[0:5]
            obs_mod_q.index=obs_mod_hour_sum_day1.index        
        
            obs_mod_q_1=pd.concat([obs_mod_q.iloc[13:],obs_mod_q[0:13]],axis=0)         
            cor_tmp=mod_dom_tdy_1.TEMP*obs_mod_q_1.MTMP.values
        
        else:
                      
            obs_mod_r=obs_mod.iloc[-24:] 
            hourt_r = pd.to_timedelta(obs_mod_r.index.hour,  unit='H')
            obs_mod_hour_sum_day1=obs_mod_r.groupby(hourt_r).sum() 
            a = obs_mod_hour_sum_day1.values ;

            obs_mod_q = a[:,5:]/a[:,0:5].astype(str).astype(float)
            obs_mod_q=pd.DataFrame(obs_mod_q)
            obs_mod_q.columns=obs_mod_hour_sum_day1.columns[0:5]
            obs_mod_q.index=obs_mod_hour_sum_day1.index        
        
            obs_mod_q_1=pd.concat([obs_mod_q.iloc[13:],obs_mod_q[0:13]],axis=0)         
            cor_tmp=mod_dom_tdy_1.TEMP*obs_mod_q_1.MTMP.values

      #################################################################################################### 

        rmse_rh =rmse_day1['MRH'] ; bias_rh=bias_day1['MRH']
            
        if ((np.sign(bias_rh) < 0 ).all()) | ((np.sign(bias_rh) > 0 ).all()):
            print("Mean bias for all days have same sign" )
               
            hourt = pd.to_timedelta(obs_mod.index.hour,  unit='H')
            obs_mod_hour_sum_day1=obs_mod.groupby(hourt).sum() 
            a = obs_mod_hour_sum_day1.values ; 
        
            obs_mod_q = a[:,5:]/a[:,0:5].astype(str).astype(float)
            obs_mod_q=pd.DataFrame(obs_mod_q)
            obs_mod_q.columns=obs_mod_hour_sum_day1.columns[0:5]
            obs_mod_q.index=obs_mod_hour_sum_day1.index        
        
            obs_mod_q_1=pd.concat([obs_mod_q.iloc[13:],obs_mod_q[0:13]],axis=0)         
            cor_rh=mod_dom_tdy_1.RH*obs_mod_q_1.MRH.values
            cor_dtmp=mod_dom_tdy_1.DTEMP*obs_mod_q_1.MDEW.values
            cor_mr=mod_dom_tdy_1.MXRATIO*obs_mod_q_1.MMRIO.values
        
        else:
                      
            obs_mod_r=obs_mod.iloc[-24:] 
            hourt_r = pd.to_timedelta(obs_mod_r.index.hour,  unit='H')
            obs_mod_hour_sum_day1=obs_mod_r.groupby(hourt_r).sum() 
            a = obs_mod_hour_sum_day1.values ;

            obs_mod_q = a[:,5:]/a[:,0:5].astype(str).astype(float)
            obs_mod_q=pd.DataFrame(obs_mod_q)
            obs_mod_q.columns=obs_mod_hour_sum_day1.columns[0:5]
            obs_mod_q.index=obs_mod_hour_sum_day1.index        
        
            obs_mod_q_1=pd.concat([obs_mod_q.iloc[13:],obs_mod_q[0:13]],axis=0)               
            cor_rh=mod_dom_tdy_1.RH*obs_mod_q_1.MRH.values
            cor_dtmp=mod_dom_tdy_1.DTEMP*obs_mod_q_1.MDEW.values
            cor_mr=mod_dom_tdy_1.MXRATIO*obs_mod_q_1.MMRIO.values
         
      #################################################################################################### 


        rmse_ws =rmse_day1['MSPD'] ; bias_ws=bias_day1['MSPD']
            
        if ((np.sign(bias_ws) < 0 ).all()) | ((np.sign(bias_ws) > 0 ).all()):
            print("Mean bias for all days have same sign" )
               
            hourt = pd.to_timedelta(obs_mod.index.hour,  unit='H')
            obs_mod_hour_sum_day1=obs_mod.groupby(hourt).sum() 
            a = obs_mod_hour_sum_day1.values ; 
        
            obs_mod_q=a[:,5:]/a[:,0:5].astype(str).astype(float)
            obs_mod_q=pd.DataFrame(obs_mod_q)
            obs_mod_q.columns=obs_mod_hour_sum_day1.columns[0:5]
            obs_mod_q.index=obs_mod_hour_sum_day1.index        
        
            obs_mod_q_1=pd.concat([obs_mod_q.iloc[13:],obs_mod_q[0:13]],axis=0)         
            cor_spd=mod_dom_tdy_1.WSPD*obs_mod_q_1.MSPD.values
        
        else:
                      
            obs_mod_r=obs_mod.iloc[-24:] 
            hourt_r = pd.to_timedelta(obs_mod_r.index.hour,  unit='H')           
            obs_mod_hour_sum_day1=obs_mod_r.groupby(hourt_r).sum() 
            a = obs_mod_hour_sum_day1.values ;

            obs_mod_q = a[:,5:]/a[:,0:5].astype(str).astype(float)
            obs_mod_q=pd.DataFrame(obs_mod_q)
            obs_mod_q.columns=obs_mod_hour_sum_day1.columns[0:5]
            obs_mod_q.index=obs_mod_hour_sum_day1.index        
        
            obs_mod_q_1=pd.concat([obs_mod_q.iloc[13:],obs_mod_q[0:13]],axis=0)         
            cor_spd=mod_dom_tdy_1.WSPD*obs_mod_q_1.MSPD.values

      #################################################################################################### 

        mod_dom_tdy_1_new=mod_dom_tdy_1              ;   mod_dom_tdy_1_new.TEMP=np.round(cor_tmp)
        
        mod_dom_tdy_1_new.RH=np.round(cor_rh )       ;   mod_dom_tdy_1_new.DTEMP=np.round(cor_dtmp ) ;  

        mod_dom_tdy_1_new.MXRATIO=np.round(cor_mr )  ;  mod_dom_tdy_1_new.WSPD=np.round(cor_spd) ;  

#########################################################################################################################################


vis_ruc=np.empty((mod_dom_tdy_1_new.RH.shape)) ;
indx1=np.where(mod_dom_tdy_1_new.RH<50) ;  indx2=np.where((mod_dom_tdy_1_new.RH >=50) & (mod_dom_tdy_1_new.RH <=85)) ;
indx3=np.where((mod_dom_tdy_1_new.RH >85) & (mod_dom_tdy_1_new.RH <=100))

#indx1=rh.where(rh<50) ; indx2=rh.where((rh>=50) & (rh <=85)) ; indx3=rh.where((rh>85) & (rh <=100))

vis_ruc[indx1]=60*np.exp(-2.5*((mod_dom_tdy_1_new.RH.iloc[indx1]-15)/80))
vis_ruc[indx2]=50*np.exp(-2.5*((mod_dom_tdy_1_new.RH.iloc[indx2]-10)/85))
vis_ruc[indx3]=50*np.exp(-2.5*((mod_dom_tdy_1_new.RH.iloc[indx3]-15)/85))
vis_ruc[vis_ruc>10]=10
mod_dom_tdy_1_new.insert(11,'Visibility',vis_ruc)

mod_dom_tdy_1_new.to_csv('/home/vkvalappil/Data/oppModel/output/cor_fcst/cor_hourly'+date+'.csv',index=False)

###########################################################################################################################################



        
        
        




