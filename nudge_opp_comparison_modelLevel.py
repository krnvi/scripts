import sys ;import os ; import numpy as np ; import pandas as pd ; import datetime as dt ; from dateutil import tz, rrule ; import pytz
import metpy.calc as mcalc ; from metpy.units import units


############################################################################################################################################

main='/home/vkvalappil/Data/modelWRF/nudging/'  ; output=main+'ARW/output/domain_02/modelLevel/stat/' ; inp=main+'ARW/output/domain_02/modelLevel'

date=str(sys.argv[1])

#date=str(sys.argv[1])
num_hours=24 ; fcs_leaddays=1 ; 

#############################################################################################################################################
def plot_line(C,parm,ds ):
    import matplotlib.pyplot as plt; from pylab import savefig ;  

    fig=plt.figure(figsize=(20,12),dpi=50,frameon=True); ax1 = fig.add_subplot(111,facecolor='white')

    pos=np.arange(0,C.shape[0],1) ; opacity = 1.0 ; #ax.hold(True);

    ax1.plot(C['obs'],pos,color='k',alpha=opacity,lw=4,linestyle='-',label='obs')
    ax1.plot(C['no_Da'],pos,color='red',alpha=opacity,lw=4,linestyle='-',label='Wrf no nudging')
    ax1.plot(C['Da'],pos,color='blue',alpha=opacity,lw=4,linestyle='-',label='Wrf nudging')

    plt.tick_params(axis='y',which='both', left='on',  right='off', labelright='off')
    ax1.tick_params(axis='y', colors='black',labelsize=16) ; ax1.yaxis.set_ticks_position('left') ;

    #ax.set_yticks(np.arange(-10,45,10)) ; 
    ax1.set_ylabel('Pressure (hPa)',color='black',fontsize=16,fontweight='bold') ;
    ax1.set_xlabel(parm,color='black',fontsize=16,fontweight='bold') ;

    ax1.set_yticks(np.arange(0,C.shape[0],1)[::3]) ; 
    yTickMarks=np.round(C.index[::3]) #pd.TimedeltaIndex.to_series(C.index).dt.components.hours 
    ytickNames =ax1.set_yticklabels(yTickMarks,rotation=0,fontsize=14,fontweight='bold')

    ax1.tick_params(axis='x', colors='black',size=14)
    #ax1.set_xticks(np.arange(min(mr_ng),max(mr_ng),0.002))
    #xtickMarks=
    #ax1.set_xticklabels(yTickMarks,rotation=0,fontsize=14,fontweight='bold')

    #plt.tick_params(axis='x',which='both', bottom='off',  top='off', labelbottom='off')
#    for n, row in enumerate(C.iloc[:]):
#        plt.text(4*n, row, np.round(row,3), ha='center', rotation=0, va='bottom',fontsize=16,fontweight='bold')

    handles, labels = ax1.get_legend_handles_labels()
    display = (0,1,2,3,4)

    leg=ax1.legend(loc='lower center',fontsize=18)

    plt.tight_layout(pad=3) ; plt.title(parm,fontsize=16)
    
    if not os.path.exists(output+date):
       os.makedirs(output+date)

    outFile=output+date+'/'+parm+'_'+ds+'.png' 
    savefig(outFile);
    plt.close(fig) ; fig.clf(fig) 
#############################################################################################################################################
def plot_line_m(C,parm,ds ):
    import matplotlib.pyplot as plt; from pylab import savefig ;  

    fig=plt.figure(figsize=(20,12),dpi=50,frameon=True); ax1 = fig.add_subplot(111,facecolor='white')

    pos=np.arange(0,C.shape[0],1) ; opacity = 1.0 ; #ax.hold(True);

    #ax1.plot(C['obs'],pos,color='k',alpha=opacity,lw=4,linestyle='-',label='obs')
    ax1.plot(C['no_Da'],pos,color='red',alpha=opacity,lw=4,linestyle='-',label='Wrf no nudging')
    ax1.plot(C['Da'],pos,color='blue',alpha=opacity,lw=4,linestyle='-',label='Wrf nudging')

    plt.tick_params(axis='y',which='both', left='on',  right='off', labelright='off')
    ax1.tick_params(axis='y', colors='black',labelsize=16) ; ax1.yaxis.set_ticks_position('left') ;

    #ax.set_yticks(np.arange(-10,45,10)) ; 
    ax1.set_ylabel('Pressure (hPa)',color='black',fontsize=16,fontweight='bold') ;
    ax1.set_xlabel(parm,color='black',fontsize=16,fontweight='bold') ;

    ax1.set_yticks(np.arange(0,C.shape[0],1)[::3]) ; 
    yTickMarks=np.round(C.index[::3]) #pd.TimedeltaIndex.to_series(C.index).dt.components.hours 
    ytickNames =ax1.set_yticklabels(yTickMarks,rotation=0,fontsize=14,fontweight='bold')

    ax1.tick_params(axis='x', colors='black',size=14)
    #ax1.set_xticks(np.arange(min(mr_ng),max(mr_ng),0.002))
    #xtickMarks=
    #ax1.set_xticklabels(yTickMarks,rotation=0,fontsize=14,fontweight='bold')

    #plt.tick_params(axis='x',which='both', bottom='off',  top='off', labelbottom='off')
#    for n, row in enumerate(C.iloc[:]):
#        plt.text(4*n, row, np.round(row,3), ha='center', rotation=0, va='bottom',fontsize=16,fontweight='bold')

    handles, labels = ax1.get_legend_handles_labels()
    display = (0,1,2,3,4)

    leg=ax1.legend(loc='lower center',fontsize=18)

    plt.tight_layout(pad=3) ; plt.title(parm,fontsize=16)
    
    if not os.path.exists(output+date):
       os.makedirs(output+date)

    outFile=output+date+'/'+parm+'_'+ds+'.png' 
    savefig(outFile);
    plt.close(fig) ; fig.clf(fig) 
###########################################################################################################################################

fcs_st_date=dt.datetime.strptime(date,'%Y%m%d%H') ; fcs_ed_date=(dt.datetime.strptime(date,'%Y%m%d%H')+dt.timedelta(hours=num_hours));
file_date_list=[x.strftime('%Y%m%d%H') for x in rrule.rrule(rrule.HOURLY,dtstart=fcs_st_date,until=fcs_ed_date)] #[::3][0:-6] ; 


for ii in range(0,len(file_date_list[:])):

    file_1=inp+'/nudge_'+date+'/modelLevel_'+file_date_list[ii]+'.csv' ;
    
    file_2=inp+'/no_nudge_'+date+'/modelLevel_'+file_date_list[ii]+'.csv' ;

    if (os.path.isfile(file_1)) & (os.path.isfile(file_2)):
    
        mod_da_1=pd.read_csv(file_1) ; mod_da_1=mod_da_1.loc[mod_da_1['ID'] == 2]   
        mod_da_2=pd.read_csv(file_2) ; mod_da_2=mod_da_2.loc[mod_da_2['ID'] == 2]   
    
        mod_da_1['Date(UTC)']=mod_da_1['Date(UTC)'].apply(pd.to_datetime, errors='ignore',format='%Y%m%d%H')          
        mod_da_1.iloc[:,3:]=mod_da_1.iloc[:,3:].apply(pd.to_numeric,errors='coerce')     
        mod_da_1.index=mod_da_1['Date(UTC)'] 
        mod_da_1.index=mod_da_1.index.tz_localize(pytz.utc)
        mod_da_1['Date(UTC)']=mod_da_1.index       
        mod_da_1['pressure']=mod_da_1['pressure']/100.0
    
        mod_da_2['Date(UTC)']=mod_da_2['Date(UTC)'].apply(pd.to_datetime, errors='ignore',format='%Y%m%d%H')          
        mod_da_2.iloc[:,3:]=mod_da_2.iloc[:,3:].apply(pd.to_numeric,errors='coerce')     
        mod_da_2.index=mod_da_2['Date(UTC)'] 
        mod_da_2.index=mod_da_2.index.tz_localize(pytz.utc)
        mod_da_2['Date(UTC)']=mod_da_2.index       
        mod_da_2['pressure']=mod_da_2['pressure']/100.0
   
############################################################################################################################################    
        obs_file_1='/home/vkvalappil/Data/masdar_station_data/wyoming/'+file_date_list[ii][0:6]+'/AbuDhabi_upperair_'+file_date_list[ii]+'.csv'
 
        if (os.path.isfile(obs_file_1)):
            
            obs_1=pd.read_csv(obs_file_1) ;obs=(obs_1.drop(obs_1.index[0])).reset_index()
            obs=obs.apply(pd.to_numeric,errors='coerce')
        
############################################################################################################################       
            x_intp=obs.PRES.values * units.hPa ;  x_i=mod_da_1.pressure.values * units.hPa ; 
            
        
            tmp_i=mod_da_1.temperature.values ;   rh_i=mod_da_1['rel.Humidity'].values
            mr_i=mod_da_1.mixRatio.values     ;   ws_i=mod_da_1.wspd.values
        
            mod_da_1_temp=mcalc.log_interp(x_intp,x_i,tmp_i)  ;  mod_da_1_rh=mcalc.log_interp(x_intp,x_i,rh_i)        
            mod_da_1_mr=mcalc.log_interp(x_intp,x_i,mr_i)    ;  mod_da_1_ws=mcalc.log_interp(x_intp,x_i,ws_i)        
        
##########################################################################################################################        
            x_i_2=mod_da_2.pressure.values * units.hPa ; 
                
            tmp_i_2=mod_da_2.temperature.values ;   rh_i_2=mod_da_2['rel.Humidity'].values
            mr_i_2=mod_da_2.mixRatio.values     ;   ws_i_2=mod_da_2.wspd.values
        
            mod_da_2_temp=mcalc.log_interp(x_intp,x_i,tmp_i_2)  ;  mod_da_2_rh=mcalc.log_interp(x_intp,x_i_2,rh_i_2)        
            mod_da_2_mr=mcalc.log_interp(x_intp,x_i,mr_i_2)    ;  mod_da_2_ws=mcalc.log_interp(x_intp,x_i_2,ws_i_2)        
                
#############################################################################################################################        
        
            tmp_=pd.concat([obs['PRES'],obs['TEMP']+273.15,pd.DataFrame(mod_da_1_temp), pd.DataFrame(mod_da_2_temp)],axis=1)    
            tmp_.columns=['PRES','obs','Da','no_Da']
            tmp_.index=tmp_.PRES

            plot_line(tmp_.iloc[0:,:],'Temperature', file_date_list[ii]) 
            
            
            rh_=pd.concat([obs['PRES'],obs['RELH'],pd.DataFrame(mod_da_1_rh), pd.DataFrame(mod_da_2_rh)],axis=1)    
            rh_.columns=['PRES','obs','Da','no_Da']
            rh_.index=rh_.PRES

            plot_line(rh_.iloc[0:,:],'Relative_humidity', file_date_list[ii]) 
 
            mr_=pd.concat([obs['PRES'],obs['MIXR']/1000.0,pd.DataFrame(mod_da_1_mr), pd.DataFrame(mod_da_2_mr)],axis=1)    
            mr_.columns=['PRES','obs','Da','no_Da']
            mr_.index=mr_.PRES

            plot_line(mr_.iloc[0:,:],'Mixing_Ratio', file_date_list[ii]) 
 
            ws_=pd.concat([obs['PRES'],obs['SKNT']*0.514,pd.DataFrame(mod_da_1_ws), pd.DataFrame(mod_da_2_ws)],axis=1)    
            ws_.columns=['PRES','obs','Da','no_Da']
            ws_.index=mr_.PRES
 
            plot_line(ws_.iloc[0:,:],'Wind_Speed', file_date_list[ii]) 
        
#################################################################################################################################        
        
        else:
        
            tmp_=pd.concat([mod_da_1['pressure'],mod_da_1['temperature'],mod_da_2['temperature']],axis=1)    
            tmp_.columns=['PRES','Da','no_Da']
            tmp_.index=tmp_.PRES 
 
            plot_line_m(tmp_.iloc[0:,:],'Temperature', file_date_list[ii]) 
           
            rh_=pd.concat([mod_da_1['pressure'],mod_da_1['rel.Humidity'],mod_da_2['rel.Humidity']],axis=1)    
            rh_.columns=['PRES','Da','no_Da']
            rh_.index=rh_.PRES 

            plot_line_m(rh_.iloc[0:,:],'Relative_humidity', file_date_list[ii]) 

            mr_=pd.concat([mod_da_1['pressure'],mod_da_1['mixRatio'],mod_da_2['mixRatio']],axis=1)    
            mr_.columns=['PRES','Da','no_Da']
            mr_.index=mr_.PRES 

            plot_line_m(mr_.iloc[0:,:],'Mixing_Ratio', file_date_list[ii]) 

            ws_=pd.concat([mod_da_1['pressure'],mod_da_1['wspd'],mod_da_2['wspd']],axis=1)    
            ws_.columns=['PRES','Da','no_Da']
            ws_.index=ws_.PRES 

            plot_line_m(ws_.iloc[0:,:],'Wind_Speed', file_date_list[ii]) 
 
            qcld_=pd.concat([mod_da_1['pressure'],mod_da_1['qcloud'],mod_da_2['qcloud']],axis=1)    
            qcld_.columns=['PRES','Da','no_Da']
            qcld_.index=qcld_.PRES 
 
            plot_line_m(qcld_.iloc[0:,:],'Qcloud', file_date_list[ii]) 
 
       
    else:        
       print file_date_list[ii] 
       print("No Data Exist")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        