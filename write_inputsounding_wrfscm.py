import numpy as np ; import pandas as pd ; import metpy.calc as mpcalc ; from metpy.units import units 

date='2017110100'
data=pd.read_csv('~/Data/masdar_station_data/wyoming/metar+rs/201711/AbuDhabi_upperair_'+date+'.csv')
data=data.apply(pd.to_numeric,errors='coerce')  

#pd.set_option('display.float_format', '{:.2E}'.format) 
#new_data=pd.concat([data['PRES'][1::][::-1],data['TEMP'][1::][::-1]+273.15,data['MIXR'][1::][::-1]/1000,data['MIXR'][1::][::-1]*0,(data['MIXR'][1::][::-1]*0).astype(int) ],axis=1 )
#new_data.to_csv('/home/vkvalappil/Data/modelWRF/LES/UCLALES-SALSA/bin/datafiles/dsrt.lay',sep=',',index=False,float_format='%.3E')  

wind_speed = (data['SKNT'] *0.514).values * units('m/s')
wind_dir = data['DRCT'].values * units.deg 

#data['u_wind'], data['v_wind'] = mpcalc.get_wind_components(data['SKNT'][1::]*0.51,np.deg2rad(data['DRCT'][1::]))
data['u_wind'], data['v_wind']  = mpcalc.get_wind_components(wind_speed,wind_dir)
data['TEMP']=data['TEMP']+273.15 ; data['PRES']=data['PRES']*100
data['THTA'].iloc[1]=data['TEMP'].iloc[1]
#u, v = mpcalc.get_wind_components(data['SKNT'][1::], data['DRCT'][1::])

surf_data=pd.DataFrame([[data['HGHT'].iloc[1],data['u_wind'].iloc[1],data['v_wind'].iloc[1],data['TEMP'].iloc[1],data['MIXR'].iloc[1],data['PRES'].iloc[1]]])
surf_data.columns=['HGHT','u_wind','v_wind','THTA','MIXR','PRES']
prof_data=pd.concat([data['HGHT'].iloc[2:],data['u_wind'].iloc[2:],data['v_wind'].iloc[2:],data['THTA'].iloc[2:],data['MIXR'].iloc[2:],data['PRES'].iloc[2:]],axis=1)

new_data=pd.concat([surf_data,prof_data],axis=0)
new_data.PRES.iloc[1:]=''  
new_data.MIXR=new_data.MIXR/1000
new_data.HGHT.iloc[0]=0
                                                  
new_data.to_csv('/home/vkvalappil/Data/modelWRF/WRF_SCM/input_sounding_'+date+'.txt',sep=' ',index=False) ; #float_format='%.3E'  

