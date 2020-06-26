
import sys ; import urllib2 as urll2 ; import calendar as cl ; import pandas as pd ; import pytaf ; import datetime as dt ;

#date='20180701'  ; 

date=str(sys.argv[1])

no_days=cl.monthrange(int(date[0:4]),int(date[4:6]))[1]

url='https://www.ogimet.com/display_metars2.php?lang=en&lugar=omaa&tipo=FT&ord=REV&nil=SI&fmt=html&ano='+date[0:4]+'&mes='+date[4:6]+'&day=01&hora=00&anof='+date[0:4]+'&mesf='+date[4:6]+'&dayf='+str(no_days)+'&horaf=23&minf=59&send=send'

out_file='/home/vkvalappil/Data/TAF_decode/ogimet/taf_weatehr_'+date[0:6]+'.csv'

web=urll2.Request(url) ; htm = urll2.urlopen(web).read() ;  

cont=pd.read_html(htm) ;  cont1=cont[2]

taf_wt=[]
for taf in cont1.iloc[:,2][::-1] :
    try:
        taf_dc=pytaf.TAF(taf) 

        decoder = pytaf.Decoder(taf_dc)
        #A=decoder.decode_taf()
        for group in taf_dc.get_groups() :
            if group["weather"]:           
                header=taf_dc.get_header()
                or_day=header['origin_date'] ; or_hr=header['origin_hours'] ;  or_min=header['origin_minutes']
                or_date=date[0:6]+or_day+or_hr+or_min
            
                if or_day==str(no_days):
                    v_date=(dt.datetime.strptime(or_date ,'%Y%m%d%H%M')+dt.timedelta(days=1)).strftime('%Y%m')
                else:
                    v_date=date[0:6]
                
                v_f_day=header['valid_from_date'] ;  v_f_hr=header['valid_from_hours'] ; 
                v_f_date=v_date+v_f_day+v_f_hr             
            
                v_t_day=header['valid_till_date'] ;  v_t_hr=header['valid_till_hours'] ;           
                v_t_date=v_date+v_t_day+v_t_hr             
                
                W=decoder._decode_weather(group["weather"])
            
                #taf_list=[or_date,date[0:6],or_day,or_hr,or_min,v_f_day,v_f_hr,v_t_day,v_t_hr,W]
                taf_list=[or_date,v_f_date,v_t_date,W]           
                taf_wt.append(taf_list) 
    except:
        pass
 
taf_wt1=pd.DataFrame(taf_wt)
taf_wt1.columns=['Date','ValidFrom','ValidTill','Weather']
taf_wt1.to_csv(out_file,index=False)

#taf_dc.get_maintenance() ; taf_dc.get_taf() ; taf_dc._weather_groups  ; A=taf_dc._raw_weather_groups
