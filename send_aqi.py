#!/usr/bin/python

import sys ; import pandas as pd ; 


def sendmail(file_taf,email_list) :
           from smtplib import SMTP ; from smtplib import SMTPException ; 
           from email.mime.multipart import MIMEMultipart ; from email.mime.text import MIMEText      

           _from         =   "fog@masdar.ac.ae" ;
           _to           =   email_list  ;  
           _sub          =   "AQI (WRF CESAM LAB)  "
           _attach       =   file_taf
           #_content      =   "WRF TAF DATA"
           _username     =   'fog' ;
           _password     =   'P@ssword321'
           _smtp         =   "mail.masdar.ac.ae:587" ;
           #_text_subtype = "plain"

           mail=MIMEMultipart()
           mail["Subject"]  =  _sub
           mail["From"]     =  _from
           mail["To"]       =  ','.join(_to) 
           #body = MIMEMultipart('alternative')
           #body.attach(MIMEText(_content, _text_subtype ))
           mail.attach(MIMEText(file(_attach).read()))
           try:
               smtpObj = SMTP(_smtp)
               #Identify yourself to GMAIL ESMTP server.
               smtpObj.ehlo()
               #Put SMTP connection in TLS mode and call ehlo again.
               smtpObj.starttls()
               smtpObj.ehlo()
               #Login to service
               smtpObj.login(user=_username, password=_password)
               #Send email
               smtpObj.sendmail(_from, _to, mail.as_string())
               #close connection and session.
               smtpObj.quit()
           except SMTPException as error:
               print "Using Gmail : Error: unable to send email :  {err}".format(err=error)   
               try:
                   _from         =   "fog.masdar@gmail.com" ;
                   _to           =   email_list;
                   _sub          =   "TAF (WRF CESAM LAB)  "
                   _attach       =   file_taf
                   #_content      =   "WRF TAF DATA"
                   _username     =   'fog.masdar' ;
                   _password     =   'fog@masdar123'
                   _smtp         =   "smtp.gmail.com:587" ;
                   #_text_subtype = "plain"
                   
                   mail=MIMEMultipart()
                   mail["Subject"]  =  _sub
                   mail["From"]     =  _from
                   mail["To"]       =  ','.join(_to) 
                   mail.attach(MIMEText(file(_attach).read()))
                   smtpObj = SMTP(_smtp)
                   #Identify yourself to GMAIL ESMTP server.
                   smtpObj.ehlo()
                   #Put SMTP connection in TLS mode and call ehlo again.
                   smtpObj.starttls()
                   smtpObj.ehlo()
                   #Login to service
                   smtpObj.login(user=_username, password=_password)
                   #Send email
                   smtpObj.sendmail(_from, _to, mail.as_string())
                   #close connection and session.
                   smtpObj.quit()

               except:
				  print 'error'
				  return "Error";
      
      
#####################################################################################################################################      
date='2018060300'      
aq_index_file='/home/vkvalappil/Data/workspace/pythonScripts/AQI_'+date+'.csv'      
email_id=sys.argv[1].split(',')
          
sendmail(aq_index_file,email_id)      
      
      
      
      
      
      
      
      