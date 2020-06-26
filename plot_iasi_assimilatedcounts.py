import pandas as pd ; import pytz



def plot_bars(C):
    
    import matplotlib.pyplot as plt; from pylab import savefig ; import numpy as np

    fig=plt.figure(figsize=(20,12),dpi=50,frameon=True); ax = fig.add_subplot(111,facecolor='white')

    pos_1  = np.arange(0,10*C.shape[0],10) ;   wdh=2.5 ; opacity = 1.0 ; #ax.hold(True);

    ax.bar(pos_1+wdh,C['ma'],width=wdh, color='b',alpha=opacity,label='ma')
    ax.bar(pos_1+2*wdh ,C['mb'],width=wdh, color='r',alpha=opacity,label='mb')
    #ax.bar(pos_1+3*wdh ,C['Ass_3'],width=wdh, color='g',alpha=opacity,label='Assimilated')


    ax.axhline(150000, color="k")

    plt.tick_params(axis='y',which='both', left='on',  right='off', labelright='off')
    ax.tick_params(axis='y', colors='black',labelsize=16) ; ax.yaxis.set_ticks_position('left') ;

    #ax.set_yticks(np.arange(-10,45,10)) ; 
    ax.set_ylabel('Count',color='black',fontsize=16,fontweight='bold') ;
    ax.set_xlabel('DATE: UTC ',color='black',fontsize=16,fontweight='bold') ;

    ax.set_xticks(np.arange(0,10*C.shape[0],10)+2*wdh) ; 
    xTickMarks=C.index #pd.TimedeltaIndex.to_series(C.index).dt.components.hours 
    xtickNames =ax.set_xticklabels(xTickMarks,rotation=90,fontsize=14,fontweight='bold')

    #plt.tick_params(axis='x',which='both', bottom='off',  top='off', labelbottom='off')
#    for n, row in enumerate(C.iloc[:]):
#        plt.text(4*n, row, np.round(row,3), ha='center', rotation=0, va='bottom',fontsize=16,fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    display = (0,1,2,3,4)

    leg=ax.legend(loc='lower right',fontsize=18)

    plt.tight_layout(pad=3) ;

    plt.title('Iasi Level C radiance: metop-B Count over Doamin',fontsize=16)
    #outFile=output+'stat/corrected/dom2_'+parm1+'_'+metric+'_'+ds+'.png' 
    outFile='/home/vkvalappil/Data/modelWRF/GSI/ARW/output/surfaceLevel/gsi_be/ass_ma_mb.png' 
    savefig(outFile);
    plt.close(fig) ; fig.clf(fig)
    


metop_a=pd.read_csv('/home/vkvalappil/Data/modelWRF/GSI/ARW/output/surfaceLevel/gsi_be/metop-a.csv') 
metop_b=pd.read_csv('/home/vkvalappil/Data/modelWRF/GSI/ARW/output/surfaceLevel/gsi_be/metop-b.csv') 

m_a_1=metop_a
m_a_1['Date']=m_a_1['Date'].apply(pd.to_datetime, errors='ignore',format='%Y%m%d%H')          
m_a_1.iloc[:,1:]=m_a_1.iloc[:,1:].apply(pd.to_numeric,errors='coerce')     
m_a_1.index=m_a_1.Date
m_a_1.index=m_a_1.index.tz_localize(pytz.utc)
m_a_1['Date']=m_a_1.index

m_b_1=metop_b
m_b_1['Date']=m_b_1['Date'].apply(pd.to_datetime, errors='ignore',format='%Y%m%d%H')          
m_b_1.iloc[:,1:]=m_b_1.iloc[:,1:].apply(pd.to_numeric,errors='coerce')     
m_b_1.index=m_b_1.Date
m_b_1.index=m_b_1.index.tz_localize(pytz.utc)
m_b_1['Date']=m_b_1.index


a_p=pd.concat([m_a_1['Date'],m_a_1['Read_3'],m_a_1['Keep_3'],m_a_1['Ass_3']],axis=1)

plot_bars(a_p)

a_p=pd.concat([m_a_1['Date'],m_a_1['Ass_3'],m_b_1['Ass_3']],axis=1)
a_p.columns=['Date','ma','mb']
plot_bars(a_p)


































