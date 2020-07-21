import pandas as pd
import numpy as np
from datetime import timedelta 
from math import sqrt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def compute_DD(feature, method, config, paras, alpha, datasets):
    """
    Compute detection delay
    Args:
        feature: feature name (speed_ratio/mean/rms/energy)
        method: methods proposed (known: known post feature distributions, ML: ML estimate, MAP: MAP estimate)
        config: intilaization configurations
        alpha: thresholds on loglikelihood ratio
        paras: feature parameters
        datasets: required datasets
    Return: 
        datasets: detection delays using 4 schemes proposed and for each threshold (alpha) seperately.
    """  
    #load datasets   
    speed_data = datasets['speed_data']
    accidents = datasets['north_accidents']
    avg_speed = datasets['avg_speed']
    sens_accu = datasets['sensor_accuaracy']
    
    
    ## extract speed data from north lane
    speed_data = speed_data[speed_data['DIRECTION'] == 0]
    ## change time format to standard 
    avg_speed['TIME'] = pd.to_datetime(avg_speed['TIME'],format='%H:%M:%S')
    speed_data = assign_time(speed_data,'TIME','%Y-%m-%d:%H:%M') 
    
    dd_ff = [[[] for _ in range(len(alpha))] for _ in range(4)]
    event_id= accidents[['EVENT_ID_NEW']].drop_duplicates()
    
    # load initilization configurations
    pi = config['pi']
    rho = config['rho']
    
    ## load feature paramters
    # pre distribution paras
    mu0 = paras['mu0']
    sigma  = paras['sigma0']
                            
    for k in range(len(event_id)):
        event_data = accidents[accidents['EVENT_ID_NEW'] == event_id.EVENT_ID_NEW.iloc[k]]
        
        # extract sensors located close and upstream from the accident location
        event_sens = event_data[(event_data.DIST > 0.1) &
                       (event_data.DIST < 2) &
                       ((event_data.BEARING <= 90) | (event_data.BEARING >= 270))] 
        
        if(len(event_sens)==0):
            continue    
        
        # sort distance 
        event_sens = event_sens.sort_values(['DIST'])
        
        # accident time
        Acc_time = pd.to_datetime(event_sens.iloc[0,6],format='%Y-%m-%d:%H:%M')
        
        # accuarcy of links on accident day
        day_accu = sens_accu[sens_accu['Day'] == Acc_time.day]
          
        #filter speed data by link 
        linkids = event_sens['LINK_ID'].values.tolist()
        Sensor_data = speed_data.query('LINK_ID == @linkids')  
        
        # extract speed data 60 mins before and 60 mins after
        start_t = Acc_time - timedelta(minutes= 65) 
        end_t = Acc_time + timedelta(minutes=65)   
        extracted_data = extract_data(Sensor_data, start_t, end_t)
        extracted_data = extracted_data.sort_values(['TIME'])
        
        dist = []
        Acc = []
        feat_array = []    
        Time = []
        
        # loop through event sensors
        for x in range(0,len(event_sens)):
            
            sensor_data = extracted_data[extracted_data['LINK_ID']== event_sens.iloc[x,7]]
            Avg_sensor = avg_speed[avg_speed['Link']== event_sens.iloc[x,7]]
                
            # check if the event has sufficient speed readings
            if(len(sensor_data)<24):
                continue
            
            dist.append(event_sens[event_sens['LINK_ID']== event_sens.iloc[x,7]].DIST.values)
            Acc.append(day_accu[day_accu['Link_ID']== event_sens.iloc[x,7]].Accuracy.values)
            
            # interpolate to get missing speed samples
            data = sensor_data[['TIME','SPEED']]
            sorted_data = data.sort_values(['TIME'])
            sorted_data1 = sorted_data.set_index(['TIME'])   
            resampled_data = sorted_data1.resample('5T').mean()  
            interpolated = resampled_data.interpolate(method='linear')  
            Time.append(interpolated.index)
            
            start_t  = interpolated.index[0].time()
            end_t = interpolated.index[len(interpolated)-1].time()
            
            Avg_data = Avg_sensor[(Avg_sensor.TIME.dt.time>=start_t) & (Avg_sensor.TIME.dt.time<=end_t)]
            
            if(len(Avg_data)!=len(interpolated)):
                continue
            
            if feature == 'speed_ratio':
                feat = (interpolated.SPEED.values - Avg_data.Speed.values)/Avg_data.Speed.values  
            else:
                feat  = interpolated.SPEED.values
                
            feat_array.append(feat)
        
        if (feature == 'mean' or feature =='energy' or feature == 'rms'):
            feat_array = window_feature(feat_array,feature)
           
        if(len(feat_array) ==0):
            continue
        
        for n in range (0,get_max(feat_array)): 
            if feature == 'speed_ratio':
                if (get_longest(Time)[n]>= Acc_time):
                     acc_sample = n
                     break
            else: #due to the rolling window size 5
                 if (get_longest(Time)[n+5]>= Acc_time):
                     acc_sample = n
                     break
    
        for j in range(0,len(alpha)):
                AL = 1
                MV = 1
                WD = 1
                SA = 1
                g_prev1 = np.zeros(len(feat_array))
                
                N1_prev = np.zeros(len(feat_array))
                N2_prev = np.zeros(len(feat_array))
                D_prev = np.full(len(feat_array),pi)
                sigma_prev = np.full(len(feat_array),sigma)
                
                for n in range (0,get_max(feat_array)+1):  
                    flag = np.zeros(len(feat_array))
               
                    for p in range(0,len(feat_array)):
                                              
                         if(len(feat_array[p])<=n-1):
                             continue
                         if(n==0):
                             g1 = np.log(pi/(1-pi))
                         else:   
                             previous = g_prev1[p]                         
                             Z = feat_array[p][n-1]
                             
                             if method == 'known':
                                 
                                 # post accident distribution
                                 mu1 = paras['mu1']
                                 sigma1 = paras['sigma1']
                                 
                                 g1 = np.log(rho + np.exp(previous)) - np.log(1 - rho) + np.log(sigma/sigma1)+\
                                         ((Z - mu0)**2)/(2*sigma) - ((Z - mu1)**2)/(2*sigma1) 
                             
                             if method == 'ML':
                                #   Calculating Log-likelihood with ML method                 
                                #   Calculating ML mean                                           
                                N1 = Z*(1 - (1-pi)*(1-rho)**(n-1)) + N1_prev[p]
                                D = (1 - (1-pi)*(1-rho)**(n-1)) + D_prev[p]
                                mu_mod = N1/D 
                                
                                #   Calculating ML variance                      
                                N2  = ((Z- mu_mod)**2)*(1 - (1-pi)*(1-rho)**(n-1)) + N2_prev[p]                     
                                sigma_mod = N2/D + sigma
                                 
                                N1_prev[p] = N1
                                N2_prev[p] = N2    
                                D_prev[p] = D                         
                                 
                                g1 = np.log(rho + np.exp(previous)) - np.log(1 - rho) + np.log(sigma/sigma_mod)+\
                                     ((Z - mu0)**2)/(2*sigma) - ((Z - mu_mod)**2)/(2*sigma_mod)        
                                
                             if method == 'MAP':
                                 
                                p_mean = paras['p_mean']
                                p_var = paras['p_var']
                                alpha1 = paras['alpha1']
                                beta1 = paras['beta1']
                            
                                #  Calculating Log-likelihood with MAP method                 
                                #  Calculating MAP mean                                           
                                N1 = Z*(1 - (1-pi)*(1-rho)**(n-1)) + N1_prev[p]
                                D = (1 - (1-pi)*(1-rho)**(n-1)) + D_prev[p]
                                 
                                num = N1 + p_mean*(sigma_prev[p]/p_var)*(1 - (1-pi)*(1-rho)**(n-1)) 
                                den = D + (sigma_prev[p]/p_var)*(1 - (1-pi)*(1-rho)**(n-1)) 
                                mu_mod = num/den 
            
                                #   Calculating MAP variance                      
                                N2  = ((Z- mu_mod)**2)*(1 - (1-pi)*(1-rho)**(n-1)) + N2_prev[p]                     
                                num1 = N2  + 2*beta1*(1 - (1-pi)*(1-rho)**(n-1)) 
                                den1 = D + 2*(alpha1 + 1)*(1 - (1-pi)*(1-rho)**(n-1)) 
                                sigma_mod = num1/den1 + sigma
                                
                                N1_prev[p] = N1
                                N2_prev[p] = N2    
                                sigma_prev[p] = sigma_mod
                                D_prev[p] = D                                                 
                                 
                                g1 = np.log(rho + np.exp(previous)) - np.log(1 - rho) + np.log(sigma/sigma_mod)+\
                                     ((Z - mu0)**2)/(2*sigma) - ((Z - mu_mod)**2)/(2*sigma_mod)
      
                                
    
                         if (~np.isnan(g1)):                    
                             g_prev1[p] = g1
                             
                         
                         if(g1>=np.log((1-alpha[j])/alpha[j])):
                             flag[p] = 1
                         else:
                             flag[p] = 0
                             
                    # atleast one scheme
                    if((np.sum(flag)>0) and (AL ==1)):    
                         delay = (n- acc_sample-1)
                         if(delay >= 0):
                             dd_ff[0][j].append(delay)
    
                         AL = 0
                    # majority vote sheme
                    if((np.sum(flag)>= len(feat_array)/2) and (MV ==1)):    
                         delay = (n- acc_sample-1)
                         if(delay >= 0):
                             dd_ff[1][j].append(delay)
    
                         MV = 0 
                         
                    # weighted distance sheme
                    if((np.nansum(flag/dist)> np.nanmean((np.full(len(feat_array),1)/dist)[:,0])) and (WD ==1)):     
                         delay = (n- acc_sample-1)
                         if(delay >= 0):
                             dd_ff[2][j].append(delay)
    
                         WD = 0
                         
                    # sensor accuracy sheme
                    if( (Acc[0]) and (np.nansum(flag*Acc)> np.nanmean(Acc))  and (SA ==1)):    
                         delay = (n- acc_sample-1)
                         if(delay >= 0):
                             dd_ff[3][j].append(delay)
     
                         SA = 0
    return dd_ff           

def window_feature(df,feature):
    # compute rolling windown features (window size = 5 samples)
    feat = []
    for j in range(0,len(df)):
        x = df[j]             
        c = []
        if feature == 'mean':
            for i in range(0,len(x)-4):
                b = np.array(x[i:i+5])
                c.append(np.mean(b))                
            feat.append(c)                     

        if feature == 'energy':
            for i in range(0,len(x)-4):
                b = np.array(x[i:i+5])
                c.append(np.mean(np.power(b,2)))               
            feat.append(c)
            
        if feature == 'rms':
            for i in range(0,len(x)-4):
                b = np.array(x[i:i+5])
                c.append(np.sqrt(np.mean(np.power(b,2))))               
            feat.append(c) 
    return np.array(feat)


def get_max(feat):  
    vv = []
    for i in range(len(feat)):
        vv.append(len(feat[i]))
 
    return np.max(vv)

def get_longest(feat):
    vv = []
    for i in range(len(feat)):
        vv.append(len(feat[i]))
 
    return feat[np.argmax(vv)]
       
def gaus_pdf(y, mean, sigma):
    x = np.exp(-((y-mean)**2)/(2*sigma))/(sqrt(2*np.pi*sigma))
    return x

def lamda(m, config):
    pi = config['pi']
    rho = config['rho']
    if(m==0):
        n = pi
    else:
        n = (1 -pi)*rho*(1-rho)**(m-1)
    return n

def assign_time(df,header,form): 
    df.loc[:,header] = pd.to_datetime(df[header],format=form)
    df = df.assign(DAY=df[header].dt.day)
    df = df.assign(DAYOFWEEK=df[header].dt.dayofweek)
    df = df.assign(HOUR_MIN=df[header].dt.time)
    return df

def extract_data(df, start_t, end_t):
    return df[(df.TIME>=start_t)&
              (df.TIME<=end_t)]
    
def reject_outliers(data, m = 2):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def plot_dd_ff(dd_ff, method, baselines, schemes, markers, alpha):
    # plot detection delay vs false alarm
    plt.figure(figsize=(9,8))
    
    delay = [[[] for _ in range(len(alpha))] for _ in range(4)]      
    for i in range(4):
        for j in range(len(alpha)):
            a = dd_ff[i][j]
            if method != 'known':
                a = reject_outliers(np.array(a))
            delay[i][j] = np.nanmean(a)
    
        plt.plot(alpha,delay[i],marker=markers[i],ms=10.0,mew = 2.0,label= schemes[i],linewidth=3) 
    
    if baselines: # to include baselines results 
        plt.plot(0.78,2.4,marker = '+',ms=8.0,mew = 3.0,c='b')
        plt.annotate('IIG ($\lambda$=0.2)', xy=(0.78, 2.4), xytext=(0.0005,2.0), 
                    arrowprops=dict(facecolor='black', width = 2,  
                                    headwidth =6,shrink=0.01),fontsize=20)
        plt.plot(0.62,2.7,marker = '+',ms=8.0,mew = 3.0,c='b')
        plt.annotate('IIG ($\lambda$=0.4)', xy=(0.62, 2.7), xytext=(0.0005,2.5),
                    arrowprops=dict(facecolor='black', width = 2, 
                                    headwidth =6,shrink=0.01),fontsize=20)
        plt.plot(0.41,2.8,marker = '+',ms=8.0,mew = 3.0,c='b')
        plt.annotate('IIG ($\lambda$=0.6)', xy=(0.41, 2.8), xytext=(0.0005,3.1),
                    arrowprops=dict(facecolor='black', width = 2,
                                    headwidth =6, shrink=0.01),fontsize=20)
        
        plt.plot(0.16,8.8,marker = 's',ms=8.0,mew = 3.0,c='g')
        plt.annotate('AFM', xy=(0.16,8.8), xytext=(0.02,8),
                    arrowprops=dict(facecolor='black', width = 2,  
                                    headwidth =6,shrink=0.04),fontsize=20)
        plt.plot(0.03,9.2,marker = '^',ms=8.0,mew = 3.0,c='y')
        plt.annotate('OM', xy=(0.03,9.2), xytext=(0.005,8.5),
                    arrowprops=dict(facecolor='black', width = 2,  
                                    headwidth =6,shrink=0.1),fontsize=20)
        plt.plot(0.81,2.9,marker = 'v',ms=8.0,mew = 3.0,c='r')
        plt.annotate('RM', xy=(0.81,2.9), xytext=(0.08,3.6),
                    arrowprops=dict(facecolor='black', width = 2,  
                                    headwidth =6,shrink=0.1),fontsize=20)
        plt.plot(0.79,3.7,marker = 'x',ms=8.0,mew = 3.0,c='slategrey')
        plt.annotate('MAM', xy=(0.79,3.7), xytext=(0.06,4.3),
                    arrowprops=dict(facecolor='black', width = 2,  
                                    headwidth =6,shrink=0.1),fontsize=20)
        plt.plot(0.75,4.4,marker = 'o',ms=8.0,mew = 3.0,c='m')
        plt.annotate('ESM', xy=(0.75,4.4), xytext=(0.06,5.3),
                    arrowprops=dict(facecolor='black', width = 2, 
                                    headwidth =6,shrink=0.1),fontsize=20)
        plt.ylim(ymin = 1.9)
    
    plt.xscale('log')
    plt.legend(loc='lower left',fontsize=20)       
    plt.xticks(alpha) 
    plt.rc('font', size=20)
    