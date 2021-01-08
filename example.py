import pandas as pd
import ATTAIN

##### Load datasets      
## speed data 
speed_data = pd.read_csv('datasets/405October.csv') 
# sensor accuracy
sens_accu = pd.read_csv('datasets/Sensor_accuracy.csv')  
# -- preprocessed dataset - accident (occured on north lane) locations in Lat and Long with respect to sensors
accidents = pd.read_csv('preprocessed_data/accidents_north.csv') 
# -- preprocessed dataset - average speed readings of sensors at times of a day
avg_speed = pd.read_csv('preprocessed_data/average_speed.csv')


# datasets
datasets = {'speed_data': speed_data,
            'north_accidents': accidents,
            'avg_speed': avg_speed,
            'sensor_accuaracy': sens_accu}

# thresholds
alpha = [1e-5,1e-4,0.001,0.01,0.1]
# configuraations
config = {'pi': 0.001, 'rho': 0.009095381665426}
#feature parameters 
paras = {'speed_ratio': {'mu0': -4.8411575518395698e-05,
                         'mu1': -0.18249676440264076,
                         'sigma0': 0.092508205830755474,
                         'sigma1': 0.092508205830755474,
                         'p_mean': -0.0017974981941875777,
                         'p_var': 0.069009694273745106,
                         'alpha1': 2.6063126688473859,
                         'beta1': 0.036799251311936912},
                'mean': {'mu0':52.71685745236524,
                         'mu1': 45.660099813699077,
                         'sigma0': 253.39935890133177,
                         'sigma1': 253.39935890133177},                        
              'energy' :{'mu0': 3063.6245178422837,
                         'mu1': 2400.8553917362997,
                         'sigma0': 2319350.3144240063,
                         'sigma1': 2319350.3144240063},
                'rms'   :{'mu0': 53.10339362104984,
                         'mu1': 45.813806089591765,
                         'sigma0': 243.65410377012697,
                         'sigma1': 243.65410377012697}}

#define feature name and method
feature = 'speed_ratio'
method = 'MAP'

# compute detection delays
dd_ff = ATTAIN.compute_DD(feature, method, config, paras[feature], alpha, datasets)
# plot the average detection delay vs false alarm
markers = ['+','o','v','s','v','p']
schemes = ['AL','MV','WD','SA']
ATTAIN.plot_dd_ff(dd_ff, method, True, schemes, markers, alpha)