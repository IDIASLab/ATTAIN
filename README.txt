# ATTAIN
## Introduction
ATTAIN is an accurate and timely traffic accident detection algorithm


## Prerequisites
python 2.7 or above and the following libraries
```
numpy
pandas
```

## Files
```
ATTAIN.py: include all the necessary functions to run ATTAIN
datasets: include speed readings and accident details from Oct 2013 of the I405 freeway and sensors accuracy
preprocessed_data: accident locations in longitudes and latitudes with respect to every sensor and average speed readings
example.py: an example code to run ATTAIN
```

## How to use
```
Step 1. Load dataset:
   	load speed and accident datasets

Step 2. Define input parameters:
	feature: feature name (speed_ratio/mean/rms/energy)
	method: methods proposed (known: known post feature distributions, ML: ML estimate, MAP: MAP estimate)
	config: intilaization configurations
	alpha: thresholds on loglikelihood ratio
	paras: feature parameters
	datasets: required datasets

Step 3. Run ATTAIN.compute_DD (compute accident detection delay)
	import ATTAIN
	dd_ff = ATTAIN.compute_DD(feature, method, config, paras[feature], alpha, datasets)
```

## Example
```
See example.py
```