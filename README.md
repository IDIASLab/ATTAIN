# Accurate and Timely Traffic Accident Detection
## Introduction
We present ATTAIN, an Accurate and Timely Traffic AccIdent detectioN method, from speed data on freeways equipped with spatially distributed speed sensors. ATTAIN is based on a novel Bayesian quickest change detection formulation for near real-time freeway accident detection, which considers both average detection delay and false alarm rate.

## Citation
To cite our paper, please use the following reference:
> Yasitha Warahena Liyanage, Daphney-Stavroula Zois, and Charalampos Chelmis. "Near Real-Time Freeway Accident Detection." IEEE Transactions on Intelligent Transportation Systems (2020). [doi: 10.1109/TITS.2020.3027494](https://doi.org/10.1109/TITS.2020.3027494).


BibTeX:
``` 
@article{liyanage2020near,
 author={Liyanage, Yasitha Warahena and Zois, Daphney-Stavroula and Chelmis, Charalampos},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Near Real-Time Freeway Accident Detection}, 
  year={2020},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TITS.2020.3027494}}
```

### Dataset
We use a corpus of 822,049 speed readings and 1,158 accident data (i.e., severity, location, time, and direction) collected by Caltrans, LA Department of Transportation, and California Highway Patrol during October 2013. Speed readings (average speed of all lanes) were collected every 5 minutes from 223 sensors placed 0.5 miles apart over 50 miles of the I405 freeway that goes through Los Angeles County. We make this dataset available for reproducibility purposes.

## Prerequisites
python 2.7 and the following libraries
```
pandas
numpy
datetime
math
matplotlib
```

## Files
```
ATTAIN.py: main algorithm
405October.csv: dataset
```
