Here can be found the implementation of the NAS for MLP-based analysis of the distances.  

It is implemeted in _mlp-distance.py_ wtih the args:
```
  --distance DISTANCE   
  Distance to use, euc or dtw  


  # Not included but should pay attention, the data path and models path
```
NNI config files includes :
config.yml and search_space.json

Usefull commands with NNI (have to cd nni)
* to start an experiment:    
nnictl create --config config.yml --port 8080

* to have a quick status check:  
nnictl status 

* to stop pause and resume:  
nnictl stop  
nnictl pause  
nnictl resume  
* The nni experiments are added to dcache
* The models as well
TODO : look at the FiP toolbox where the different approaches are supposed to be made cleaner and "merged", if further R&D needed.