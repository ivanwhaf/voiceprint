# Voiceprint recognition project

## Architecture
.wav/.npy->cut->fbank->resnet->embeddings
* training:amsoft
* test/eval:cosine socre EER

## Usage
* 1.Train clean baseline
```
python train.py
```
* 2.Inference score
```
python inference.py
```
* 3.Retrain
```
python train_sort.py
```

## Experiment Record
* resnet18 lr:0.005 bs:24 optim:sgd data:100utt/person aug:specaugment data:wav loss:am-softmax(s=15)
* resnet18 lr:0.005 bs:24 optim:sgd data:100utt/person aug:specaugment data:npy loss:am-softmax(s=15)
* resnet18 lr:0.001 bs:24 optim:adam data:100utt/person aug:specaugment data:npy loss:am-softmax(s=15)
* resnet18 lr:0.001 bs:24 optim:adam data:200utt/person aug:specaugment data:npy loss:am-softmax(s=15) eer:0.286
* resnet18 lr:0.001 bs:24 optim:adam data:300utt/person aug:specaugment data:npy loss:am-softmax(s=15) eer:0.37
* resnet18 lr:0.001 bs:24 optim:adam data:100utt/person aug:specaugment data:npy loss:am-softmax(s=30) eer:0.281
* resnet18 lr:0.001 bs:24 optim:sgd data:100utt/person aug:specaugment data:npy loss:am-softmax(s=30) eer:0.255
* resnet18 lr:0.001 bs:24 optim:sgd data:200utt/person aug:specaugment data:npy loss:am-softmax(s=30) eer:0.2198
* resnet18 lr:0.001 bs:24 optim:sgd data:300utt/person aug:specaugment data:npy loss:am-softmax(s=30) eer:0.268/0.24
* resnet18 lr:0.001 bs:24 optim:sgd data:100utt/person aug:specaugment data:wav loss:am-softmax(s=30) eer:0.245
* resnet18 lr:0.001 bs:24 optim:sgd data:100utt/person aug:specaugment data:npy loss:aam-softmax(s=30,m=0.3) eer:0.263
* resnet18 lr:0.001(scheduler 0.97) bs:24 optim:sgd data:200utt/person aug:specaugment data:npy loss:aam-softmax(s=30,m=0.3) eer:0.201
* resnet18 lr:0.001(scheduler 0.97) bs:24 optim:adam data:150utt/person aug:specaugment data:npy loss:aam-softmax(s=30,m=0.3) eer:0.316

* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:sgd data:100utt/person aug:specaugment data:npy loss:aam-softmax(s=30,m=0.3) eer:0.079
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:100utt/person aug:specaugment data:npy loss:aam-softmax(s=30,m=0.3) eer:0.084
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:100utt/person aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.069
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:150utt/person aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.072/0.063
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:300utt/person aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.067/0.069
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:0-98utt/person aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.0755/0.0672
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:2-98utt/person aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.0666/0.086
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:1-99utt/person aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.084/0.075
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:t/person aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.0666/02-100ut.0778
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:0-100utt/person aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.078/0.071
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:4-96utt/person aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.0727/0.073
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:0-100utt/person noise:s0.2 aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.217/0.191
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:0-100utt/person noise:s0.02 aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.087/0.099
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:0-98utt/person noise:s0.02 aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.089/0.074
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:0-100utt/person noise:s0.05 aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.105/0.105
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:0-95utt/person noise:s0.05 aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.105/0.105
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:0-95utt/person(15epoch) noise:s0.05 aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.101/0.119
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:5-95utt/person(20epoch) noise:s0.05 aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.0966/0.113
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:2-98utt/person(20epoch) noise:s0.02 aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.0793/0.087
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:0-100utt/person clean aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.081/0.069
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:2-98utt/person(50epoch) clean aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.072/0.074
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:0-100utt/person noise:s0.1 aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.119/0.138
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:10-90utt/person(50epoch) noise:s0.1 aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.136/0.130
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:5-95utt/person(20epoch) noise:s0.1 aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.121/0.118
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:2-98utt/person(20epoch) noise:s0.1 aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.153/0.141
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:2-98utt/person(50epoch) noise:s0.1 aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.139/0.134
* 
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:0-150utt/person clean aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.075/0.078
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:2-148utt/person(50epoch) clean aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.076/0.076
* **ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:0-150utt/person noise:s0.05 aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.113/0.113**
* **ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:5-145utt/person noise:s0.05 aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.093/0.099**
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:10-140utt/person noise:s0.05 aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.114/0.107
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:2-148utt/person noise:s0.05 aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.104/0.121
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:0-150utt/person noise:s0.1 aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.133/0.154
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:(0.05-0.95)/person noise:s0.1 aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.139/0.153
* ECAPA-TDNN lr:0.0005(scheduler 0.97) bs:64 optim:adam data:(0.1-0.9)/person noise:s0.1 aug:specaugment data:npy loss:aam-softmax(s=30,m=0.2) eer:0.113/0.133