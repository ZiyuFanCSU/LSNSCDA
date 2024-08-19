# LSNSCDA: Unraveling CircRNA–Drug Sensitivity via Local Smoothing Graph Neural Network and Credible Negative Samples
## Introduction
We propose LSNSCDA, a novel prediction algorithm leveraging Local Smoothing Graph Neural Networks and Credible Negative Samples to enhance accuracy in identifying circRNA-drug associations. Our method addresses the limitations of fixed-length propagation in graph neural networks and the unreliability of randomly sampled negative instances. Experimental results demonstrate that LSNSCDA outperforms existing models, providing robust predictions and valuable insights for improving cancer treatment outcomes.
## Requirements

To run the codes, You can configure dependencies by restoring our environment:
```
conda env create -f environment.yml -n $Your_env_name$
```

and then：

```
conda activate $Your_env_name$
```

## Structure
The code includes data processing and data enhancement, model construction, model training, experimental results, case stydies, and various visualisations and interpretive analyses. The directory structure of our uploaded code is as follows:

```
LSNSCDA
├── case study                  # Result of case study
├── code                        # some codes and figures of the paper
├── dataset                     # including circRNA similarity matrix, drug similarity matrix and associations
├── paper_info
│   ├── image                   # Model overview
│   └── Supplementary Materials # Appendix of this paper
├── other_methods               # All baseline models for comparison
├── result_files                # Experimental results
└── train.py                    # Training and validation code
└── environment.yml             # Dependent Package Version
``` 


## Model
Overview of LSNSCDA. (a) The overall model workflow proceeds from left to right, encompassing initial node feature encoding, node feature updating based on the local smoothing algorithm of the graph neural network, and concatenating the updated circRNA and drug node encodings for input into the predictor for prediction. (b) The node-dependent local smoothing algorithm. (c) High-credible negative sampling strategy.
![1.png](paper_info%2Fmodeloverview.png)

## Training and testing

Run `train.py` using the following command:
```bash
python train.py --device <your_device>
```

Other configurations need to be changed inside `train.py`, including model settings and the data directory.



## Contact

We thank all the researchers who contributed to this work.

If you have any questions, please contact fzychina@csu.edu.cn.
