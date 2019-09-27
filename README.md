Disfluency Detection and Constituency Parsing
------------------------------------------------------------
A joint disfluency detection and constituency parsing model for transcribed speech based on [Neural Constituency Parsing of Speech Transcripts](https://www.aclweb.org/anthology/N19-1282).

## Contents
1. [Requirements for Training](#requirement)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Training Instructions](#training)
5. [Citation](#citation)
6. [Contact](#contact)
7. [Credits](#credits)


### Requirements for Training
* Python 3.6 or higher.
* Cython 0.25.2 or any compatible version.
* [PyTorch](http://pytorch.org/) 0.4.1, 1.0/1.1, or any compatible version.
* [EVALB](http://nlp.cs.nyu.edu/evalb/). Before starting, run `make` inside the `EVALB/` directory to compile an `evalb` executable. This will be called from Python for evaluation.
* [AllenNLP](http://allennlp.org/) 0.7.0 or any compatible version (only required when using ELMo word representations)

### Installation
```bash
$ pip install cython numpy
$ pip install benepar[cpu]
$ python -m spacy download en
```

```
$ git clone https://github.com/pariajm/naacl2019
$ cd naacl2019/EVALB
$ make evalb
$ cd .. && mkdir data
$ cd data
$ wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
$ wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
$ cd ..
```
### Usage
Use our model to parse and disfluency label your own sentences. Befor running the following commands, make sure you meet the requirements for training and installation. 
1. Use our best model trained on the original Switchboard treebank:
```
$ cd best_models
$ wget https://cloudstor.aarnet.edu.au/plus/s/KW6ndLh8hfuilOg/download -O best_nopunct_nopw_Edev=0.872.pt
$ cd ..
$ python3 src/parse.py --config best_models/best_nopunct_nopw_config.json --eval-path best_models/eval.txt >best_models/out_and_error.txt
```

2. Use our best model trained on the tree transformed Switchboard treebank (recommended):
```
$ cd best_models
$ wget https://cloudstor.aarnet.edu.au/plus/s/KW6ndLh8hfuilOg/download -O best_tree_transformation_Edev=0.8838.pt
$ cd ..
$ python3 src/parse.py --config best_models/best_tree_transformation_config.json --eval-path best_models/eval.txt >best_models/out_and_error.txt
```
### Training Instructions
```
$ python3 src/train_parser.py --config results/best_nopunct_nopw_config.json --eval-path results/eval.txt >results/out_and_error.txt
```
### Citation
If you use this model, please cite our paper:
```
@inproceedings{jamshid-lou-etal-2019-neural,
    title = {Neural Constituency Parsing of Speech Transcripts},
    author = {Jamshid Lou, Paria and Wang, Yufei and Johnson, Mark},
    booktitle = {Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
    month = {June},
    year = {2019},
    address = {Minneapolis, Minnesota},
    publisher = {Association for Computational Linguistics},
    url = {https://www.aclweb.org/anthology/N19-1282},
    doi = {10.18653/v1/N19-1282},
    pages = {2756--2765}
}
```

### Contact
Paria Jamshid Lou <paria.jamshid-lou@hdr.mq.edu.au>

 
### Credits
The code for self-attentive parser and part of the README file are based on https://github.com/nikitakit/self-attentive-parser.


