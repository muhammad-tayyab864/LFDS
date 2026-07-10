# LFDS
Luminance-aware Feature Integration with Disparity-aware Stereo enhancement (LFDS), an end-to-end quality assessment and enhancement framework that jointly optimizes luminance reduction and binocular consistency under a target power constraint.

# Stereo OLED Display 

## Comparison
1. ACE
2. RACE
3. Meur
4. PAVE
5. SPACE
6. Proposed


## Install

## Train
### ACE
- Modify dataset path in train.py to flickr10124 dataset in your computer.
- Execute following command:
```Python
python train.py
```
### RACE
- Modify dataset path in train.py to flickr10124 dataset in your computer.
- Execute following command:
```Python
python train.py
```
### Meur
- Modify dataset path in train.py to flickr10124 dataset in your computer.
- Execute following command:
```Python
python train.py
```
### SPACE
- Do not need training (have pretrain weight).
### Proposed
- Modify dataset path in train.py to flickr10124 dataset in your computer.
- Execute following command:
```Python
python train.py
```

## Inference (produce test data)
### ACE
- Modify dataset path in Inference.py to test dataset in your computer.
- Execute following command:
```Python
python Inference.py
```
### RACE
- Modify dataset path in Inference.py to test dataset in your computer.
- Execute following command:
```Python
python Inference.py
```
### Meur
- Modify dataset path in Inference.py to test dataset in your computer.
- Execute following command:
```Python
python Inference.py
```
### SPACE
- Modify dataset path in Inference.py to test dataset in your computer.
- Execute following command:
```Python
python Inference.py
```
### Proposed
- Modify dataset path in Inference.py to test dataset in your computer.
- Execute following command:
```Python
python Inference.py
```

## Test (Image Quality Assessment)
- Modify test model and dataset in iqa.py.
- Execute following command:
```Python
python iqa.py
```
