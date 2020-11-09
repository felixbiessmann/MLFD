# MLFD
With this repository I try to leverage machine learning for Approximate Functional Dependency (AFD) detection, based on Philipp Jung's Master thesis at Beuth University of Applied Sciences, under the supervision of Prof. Felix Biessmann.

## Installation
The repo requires python version 3.7.3 -- this is due to `datawig`'s dependency with mxnet.
To clone the project and install dependencies, do as follows:

```
git clone https://github.com/felixbiessmann/MLFD
cd MLFD
pip install -r requirements.txt
```

## Detect FDs Conventionally
To measure performance of conventional FD-detection algorithms, we use [metanome]((https://github.com/HPI-Information-Systems/Metanome) and [metanome-cli](https://github.com/sekruse/metanome-cli). In `backend/WEB-INF` there is a subfolder `classes` where all algorithms(`algorithms/`) and datasets(`inputData/`) are stored.

To start Metanome, you call `./run.sh` within the `performance/` folder. Open the graphical interface in your browser and use it to add algorithms and datasets.

## Detect AFDs with MLFD
In the project's root folder, run
```python
python computer.py --model greedy_detect --data iris
```
to use the greedy strategy on the iris dataset.

## Plot results
WIP
