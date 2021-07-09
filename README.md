# MLFD
This repository started out as the codebase for my master thesis, titled "Robust Machine-Learning Approaches for Efficient Functional Dependency Approximation", which I've written at Beuth University of Applied Sciences under the supervision of Prof. Felix Biessmann.

The code changed substantially ever since. If you're interested in the thesis' state of the repository, have a look at the `abgabe` branch.

## Installation
The repo requires python version 3.7.10. More recent python versions are not supported, mainly the `shap` and `AutoGluon` dependencies prevent this.

Run the following commands to clone and install the dependencies:

```
git clone https://github.com/felixbiessmann/MLFD
cd MLFD
python -m pip install -r requirements.txt
python -m pip install .
```

To fetch the metanome binaries and download the test datasets, run

```
python get_binaries.py
```

which will create a new folder calles `MLFD_fd_detection/` and download the metanome binaries and test datasets.

## Testing
I use `pytest` to carry out all tests. The tests are stored in the `lib/` directory and are named `test_$LIBRARY.py`.

The easiest way to run the tests is to install the repository as an editable package with `pip`. Run in the repository's root folder the following command:

```
python -m pip install -e .
```

Once that is done, simply run `pytest` in the project's root folder to execute all tests.


## Detect FDs Conventionally
To measure performance of conventional FD-detection algorithms, we use [metanome]((https://github.com/HPI-Information-Systems/Metanome) and [metanome-cli](https://github.com/sekruse/metanome-cli). In `backend/WEB-INF` there is a subfolder `classes` where all algorithms(`algorithms/`) and datasets(`inputData/`) are stored.

To start Metanome, you call `./run.sh` within the `performance/` folder. Open the graphical interface in your browser and use it to add algorithms and datasets.


## Detect AFDs with MLFD
In the project's root folder, run
```python
python computer.py --model greedy_detect --data iris
```
to use the greedy strategy on the iris dataset.

## Minimal Dependencies (GER)

Das Ergebnis der `DepOptimizer.search_dependencies()` Methode ist ein Baum der folgenden Form:

```
0 38.4
└── [0, 1, 2] 11.8
    ├── [1, 2] 12.2
    └── [0, 2] 9.9
        ├── [2] 9.9
        └── [0] 2028.5
```
Der Knoten `0`, von dem alle anderen Knoten ausgehen, ist die potentielle rechte Seite der Abhängigkeit. `0` referenziert die `0`-te Tabellenspalte. Die Zahl `38.4` ist der zum Ursprungsknoten zugehörige Schwellenwert. Dieser ist im kontinuierlichen Fall der Mean Squared Error (MSE), im diskreten Fall der F1-Score. Im Beispiel wird ein MSE angezeigt.

Alle Knoten, die vom Ursprungsknoten abgehen, sind potentielle linke Seiten der Abhängigkeit. Im Beispiel ist `[0, 1, 2]` das erste Kind der potentiellen rechten Seite, der zugehörige MSE ist `11.8`. Das bedeutet, dass ein Modell mit den Daten der Spalten `0, 1, 2` der Tabelle trainiert wurde, und dass dieses Modell auf den Validierungsdaten einen MSE von `11.8` erzielt, wenn es die Inhalte der Spalte `0` berechnet.

Liegt der Schwellenwert eines Kindsknotens über [^0] dem des Elternknotens, wird dieser Ast das Baumdiagramms nicht weiter verfolgt. Zu sehen ist das für den Knoten `[1, 2]`, dessen Schwellwert mit `12.2` über dem Schwellwert des Elternknotens liegt.

2        1.2216000000000002
└── [0, 1, 3, 4, 5] 0.1026823543956827
    ├── [1, 3, 4, 5] 0.10134767167251084
    │   ├── [3, 4, 5] 0.10138946277790035
    │   ├── [1, 4, 5] 0.10015266454080729
    │   │   ├── [4, 5] 0.0981834771274491
    │   │   │   ├── [5] 0.14659046298195563
    │   │   │   └── [4] 0.07602522188974341
    │   │   ├── [1, 5] 0.12380707040733421
    │   │   └── [1, 4] 0.10866514269674026
    │   ├── [1, 3, 5] 0.13809114573604503
    │   └── [1, 3, 4] 0.11313579374739072
    ├── [0, 3, 4, 5] 0.12994118710701896
    ├── [0, 1, 4, 5] 0.10100902575172241
    │   ├── [1, 4, 5] 0.10015266454080729
    │   │   ├── [4, 5] 0.0981834771274491
    │   │   │   ├── [5] 0.14659046298195563
    │   │   │   └── [4] 0.07602522188974341
    │   │   ├── [1, 5] 0.12380707040733421
    │   │   └── [1, 4] 0.10866514269674026
    │   ├── [0, 4, 5] 0.12167021654231883
    │   ├── [0, 1, 5] 0.12123897601686776
    │   └── [0, 1, 4] 0.10557385655521863
    ├── [0, 1, 3, 5] 0.12836826731975562
    └── [0, 1, 3, 4] 0.11918173381240424


[^0]Im diskreten Fall unter
