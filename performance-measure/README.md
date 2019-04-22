# Performance Measure for MLFD
To measure performance of conventional FD-detection algorithms, we use [metanome]((https://github.com/HPI-Information-Systems/Metanome) and [metanome-cli](https://github.com/sekruse/metanome-cli). In `backend/WEB-INF` there is a subfolder `classes` where all algorithms(`algorithms/`) and datasets(`inputData/`) are stored.

## Metanome
Metanome can be started, if you call `./run.sh` within the `performance/measure/` folder. The GUI is self-explaining.

## Metanome-cli
In `backend/WEB-INF/meta-cli` is the metanome-cli executable stored. Within the `meta-cli` subfolder, metanome-cli can be executed by calling

```$ java -cp metanome-cli-1.1.0.jar:../classes/algorithms/pyro-distro-1.0-SNAPSHOT-distro.jar de.metanome.cli.App --algorithm de.hpi.isg.pyro.algorithms.TaneX --files ../classes/inputData/adult.csv --file-key inputFile```
