# Überlegungen über minimale Abhängigkeiten

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
