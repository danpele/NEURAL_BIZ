# NN Quantlets

One Quantlet per chart of the lectures of *Neural Networks and Deep Learning*, as in the MFM, SFM and EMQA courses:

```
Quantlets/Lk/NN_Lk_<topic>/
    Metainfo.txt            name, description, keywords, data file, output (QuantLet format)
    NN_Lk_<topic>.ipynb     self-contained notebook (executed): loads the course data, computes, trains, draws the chart
    <topic>.png             the chart
```

* `<topic>` is the chart name used in the deck (`charts/en/<topic>.pdf`), e.g. `NN_L3_dropout_curves`.
* Each notebook runs on its own, locally or on Google Colab: it reads `data/clean` of a local copy of the repository,
  or the same files from the public repository on GitHub; it imports no course module (the chart style is inlined).
* Numbers printed with `show(key, label, value)` are those on the slide; `make_quantlets_Lk.py` compares each of them
  with `lectures/Lk/Lk_numbers.json` (`nn_quantlet.check_numbers`) and stops on any mismatch.
* Every slide with a chart links the Quantlet of that chart (bottom right) automatically, as soon as the folder exists.
* Built by `lectures/Lk/make_quantlets_Lk.py` with `lectures/common/nn_quantlet.py`.
