# C++ Visual Transformer

This project contains an implementation of all the necessary components as well as a full ViT model. In particular:
- A stand-alone single-threaded serial implementation (`src/`)
- A OpenMP implementation (`omp_src/`)
- PyTorch-like C++ basic components (`include/`)
- A pre-defined model to be used for inference (`models/`)
- TODO --> do we include datasets

# How to Compile
Stand-alone single-threaded serial implementation:
```
make bin/vit.exe
```

OpenMP implementation:
```
make bin/omp/vit.exe
```

Unit tests:
```
make bin/test/test_<module_name>.exe
```

Modules are: `datatypes`, `modules`, `mlp`, `conv2d`, `attention`, `block`, `patch_embed`, `vision_transformer` and `utils`.

# How to Run (locally)

```
./bin/vit.exe <model.cvit> <dataset.cpic> <out_file.cprd> <measures_file.csv>
```

TODO OpenMP version

For example:
```
./bin/vit.exe models/vit_1.cvit datasets/pic_1.cpic output/test_1.cprd measures/test_1.csv
```

# Run on Leonardo

# Unit Tests

To test a module, simply run:
```
bin/test/test_<module_name>.exe
```

# Performance Analysis
