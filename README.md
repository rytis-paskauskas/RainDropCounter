# Rain drop counter data and code

## Directory structure
- inputs :: contains audio recordings (wav) and annotation (txt) files. The recordings and annotations can be loaded using the Audacity software.fkfx
- augment :: encoded recordings with various parameters in `TFRecord` format. The datasets are generated from these files
- config :: data splitting
- ds :: datasets (only test is included at the moment to save space)
- m :: models in `keras` format.


