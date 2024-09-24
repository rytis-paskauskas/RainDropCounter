
## data folder structure
The **inputs** folder contains audio recordings (wav) and Audacity-ready annotation (txt) files. The recordings and annotations can be loaded using the Audacity software.

The **augment** folder contains encoded recordings with various parameters in `TFRecord` format. The datasets are generated from these files
- config :: data splitting

The **ds** folder contains training-ready datasets

The **m** folder contains trained models in Keras and H5 formats.
