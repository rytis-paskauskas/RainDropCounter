# Rain drop counter data and code

## Software and hardware
Most of the work was done in `Python 3.11`  on a standard CPU-laptop running Linux operating system. The `augment` routine uses system calls to the `sox` audio processing utility for resampling and other audio processing (```import audio_encoder.augment`). Note that `Tensorflow 2.15` python package is required at this time for loading models in keras format, as v2.16 introduced some breaking changes.
## Directory structure
- inputs :: contains audio recordings (wav) and annotation (txt) files. The recordings and annotations can be loaded using the Audacity software.
- augment :: encoded recordings with various parameters in `TFRecord` format. The datasets are generated from these files
- config :: data splitting
- ds :: datasets (only test is included at the moment to save space)
- m :: models in `keras` format.


