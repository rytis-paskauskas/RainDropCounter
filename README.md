# Rain drop counter data and code

## Software and hardware
Most of the work was done in `Python 3.11`  on a standard CPU-laptop running Linux operating system. 
### Custom python packages
Other than sample code, the reusable code is organised in two packages 
```python
import audio_encoder # custom signal processing
import rainml # study specific code, most notably
import rainml.tfmodels # TensorFlow model definitions
```

```python
from audio_encoder.augment import augment
```

The `augment` routine uses system calls to the `sox` audio processing utility. It should be available for Windows and OSX.

Note that `Tensorflow 2.15` python package is required at this time for loading models in keras format, as v2.16 introduced some breaking changes.
## Directory structure
- inputs :: contains audio recordings (wav) and annotation (txt) files. The recordings and annotations can be loaded using the Audacity software.
- augment :: encoded recordings with various parameters in `TFRecord` format. The datasets are generated from these files
- config :: data splitting
- ds :: datasets (only test is included at the moment to save space)
- m :: models in `keras` format.


