# Keras to TensorFlow
The **keras_to_tensorflow** is a tool that converts a trained keras model into a ready-for-inference TensorFlow model. *The original tool does not support TF2.0 very well, so I tailored it to support TF2.0*

If you are looking for a convertion tool that supports TF1.x, you can checkout this repo: https://github.com/amir-abdi/keras_to_tensorflow

#### Summary
- In the default behaviour, this tool **freezes** the nodes (converts all TF variables to TF constants), and saves the inference graph and weights into a binary protobuf (.pb) file. During freezing, TensorFlow also applies node pruning which removes nodes with no contribution to the output tensor.

## How to use
Keras models can be saved as a single [`.hdf5` or `h5`] file, which stores both the architecture and weights, using the `model.save()` function.
 This model can be then converted to a TensorFlow model by calling this tool as follows:
    
    python keras_to_tensorflow.py 
        --input_model="path/to/keras/model.h5" 
        --output_model="path/to/save/model.pb"
     


## Dependencies
- keras
- tensorflow
- pprint
- pathlib
- argparse

