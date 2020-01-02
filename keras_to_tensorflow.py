#!/usr/bin/env python
"""
Copyright (c) 2019, by the Authors: Shuai Chen
This script is freely available under the MIT Public License.
Please see the License file in the root for details.

The following code snippet will convert the keras model files
to the freezed .pb tensorflow weight file. The resultant TensorFlow model
holds both the model architecture and its associated weights.
"""

import tensorflow as tf
from tensorflow.compat.v1 import graph_util
from tensorflow.python.keras import backend as K
import argparse
from pathlib import Path
import pprint
import sys

#necessary !!!
tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()
req_grp = parser.add_argument_group('required')
req_grp.add_argument('--input_model',   default=False, type=str, help='Path to the input model.')
req_grp.add_argument('--output_model',  default=False, type=str, help='Path where the converted model will.')

args = parser.parse_args()
pprint.pprint(vars(args))

# input model path
model_path = args.input_model

# If output_model path is relative and in cwd, make it absolute from root
output_model = args.output_model
if str(Path(output_model).parent) == '.':
    output_model = str((Path.cwd() / output_model))

output_fld = Path(output_model).parent
output_model_name = Path(output_model).name

K.set_learning_phase(0)

restored_model = tf.keras.models.load_model(model_path)
print(restored_model.outputs)
print(restored_model.inputs)

restored_model.summary()

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

frozen_graph = freeze_session(tf.compat.v1.keras.backend.get_session(),
                              output_names=[out.op.name for out in restored_model.outputs],
                              clear_devices=True)

tf.io.write_graph(frozen_graph, str(output_fld), output_model_name, as_text=False)
print("save pb successfully! ")
