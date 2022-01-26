import os
import importlib
import subprocess
import argparse
import numpy as np
import tensorflow as tf
from GraphicsDL.modules import config


def upgrade_tf_model(in_example_path, in_ckp_path, in_net_yaml_path, out_ckp_path, batch_size=None):
    """
    load ONNX support network -- replace custom ops

    usage example:
    upgrade_tf_model(
        'examples/mnist_deploy',
        'examples/mnist_deploy/ckp/model_iters.050000.ckpt',
        'examples/mnist_deploy/mnist.yaml',
        'examples/mnist_deploy/output/mnist_deploy')

    Args:
        in_example_name:        example name relative to root directory
        in_ckp_path:            in_ckp_path.meta, in_ckp_path.index, in_ckp_path.data-*
        in_net_yaml_path:       the filename of network.yaml
        out_ckp_path:           out_ckp_path.meta, out_ckp_path.index, out_ckp_path.data-*
        batch_size:             batch size for inference, None means accept any size

    Returns:
        onnx_inputs:            list of inputs, [[string, shape, type]]
        onnx_outputs:           list of outputs, [[string]]
    """

    # get net_def
    in_example_name = in_example_path.replace('/', '.').replace('\\', '.')
    cfg = config.RunnerConfigurator()
    cfg.load_from_yaml(in_net_yaml_path)
    net_cfg = cfg.solver.net
    net_module = importlib.import_module(f'{in_example_name}.net_def')
    net_type = getattr(net_module, net_cfg.type)    # now, the network defined in net_def is net_type

    # get inputs
    onnx_inputs = []
    inputs = []
    for in_param in cfg.reader.in_params:
        in_type = getattr(tf.dtypes, in_param.type)
        in_shape = in_param.raw_shape
        in_shape = [batch_size] + in_shape
        in_name = in_param.name
        onnx_inputs.append([in_name+':0', in_shape, in_type])
        inputs.append(tf.placeholder(in_type, in_shape, name=in_name))

    # build the network
    net_kargs = net_cfg.match_function_args({'is_training': 0}, net_type.__init__)
    net_ins = net_type(**net_kargs)
    net_ins.process_inputs(inputs)
    net_ins.build_network()
    outputs = net_ins.outputs()
    onnx_outputs = [x.name for x in outputs]

    # load saved checkpoint
    assign_ops = list()
    reader = tf.contrib.framework.load_checkpoint(in_ckp_path)
    all_vars = tf.contrib.framework.list_variables(in_ckp_path)
    selected_vars = [v for v in all_vars if v[0].find('Adam') == -1]
    all_net_vars = tf.get_collection(tf.GraphKeys.VARIABLES)
    for n_v in all_net_vars:
        s_v, = [v_ for v_ in selected_vars if n_v.name.find(v_[0]) != -1]
        s_t = reader.get_tensor(s_v[0])
        if not np.all(np.asarray(s_v[1]) == np.asarray(n_v.shape.as_list())):
            s_t = tf.transpose(s_t, [2, 3, 1, 0])
        assign_ops.append(tf.assign(n_v, s_t))

    # save upgraded tensorflow model
    out_dir, out_file = os.path.split(out_ckp_path)
    os.makedirs(out_dir, exist_ok=True)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(assign_ops)
        saver.save(sess, out_ckp_path)

    return onnx_inputs, onnx_outputs


def create_onnx(in_example_path, in_ckp_path, in_net_yaml_path, out_ckp_path, out_onnx_path, batch_size=None):
    onnx_inputs, onnx_outputs = upgrade_tf_model(
        in_example_path,
        in_ckp_path,
        in_net_yaml_path,
        out_ckp_path,
        batch_size)

    # onnx string
    onnx_inputs_str = ','.join(x[0] for x in onnx_inputs)
    onnx_outputs_str = ','.join(onnx_outputs)

    # create onnx
    cmd = 'python -m tf2onnx.convert ' \
          f'--checkpoint {out_ckp_path}.meta ' \
          f'--inputs {onnx_inputs_str} ' \
          f'--outputs {onnx_outputs_str} ' \
          f'--output {out_onnx_path} ' \
          '--opset 7'
    subprocess.call(cmd, shell=True)

    # write onnx io
    onnx_io_filename = out_onnx_path+'.io.txt'
    with open(onnx_io_filename, 'w') as f:
        f.write('onnx_network_in:\n')
        for in_data in onnx_inputs:
            f.write('%s    %s    %s\n' % (in_data[0], str(in_data[1]), str(in_data[2])))

        f.write('\nonnx_network_out:\n')
        for out_data in onnx_outputs:
            f.write(out_data)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='all path need to be absolute, or with respect to workspace root')
    parser.add_argument('--in_example_path', required=True, type=str, help='the input example path')
    parser.add_argument('--in_ckp_path', required=True, type=str, help='the input ckp files path, format is in_ckp_path.meta')
    parser.add_argument('--in_net_yaml_path', required=True, type=str, help='path and filename of .yaml file')
    parser.add_argument('--out_ckp_path', required=True, type=str, help='the output ckp files path, format is out_ckp_path.meta')
    parser.add_argument('--out_onnx_path', required=True, type=str, help='the output onnx file path')
    parser.add_argument('--batch_size', required=False, type=str, help='batch size, None means any size is acceptable', default='None')
    args = parser.parse_args()

    try:
        batch_size = int(args.batch_size)
    except ValueError:
        batch_size = None

    create_onnx(
        args.in_example_path,
        args.in_ckp_path,
        args.in_net_yaml_path,
        args.out_ckp_path,
        args.out_onnx_path,
        batch_size)
