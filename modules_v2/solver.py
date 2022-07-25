"""
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
"""
# pylint: disable=no-name-in-module
import logging
import inspect
from queue import LifoQueue
from typing import List, Dict, Optional, Tuple
from abc import abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras import layers

from graphics_dl.utils.log import LogOnce

from .config import SolverConfigurator, ValidatorConfigurator, FlowConfigurator
from ..graphicstf.basic import BasicLossProxy


class LossScalarStatistics(object):
    """
    The loss scalar statistics
    """
    def __init__(self, losses: List[BasicLossProxy], phase: str, alias: str):
        stat_func = tf.keras.metrics.Mean
        self.losses_dict: Dict[BasicLossProxy, List[tf.keras.metrics.Metric]] = dict()
        for idx, loss in enumerate(losses):
            prefix = f'{phase}_{alias}'
            self.losses_dict[loss] = [
                stat_func(f'{prefix}_{idx}_{_i}') for _i in range(loss.num_losses)
            ]

    @tf.function
    def update(self, losses, losses_type: BasicLossProxy):
        """
        Update a tick value
        """
        losses_op = list()
        for l_type in losses_type:
            losses_op.extend(self.losses_dict[l_type])
        for loss_op, loss in zip(losses_op, losses):
            if isinstance(loss, (tuple, list)):
                loss, weight = loss[:2]
            else:
                loss = tf.where(tf.math.is_finite(loss), loss, loss_op.result())
                weight = tf.ones_like(loss)
            loss_op(loss, sample_weight=weight)

    def apply(self, writer, step, refresh=False, prefix=''):
        """
        Compute the loss value
        """
        out_str = f'{prefix} - {step}'
        out_dict = dict()
        for l_key, l_value in self.losses_dict.items():
            for l_op, l_name in zip(l_value, l_key.losses_name):
                if writer:
                    with writer.as_default():
                        tf.summary.scalar(l_name, l_op.result(), step)
                l_result = l_op.result().numpy()
                out_str += f' - {l_name}: {l_result:.5}'
                out_dict[l_name] = l_result
                if refresh:
                    l_op.reset_states()
        logging.info(out_str)
        return out_dict

    def reset(self):
        """
        Reset the loss state
        """
        for _, l_value in self.losses_dict.items():
            for l_op in l_value:
                l_op.reset_states()


class ImageScalarStatistics(layers.Layer):  # pylint: disable=abstract-method
    """
    Image visualization interface for TensorBoard
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img_meta = None

    def build(self, input_shape):
        self.img_meta = self.add_weight('img_content',
                                        shape=input_shape,
                                        dtype=tf.float32)
        self.built = True

    def call(self, inputs, *args, **kwargs):
        del args, kwargs
        self.img_meta.assign(tf.cast(inputs, self.img_meta.dtype))
        return self.img_meta


class ExecutionTree(object):
    """
    The execution tree will parse the string-defined flow and serialize into multiple
        forward passes
    """
    def __init__(self, flow_str: str, dict_mode: bool = False):
        self.flow_str = flow_str
        self.built = False
        self.dict_mode = dict_mode

        self.ops_flow: List[Tuple[str, List]] = list()

    @staticmethod
    def traversal_path(flow_str):
        """
        Serialize the string-defined flow
        """
        flow = LifoQueue()
        flow_runtime: str = flow_str
        flow_ops = list()
        index = 0
        mediate_mask = 0
        while index < len(flow_runtime):
            if flow_runtime[index] == '(':
                flow.put(index)
            elif flow_runtime[index] == ')':
                s_index = flow.get()
                ctx = flow_runtime[s_index + 1:index]
                flow_ops.append(ctx)
                flow_runtime = flow_runtime.replace(f'({ctx})',
                                                    f'M{mediate_mask}', 1)
                index = s_index + 2
                mediate_mask += 1
            else:
                pass
            index += 1
        flow_ops.append(flow_runtime)
        return flow_ops

    def build(self, nets):
        """
        Build the network passes
        """
        logging.warning('Experimental function')
        if not self.built:
            flow_ops: List[str] = self.traversal_path(self.flow_str)
            for f_op in flow_ops:
                ops_desc_seq = f_op.split('-')
                in_desc = ops_desc_seq[0]
                nets_desc = ops_desc_seq[1:]
                net_ins = list()
                for net_desc in nets_desc:
                    net_ins.append(nets[int(net_desc[1:])])
                self.ops_flow.append((in_desc, net_ins))
        self.built = True

    def execute(self, data_list, gpu_id, training):
        """
        Execute the predefined workflow
        """
        i_kargs = dict(I='inputs', O='outputs', W='weights', A='alias')
        mediates: List = [None] * len(self.ops_flow)
        vis_ports: Dict[str, tf.Tensor] = dict()
        for flow_index, op_flow in enumerate(self.ops_flow):
            in_deps, op_defs = op_flow
            in_tensors = dict() if self.dict_mode else list()
            for in_dep in in_deps.split('&'):
                if in_dep.startswith('R'):
                    all_inputs = data_list[int(in_dep[1:-1])][i_kargs[in_dep[-1]]]
                    if self.dict_mode:
                        for a_in in all_inputs.items():
                            in_tensors[a_in[0]] = a_in[1][gpu_id]
                    else:
                        in_tensors.extend([a_i_[gpu_id] for a_i_ in all_inputs])
                elif in_dep.startswith('M'):
                    mediate_inputs = mediates[int(in_dep[1:])]
                    if self.dict_mode:
                        in_tensors = dict(in_tensors, **mediate_inputs)
                    else:
                        if not isinstance(mediate_inputs, list):
                            mediate_inputs = [mediate_inputs]
                        in_tensors.extend(mediate_inputs)
                else:
                    raise NotImplementedError
            out = in_tensors
            for op_index, op_def in enumerate(op_defs):
                out = op_def(out, training=training)
                if not self.dict_mode and isinstance(out, dict):
                    in_next = out['out']
                    del out['out']
                    out = {f'M{flow_index}_O{op_index}_{k}': v for k, v in out.items()}
                    vis_ports = dict(**vis_ports, **out)
                    out = in_next
                elif isinstance(out, dict):
                    vis_ports = dict(**vis_ports, **out)
            mediates[flow_index] = out
        return mediates[-1], vis_ports


class BaseFlower(object):
    """
    The base flower with unimplemented method `step_kernel`
    """
    def __init__(self, args: FlowConfigurator, nets: List[tf.keras.Model],
                 losses: List[BasicLossProxy], metrics: List[BasicLossProxy],
                 dict_mode: bool = False):
        self.args = args
        self.nets: Dict[str, tf.keras.Model] = {
            _n.name: _n
            for _n in nets if _n.name in args.sequence.nets
        }

        self.losses: Optional[BasicLossProxy] = list()
        self.stats_loss: Optional[LossScalarStatistics] = None
        if args.sequence.loss:
            self.losses.extend([l_ for l_ in losses if l_.name in args.sequence.loss])
            self.stats_loss = LossScalarStatistics(self.losses,
                                                   f'{self.args.name}_Train',
                                                   'losses')
        self.metrics: Optional[BasicLossProxy] = list()
        self.stats_metric: Optional[LossScalarStatistics] = None
        if args.sequence.metric:
            self.metrics.extend([
                m_ for m_ in metrics if m_.name in args.sequence.metric
            ])
            self.stats_metric = LossScalarStatistics(self.metrics,
                                                     f'{self.args.name}_Train',
                                                     'metrics')

        self.execution_tree = ExecutionTree(self.args.sequence.flow, dict_mode)
        self.execution_tree.build(
            [self.nets[k_] for k_ in self.args.sequence.nets])

        self.vis_port: Dict[str, ImageScalarStatistics] = dict()

        if self.args.g_checker:
            logging.info("Gradient checker is enable. NaN/INF gradients will be\
                ignored automatically!")

    @staticmethod
    def average_and_rename_vars(tower_vars):
        """
        Compute the average results from multiple towers
        """
        avg_vars = list()
        for t_var in zip(*tower_vars):
            if not isinstance(t_var[0], (tuple, list)):
                avg_vars.append(tf.reduce_mean(tf.stack(t_var, axis=0), axis=0))
                continue
            a_vars_v = tf.stack([_a_v[0] for _a_v in t_var], axis=0)
            a_vars_w = tf.stack([_a_v[1] for _a_v in t_var], axis=0)
            a_vars_sum = tf.reduce_sum(a_vars_w, axis=0)
            a_vars_wr = tf.cast(a_vars_w, tf.float32) / (tf.cast(a_vars_sum, tf.float32) + 1e-8)
            a_vars_avg = tf.reduce_sum(a_vars_v * tf.cast(a_vars_wr, a_vars_v.dtype), axis=0)
            avg_vars.append([a_vars_avg, a_vars_sum])
        return avg_vars

    def tower_train_step_with_debug(self, next_data):
        """
        For multiple GPU traininng
        """
        tower_grads = list()
        tower_losses = list()
        tower_metrics = list()
        tower_vis_dict = list()
        for g_id in range(self.args.num_devices):
            with tf.device(f'/gpu:{g_id}'):
                grads, losses, metrics, vis_dict = self.step_kernel(next_data, g_id)
            tower_grads.append(grads)
            tower_losses.append(losses)
            tower_metrics.append(metrics)
            tower_vis_dict.append(vis_dict)
        return tower_grads, tower_losses, tower_metrics, tower_vis_dict

    @abstractmethod
    def step_kernel(self, next_data, gpu_id):
        """
        The logic of iteration definition
        """
        pass

    def apply_statistics(self, writer, step, refresh, prefix):
        """
        Add a statistics item
        """
        out_dict = dict()
        if self.stats_loss:
            loss_dict = self.stats_loss.apply(writer, step, refresh, prefix)
            out_dict = dict(**loss_dict, **out_dict)
        if self.stats_metric:
            metric_dict = self.stats_metric.apply(
                writer, step, refresh, prefix)
            out_dict = dict(**metric_dict, **out_dict)
        return out_dict

    def reset_statistics(self):
        """
        Reset the statistics
        """
        if self.stats_loss:
            self.stats_loss.reset()
        if self.stats_metric:
            self.stats_metric.reset()

    def apply_vis_port(self, writer, step):
        """
        Write visualizable data into TensorBoard
        """
        if not writer:
            return
        with writer.as_default():
            for key, value in self.vis_port.items():
                img_meta: tf.Tensor = value.img_meta
                if img_meta.shape.ndims == 5:
                    img_meta = img_meta[0]
                elif img_meta.shape.ndims > 5:
                    LogOnce(f'Unsupport visualized for {key} with shape {img_meta.shape}')
                    continue

                if img_meta.shape.as_list()[-1] not in [1, 3, 4]:
                    LogOnce(f'Visualization failed for unsupported channels of {key}')
                    continue
                tf.summary.image(key, img_meta, step, max_outputs=4)


class BaseSolverV2(BaseFlower):
    """
    Base solver
    """
    def __init__(self, args: SolverConfigurator, nets: List[tf.keras.Model],
                 losses: List[BasicLossProxy], metrics: List[BasicLossProxy],
                 dict_mode: bool = False):
        super().__init__(args, nets, losses, metrics, dict_mode)
        self.optimizer_name = args.optimizer.name
        self.partial_mode = args.partial_mode
        self.optimizer: tf.keras.optimizers.Optimizer = self.parse_optimizer()
        self.trainable_vars = list()

    def parse_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """
        Parse the optimizer description into optimizer
        """
        opt_args = np.fromstring(self.args.optimizer.params, sep='-')
        if self.args.optimizer.lr_decay:
            opt_lr_decay = self.args.optimizer.lr_decay.split('_')
            lr_decay_type = getattr(tf.keras.optimizers.schedules,
                                    opt_lr_decay[0])
            lr_decay_args = [
                np.fromstring(s_, sep='-') for s_ in opt_lr_decay[1:]
            ]
            if lr_decay_type == tf.keras.optimizers.schedules.PiecewiseConstantDecay:
                lr_decay_steps, lr_decay_mul = lr_decay_args
                lr_decay_steps = np.cumsum(lr_decay_steps).astype(
                    np.int32).tolist()
                lr_decay_mul = (opt_args[0] * lr_decay_mul).tolist()
                lr_scheme = lr_decay_type(lr_decay_steps, lr_decay_mul)
            else:
                raise NotImplementedError
        else:
            lr_scheme = opt_args[0]
        try:
            opt_type = getattr(tf.keras.optimizers, self.args.optimizer.type)
            return opt_type(lr_scheme, *opt_args[1:])
        except AttributeError:
            if self.args.optimizer.type == 'Range':
                learning_rate = tf.convert_to_tensor(lr_scheme, tf.float32)
                return tfa.optimizers.Lookahead(
                    tfa.optimizers.RectifiedAdam(learning_rate),
                    sync_period=6,
                    slow_step_size=0.5)
            elif self.args.optimizer.type == 'AdamW':
                step = tf.Variable(0, trainable=False)
                return tfa.optimizers.AdamW(
                    weight_decay=1e-4 * lr_scheme(step), learning_rate=lr_scheme(step))
            else:
                raise NotImplementedError  # pylint: disable=raise-missing-from

    def get_trainable_vars(self):
        """
        Get all trainable variables from given name scope
        """
        if not self.trainable_vars:
            for t_scope in self.args.sequence.trainable:
                scopes = t_scope.split('.')
                var_scopes = scopes[1:] if len(scopes) > 1 else list()
                if var_scopes:
                    assert self.partial_mode == 1
                cur_net = self.nets[scopes[0]]
                if not var_scopes:
                    LogOnce(f'{__name__} fetches trainable vars with net -- {scopes[0]}')
                    self.trainable_vars.extend(cur_net.trainable_variables)
                    continue
                cur_scopes = list()
                for v_scope in var_scopes:
                    for m_name, m_type in inspect.getmembers(cur_net):
                        if not isinstance(m_type, (tf.keras.Model, tf.keras.layers.Layer)):
                            continue
                        if m_type.name != v_scope and m_name != v_scope:
                            continue
                        cur_scopes.append(v_scope)
                        cur_net = m_type
                cur_scope = '.'.join(tuple(cur_scopes))
                if f'{scopes[0]}.{cur_scope}' == t_scope:
                    LogOnce(f'{__name__} fetches trainable vars with scope -- {t_scope}')
                    self.trainable_vars.extend(cur_net.trainable_variables)
                else:
                    LogOnce(f'{__name__} unable to fetch trainable vars with scope -- ' +\
                        f'{t_scope} (nearest queried: {cur_scope})')
        return self.trainable_vars

    def average_tower_grads(self, tower_grads):
        """
        Average multiple GPUs gradients
        """
        avg_grads = list()
        none_gradient = False
        for g_idx, grads in enumerate(zip(*tower_grads)):
            if grads[0] is None:
                none_gradient = True
                LogOnce(f'None gradients to {self.get_trainable_vars()[g_idx].name}')
            else:
                LogOnce(f'Apply gradients to {self.get_trainable_vars()[g_idx].name}')
                avg_grads.append(tf.reduce_mean(tf.stack(grads, axis=0), axis=0))
        assert not none_gradient
        return avg_grads

    @staticmethod
    def multi_gpus_splitter(in_data, gpu_id):
        """
        Split data into GPUs
        """
        o_data = list()
        for i in in_data:
            o_data.append(i[gpu_id])
        return o_data

    @tf.function
    def train_step(self, next_data):
        """
        Train a iteration with debug off
        """
        self.train_step_with_debug(next_data)

    def train_step_with_debug(self, next_data):
        """
        Train a iteration with debug on
        """
        tower_grads, tower_losses, tower_metrics, _ = self.tower_train_step_with_debug(
            next_data)
        avg_grads = self.average_tower_grads(tower_grads)
        for grad, var in zip(avg_grads, self.get_trainable_vars()):
            if grad.shape == var.shape:
                continue
            LogOnce(f'Unmatched gradient and variable shape {var.name}')
        self.optimizer.apply_gradients(
            zip(avg_grads, self.get_trainable_vars()))
        self.stats_loss.update(self.average_and_rename_vars(tower_losses),
                               self.losses)
        if self.stats_metric:
            self.stats_metric.update(
                self.average_and_rename_vars(tower_metrics), self.metrics)

    def step_kernel(self, next_data, gpu_id):
        with tf.GradientTape() as tape:
            training = True if not self.partial_mode else self.args.sequence.trainable
            final_out, vis_po = self.execution_tree.execute(next_data, gpu_id, training)
            losses = list()
            vis_pl = dict()
            for loss in self.losses:
                loss_res, loss_vis = loss(final_out)
                losses.extend(loss_res)
                vis_pl = dict(vis_pl, **loss_vis)
            losses_sum = tf.add_n(losses)
        grads = tape.gradient(losses_sum, self.get_trainable_vars())
        if self.args.g_checker:
            grads = [
                tf.where(tf.reduce_all(tf.math.is_finite(g_)), g_,
                         tf.zeros_like(g_)) for g_ in grads
            ]
        vis_pm = dict()
        metrics = list()
        if self.metrics:
            for metric in self.metrics:
                metric_res, metric_vis = metric(final_out)
                metrics.extend(metric_res)
            vis_pm = dict(vis_pm, **metric_vis)
        vis_out = dict(**vis_po, **vis_pl, **vis_pm)
        if vis_out:
            extra_vis = {
                key_: ImageScalarStatistics()
                for key_ in vis_out if key_ not in self.vis_port.keys()
            }
            self.vis_port = dict(**self.vis_port, **extra_vis)
        for key, value in vis_out.items():
            self.vis_port[key](value)
        return grads, losses, metrics, vis_out


class BaseValidatorV2(BaseFlower):
    """
    The validator
    """
    def __init__(self, args: ValidatorConfigurator, nets: List[tf.keras.Model],
                 losses: List[BasicLossProxy], metrics: List[BasicLossProxy],
                 dict_mode: bool = False):
        super().__init__(args, nets, losses, metrics, dict_mode)

    @tf.function
    def val_step(self, next_data):
        """
        Start validation
        """
        return self.val_step_with_debug(next_data)

    def val_step_with_debug(self, next_data):
        """
        Start validation with debug information
        """
        outs, losses, metrics, vis_results = self.tower_train_step_with_debug(
            next_data)
        if self.stats_loss:
            self.stats_loss.update(self.average_and_rename_vars(losses), self.losses)
        if self.stats_metric:
            self.stats_metric.update(self.average_and_rename_vars(metrics), self.metrics)
        return outs, losses, metrics, vis_results

    def step_kernel(self, next_data, gpu_id):
        final_out, vis_po = self.execution_tree.execute(next_data, gpu_id, False)

        losses_out = list()
        vis_pl = dict()
        for loss in self.losses:
            loss_out, loss_vis = loss(final_out)
            losses_out.extend(loss_out)
            vis_pl = dict(vis_pl, **loss_vis)

        metrics_out = list()
        vis_pm = dict()
        for metric in self.metrics:
            metric_out, metric_vis = metric(final_out)
            metrics_out.extend(metric_out)
            vis_pm = dict(vis_pm, **metric_vis)

        vis_out = dict(**vis_po, **vis_pl, **vis_pm)
        return final_out, losses_out, metrics_out, vis_out
