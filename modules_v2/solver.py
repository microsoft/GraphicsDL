import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import logging

from queue import LifoQueue
from typing import List, Dict, Optional, Tuple
from abc import abstractmethod
from .config import SolverConfigurator, ValidatorConfigurator, FlowConfigurator
from ..graphicstf.basic import BasicLossProxy


class LossScalarStatistics(object):
    def __init__(self, losses: List[BasicLossProxy], phase: str, alias: str):
        self.losses_dict = dict()
        for n_, loss in enumerate(losses):
            prefix = f'{phase}_{alias}'
            self.losses_dict[loss] = [
                tf.keras.metrics.Mean(f'{prefix}_{n_}_{i_}')
                for i_ in range(loss.num_losses)
            ]

    @tf.function
    def update(self, losses, loss_type: BasicLossProxy):
        losses_op = self.losses_dict[loss_type]
        [l_o_(l_) for l_o_, l_ in zip(losses_op, losses)]

    def apply(self, writer, step, refresh=False, prefix=''):
        out_str = f'{prefix} - {step}'
        for l_key in self.losses_dict.keys():
            for l_op, l_name in zip(self.losses_dict[l_key],
                                    l_key.losses_name):
                if writer:
                    with writer.as_default():
                        tf.summary.scalar(l_name, l_op.result(), step)
                out_str += f' - {l_name}: {l_op.result().numpy():.5}'
                if refresh:
                    l_op.reset_states()
        logging.info(out_str)

    def reset(self):
        for l_key in self.losses_dict.keys():
            for l_op, _ in zip(self.losses_dict[l_key], l_key.losses_name):
                l_op.reset_states()


class ImageScalarStatistics(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img_meta = None

    def build(self, input_shape):
        self.img_meta = self.add_weight('img_content',
                                        shape=input_shape,
                                        dtype=tf.float32)
        self.built = True

    def call(self, inputs, **kwargs):
        self.img_meta.assign(inputs)
        return self.img_meta


class ExecutionTree(object):
    def __init__(self, flow_str: str):
        self.flow_str = flow_str
        self.built = False

        self.ops_flow: List[Tuple[str, List]] = list()

    @staticmethod
    def traversal_path(flow_str):
        flow = LifoQueue()
        flow_runtime = flow_str
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
                pass
            else:
                pass
            index += 1
        flow_ops.append(flow_runtime)
        return flow_ops

    def build(self, nets):
        logging.warning('Experimental function')
        if not self.built:
            flow_ops = self.traversal_path(self.flow_str)
            for f_o in flow_ops:
                op_lists = f_o.split('-')
                in_list = op_lists[0]
                net_list = op_lists[1:]
                net_ins = list()
                for n in net_list:
                    net_ins.append(nets[int(n[1:])])
                self.ops_flow.append((in_list, net_ins))
        self.built = True

    def execute(self, data_list, gpu_id):
        i_kargs = dict(I='inputs', O='outputs', W='weights', A='alias')
        mediates: List = [None] * len(self.ops_flow)
        vis_ports: Dict[str, tf.Tensor] = dict()
        for o_id, o_f in enumerate(self.ops_flow):
            in_d, op_d = o_f
            in_lists = list()
            for i_ in in_d.split('&'):
                if i_.startswith('R'):
                    all_inputs = data_list[int(i_[1:-1])][i_kargs[i_[-1]]]
                    in_lists.extend([a_i_[gpu_id] for a_i_ in all_inputs])
                elif i_.startswith('M'):
                    mediate_inputs = mediates[int(i_[1:])]
                    if isinstance(mediate_inputs, list):
                        in_lists.extend(mediate_inputs)
                    else:
                        in_lists.append(mediate_inputs)
                else:
                    raise NotImplementedError
            out = in_lists
            for o_i, o_ in enumerate(op_d):
                out = o_(out)
                if isinstance(out, dict):
                    in_next = out['out']
                    del out['out']
                    out = {f'M{o_id}_O{o_i}_{k}': v for k, v in out.items()}
                    vis_ports = dict(**vis_ports, **out)
                    out = in_next
            mediates[o_id] = out
        return mediates[-1], vis_ports


class BaseFlower(object):
    def __init__(self, args: FlowConfigurator, nets: List[tf.keras.Model],
                 losses: List[BasicLossProxy], metrics: List[BasicLossProxy]):
        self.args = args
        self.nets: Dict[str, tf.keras.Model] = {
            n_.name: n_
            for n_ in nets if n_.name in args.sequence.nets
        }

        self.loss: Optional[BasicLossProxy] = None
        self.stats_loss: Optional[LossScalarStatistics] = None
        if args.sequence.loss:
            self.loss = [l_ for l_ in losses
                         if l_.name == args.sequence.loss][0]
            self.stats_loss = LossScalarStatistics([self.loss],
                                                   f'{self.args.name}_Train',
                                                   'losses')
        self.metric: Optional[BasicLossProxy] = None
        self.stats_metric: Optional[LossScalarStatistics] = None
        if args.sequence.metric:
            self.metric = [
                m_ for m_ in metrics if m_.name == args.sequence.metric
            ][0]
            self.stats_metric = LossScalarStatistics([self.metric],
                                                     f'{self.args.name}_Train',
                                                     'metrics')

        self.execution_tree = ExecutionTree(self.args.sequence.flow)
        self.execution_tree.build(
            [self.nets[k_] for k_ in self.args.sequence.nets])

        self.vis_port: Dict[str, ImageScalarStatistics] = dict()

        if self.args.g_checker:
            logging.info(
                "Gradient checker is enable. NaN or Infinite gradients will be ignored automatically!")

    @staticmethod
    @tf.function()
    def average_and_rename_vars(tower_vars):
        avg_vars = list()
        for vs in zip(*tower_vars):
            avg_vars.append(
                tf.identity(tf.reduce_mean(tf.stack(vs, axis=0), axis=0)))
        return avg_vars

    def tower_train_step_with_debug(self, next_data):
        tower_grads = list()
        tower_losses = list()
        tower_metrics = list()
        for g in range(self.args.num_devices):
            with tf.device(f'/gpu:{g}'):
                grads, losses, metrics = self.step_kernel(next_data, g)
            tower_grads.append(grads)
            tower_losses.append(losses)
            tower_metrics.append(metrics)
        return tower_grads, tower_losses, tower_metrics

    @abstractmethod
    def step_kernel(self, next_data, gpu_id):
        pass

    def apply_statistics(self, writer, step, refresh, prefix):
        if self.stats_loss:
            self.stats_loss.apply(writer, step, refresh, prefix)
        if self.stats_metric:
            self.stats_metric.apply(writer, step, refresh, prefix)

    def reset_statistics(self):
        if self.stats_loss:
            self.stats_loss.reset()
        if self.stats_metric:
            self.stats_metric.reset()

    def apply_vis_port(self, writer, step):
        if not writer:
            return
        with writer.as_default():
            [
                tf.summary.image(key,
                                 self.vis_port[key].img_meta,
                                 step,
                                 max_outputs=4) for key in self.vis_port
            ]


class BaseSolverV2(BaseFlower):
    def __init__(self, args: SolverConfigurator, nets: List[tf.keras.Model],
                 losses: List[BasicLossProxy], metrics: List[BasicLossProxy]):
        super().__init__(args, nets, losses, metrics)
        self.optimizer_name = args.optimizer.name
        self.optimizer: tf.keras.optimizers.Optimizer = self.parse_optimizer()
        self.trainable_vars = list()

    def parse_optimizer(self) -> tf.keras.optimizers.Optimizer:
        try:
            opt_type = getattr(tf.keras.optimizers, self.args.optimizer.type)
        except AttributeError:
            if self.args.optimizer.type == 'Range':
                def opt_type(x): return tfa.optimizers.Lookahead(
                    tfa.optimizers.RectifiedAdam(x),
                    sync_period=6,
                    slow_step_size=0.5)
            else:
                raise NotImplementedError
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
        return opt_type(lr_scheme, *opt_args[1:])

    def get_trainable_vars(self):
        if not self.trainable_vars:
            for t in self.args.sequence.trainable:
                self.trainable_vars.extend(self.nets[t].trainable_variables)
        return self.trainable_vars

    @staticmethod
    def average_tower_grads(tower_grads):
        avg_grads = list()
        for grads in zip(*tower_grads):
            avg_grads.append(tf.reduce_mean(tf.stack(grads, axis=0), axis=0))
        return avg_grads

    @staticmethod
    def multi_gpus_splitter(in_data, gpu_id):
        o_data = list()
        for i in in_data:
            o_data.append(i[gpu_id])
        return o_data

    @tf.function
    def train_step(self, next_data):
        self.train_step_with_debug(next_data)

    def train_step_with_debug(self, next_data):
        tower_grads, tower_losses, tower_metrics = self.tower_train_step_with_debug(
            next_data)
        avg_grads = self.average_tower_grads(tower_grads)
        for g, v in zip(avg_grads, self.get_trainable_vars()):
            assert g.shape == v.shape
        self.optimizer.apply_gradients(
            zip(avg_grads, self.get_trainable_vars()))
        self.stats_loss.update(self.average_and_rename_vars(tower_losses),
                               self.loss)
        if self.stats_metric:
            self.stats_metric.update(
                self.average_and_rename_vars(tower_metrics), self.metric)

    def step_kernel(self, next_data, gpu_id):
        with tf.GradientTape() as tape:
            final_out, vis_out = self.execution_tree.execute(next_data, gpu_id)
            losses = self.loss(final_out)
            losses_sum = tf.add_n(losses)
        grads = tape.gradient(losses_sum, self.get_trainable_vars())
        if self.args.g_checker:
            grads = [
                tf.where(tf.reduce_all(tf.math.is_finite(g_)), g_,
                         tf.zeros_like(g_)) for g_ in grads
            ]
        metrics = self.metric(final_out) if self.metric else None
        if vis_out:
            extra_vis = {
                key_: ImageScalarStatistics()
                for key_ in vis_out if key_ not in self.vis_port.keys()
            }
            self.vis_port = dict(**self.vis_port, **extra_vis)
        for key in vis_out:
            self.vis_port[key](vis_out[key])
        return grads, losses, metrics


class BaseValidatorV2(BaseFlower):
    def __init__(self, args: ValidatorConfigurator, nets: List[tf.keras.Model],
                 losses: List[BasicLossProxy], metrics: List[BasicLossProxy]):
        super().__init__(args, nets, losses, metrics)

    @tf.function
    def val_step(self, next_data):
        return self.val_step_with_debug(next_data)

    def val_step_with_debug(self, next_data):
        tower_outs, tower_losses, tower_metrics = self.tower_train_step_with_debug(
            next_data)
        if self.stats_loss:
            self.stats_loss.update(self.average_and_rename_vars(tower_losses),
                                   self.loss)
        if self.stats_metric:
            self.stats_metric.update(
                self.average_and_rename_vars(tower_metrics), self.metric)
        return tower_outs

    def step_kernel(self, next_data, gpu_id):
        final_out, vis_out = self.execution_tree.execute(next_data, gpu_id)
        losses = self.loss(final_out) if self.loss else None
        metrics = self.metric(final_out) if self.metric else None
        if vis_out:
            extra_vis = {
                key_: ImageScalarStatistics()
                for key_ in vis_out if key_ not in self.vis_port.keys()
            }
            self.vis_port = dict(**self.vis_port, **extra_vis)
        for key in vis_out:
            self.vis_port[key](vis_out[key])
        return final_out, losses, metrics
