import importlib
import tensorflow as tf
import logging
import os
import time

from typing import List, Dict

from tensorflow.keras import backend as K

from GraphicsDL.graphicsutils import g_cfg
from ..graphicstf import basic
from . import reader, solver, config


class CheckpointManager(object):
    def __init__(self, model_dir, pretrain_dir, nets: List[tf.keras.Model], solvers: List[solver.BaseSolverV2],
                 max_to_keep=60):
        self.nets_dict = {n_.name: n_ for n_ in nets}
        self.opts_dict = {s_.optimizer_name: s_.optimizer for s_ in solvers}
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), **self.opts_dict, **self.nets_dict)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, model_dir, max_to_keep=max_to_keep)
        self.pretrain_dir = pretrain_dir

    def restore_continuous(self, allow_unused=False) -> int:
        last_ckpt = self.ckpt_manager.latest_checkpoint
        epoch_trained = 0
        if last_ckpt:
            epoch_trained = int(last_ckpt.split('-')[-1])
            self.ckpt.restore(last_ckpt) if allow_unused else self.ckpt.restore(last_ckpt).expect_partial()
            logging.info(f'Load previous models from {last_ckpt} with continuous mode')
        return epoch_trained

    def restore_pretrain(self) -> int:
        if not os.path.exists(self.pretrain_dir):
            logging.info(f'There is no pretrain model dir, skip restore')
            return 0
        targets = [os.path.splitext(f_)[0] for f_ in os.listdir(self.pretrain_dir) if f_.endswith('index')]
        vars_dict = {}
        for t in targets:
            trained_vars = tf.train.load_checkpoint(os.path.join(self.pretrain_dir, t))
            trained_names = [t_[0] for t_ in tf.train.list_variables(os.path.join(self.pretrain_dir, t))
                             if t_[0].find('.ATTRIBUTES') != -1 and t_[0].find('.OPTIMIZER_SLOT') == -1]
            for t_n in trained_names:
                if t_n in vars_dict.keys():
                    logging.warning(f'Found duplicate keys in checkpoints {t_n}')
                    raise KeyError
                t_v = trained_vars.get_tensor(t_n)
                vars_dict[t_n] = t_v
        index = 0
        for k in vars_dict.keys():
            k_split = k.split('/')
            if k_split[0] in self.nets_dict.keys():
                k_attr = self.nets_dict[k_split[0]]
                for v_ in k_split[1:]:
                    if v_ == '.ATTRIBUTES':
                        break
                    if isinstance(k_attr, list):
                        k_attr = k_attr[int(v_)]
                    else:
                        k_attr = getattr(k_attr, v_)
                k_attr.assign(vars_dict[k])
                logging.info(f'{index}: Auto fetch variable from {k} to {k_attr.name}')
                index += 1
        return 0

    def restore(self, mode='auto') -> int:
        if mode == 'auto':
            epoch_trained = self.restore_continuous()
            if not epoch_trained:
                self.restore_pretrain()
        else:
            raise NotImplementedError
        return epoch_trained

    def save(self):
        self.ckpt_manager.save()


class BaseRunnerV2(object):
    def __init__(self, args: config.RunnerConfigurator):
        super().__init__()
        self.args = args

        self.reader_module = self.load_external_module('custom_reader')
        self.net_module = self.load_external_module('net_def')

        self.readers: List[reader.BaseReaderV2] = list()
        self.nets: List[tf.keras.Model] = list()
        self.losses: List[basic.BasicLossProxy] = list()
        self.metrics: List[basic.BasicLossProxy] = list()
        self.built = False
        self.initialize_environment()

        self.solvers: List[solver.BaseSolverV2] = list()
        self.validators: List[solver.BaseValidatorV2] = list()

        self.ckpt = None

    def load_external_module(self, module_name):
        try:
            example_module = self.args.example.replace('/', '.').replace('\\', '.')
            reader_module = importlib.import_module(f'{example_module}.{module_name}')
        except ModuleNotFoundError:
            reader_module = reader
        return reader_module

    @staticmethod
    def instance_case(cls_type, arg_dict: g_cfg.DictRecursive, external_dict=None):
        if external_dict is None:
            external_dict = dict()
        cls_kargs: Dict = arg_dict.match_function_args(external_dict, cls_type.__init__)
        logging.info(f'-- Instance: {cls_type.__init__}')
        for k in cls_kargs.keys():
            logging.info(f'---- Args: {k} / Value: {cls_kargs[k]}')
        return cls_type(**cls_kargs)

    def initialize_environment(self):
        # Initialize all reader
        if self.built:
            return
        [self.readers.append(self.instance_case(getattr(self.reader_module, r_.type), r_)) for r_ in self.args.readers]
        [self.nets.append(self.instance_case(getattr(self.net_module, n_.type), n_)) for n_ in self.args.nets]
        [self.losses.append(self.instance_case(getattr(self.net_module, l_.type), l_)) for l_ in self.args.losses]
        [self.metrics.append(self.instance_case(getattr(self.net_module, l_.type), l_)) for l_ in self.args.metrics]
        self.built = True

    def initialize_solvers(self):
        for arg_s in self.args.solvers:
            self.solvers.append(solver.BaseSolverV2(arg_s, self.nets, self.losses, self.metrics))

    def initialize_validator(self):
        for arg_v in self.args.validators:
            self.validators.append(solver.BaseValidatorV2(arg_v, self.nets, self.losses, self.metrics))

    def log_manager(self):
        train_log_dir = self.args.log_dir + '/train'
        val_log_dir = self.args.log_dir + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        return train_summary_writer, val_summary_writer

    def get_solvers_reader(self, instance):
        solvers_reader = list()
        for s in instance:
            solvers_reader.extend(s.sequence.readers)
        reader_dict = dict()
        for s_r in solvers_reader:
            if s_r in reader_dict.keys():
                continue
            reader_dict[s_r] = [r_ for r_ in self.readers if r_.name == s_r][0]
        return reader_dict

    def run_validate(self, validators_reader_dict: Dict[str, reader.BaseReaderV2],
                     out_callback, epoch, summary_writer=None, print_info=False):
        K.set_learning_phase(0)
        profiling_dir = 'prof_log' if self.args.profiling else None
        counter = 0
        total = time.time()
        while self.args.validators and epoch % self.args.validate_stride == 0 and epoch > 0:
            try:
                out_dict = dict()
                for a_v, v in zip(self.args.validators, self.validators):
                    cur = time.time()
                    if self.args.profiling:
                        tf.profiler.experimental.start(profiling_dir)
                    v_func = v.val_step_with_debug if self.args.debug else v.val_step
                    v_data = [validators_reader_dict[v_].next() for v_ in a_v.sequence.readers]
                    pre_time = time.time() - cur
                    cur = time.time()
                    out = [k_ for k_ in v_func(v_data)]
                    out_list = list()
                    for o in out:
                        if isinstance(o, list):
                            out_list.extend(o)
                        else:
                            out_list.append(o)
                    [validators_reader_dict[v_].post(out_list)
                     for v_ in a_v.sequence.readers]
                    for o_ in out:
                        [o_i_.numpy() for o_i_ in o_]
                    runtime = time.time() - cur
                    if self.args.profiling:
                        tf.profiler.experimental.stop()
                    if out_callback:
                        out_dict[a_v.name] = out
                    if print_info:
                        print(f'Preload: {pre_time} / Runtime: {runtime}')
                if out_callback:
                    out_callback.update(out_dict)
                if self.args.profiling and counter > 5:
                    raise StopIteration
                counter += 1
            except StopIteration:
                if print_info:
                    print(f'Total: {time.time() - total}')
                for v in self.validators:
                    v.apply_statistics(summary_writer, epoch, True, 'Val Epoch')
                    v.apply_vis_port(summary_writer, epoch)
                if out_callback:
                    out_callback.analysis(epoch)
                break

    def train(self):
        assert self.args.solver_mode == 'sequential'
        K.set_learning_phase(1)
        self.initialize_solvers()
        self.initialize_validator()
        train_summary_writer, val_summary_writer = self.log_manager()
        train_iter = 0
        solvers_reader_dict = self.get_solvers_reader(self.args.solvers)
        validators_reader_dict = self.get_solvers_reader(self.args.validators)
        validator_type = self.args.validator_type if self.args.validator_type else 'ValidatorCallback'
        out_callback = None if not hasattr(self.net_module, validator_type) \
            else getattr(self.net_module, validator_type)(self.args)
        for a_s, s in zip(self.args.solvers, self.solvers):
            s_func = s.train_step_with_debug if self.args.debug else s.train_step
            s_func([solvers_reader_dict[r_].next() for r_ in a_s.sequence.readers])
            s.reset_statistics()
        ckpt_manager = CheckpointManager(self.args.model_dir, self.args.pretrain_dir, self.nets, self.solvers)
        epoch_trained = ckpt_manager.restore()
        epoch_trained = epoch_trained if epoch_trained == 0 else (epoch_trained - 1) * self.args.epoch_stride
        # self.run_validate(validators_reader_dict, out_callback, 0, val_summary_writer)
        for e in range(epoch_trained, self.args.epochs):
            while True:
                K.set_learning_phase(1)
                try:
                    for a_s, s, s_r in zip(self.args.solvers, self.solvers, self.args.solver_ratio):
                        s_func = s.train_step_with_debug if self.args.debug else s.train_step
                        for _ in range(s_r):
                            s_func([solvers_reader_dict[r_].next() for r_ in a_s.sequence.readers])
                        if train_iter % self.args.show_iters == 0:
                            s.apply_statistics(None if self.args.epochs_stat else train_summary_writer, train_iter,
                                               False if self.args.epochs_stat else True, '-- Iter')
                            if not self.args.epochs_stat:
                                s.apply_vis_port(train_summary_writer, train_iter)
                    train_iter += 1
                except StopIteration:
                    if self.args.epochs_stat:
                        for s in self.solvers:
                            s.apply_statistics(train_summary_writer, e, True, 'Train Epoch')
                            s.apply_vis_port(train_summary_writer, train_iter)
                    self.run_validate(validators_reader_dict, out_callback, e, val_summary_writer, False)
                    if e % self.args.epoch_stride == 0:
                        ckpt_manager.save()
                    break

    def test(self, profiling=False):
        self.initialize_validator()
        validators_reader_dict = self.get_solvers_reader(self.args.validators)
        validator_type = self.args.validator_type if self.args.validator_type else 'TestCallback'
        out_callback = None if not hasattr(self.net_module, validator_type) \
            else getattr(self.net_module, validator_type)(self.args)
        ckpt_manager = CheckpointManager(self.args.model_dir, self.args.pretrain_dir, self.nets, self.solvers)
        epoch_trained = ckpt_manager.restore()
        self.run_validate(validators_reader_dict, out_callback, epoch_trained, None, False)

    def profiling(self):
        self.test(profiling=True)

    def perf(self):
        raise NotImplementedError
