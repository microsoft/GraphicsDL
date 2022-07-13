"""
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
"""
import importlib
import logging
import os

from typing import List, Dict

import tensorflow as tf

from graphics_dl.graphicsutils import config as g_cfg
from graphics_dl.modules_v2.reader.base_reader import BaseReaderV2
from ..graphicstf import basic
from ..toolbox.data_dump import DataDump
from . import reader, solver, config


class CheckpointManager(object):
    """
    Checkpoint manager
    """
    def __init__(self, model_dir, pretrain_dir, nets: List[tf.keras.Model],
                 solvers: List[solver.BaseSolverV2], max_to_keep=200):
        self.nets_dict = {n_.name: n_ for n_ in nets}
        self.opts_dict = {s_.optimizer_name: s_.optimizer for s_ in solvers}
        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(1), **self.opts_dict, **self.nets_dict)
        if not os.path.exists(f'{model_dir}.index'):
            self.model_path = None
        else:
            self.model_path = model_dir
            model_dir = os.path.dirname(model_dir)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, model_dir, max_to_keep=max_to_keep)
        self.pretrain_dir = pretrain_dir

    def restore_continuous(self, allow_unused=False) -> int:
        """
        Restore the unfinished training
        """
        if self.model_path is None:
            last_ckpt = self.ckpt_manager.latest_checkpoint
        else:
            last_ckpt = self.model_path
        epoch_trained = 0
        if last_ckpt:
            epoch_trained = int(last_ckpt.split('-')[-1])
            self.ckpt.restore(last_ckpt) if allow_unused else self.ckpt.restore(
                last_ckpt).expect_partial()
            logging.info(
                f'Load previous models from {last_ckpt} with continuous mode')
        return epoch_trained

    def restore_pretrain(self) -> int:
        """
        Restore from a trained model
        """
        if self.pretrain_dir is None or not os.path.exists(self.pretrain_dir):
            logging.info(f'There is no pretrain model dir, skip restore')
            return 0
        targets = [os.path.splitext(f_)[0] for f_ in os.listdir(
            self.pretrain_dir) if f_.endswith('index')]
        vars_dict = {}
        for t in targets:
            trained_vars = tf.train.load_checkpoint(os.path.join(self.pretrain_dir, t))
            trained_names = [t_[0] for t_ in tf.train.list_variables(os.path.join(self.pretrain_dir, t))
                             if t_[0].find('.ATTRIBUTES') != -1 and t_[0].find('.OPTIMIZER_SLOT') == -1]
            for t_n in trained_names:
                if t_n in vars_dict.keys():
                    logging.warning(
                        f'Found duplicate keys in checkpoints {t_n}')
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
                logging.info(
                    f'{index}: Auto fetch variable from {k} to {k_attr.name}')
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
    """
    Runner will manage the whole training/test
    """
    def __init__(self, args: config.RunnerConfigurator):
        super().__init__()
        self.args = args

        self.reader_module = self._load_module('custom_reader', reader)
        self.net_module = self._load_module('net_def')
        self.loss_module = self._load_module('loss_def')
        self.callback_module = self._load_module('callback_def')

        self.readers: List[reader.BaseReaderV2] = list()
        self.nets: List[tf.keras.Model] = list()
        self.losses: List[basic.BasicLossProxy] = list()
        self.metrics: List[basic.BasicLossProxy] = list()
        self.built = False
        self._initialize_environment()

        self.solvers: List[solver.BaseSolverV2] = list()
        self.validators: List[solver.BaseValidatorV2] = list()

        self.ckpt = None

    def _load_module(self, module_name, default=None):
        try:
            example_module = self.args.example.replace(
                '/', '.').replace('\\', '.')
            loaded_module = importlib.import_module(
                f'{example_module}.{module_name}')
        except ModuleNotFoundError:
            loaded_module = default
        return loaded_module

    @staticmethod
    def _instance_case(cls_type, arg_dict: g_cfg.DictRecursive, external_dict=None):
        if external_dict is None:
            external_dict = dict()
        cls_kargs: Dict = arg_dict.match_function_args(external_dict, cls_type.__init__)
        return cls_type(**cls_kargs)

    def _initialize_environment(self):
        """
        Initialize all reader
        """
        if self.built:
            return
        for r_args in self.args.readers:
            reader_instance = getattr(self.reader_module, r_args.type)
            reader_case: BaseReaderV2 = self._instance_case(reader_instance, r_args)
            if self.args.dict_mode:
                reader_case.enable_dict_mode()
            self.readers.append(reader_case)

        [self.nets.append(self._instance_case(
            getattr(self.net_module, n_.type), n_)) for n_ in self.args.nets]
        [self.losses.append(self._instance_case(
            getattr(self.loss_module, l_.type), l_)) for l_ in self.args.losses]
        [self.metrics.append(self._instance_case(
            getattr(self.loss_module, l_.type), l_)) for l_ in self.args.metrics]
        self.built = True

    def _initialize_solvers(self):
        for arg_s in self.args.solvers:
            self.solvers.append(
                solver.BaseSolverV2(
                    arg_s,
                    self.nets,
                    self.losses,
                    self.metrics,
                    self.args.dict_mode))

    def _initialize_validator(self):
        for arg_v in self.args.validators:
            self.validators.append(
                solver.BaseValidatorV2(
                    arg_v,
                    self.nets,
                    self.losses,
                    self.metrics,
                    self.args.dict_mode))

    def _log_manager(self):
        train_log_dir = self.args.log_dir + '/train'
        val_log_dir = self.args.log_dir + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        return train_summary_writer, val_summary_writer

    def _get_solvers_reader(self, instance):
        solvers_reader = list()
        for s in instance:
            solvers_reader.extend(s.sequence.readers)
        reader_dict = dict()
        for s_r in solvers_reader:
            if s_r in reader_dict.keys():
                continue
            reader_dict[s_r] = [r_ for r_ in self.readers if r_.name == s_r][0]
        return reader_dict

    def _run_validate(self, validators_reader_dict: Dict[str, reader.BaseReaderV2],
                     out_callback, epoch, summary_writer=None):
        """
        Run a validation pass
        """
        while self.args.validators and epoch % self.args.validate_stride == 0:
            try:
                out_dict = dict()
                in_dict = dict()
                for val_args, val_case in zip(self.args.validators, self.validators):
                    v_func = val_case.val_step_with_debug if self.args.debug else val_case.val_step

                    v_data = [validators_reader_dict[v_].next() for v_ in val_args.sequence.readers]
                    outs = list(v_func(v_data))
                    for val_reader in validators_reader_dict.values():
                        val_reader.post(outs[0])
                    if out_callback:
                        out_dict[val_args.name] = outs
                        in_dict[val_args.name] = v_data
                if out_callback:
                    out_callback.update(out_dict, in_dict)
                DataDump().increment()
            except StopIteration:
                out_dict = dict()
                for val_case in self.validators:
                    v_dict = val_case.apply_statistics(summary_writer, epoch, True,
                                                'Val Epoch')
                    out_dict = dict(**v_dict, **out_dict)
                    val_case.apply_vis_port(summary_writer, epoch)
                if out_callback:
                    out_callback.close(out_dict)
                break

    def _run_optimize(self, solvers_reader_dict: Dict[str, reader.BaseReaderV2],
        train_iter: int, summary_writer):
        solver_zips = zip(self.args.solvers, self.solvers, self.args.solver_ratio)
        for s_args, s_case, s_readers in solver_zips:
            s_func = s_case.train_step_with_debug if self.args.debug else s_case.train_step
            for _ in range(s_readers):
                s_func([solvers_reader_dict[r_].next() for r_ in s_args.sequence.readers])
            if train_iter % self.args.show_iters == 0 and not self.args.epochs_stat:
                s_case.apply_statistics(summary_writer, train_iter, True, '-- Iter')
                s_case.apply_vis_port(summary_writer, train_iter)

    def _run_epoch_statistics(self, val_reader_dict: Dict[str, reader.BaseReaderV2],
        epoch, out_callback, summary_writer, ckpt_manager):
        if self.args.epochs_stat:
            for s in self.solvers:
                s.apply_statistics(summary_writer, epoch, True, 'Train Epoch')
                s.apply_vis_port(summary_writer, epoch)
        self._run_validate(val_reader_dict, out_callback, epoch, summary_writer)
        if epoch % self.args.epoch_stride == 0:
            ckpt_manager.save()

    def train(self):
        """
        The training phase
        """
        assert self.args.solver_mode == 'sequential'
        self._initialize_solvers()
        self._initialize_validator()

        train_summary_writer, val_summary_writer = self._log_manager()

        slv_reader_dict = self._get_solvers_reader(self.args.solvers)
        val_reader_dict = self._get_solvers_reader(self.args.validators)
        validator_type = self.args.validator_type if self.args.validator_type\
            else 'ValidatorCallback'

        out_callback = None if not hasattr(self.net_module, validator_type)\
            else getattr(self.net_module, validator_type)(self.args)

        ckpt_manager = CheckpointManager(self.args.model_dir, self.args.pretrain_dir,
            self.nets, self.solvers)
        epoch_trained = ckpt_manager.restore()
        epoch_trained = epoch_trained * self.args.epoch_stride
        train_iter = 0
        for epoch in range(epoch_trained, self.args.epochs):
            while True:
                try:
                    self._run_optimize(slv_reader_dict, train_iter, train_summary_writer)
                    train_iter += 1
                except StopIteration:
                    self._run_epoch_statistics(val_reader_dict, epoch, out_callback,\
                        val_summary_writer, ckpt_manager)
                    break

    def test(self, profiling=False):
        """
        The test phase
        """
        assert not profiling
        self._initialize_validator()
        validators_reader_dict = self._get_solvers_reader(self.args.validators)
        out_callback = None
        if len(self.args.callbacks):
            out_callback_cls = getattr(self.callback_module, 'TestCallback')
            out_callback = self._instance_case(out_callback_cls, self.args.callbacks[0])
        ckpt_manager = CheckpointManager(
            self.args.model_dir,
            self.args.pretrain_dir,
            self.nets,
            self.solvers)
        epoch_trained = ckpt_manager.restore()
        if out_callback is not None and not out_callback.initialize(epoch_trained):
            logging.info('Skip evaluate Epoch: %d...', epoch_trained)
            return
        self._run_validate(
            validators_reader_dict,
            out_callback,
            epoch_trained,
            None)
