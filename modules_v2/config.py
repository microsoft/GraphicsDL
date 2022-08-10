"""
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
"""

from ..graphicsutils import config as g_cfg


class DataItemConfigurator(g_cfg.DictRecursive):
    """
        The description of each key in readers/datasets
    """
    def __init__(self):
        super().__init__()
        self.name = str()
        self.raw_name = str()
        self.type = str()
        self.quantized = int(0)
        self.raw_shape = list([0])
        self.crop_shape = list([0])
        self.process_group = int(0)
        # pre-process will process data before batching with CPU context
        self.preprocess = str()
        # post-process will process data after batching with GPU context
        self.postprocess = list([str()])

    def get_raw_name(self):
        """
        get compatible raw name
        """
        return self.raw_name if self.raw_name else self.name


class ReaderConfigurator(g_cfg.DictRecursive):
    """
    The description of reader
    """
    def __init__(self):
        super().__init__()
        self.type = str()
        self.name = str()
        self.compress = str()
        self.rel_path = str()
        self.split = int(0)
        self.data_dir = str()
        self.batch_size = int(0)
        self.num_devices = int(0)
        self.shuffle = int(0)
        self.interleave = int(0)
        self.prefix = str()
        self.num_samples = int(0)
        self.infinite = int(0)
        self.in_params = list([DataItemConfigurator()])
        self.out_params = list([DataItemConfigurator()])
        self.w_params = list([DataItemConfigurator()])
        self.data_format = 'channels_last'


class NetworkConfigurator(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.type = str()
        self.name = str()
        self.out_channels = int(0)
        self.data_format = 'channels_last'


class LossConfigurator(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.type = str()
        self.name = str()
        self.flag = int(0)


class OptimizerConfigurator(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.type = str()
        self.name = str()
        self.params = str()
        self.lr_decay = str()


class SequenceConfigurator(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.nets = list([str()])
        self.readers = list([str()])
        self.loss = list([str()])
        self.trainable = list([str()])
        self.metric = list([str()])
        self.flow = str()


class FlowConfigurator(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.type = str()
        self.name = str()
        self.num_devices = int(0)
        self.g_checker = int(0)
        self.sequence = SequenceConfigurator()
        self.optimizer = OptimizerConfigurator()


class SolverConfigurator(FlowConfigurator):
    def __init__(self):
        super().__init__()
        self.partial_mode = 0


class ValidatorConfigurator(FlowConfigurator):
    def __init__(self):
        super().__init__()


class CallbackConfigurator(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.type = str()
        self.compress = str()
        self.eval_keys = list([str()])
        self.result_root = str()
        self.dataset_name = str()


class RunnerConfigurator(g_cfg.DictRecursive):
    """
    RunnerConfigurator is the root entry. It will organize all necessary information.

    Attributes:
        log_dir(str): the directory to save the tensorboard logs
        model_dir(str): the directory to save the trained models
        pretrain_dir(str): the directory to save the pretrain models, used for fine-tune
            networks
        example(str): the directory to the target example, which will help system to
            load custom modules (reader, config).
        type(str): the runner type used for training, current only support `BaseRunnerV2`.
        epochs(int): total epochs to train the model.
        epochs_stat(int): will be disabled, please do not use.
        show_iters(int): iterations stride to show the progress.
        validate_stride(int): per validation stride, counted by epochs.
        debug(int): 0 for runtime mode, 1 for debug mode. The whole training will be
            executed in non-eager mode in debug mode. Breakpoints are free to use in
            network forward pass or even during optimizing.
        readers(list): a list of ``ReaderConfigurator``, the properties of all readers
            used in the system.
        nets(list): a list of ``NetworkConfigurator``, the properties of all networks
            used in the system.
        losses(list): a list of ``LossConfigurator``, the properties of all losses
            functions used in the system.
        metrics(list): a list of ``LossConfigurator``, the properties of all metrics
            functions used in the system.
        solvers(list): a list of ``SolverConfigurator``, the properties of all solvers
            used in the system.
        validators(list): a list of ``ValidatorConfigurator``, the properties of all
            validators used in the system.
        solver_mode(str): the collaborate mode among all solvers, current only support
            `sequential` mode.
        solver_ratio(list): a list of ``int``, the optimized ratio among all solvers
            for a single step.
    """

    def __init__(self):
        super().__init__()
        #
        self.profiling = int(0)
        # System parameters, loading from command line
        self.log_dir = str()
        self.model_dir = str()
        self.pretrain_dir = str()
        self.result_dir = str()
        self.example = str()
        # Custom parameters, loaded from custom config
        self.type = str()
        self.epochs = int(0)
        # 1 for epoch mode, 0 for iteration mode
        self.epochs_stat = int(1)
        self.epoch_stride = int(1)              # epochs to save model
        # iterations to print log on screen
        self.show_iters = int(0)
        self.validate_stride = int(1)
        self.validator_type = str()
        # whether to enable tf.function within runner
        self.debug = int(0)
        self.readers = [ReaderConfigurator()]
        self.nets = [NetworkConfigurator()]
        self.losses = [LossConfigurator()]
        self.metrics = [LossConfigurator()]
        self.solvers = [SolverConfigurator()]
        self.validators = [ValidatorConfigurator()]
        self.callbacks = [CallbackConfigurator()]
        # Flow controller
        self.solver_mode = str()
        self.solver_ratio = list([int(0)])
        # Key mode
        self.dict_mode = False
