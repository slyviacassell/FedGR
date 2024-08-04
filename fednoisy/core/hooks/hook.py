# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/hook.py


class Hook:
    stages = ('before_run', 
              'before_train_epoch', 'before_train_step', 'after_train_step', 'after_train_epoch',
              'after_run')

    def before_train_epoch(self, algorithm):
        pass

    def after_train_epoch(self, algorithm):
        pass

    def before_train_step(self, algorithm):
        pass

    def after_train_step(self, algorithm):
        pass
    
    def before_run(self, algorithm):
        pass

    def after_run(self, algorithm):
        pass

    def every_n_epochs(self, algorithm, n):
        return (algorithm.epoch + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, algorithm, n):
        return (algorithm.it + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, algorithm):
        return algorithm.it + 1 % len(algorithm.data_loader['train_lb']) == 0

    def is_last_epoch(self, algorithm):
        return algorithm.epoch + 1 == algorithm.epochs

    def is_last_iter(self, algorithm):
        return algorithm.it + 1 == algorithm.num_train_iter
    
    def every_n_round(self, client_trainer_or_sever_handler, n):
        return ((client_trainer_or_sever_handler.round + 1) % n == 0) or (client_trainer_or_sever_handler.round == 0) if n > 0 else False


class SerialClientTrainerHook(Hook):
    stages = (
        'on_init',
        'on_local_process_start',
        'on_local_process_end',
        'on_client_serial_process_start',
        'on_client_serial_process_end',
        'on_client_training_start',
        'on_client_training_end',
        'on_training_epoch_start',
        'on_training_epoch_end',
        'on_training_batch_start',
        'on_training_batch_end',
        'on_training_step_start',
        'on_training_step_end',
        'on_final_epoch_start',
        'on_final_epoch_end',
        'on_final_iter_start',
        'on_final_iter_end',
    )

    def on_init(self, algorithm, *args, **kwargs):
        pass

    # hook entries
    def on_local_process_start(self, algorithm, *args, **kwargs):
        pass

    def on_local_process_end(self, algorithm, *args, **kwargs):
        pass

    def on_client_serial_process_start(self, algorithm, *args, **kwargs):
        pass

    def on_client_serial_process_end(self, algorithm, *args, **kwargs):
        pass

    def on_client_training_start(self, algorithm, *args, **kwargs):
        pass

    def on_client_training_end(self, algorithm, *args, **kwargs):
        pass

    def on_training_epoch_start(self, algorithm, *args, **kwargs):
        pass

    def on_training_epoch_end(self, algorithm, *args, **kwargs):
        pass

    def on_training_batch_start(self, algorithm, *args, **kwargs):
        pass

    def on_training_batch_end(self, algorithm, *args, **kwargs):
        pass

    def on_training_step_start(self, algorithm, *args, **kwargs):
        pass

    def on_training_step_end(self, algorithm, *args, **kwargs):
        pass

    def on_optimize_step_start(self, algorithm, *args, **kwargs):
        pass

    def on_optimize_step_end(self, algorithm, *args, **kwargs):
        pass

    def on_final_epoch_start(self, algorithm, *args, **kwargs):
        pass

    def on_final_epoch_end(self, algorithm, *args, **kwargs):
        pass

    def on_final_iter_start(self, algorithm, *args, **kwargs):
        pass

    def on_final_iter_end(self, algorithm, *args, **kwargs):
        pass


class SyncServerHook(Hook):
    stages = (
        'on_global_update_start',
        'on_global_update_end',
    )

    # hook entries
    def on_global_update_start(self, algorithm, *args, **kwargs):
        pass
    
    def on_global_update_end(self, algorithm, *args, **kwargs):
        pass


class StandalonePipelineHook(Hook):
    stages = (
        'on_init',
        'on_round_start',
        'on_round_end',
    )

    def on_init(self, algorithm, *args, **kwargs):
        pass

    def on_round_start(self, algorithm, *args, **kwargs):
        pass

    def on_round_end(self, algorithm, *args, **kwargs):
        pass