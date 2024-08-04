from collections import OrderedDict

from fednoisy.core.hooks import (
    Hook, 
    get_priority,
)

class AlogrithmBase:
    def __init__(self, *args, **kwargs) -> None:
        self.hooks_dict = OrderedDict()
        self._hooks = []

    def register_hooks(self, hook, name=None, priority="NORMAL"):
        """
        Ref: https://github.com/open-mmlab/mmcv/blob/a08517790d26f8761910cac47ce8098faac7b627/mmcv/runner/base_runner.py#L263
        Register a hook into the hook list.
        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            hook_name (:str, default to None): Name of the hook to be registered. Default is the hook class name.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority  # type: ignore
        hook.name = name if name is not None else type(hook).__name__

        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:  # type: ignore
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        
        if not inserted:
            self._hooks.insert(0, hook)

        # call set hooks
        self.hooks_dict = OrderedDict()
        for hook in self._hooks:
            self.hooks_dict[hook.name] = hook

    def call_hook(self, fn_name, hook_name=None, *args, **kwargs):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
            hook_name (str): The specific hook name to be called, such as
                "param_update" or "dist_align", uesed to call single hook in train_step.
        """
        
        if hook_name is not None:
            return getattr(self.hooks_dict[hook_name], fn_name)(self, *args, **kwargs)
        
        # call fn_name in all hooks
        for hook in self.hooks_dict.values():
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self, *args, **kwargs)

    def set_hooks(self):
        pass


class SerialClientAlogrithmBase(AlogrithmBase):
    def __init__(self, *args, **kwargs) -> None:
        super(SerialClientAlogrithmBase, self).__init__(*args, **kwargs)

    def on_init(self, *args, **kwargs):
        self.call_hook("on_init", None, *args, **kwargs)

    # callback & hook entries
    def on_local_process_start(self, *args, **kwargs):
        self.call_hook("on_local_process_start", None, *args, **kwargs)

    def on_local_process_end(self, *args, **kwargs):
        self.call_hook("on_local_process_end", None, *args, **kwargs)

    def on_client_serial_process_start(self, *args, **kwargs):
        self.call_hook("on_client_serial_process_start", None, *args, **kwargs)

    def on_client_serial_process_end(self, *args, **kwargs):
        self.call_hook("on_client_serial_process_end", None, *args, **kwargs)

    def on_client_training_start(self, *args, **kwargs):
        self.call_hook("on_client_training_start", None, *args, **kwargs)

    def on_client_training_end(self, *args, **kwargs):
        self.call_hook("on_client_training_end", None, *args, **kwargs)

    def on_training_epoch_start(self, *args, **kwargs):
        self.call_hook("on_training_epoch_start", None, *args, **kwargs)

    def on_training_epoch_end(self, *args, **kwargs):
        self.call_hook("on_training_epoch_end", None, *args, **kwargs)

    def on_training_batch_start(self, *args, **kwargs):
        self.call_hook("on_training_batch_start", None, *args, **kwargs)

    def on_training_batch_end(self, *args, **kwargs):
        self.call_hook("on_training_batch_end", None, *args, **kwargs)

    def on_optimize_step_start(self, *args, **kwargs):
        self.call_hook("on_optimize_step_start", None, *args, **kwargs)

    def on_optimize_step_end(self, *args, **kwargs):
        self.call_hook("on_optimize_step_end", None, *args, **kwargs)

    def on_training_step_start(self, *args, **kwargs):
        self.call_hook("on_training_step_start", None, *args, **kwargs)

    def on_training_step_end(self, *args, **kwargs):
        self.call_hook("on_training_step_end", None, *args, **kwargs)


class SynServerAlogrithmBase(AlogrithmBase):
    def __init__(self, *args, **kwargs) -> None:
        super(SynServerAlogrithmBase, self).__init__(*args, **kwargs)

    def on_init(self, *args, **kwargs):
        self.call_hook("on_init", None, *args, **kwargs)

    def on_global_update_start(self, *args, **kwargs):
        self.call_hook("on_global_update_start", None, *args, **kwargs)

    def on_global_update_end(self, *args, **kwargs):
        self.call_hook("on_global_update_end", None, *args, **kwargs)