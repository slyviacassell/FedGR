"""
This file is for learning rate scheduler for different algorithms
"""

from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR, LambdaLR
import math

def get_lr_scheduler(args, optimizer,last_epoch):
    if args.lr_scheduler=="none":
        return None
    elif args.lr_scheduler=="step":
        return StepLR(optimizer=optimizer,step_size=args.step_size,gamma=args.step_gamma,last_epoch=last_epoch)
    elif args.lr_scheduler=="multistep":
        return MultiStepLR(optimizer=optimizer,milestones=args.multistep_milestone,gamma=args.step_gamma,last_epoch=last_epoch)
    elif args.lr_scheduler=="cosine":
        return get_cosine_schedule_with_warmup(optimizer=optimizer,num_training_steps=args.com_round,num_warmup_steps=args.warmup_round,last_epoch=last_epoch)
    else:
        raise ValueError(
            f"args.lr_scheduler='{args.lr_scheduler}' is not supported. Only support 'step'."
        )
    
def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''
    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an integer parameter epochs.
        Decaying criteria: last_epoch
        '''

        if current_step < num_warmup_steps:
            # _lr = float(current_step) / float(max(1, num_warmup_steps))
            _lr = 1.0
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)