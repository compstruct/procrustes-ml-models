import torch
from qe_cpp import qe


class Dropback(torch.optim.SGD):
    '''
    Dropback only support SGD and SGD with momentum
    Does not currently support Nesterov
    '''

    def __init__(self, params, lr, track_size=0, init_decay=1, proper_decay=False,
                 q=None, q_init=1e-2, q_step=1e-6, sf=False, ulp=False, beta=0.1,
                 momentum=0, weight_decay=0):
        '''
        weight_decay: gamma in lr decay setting
        decay_rate is the actual ratio that applies on init_param (lr in lr decay setting)
        q: target quantile (default None, corresponds to not apply qe)
        sf: stop fixed init scheme: quantile estimation init change based on mean of runtime estimation
        ulp: use last prediction: quantile estimation init change based on last value of runtime estimation
        '''
        super(Dropback, self).__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        # TODO: check if input values are valid

        self.debug_flag = True
        self.debug = {
            "tracked_weights": 0,
            "tracked_est": 0,
            "th_val": 0
        }

        for group in self.param_groups:
            init_params = []
            for p in group['params']:
                init_params.append(p.clone().detach())
            group['init_params'] = init_params
            group['track_size'] = track_size
            group['first_iter'] = True
            group['init_decay'] = init_decay
            group['decay_rate'] = 1
            group['proper_decay'] = proper_decay
            group['q'] = q
            group['q_init'] = q_init
            group['q_step'] = q_step
            group['sf'] = sf
            group['ulp'] = ulp
            group['beta'] = beta
        # save init weights to check?

    def get_decay_rate(self):
        '''Get decay rate of the optimizer'''
        return self.param_groups[0]['decay_rate']

    def step(self, closure=None, overwritten_decay_rate=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            overwritten_decay_rate (optional): Overwrite the decay rate. 
                The model will decay no matter if init_decay < 1
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # decay init weights
            if overwritten_decay_rate is not None:
                group['decay_rate'] = overwritten_decay_rate
            else:
                if not group['first_iter'] and group['init_decay'] < 1:
                    group['decay_rate'] *= group['init_decay']

            if group['first_iter']:
                group['first_iter'] = False

        super(Dropback, self).step(closure)
        # think and make sure it is a way that can be done in HW
        # evaluate and sort accumulated gradients (as an metric of importance)
        # mask off the non important weights back to initial weights
        for group in self.param_groups:
            abs_accumulated_all = []  # absolute value of accumulated gradients of the entire network
            for p, init_p in zip(group['params'], group['init_params']):
                if p.grad is None:
                    continue
                abs_accumulated_all.append(torch.abs(p - group['decay_rate'] * init_p).flatten().clone().detach())
            abs_accumulated_flatten = torch.cat(abs_accumulated_all)
            if group['q'] is not None:
                flattened_mask, flattened_est = qe(abs_accumulated_flatten.cpu(), group['q_init'], group['q_step'], group['q'])
                self.debug['tracked_weights'] = torch.mean(flattened_est)
                self.debug['tracked_est'] = torch.sum(flattened_mask)

                if self.debug_flag:
                    self.debug['tracked_est'] = torch.mean(flattened_est)
                    self.debug['tracked_weights'] = torch.sum(flattened_mask)
                    elements, ind = torch.topk(abs_accumulated_flatten, group['track_size'])
                    self.debug['th_val'] = torch.min(elements)

                # update init estimation for quantile
                if group['ulp']:
                    group['q_init'] = flattened_est[0]
                elif group['sf']:
                    group['q_init'] = group['beta'] * group['q_init'] + (1 - group['beta']) * torch.mean(flattened_est)
            else:
                elements, ind = torch.topk(abs_accumulated_flatten, group['track_size'])
                # create a mask that selects topk values
                flattened_mask = torch.zeros_like(abs_accumulated_flatten, dtype=torch.bool)
                flattened_mask.scatter_(0, ind, 1.)

                if self.debug_flag:
                    self.debug['th_val'] = torch.min(elements)

            start = 0
            for p, init_p in zip(group['params'], group['init_params']):
                if p.grad is None:
                    continue
                end = start + p.data.numel()
                mask = flattened_mask[start:end].view(p.size())
                p.data[~mask] = group['decay_rate'] * init_p.data[~mask]
                # param is decayed for next iteration inference
                if group['proper_decay'] and group['init_decay'] < 1:
                    p.data.add_(group['init_decay'] - 1, group['decay_rate'] * init_p)
                start = end
