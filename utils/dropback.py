import torch
import numpy as np

class Dropback(torch.optim.SGD):
    '''
    Dropback only support SGD and SGD with momentum
    Does not currently support Nesterov
    '''

    def __init__(self, params, lr, track_size=0, init_decay=1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, named_params=[]):
        super(Dropback, self).__init__(params, lr=lr, momentum=momentum, dampening=dampening,
                                       weight_decay=weight_decay, nesterov=nesterov)
        # TODO: check if input values are valid

        self.named_params = named_params
        self.dump_path= './'
        self.dump_inited= False
        self.dump_flag=False

        for group in self.param_groups:
            init_params = []
            for p in group['params']:
                init_params.append(p.clone().detach())
            group['init_params'] = init_params
            group['track_size'] = track_size
            group['first_iter'] = True
            group['init_decay'] = init_decay

        # save init weights to check?

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            is_init_decay = group['init_decay'] < 1
            # decay init weights
            if not group['first_iter'] and is_init_decay:
                for init_p in group['init_params']:
                    init_p *= group['init_decay']

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
                abs_accumulated_all.append(torch.abs(p - init_p).flatten().clone().detach())
            abs_accumulated_flatten = torch.cat(abs_accumulated_all)
            _, ind = torch.topk(abs_accumulated_flatten, group['track_size'])
            # create a mask that selects topk values
            flattened_mask = torch.zeros_like(abs_accumulated_flatten, dtype=torch.bool)
            flattened_mask.scatter_(0, ind, 1.)

            start = 0
            layer_id=0
            total_non_zero=0
            total_size= sum([param.nelement() for param in group['params']])
            for p, init_p, (n_p,p_p) in zip(group['params'], group['init_params'], self.named_params):
                if p.grad is None:
                    continue
                end = start + p.data.numel()
                mask = flattened_mask[start:end].view(p.size())
                p.data[~mask] = init_p.data[~mask]
                start = end
                if self.dump_flag:
                    mask_to_dump = np.array(mask.cpu().detach())
                    mask_nonZero = np.sum(mask_to_dump)
                    total_non_zero += mask_nonZero
                    mask_sparsity = mask_nonZero / mask_to_dump.size # how many nonZero elements
                    mask_portion = mask_to_dump.size / total_size # What portion of total weights are these layer's Ws
                    layer_id_name=str(layer_id)+n_p.replace('module','').replace('.','_')
                    #print(layer_id_name, '\t,#nz', mask_nonZero)
                    self.dump_array(mask_to_dump ,layer_id_name)
                    self.dump_summary_sparsity(mask_sparsity , mask_portion, layer_id_name)

                layer_id = layer_id + 1
            #print('Total Nonzeros,', total_non_zero)
    def dump_summary_sparsity(self, sp=0 , mp=0, layer_name='x_def'):
            f = open(self.dump_path+'_summary_sparsity.txt', 'a+')
            f.write(layer_name+', '+ str(sp)+', '+ str(mp)+'\n')

    def dump_array(self, w_arr, layer_name='x_def'):
        if w_arr is None:
            print(layer_name, 'of layer', layer_name, 'is None!')
        else:
            np.save(self.dump_path+'_'+layer_name+'_W_mask', w_arr)
    def dump_init(self, dump_path):
        if not self.dump_inited:
            self.dump_path = dump_path
            self.dump_inited = True
            print("Weights masks are under:", dump_path)
            f = open(self.dump_path+'_summary_sparsity.txt', 'w+')
            f.write('layer_name, nz_portion, w_portion\n')
    def enable_dumping(self):
        self.dump_flag = True
    def disable_dumping(self):
        self.dump_flag = False
    
