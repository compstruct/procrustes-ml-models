import numpy
import torch
# A simple hook class that returns the input and output of a layer during forward/backward pass
class Hook():
    def __init__(self, module, module_name='undef', enabled= False, backward=False):       
        self.name= module_name
        #attaching hooks
        self.hook = module.register_forward_hook(self.fw_hook_fn)
        self.hook = module.register_backward_hook(self.bw_hook_fn)
        # collecting ?
        self.enabled = enabled
        # lists
        self.in_acts = []
        self.in_acts_mask = []
        self.out_acts = []
        self.out_acts_mask = []
        self.out_grads = []
        self.out_grads_mask = []
        self.in_grads = []
        self.in_grads_mask = []
        #
        self.gradients = self.out_grads
        self.activations = self.in_acts
        # dumping 
        self.dump_inited = False
        self.dump_path = './'
        self.act_bl_file= None
        self.grd_bl_file= None
    def fw_hook_fn(self, module, input, output):
        if self.enabled:
            #IN
            self.in_act = input[0] # input is a tuple
            self.in_acts.append(self.in_act)
            self.in_acts_mask.append(self.get_mask(self.in_act))
            #OUT
            self.out_act = output.data # output is a tensor
            self.out_acts.append(self.out_act)
            self.out_acts_mask.append(self.get_mask(self.out_act))
            
    def bw_hook_fn(self, module, input, output):
        if self.enabled:
            #OUT
            self.out_grad = output[0] # grad output is a tuple
            self.out_grads.append(self.out_grad)
            self.out_grads_mask.append(self.get_mask(self.out_grad))
            #IN
            self.in_grad = input[0] # grad input is a tuple
            self.in_grads.append(self.in_grad)
            self.in_grads_mask.append(self.get_mask(self.in_grad))
            
    def close(self):
        self.hook.remove()
    def print_act(self):
        print(self.activations)
    def print_grad(self):
        print(self.gradients)
    def act(self):
        return self.activations
    def grd(self):
        return self.gradients
    def enable(self):
        self.enabled = True
    def disable(self):
        self.enabled = False
    def dump_init(self, dump_path, numpy_only=True):
        if not self.dump_inited:
            self.dump_path = dump_path
            #if not numpy_only:
            #    self.act_bl_file = open(dump_path+'_layer-' + self.name + '_act_val_b.t', 'a+')
            #    self.grd_bl_file = open(dump_path+'_layer-' + self.name + '_grd_val_b.t', 'a+')
            self.dump_inited = True

    def get_mask(self, in_tensor):
        if in_tensor is None:
            print("One tensor is None!")
            return None
        in_t_size = in_tensor.size()
        in_flat_t = in_tensor.clone().detach().view(1,-1)
        idx = in_flat_t.nonzero(as_tuple=True)
        mask_t = torch.zeros(in_flat_t.size(),dtype=torch.bool)
        mask_t[idx] = 1
        mask_tensor = mask_t.reshape(in_t_size)

        return mask_tensor

    def dump_array(self, inters, inter_name='_undef_act'):
        if inters is None:
            print(inter_name, 'of layer', self.name, 'is None!')
        elif None in  inters:
            print(inter_name, 'of layer', self.name, 'has a None member!!')
        else:
            inter_arr = [numpy.array(inter_t.cpu().detach()) for inter_t in inters] # convert array of tensors to numpy array
            numpy.save(self.dump_path+'_layer-' + self.name + inter_name, inter_arr)
    
    def dump_arrays(self):
        # save all the array
        self.dump_array(self.in_acts, '_in_acts')
        self.dump_array(self.in_acts_mask, '_in_acts_mask')
        self.dump_array(self.out_acts, '_out_acts')
        self.dump_array(self.out_acts_mask, '_out_acts_mask')
        self.dump_array(self.out_grads, '_out_grads')
        self.dump_array(self.out_grads_mask, '_out_grads_mask')
        self.dump_array(self.in_grads, '_in_grads')
        self.dump_array(self.in_grads_mask, '_in_grads_mask')
        

'''
    def dump_itt(self, itt_idx):
        self.dump_act(itt_idx)
        self.dump_grd(itt_idx)

    def dump_act(self, itt_idx):
        """
                # 1D how many itterations
                # 2D how many inputs in each batch
                # 3D activation_map size 
        """
        self.act_bl_file.write( '------------------------------------------------ itteration_%d\n' % itt_idx)
        h_act = self.activations
        print("Act_len is:", len(h_act))
        print(type(self.activations))
        h_act_last = h_act[len(h_act)-1] #last itteration
        print(type(h_act_last))
        # for all
        batch = h_act_last
        image_idx = -1
        for _image in batch:
            print(type(_image))
            image_idx = image_idx +1
#                    act_vl_file.write("# ================== image_id_in_batch %d\n" % image_idx )
            self.act_bl_file.write("# ================== image_id_in_batch %d\n" % image_idx )

            # for each  channels in that batch
            channels = _image
            for _channel in channels:
                print(type(_channel))
                flattened_channel = _channel.view(1,-1)
                flattened_channel_b = flattened_channel != 0

                flattened_channel_b = [int(el) for el in flattened_channel_b.data.tolist()[0]] # mask
                self.act_bl_file.write("%s\n" % str(flattened_channel_b) )

    def dump_grd(self, itt_idx):
        """
                # 1D how many itterations
                # 2D how many inputs in each batch
                # 3D activation_map size 
        """
        self.grd_bl_file.write( '------------------------------------------------ itteration_%d\n' % itt_idx)
        h_grd = self.gradients
        print("Grd_len is:", len(h_grd))
        h_grd_last = h_grd[len(h_grd)-1] #last itteration
        # for all
        batch = h_grd_last
        image_idx = -1
        for _image in batch:
            image_idx = image_idx +1
            self.grd_bl_file.write("# ================== image_id_in_batch %d\n" % image_idx )
            # for each  channels in that batch
            channels = _image
            for _channel in channels:
                flattened_channel = _channel.view(1,-1)
                flattened_channel_b = flattened_channel != 0
                flattened_channel_b = [int(el) for el in flattened_channel_b.data.tolist()[0]] # mask
                self.grd_bl_file.write("%s\n" % str(flattened_channel_b) )

'''
