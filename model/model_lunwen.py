import logging
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
from torch.cuda.amp import autocast
logger = logging.getLogger('base')
import math
from torch.cuda.amp import GradScaler
import numpy as np
class AFD(nn.Module):
    '''
    Pay Attention to Features, Transfer Learn Faster CNNs
    https://openreview.net/pdf?id=ryxyCeHtPB
    '''

    def __init__(self, in_channels, att_f):
        super(AFD, self).__init__()
        mid_channels = int(in_channels * att_f)

        self.attention = nn.Sequential(*[
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, 1, 0, bias=True)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @autocast()
    def forward(self, fm_s, fm_t, eps=1e-6):
        fm_t_pooled = F.adaptive_avg_pool2d(fm_t, 1)
        rho = self.attention(fm_t_pooled)
        # rho = F.softmax(rho.squeeze(), dim=-1)
        rho = torch.sigmoid(rho.squeeze())
        rho = rho / torch.sum(rho, dim=1, keepdim=True)

        fm_s_norm = torch.norm(fm_s, dim=(2, 3), keepdim=True)
        fm_s = torch.div(fm_s, fm_s_norm + eps)
        fm_t_norm = torch.norm(fm_t, dim=(2, 3), keepdim=True)
        fm_t = torch.div(fm_t, fm_t_norm + eps)

        loss = rho * torch.pow(fm_s - fm_t, 2).mean(dim=(2, 3))
        loss = loss.sum(1).mean(0)

        return loss
class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None
        self.schedule_phase_tc = None
        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
            self.loss_func = nn.L1Loss(reduction='sum').to(self.device)
####################################################################
       ###设置teach参数
        self.opt_tc=opt['model']['beta_schedule']['train'].copy()
        self.opt_tc['n_timestep']=2000
        self.netG_tc = self.set_device(networks.define_G(opt))
        self.set_loss_tc()
        self.set_new_noise_schedule_tc(
            self.opt_tc, schedule_phase='train')

        if self.opt['phase'] == 'train':
            self.netG_tc.eval()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG_tc.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params_tc = list(self.netG_tc.parameters())

            self.optG_tc = torch.optim.Adam(
                optim_params_tc, lr=opt['train']["optimizer"]["lr"])
            # self.log_dict = OrderedDict()
####################################################################################


        self.load_network()
        self.print_network()
        self.load_network_tc()

        self.AFD = AFD(in_channels=512, att_f=1).to(self.device)
        self.scaler = GradScaler()
    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self,current_step=None):
        b, c, h, w = self.data['HR'].shape
        with autocast():  # 建立autocast的上下文语句
                t_tc = np.random.randint(1, self.netG_tc.num_timesteps +1)
                time_s = math.ceil(
                    t_tc / (self.opt_tc['n_timestep'] / self.opt['model']['beta_schedule']['train']['n_timestep']))
                # self.scaler.step(self.optG)
                # # scaler factor更新
                # self.scaler.update()

                l_pix=self.netG_tc.distill_loss( self.netG, self.data, tt=t_tc,ts=time_s)
                # self.netG()
                self.scaler.scale(l_pix).backward()
                # scaler 更新参数，会先自动unscale梯度
                # 如果有nan或inf，自动跳过
                self.scaler.step(self.optG)
                self.scaler.update()

        self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['SR'], continous)
            else:
                self.SR = self.netG.super_resolution(
                    self.data['SR'], continous)
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)


    def set_loss_tc(self):
        if isinstance(self.netG_tc, nn.DataParallel):
            self.netG_tc.module.set_loss(self.device)
        else:
            self.netG_tc.set_loss(self.device)
    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def set_new_noise_schedule_tc(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase_tc is None or self.schedule_phase_tc  != schedule_phase:
            self.schedule_phase_tc  = schedule_phase
            if isinstance(self.netG_tc, nn.DataParallel):
                self.netG_tc.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG_tc.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']

    def load_network_tc(self):
            logger.info(
                'Loading teacher Network model for G ...')
            gen_path ="/home/hyyjs/yanjingke/OnlieImage-Super-Resolution-via-Iterative-Refinement-master/pre/I670000_E42_gen.pth"
            # gen
            network = self.netG_tc
            if isinstance(self.netG_tc, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            print(self.opt['model']['finetune_norm'])
