"""
Created on 2023/11/23
@author: Chen Qian
@e-mail: chenqian2020@sjtu.edu.cn
"""

from Models.TFconvlayer import *
from Models.WaveletRLConv import WaveletShrinkageConv1d
from Models.BackboneCNN import CNN
from torch.nn import Conv1d
from utils.mysummary import summary


class Base_FUNC_CNN(CNN):
    """
    the base class of TFN
    """
    FuncConv1d = BaseFuncConv1d
    funckernel_size = 21

    def __init__(self, in_channels=1, out_channels=10, kernel_size=15, clamp_flag=True, mid_channel=16):
        super().__init__(in_channels, out_channels, kernel_size)
        # Reinitialize the first layer by changing the in_channels
        args = {x: getattr(self.layer1[0], x) for x in
                ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'bias']}
        args['bias'] = None if (args['bias'] is None) else True
        args['in_channels'] = mid_channel
        self.layer1[0] = nn.Conv1d(**args)
        # use the TFconvlayer as the first preprocessing layer
        funconv_kwargs = getattr(self, 'funconv_kwargs', {})
        self.funconv = self.FuncConv1d(in_channels, mid_channel, self.funckernel_size,
                                       padding=self.funckernel_size // 2,
                                       bias=False, clamp_flag=clamp_flag,
                                       **funconv_kwargs)
        self.superparams = self.funconv.superparams

    def forward(self, x):
        x = self.funconv(x)
        return super().forward(x)

    def getweight(self):
        """
        get the weight and superparams of the first preprocessing layer (for recording)
        """
        weight_tensor = getattr(self.funconv, 'weight', None)
        weight = weight_tensor.cpu().detach().numpy() if weight_tensor is not None else None
        superparams = self.funconv.superparams.cpu().detach().numpy()
        return weight, superparams

    def update_reinforcement(self, reward: float, done: bool = False, loss: float = None, phase: str = 'train'):
        if hasattr(self.funconv, 'update_agent') and self.training and phase == 'train':
            self.funconv.update_agent(reward=reward, done=done, loss=loss)

    def on_epoch_end(self, phase: str = 'train'):
        if hasattr(self.funconv, 'agent_cache') and self.training and phase == 'train':
            # Flush any pending cache to avoid stale references between epochs
            self.funconv.agent_cache = None


class TFN_STTF(Base_FUNC_CNN):
    """
    TFN with TFconv-STTF as the first preprocessing layer
    FuncConv1d = TFconv_STTF
    kernel_size = mid_channel * 2 - 1
    """
    FuncConv1d = TFconv_STTF
    def __init__(self, mid_channel=16, **kwargs):
        self.funckernel_size = mid_channel * 2 - 1
        super().__init__(mid_channel=mid_channel, **kwargs)


class TFN_Chirplet(Base_FUNC_CNN):
    """
    TFN with TFconv-Chirplet as the first preprocessing layer
    FuncConv1d = TFconv_Chirplet
    kernel_size = mid_channel * 2 - 1
    """
    FuncConv1d = TFconv_Chirplet
    def __init__(self, mid_channel=16, **kwargs):
        self.funckernel_size = mid_channel * 2 - 1
        super().__init__(mid_channel=mid_channel, **kwargs)


class TFN_Morlet(Base_FUNC_CNN):
    """
    TFN with TFconv-Morlet as the first preprocessing layer
    FuncConv1d = TFconv_Morlet
    kernel_size = mid_channel * 10 - 1
    """
    FuncConv1d = TFconv_Morlet
    def __init__(self, mid_channel=16, **kwargs):
        self.funckernel_size = mid_channel * 10 - 1
        super().__init__(mid_channel=mid_channel, **kwargs)


class Random_conv(Conv1d):
    """
    traditional Conv1d with random weight
    """
    def __init__(self, *pargs, **kwargs):
        new_kwargs = {k:v for k,v in kwargs.items() if k in ['in_channels','out_channels','kernel_size','stride','padding','bias']}
        super().__init__(*pargs, **new_kwargs)
        self.superparams = self.weight

class Random_CNN(Base_FUNC_CNN):
    """
    Backbone-CNN with Random_conv as the first preprocessing layer
    """
    FuncConv1d = Random_conv
    def __init__(self, mid_channel=16, **kwargs):
        self.funckernel_size = mid_channel * 2 - 1
        super().__init__(mid_channel=mid_channel, **kwargs)


class TFN_WaveletRL(Base_FUNC_CNN):
    """TFN variant using the wavelet shrinkage layer with D3QN selection."""

    FuncConv1d = WaveletShrinkageConv1d

    def __init__(self, mid_channel=16, wavelet_types=("morlet", "mexhat", "laplace"),
                 agent_hidden=128, agent_gamma=0.98, agent_buffer_size=2048,
                 agent_batch_size=64, agent_lr=1e-3, agent_epsilon_start=1.0,
                 agent_epsilon_end=0.05, agent_epsilon_decay=0.995, agent_tau=0.01,
                 threshold_init=0.1, **kwargs):
        self.funckernel_size = mid_channel * 2 - 1
        self.funconv_kwargs = dict(
            wavelet_types=wavelet_types,
            agent_hidden=agent_hidden,
            agent_gamma=agent_gamma,
            agent_buffer_size=agent_buffer_size,
            agent_batch_size=agent_batch_size,
            agent_lr=agent_lr,
            agent_epsilon_start=agent_epsilon_start,
            agent_epsilon_end=agent_epsilon_end,
            agent_epsilon_decay=agent_epsilon_decay,
            agent_tau=agent_tau,
            threshold_init=threshold_init,
        )
        super().__init__(mid_channel=mid_channel, **kwargs)





if __name__ == '__main__':
    # test all the models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for Model in [Base_FUNC_CNN,TFN_STTF,TFN_Morlet,TFN_Chirplet,Random_CNN]:
        print('\n\n'+"-"*50+'\n'+Model.__name__+'\n'+"-"*50)
        models = Model()
        models.to(device)
        summary(models, (1, 1024), batch_size=-1, device="cuda")
        print("\n\n")