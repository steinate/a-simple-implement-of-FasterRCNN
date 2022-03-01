from pprint import pprint
import torch as t

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if t.cuda.device_count() >= i + 1:
        return t.device(f'cuda:{i}')
    return t.device('cpu')

class Config:
    data_dir = r'C:\Users\pjq\Desktop\deeplearning\VOCdevkit\VOC0712'
    min_size = 600
    max_size = 1000
    num_workers = 0
    test_num_workers = 0

    weight_decay = 0.0005
    lr_decay = 0.1
    lr = 1e-3

    use_adam = False

    rpn_sigma = 3.
    roi_sigma = 1.

    # env = 'faster-rcnn'
    port = 8097
    plot_every = 50

    data = 'voc'
    epoch = 1
    load_path = ''

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

    def _parse(self):
        print('======user config========')
        pprint(self._state_dict()) # 分行打印输出
        print('==========end============')

opt = Config()
device = try_gpu()
