# -*- coding:utf-8 -*-


class DefaultConfig(object):
    split = 'train'
    model = 'FusionDenseNetBev'
    # model = 'TestNet'
    root = '/media/yangfei/Repository/KITTI/data_road'

    device = '0'

    # load_model_path = 'checkpoints/FusionDenseNet-trainval-stride16-booststrap-epoch-170.pth'
    # load_model_path = 'checkpoints/FusionDenseNet-trainval-stride16-booststrap-sgd-epoch-170.pth'
    load_model_path = 'checkpoints/best-model-baseline@K_0.20.pth'
    # load_model_path = ''
    run_dir = 'RUN/test_baseline'
    # model_prefix = 'checkpoints/FusionDenseNet-trainval-stride16-booststrap-epoch'
    model_prefix = 'checkpoints/best-model-st@'
    optim = 'Adam'

    IMG_SIZE = 288*1216
    LAMBDA = 0.3
    # K_RATE = 5      # percentage
    K1 = 0.1
    K2 = 0.1
    batch_size = 3
    use_gpu = True
    num_workers = 8
    print_freq = 10
    test_freq = 1

    max_epoch = 200
    learning_rate = 1e-4
    lr_decay = 0.95
    weight_decay = 1e-4


config = DefaultConfig()
