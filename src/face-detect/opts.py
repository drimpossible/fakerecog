import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Retinaface based face detection')
    parser.add_argument('--data_dir', default='/media/anarchicorganizer/Qiqi//dfdc_small/', type=str, help='Full path to where all datasets are stored')
    parser.add_argument('--lib_dir', default='/media/anarchicorganizer/Emilia/fakerecog/libs/', type=str, help='Directory where all pretrained models and libraries are stored. Limit: 1GB')
    parser.add_argument('--log_dir', default='/media/anarchicorganizer/Emilia/fakerecog/logs/', type=str, help='Directory where all training/detection logs are stored')
    
    parser.add_argument('--exp_name', type=str, default='test', help='Experiment name to differentiate logs')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size to be used in detection/training')
    parser.add_argument('--model', default='Mobilenet0.25', type=str, choices=['Resnet50','Mobilenet0.25'], help='Trained state_dict file path to open')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID used for training')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--current_idx', type=int, default=0, help='Idx from the json file to begin training')
    # Restart arguments
    

    #Default arguments
    parser.add_argument('--resize', default=1.0, type=float, help='Resize an image')
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel worker threads')
    opt = parser.parse_args()
    return opt


cfg_mnet = {
    'name': 'Mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_res50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}
