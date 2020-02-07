import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Retinaface based face detection')
    parser.add_argument('--data_dir', default='/media/anarchicorganizer/Qiqi/', type=str, help='Full path to where all datasets are stored')
    parser.add_argument('--out_dir', default='/media/anarchicorganizer/Qiqi/dfdc_bursted_small/', type=str, help='Full path to where bursted datasets are stored')
    parser.add_argument('--lib_dir', default='/media/anarchicorganizer/Emilia/fakerecog/libs/', type=str, help='Directory where all pretrained models and libraries are stored. Limit: 1GB')
    parser.add_argument('--log_dir', default='/media/anarchicorganizer/Emilia/fakerecog/logs/', type=str, help='Directory where all datasets are stored')
    parser.add_argument('--exp_name', type=str, default='test', help='Experiment Name')
    parser.add_argument('--dataset', type=str, default='dfdc_preview', choices=['dfdc_preview','dfdc_full','FF++40','FF++24'], help='Experiment Name')
    parser.add_argument('--loader_type', type=str, default='burst', choices=['burst','video'], help='Type of dataloader')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size to be used in training')
    parser.add_argument('--model', default='Mobilenet0.25', type=str, choices=['Resnet50','Mobilenet0.25'], help='Trained state_dict file path to open')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU used for training')
    parser.add_argument('--worker_id', type=int, default=0, help='GPU used for training')
    parser.add_argument('--confidence_threshold', default=0.75, type=float, help='confidence_threshold')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--current_idx', type=int, default=0, help='Start from the beginning')
    parser.add_argument('--total_processes', type=int, default=0, help='Start from the beginning')
    parser.add_argument('--max_track_age', type=int, default=20, help='Start from the beginning')
    parser.add_argument('--min_track_hits', type=int, default=3, help='Start from the beginning')
    parser.add_argument('--frame_rate', type=int, default=12, help='Frame rate to burst videos')
    parser.add_argument('--paths_to_process', type=str, help='Choose json file with video paths to process') # Merge this with data_dir + dataset?
    # Restart arguments
    

    #Default arguments
    parser.add_argument('--resize', default=1.0, type=float, help='Resize an image')
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel worker threads')
    parser.add_argument('--scale', type=float, default=1.2, help='Enlarged crops')
    opt = parser.parse_args()
    return opt

