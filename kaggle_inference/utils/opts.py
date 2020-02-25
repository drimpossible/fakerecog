import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Retinaface based face detection')
    parser.add_argument('--lib_dir', default='ckpt', type=str,
                        help='Directory where all pretrained models and libraries are stored. Limit: 1GB')
    parser.add_argument('--log_dir', default='/media/anarchicorganizer/Emilia/fakerecog/logs/', type=str,
                        help='Directory where all datasets are stored')
    parser.add_argument('--model', default='Mobilenet0.25', type=str, choices=['Resnet50', 'Mobilenet0.25'],
                        help='Trained state_dict file path to open')
    parser.add_argument('--confidence_threshold', default=0.75, type=float, help='confidence_threshold')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--max_track_age', type=int, default=20, help='Start from the beginning')
    parser.add_argument('--min_track_hits', type=int, default=3, help='Start from the beginning')
    parser.add_argument('--frame_rate', type=int, default=12, help='Frame rate to burst videos')
    parser.add_argument('--num_frames', type=int, default=32, help='Number of frames used per videos')
    parser.add_argument('--ckpt', type=str, default='ckpt/ckpt.pth.tar')

    # Default arguments
    parser.add_argument('--resize', default=1.0, type=float, help='Resize an image')
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel worker threads')
    parser.add_argument('--scale', type=float, default=1.2, help='Enlarged crops')
    opt = parser.parse_args()
    return opt

