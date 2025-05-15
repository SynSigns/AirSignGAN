import argparse
import torch
import logging
import signal
import sys
import time
import torch.backends.cudnn as cudnn
from trainer import Trainer
from datetime import datetime
from os import path
from utils import misc
from random import randint

def get_arguments():
    parser = argparse.ArgumentParser(description='AirSignGAN: Learning a Generative Model to generate synthetic air signatures')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('--device-ids', default=[0], type=int, nargs='+', help='device ids assignment (e.g 0 1 2 3)')

    parser.add_argument('--gen-model', default='g_basiclstm', 
                        choices=['g_basiclstm', 'g_basicgru', 'g_basiccnn1d', 'g_attentionlstm', 'g_attentioncnn1d'],
                        help='generator architecture (default: g_basiclstm)')
    parser.add_argument('--dis-model', default='d_basiclstm', 
                        choices=['d_basiclstm', 'd_basicgru', 'd_basiccnn1d', 'd_attentionlstm', 'd_attentioncnn1d'],
                        help='discriminator architecture (default: d_basiclstm)')
    parser.add_argument('--input-size', default=6, type=int, help=' (default: 6)')
    parser.add_argument('--hidden-size', default=32, type=int, help=' (default: 32)')
    parser.add_argument('--num-layers', default=3, type=int, help='number of layers if the base block is based on an LSTM/GRU (default: 3)')
    parser.add_argument('--kernel-size', default=3, type=int, help='kernel size if base block is based on a 1DCNN (default: 3)')
    parser.add_argument('--dropout', default=0, type=int, help=' (default: 0)')
    
    parser.add_argument('--root', default='', help='timeseries source')
    parser.add_argument('--dir-name', default='', help='directory name')
    parser.add_argument('--min-size', default=50, type=int, help='minimum scale size (default: 50)')
    parser.add_argument('--max-size', default=500, type=int, help='maximum scale size  (default: 500)')
    parser.add_argument('--scale-factor-init', default=0.75, type=float, help='initilize scaling factor (default: 0.75)')
    parser.add_argument('--noise-weight', default=0.1, type=float, help='noise amplitude (default: 0.1)')
    parser.add_argument('--normalization', default=0.1, type=float, help='normalization for discriminator(spectral norm)')

    parser.add_argument('--batch-size', default=1, type=int, help='batch-size (default: 1)')
    parser.add_argument('--truncate-size', default=500, type=int, help='truncate-size of last scale (default: 500)')
    parser.add_argument('--num-steps', default=4000, type=int, help='number of steps per scale (default: 4000)')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate (default: 5e-4)')
    parser.add_argument('--gen-betas', default=[0.5, 0.9], nargs=2, type=float, help='adam betas (default: 0.5 0.9)')
    parser.add_argument('--dis-betas', default=[0.5, 0.9], nargs=2, type=float, help='adam betas (default: 0.5 0.9)')
    parser.add_argument('--num-critic', default=1, type=int, help='critic iterations (default: 1)')
    parser.add_argument('--step-size', default=2000, type=int, help='scheduler step size (default: 2000)')
    parser.add_argument('--gamma', default=0.1, type=float, help='scheduler gamma (default: 0.1)')
    parser.add_argument('--penalty-weight', default=0.1, type=float, help='gradient penalty weight (default: 0.1)')
    parser.add_argument('--reconstruction-weight', default=10., type=float, help='reconstruction-weight (default: 10)')
    parser.add_argument('--adversarial-weight', default=1., type=float, help='adversarial-weight (default: 1)')
    parser.add_argument('--euclidean-weight', default=10., type=float, help='euclidean-weight to ensure the distance between the points are constants(default: 1.0)')
    parser.add_argument('--penalty-distance', default=0.15, type=float, help='penalty distance... penalises if the distance between the points arent this value (default: 0.15)')

    parser.add_argument('--seed', default=-1, type=int, help='random seed (default: random)')
    parser.add_argument('--print-every', default=200, type=int, help='print-every (default: 200)')
    parser.add_argument('--eval-every', default=1000, type=int, help='eval-every (default: 1000)')
    parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results', help='results dir')
    parser.add_argument('--save', metavar='SAVE', default='', help='saved folder')
    parser.add_argument('--evaluation', default=False, action='store_true', help='evaluate a model (default: false)')
    parser.add_argument('--num-synthetic-users', default = None, type = int, help='number of synthetic users to generate if eval is true (default: None)')
    parser.add_argument('--model-to-load', default='', help='evaluating from file (default: None)')
    parser.add_argument('--amps-to-load', default='', help='evaluating from file (default: None)')
    parser.add_argument('--use-tb', default=False, action='store_true', help='use torch.utils.tensorboard (default: false)')
    args = parser.parse_args()

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.save == '':
        #args.save = time_stamp
        #args.save = f"{args.dir_name}_{time_stamp}" if args.dir_name else time_stamp
        args.save = f"{args.dir_name}" if args.dir_name else time_stamp
    args.save_path = path.join(args.results_dir, args.save)
    if args.seed == -1:
        args.seed = randint(0, 12345)
    return args

def main():
    # arguments
    args = get_arguments()

    torch.manual_seed(args.seed)
    # cuda
    if 'cuda' in args.device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.device_ids[0])
        cudnn.benchmark = True
    else:
        args.device_ids = None

    # this is for when we want to synthesise new users and want to give a custom name to the generated files
    #this value is updated in the evalCreateSynthetic function
    args.save_name = None

    # set logs
    misc.mkdir(args.save_path)
    misc.setup_logging(path.join(args.save_path, 'log.txt'))

    # print logs
    logging.info(args)

    # trainer
    trainer = Trainer(args)
    start_time = time.time()
    logging.info(f"Start time: {start_time}")
    logging.info(f"Start training/eval...")
    if not args.evaluation:
        trainer.train()
    else:
        if args.num_synthetic_users is not None:
            # args.root actually points to a folder not file when num_synthetic_users is not None
            trainer.evalCreateSynthetic()
        else:
            trainer.eval()
        
    end_time = time.time()
    logging.info(f"End time: {end_time}")
    logging.info(f"Total time: {end_time - start_time}")
    logging.info(f"End training/eval...")

if __name__ == '__main__':
    # enables a ctrl-c without triggering errors
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))
    main()