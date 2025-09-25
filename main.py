import os
import torch
import argparse
import numpy as np

from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config
from Extreme.Extreme_Train import ExtremeTrain
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--config_file', type=str, default=None,
                        help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT',
                        help='directory to save the results')
    parser.add_argument('--tensorboard', action='store_true',
                        help='use tensorboard for logging')

    # args for random

    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=12345,
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                             ' used, and ddp will be disabled')

    # args for training
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--sample', type=int, default=0,
                        choices=[0, 1], help='Condition or Uncondition.')
    parser.add_argument('--mode', type=str, default='infill',
                        help='Infilling or Forecasting.')
    parser.add_argument('--milestone', type=int, default=10)

    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.')
    parser.add_argument('--pred_len', type=int, default=0, help='Length of Predictions.')
    parser.add_argument('--pred_step', type=int, default=100, help='Length of Predictions.')
    parser.add_argument('--topk', type=int, default=2,
                        help='Top-K frequency to use in model / save dir naming.')

    # args for modify config
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, f'{args.name}({args.topk})')

    return args


def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)
    config["model"]["params"]["top_k_frequency"] = args.topk

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).cuda()  # EFDiff\
    if args.sample == 1 and args.mode in ['infill', 'predict','longpredict']:
        test_dataloader_info = build_dataloader_cond(config, args)
    dataloader_info = build_dataloader(config, args)



    extreme_cfg = config.get("extreme_train", {})
    ext_trainer = ExtremeTrain(extreme_cfg)
    samples = dataloader_info["dataset"].samples
    pairs = ext_trainer.make_pairs_from_dataset_cfg(samples)
    n = len(pairs)
    split = int(0.8 * n)
    train_pairs, val_pairs = pairs[:split], pairs[split:]
    report = ext_trainer.fit(train_pairs, val_pairs)
    results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}_topk({model.top_k_frequency})')
    results_folder.mkdir(parents=True, exist_ok=True)
    ext_model_path = results_folder / "extreme_model.pth"
    ext_trainer.save(str(ext_model_path))
    print("ExtremeTrain finished. Report:", report)

    model.set_ext_trainer(ext_trainer)
    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger,ext_trainer=ext_trainer)
    if args.train:
        trainer.train()
    elif args.sample == 1 and args.mode in ['infill', 'predict']:
        trainer.load(args.milestone)
        dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']
        coef = config['dataloader']['test_dataset']['coefficient']
        stepsize = config['dataloader']['test_dataset']['step_size']
        sampling_steps = config['dataloader']['test_dataset']['sampling_steps']
        samples, *_ = trainer.restore(dataloader, [dataset.window, dataset.var_num], coef, stepsize, sampling_steps)
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        np.save(os.path.join(args.save_dir, f'ddpm_{args.mode}_{args.name}.npy'), samples)
    elif args.sample == 1 and args.mode in ['longpredict']:
        trainer.load(args.milestone)
        dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']
        coef = config['dataloader']['long_predict_dataset']['coefficient']
        stepsize = config['dataloader']['long_predict_dataset']['step_size']
        sampling_steps = config['dataloader']['long_predict_dataset']['sampling_steps']
        predict_steps = config['dataloader']['long_predict_dataset']['params']['predict_step']
        whole_samples = np.empty([dataset.sample_num, dataset.window + (predict_steps - 1) * dataset.pred_len, dataset.var_num])
        for predict_step_count in range(predict_steps):
            samples, *_ = trainer.restore(dataloader, [dataset.window, dataset.var_num], coef, stepsize, sampling_steps)
            samples = unnormalize_to_zero_to_one(samples)
            if predict_step_count == 0:
                whole_samples[:, :dataset.window, :] = samples
            else:
                whole_samples[:, dataset.window + (predict_step_count - 1) * dataset.pred_len:
                               dataset.window + (predict_step_count)* dataset.pred_len,:] = samples[:,-dataset.pred_len:,:]
            step_samples = samples[:, -(dataset.window - dataset.pred_len):, :]
            padding_samples = samples[:,:dataset.pred_len,:].copy()
            step_samples= np.concatenate([step_samples, padding_samples], axis=1)
            test_dataloader_info = build_dataloader_cond(config, args, step_samples)
            dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']
        np.save(os.path.join(args.save_dir, f'ddpm_fake_long_predict_{args.name}.npy'), whole_samples)
    else:
        trainer.load(args.milestone)
        dataset = dataloader_info['dataset']
        samples = trainer.sample(num=len(dataset), size_every=2001, shape=[dataset.window, dataset.var_num])
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
        np.save(os.path.join(args.save_dir, f'ddpm_fake_{args.name}.npy'), samples)


if __name__ == '__main__':
    main()