import os
import ast
import argparse

import torch

from dblock import datasets, models, trainer
from dblock.datasets.transforms import List


def get_dataset(base_path, args):
    flatten = not ast.literal_eval(args.load_spatial_dims)
    use_augmentation = ast.literal_eval(args.use_augmentation)
    print(f"flatten = {flatten} {type(flatten)}")
    print(f"use_augmentation = {use_augmentation} {type(use_augmentation)}")

    if args.dataset == "nmnist":
        transform = List.get_nmnist_transform(args.t_len)
        dataset = datasets.NMNISTDataset(os.path.join(base_path, "data", "N-MNIST"), dt=1, transform=transform)

    elif args.dataset == "shd":
        transform = List.get_shd_transform(args.t_len)
        dataset = datasets.SHDDataset(os.path.join(base_path, "data", "SHD"), dt=2, transform=transform)

    elif args.dataset == "ssc":
        transform = List.get_ssc_transform(args.t_len)
        dataset = datasets.SSCDataset(os.path.join(base_path, "data", "SSC"), dt=2, transform=transform)

    return dataset


def get_model(t_len, args):
    load_conv_model = ast.literal_eval(args.load_spatial_dims)
    single_spike = ast.literal_eval(args.single_spike)
    beta_requires_grad = ast.literal_eval(args.beta_requires_grad)
    readout_max = ast.literal_eval(args.readout_max)
    print(f"single_spike = {single_spike} {type(single_spike)}")
    print(f"load_conv_model = {load_conv_model} {type(load_conv_model)}")
    recurrent = ast.literal_eval(args.recurrent)
    detach_recurrent_spikes = ast.literal_eval(args.detach_recurrent_spikes)
    print(f"recurrent = {recurrent}")
    print(f"detach_recurrent_spikes = {detach_recurrent_spikes}")

    if args.dataset == "nmnist":
        milestones = [30, 60, 90]
        if args.d == 0:
            model = models.NMNISTModel(args.method, t_len, heterogeneous_beta=True, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike, recurrent=recurrent, n_layers=args.n_layers, n_neurons=args.n_neurons)
        else:
            model = models.NMNISTDModel(args.d, recurrent, args.method, t_len, args.n_neurons, args.n_layers, detach_recurrent_spikes, heterogeneous_beta=True, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike)
    elif args.dataset == "shd":
        milestones = [30, 60, 90]
        if args.d == 0:
            model = models.SHDModel(args.method, t_len, heterogeneous_beta=True, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike, recurrent=recurrent, n_layers=args.n_layers, n_neurons=args.n_neurons)
        else:
            model = models.SHDDModel(args.d, recurrent, args.method, t_len, args.n_neurons, args.n_layers, detach_recurrent_spikes, heterogeneous_beta=True, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike)
    elif args.dataset == "ssc":
        milestones = [30, 60]
        if args.d == 0:
            model = models.SSCModel(args.method, t_len, heterogeneous_beta=True, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike, recurrent=recurrent, n_layers=args.n_layers, n_neurons=args.n_neurons)
        else:
            model = models.SSCDModel(args.d, recurrent, args.method, t_len, args.n_neurons, args.n_layers, detach_recurrent_spikes, heterogeneous_beta=True, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike)

    return model, milestones


def main():
    torch.backends.cudnn.benchmark = True

    # Building settings
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--method", type=str)
    parser.add_argument("--t_len", type=int)
    parser.add_argument("--beta_requires_grad", type=str)
    parser.add_argument("--readout_max", type=str, default="False")
    parser.add_argument("--single_spike", type=str, default="True")
    # For d-model
    parser.add_argument("--d", type=int, default=0)
    parser.add_argument("--recurrent", type=str, default="True")
    parser.add_argument("--n_neurons", type=int, default=300)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--detach_recurrent_spikes", type=str, default="True")

    # Training arguments
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--load_spatial_dims", type=str, default="False")
    parser.add_argument("--use_augmentation", type=str, default="False")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--device", type=str, default="cuda")

    # Load arguments
    args = parser.parse_args()
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Instantiate the dataset
    print("Building dataset...")
    dataset = get_dataset(base_path, args)

    # Instantiate the model
    print("Building model...")
    model, milestones = get_model(args.t_len, args)

    # Instantiate the trainer
    print("Started training...")
    model_results_path = os.path.join(base_path, f"temp/datasets/{args.dataset}")

    snn_trainer = trainer.Trainer(model_results_path, model, dataset, args.epoch, args.batch, args.lr, milestones=milestones, device=args.device)
    snn_trainer.train(save=True)


if __name__ == "__main__":
    main()
