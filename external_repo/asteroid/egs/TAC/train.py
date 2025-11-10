import os
import sys
import json
import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from pystoi import stoi
from local.tac_dataset import TACDataset
from asteroid.engine.system import System
from asteroid.models import save_publishable
from asteroid.models.fasnet import FasNetTAC
from asteroid.engine.optimizers import make_optimizer

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
from auditus.evaluation import Auditus

# Fix the seed
torch.manual_seed(42)

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")
parser.add_argument('--loss', required=True, help='Enter a loss function tag')
parser.add_argument('--scale', required=False, help='Enter a scale function tag')
parser.add_argument('--n_fft', required=False, default=1024, help='Enter a n_fft for STFT')
parser.add_argument('--hop_length', required=False, default=256, help='Enter a hop_length for STFT')
parser.add_argument('--n_bins', required=False, help='Enter a n_bins for STFT')
parser.add_argument('--weights_type', required=False, default=None, help='Enter a weights type for STFT')

class TACSystem(System):
    def common_step(self, batch, batch_nb, train=True):
        inputs, targets, valid_channels = batch
        # valid_channels contains a list of valid microphone channels for each example.
        # each example can have a varying number of microphone channels (can come from different arrays).
        # e.g. [[2], [4], [1]] three examples with 2 mics 4 mics and 1 mics.
        est_targets = self.model(inputs, valid_channels) #@ est_targets: [batch_size, 2-sources, n_samples] and targets: [batch_size, 4-mics, 2-sources, n_samples]
        # print("Estimated Targets:", est_targets)
        # print("Ground Truth Targets:", targets)
        loss = self.loss_func(est_targets, targets[:, 0])  # first channel is used as ref #@ removed .mean() because we select only the first source (speech)
        
        # Log the training loss
        if train:
            self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_nb):
        inputs, targets, valid_channels = batch
        with torch.no_grad():  # Ensure no gradient calculation
            est_targets = self.model(inputs, valid_channels)

            auditus = Auditus()
            metrics_in = auditus.se_eval(
                speech=targets[:, 0, 0], 
                noise=targets[:, 0, 1], 
                est_speech=inputs[:, 0, :],
                metrics=['si-sdr', 'si-sir', 'si-sar', 'fw-si-sdr', 'fw-si-sir', 'fw-si-sar', 'stoi'])

            metrics = auditus.se_eval(
                speech=targets[:, 0, 0],
                noise=targets[:, 0, 1],
                est_speech=est_targets[:, 0, :],
                metrics=['si-sdr', 'si-sir', 'si-sar', 'fw-si-sdr', 'fw-si-sir', 'fw-si-sar', 'stoi'])
            
            # si-sdr
            self.log("val_loss", metrics['si-sdr'], prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

            # Monitoring other metrics
            self.log("si-sir(in)", metrics_in['si-sir'], prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            self.log("si-sir(out)", metrics['si-sir'], prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            self.log("si-sar(out)", metrics['si-sar'], prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            self.log("si-sdr(out)", metrics['si-sdr'], prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            
            self.log("fw-si-sir(in)", metrics_in['fw-si-sir'], prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            self.log("fw-si-sir(out)", metrics['fw-si-sir'], prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            self.log("fw-si-sar(out)", metrics['fw-si-sar'], prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            self.log("fw-si-sdr(out)", metrics['fw-si-sdr'], prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            
            self.log("stoi(in)", metrics_in['stoi'], prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            self.log("stoi(out)", metrics['stoi'], prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

            return metrics['si-sdr']

# @Nasser: added a padding function as I increased the batch size to 2 
def collate_fn_padd(batch):
    # Separate inputs, targets, and valid_channels
    inputs, targets, valid_channels = zip(*batch)

    # Pad inputs and targets along the time dimension to match the longest sequence in the batch
    max_len_inputs = max([input.shape[1] for input in inputs])
    max_len_targets = max([target.shape[2] for target in targets])  # Assuming targets have shape [channels, time]
    
    padded_inputs = [torch.nn.functional.pad(input, (0, max_len_inputs - input.shape[1])) for input in inputs]
    padded_targets = [torch.nn.functional.pad(target, (0, max_len_targets - target.shape[2])) for target in targets]
    
    # Convert valid_channels from tuple to tensor
    valid_channels = torch.tensor(valid_channels, dtype=torch.int)

    # Stack padded sequences into tensors for the batch
    inputs = torch.stack(padded_inputs, dim=0)
    targets = torch.stack(padded_targets, dim=0)
    
    return inputs, targets, valid_channels

def main(conf):

    # Exp 
    tag = conf['main_args']['exp_dir'].split('/')[-1].split('_')[-1]
    print(
        f"\033[1;34m\nExperiment: {tag} | Loss: {conf['main_args']['loss']} | n_fft: {conf['main_args']['n_fft']} | "
        f"Hop size: {conf['main_args']['hop_length']} | Scale: {conf['main_args'].get('scale', 'None')} | "
        f"n_bins: {conf['main_args'].get('n_bins', 'None')} | Weighting: {conf['main_args'].get('weights_type', 'None')}\033[0m\n"
    )

    # Create trainset and validset
    train_set = TACDataset(conf["data"]["train_json"], conf["data"]["segment"], max_mics=4, train=True)
    val_set = TACDataset(conf["data"]["dev_json"], conf["data"]["segment"], max_mics=4, train=False)

    print(f"[INFO] Train set size: {len(train_set)} | Validation set size: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
        pin_memory=True, # Optimize data transfer to GPU
        collate_fn=collate_fn_padd, # for padding
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=1,
        num_workers=conf["training"]["num_workers"],
        drop_last=False, # changed to False
        pin_memory=True, # Optimize data transfer to GPU
        persistent_workers=True,
    )

    # Compute effective learning rate
    num_gpus = torch.cuda.device_count()
    print(f"[INFO] GPUs available: {num_gpus}")
    print("[INFO] Learning Rate: ", conf["optim"]["lr"])
    
    original_batch_size = 1 # in the asteroid config file
    effective_batch_size = conf["training"]["batch_size"] * num_gpus * conf["training"]["accumulate_batches"]
    scaling_factor = effective_batch_size / original_batch_size
    conf["optim"]["lr"] *= scaling_factor
    print(f"[INFO] Effective Learning Rate:", conf["optim"]["lr"], end='\n\n')

    model = FasNetTAC(**conf["net"], sample_rate=conf["data"]["sample_rate"])
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Define scheduler
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer, factor=0.5, patience=conf["training"]["patience"]
        )
    else:
        scheduler = None
    
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Loss Functions
    if conf['main_args']['loss'] == 'xp1':
        from asteroid.losses.sdr_xp import XP1Loss
        loss_func = XP1Loss()
    
    elif conf['main_args']['loss'] == 'neg_sisdr':
        from asteroid.losses.sdr_xp import NegSISDRLoss
        loss_func = NegSISDRLoss(
            scale=conf['main_args']['scale'], n_fft=conf['main_args']['n_fft'], 
            hop_length=conf['main_args']['hop_length'], n_bins=conf['main_args']['n_bins'],
            weights_type=conf['main_args']['weights_type']
            )
    
    else:
        print("The loss function is not valid")
        sys.exit()

    system = TACSystem(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )
    
    # Initialize the TensorBoard logger
    tb_logger = TensorBoardLogger(save_dir=exp_dir, name="tensorboard_logs")

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="val_loss",
        mode="max",
        save_top_k=conf["training"]["save_top_k"],
        verbose=True,
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss", mode="max", patience=conf["training"]["patience"], verbose=True
            )
        )

    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu",  # Ensure this is "gpu" for GPU acceleration
        devices="auto",  # Or "auto" for automatic detection of available GPUs
        strategy="ddp", # if torch.cuda.device_count() > 1 else None,  # Distributed strategy if using multiple GPUs
        gradient_clip_val=conf["training"]["gradient_clipping"],
        logger=tb_logger,
        num_sanity_val_steps=0,
        log_every_n_steps=100,  # Ensures that logs are not generated per batch
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))
    save_publishable(
        os.path.join(exp_dir, "publish_dir"),
        to_save,
        metrics=dict(),
        train_conf=conf,
        recipe="asteroid/TAC",
    )

if __name__ == "__main__":
    import yaml
    from pprint import pprint #as print
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("./local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)

    # ./run_.sh --tag TEST --id 0,1,2,3 --loss neg_sisdr --scale critical --n_fft 1024 --hop_length 256 --n_bins 18