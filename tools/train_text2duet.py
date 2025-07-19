import sys
sys.path.append(sys.path[0] + r"/../")
import torch
import lightning.pytorch as pl
import torch.optim as optim
from collections import OrderedDict
from datasets import DataModule
from configs import get_config
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter
from models import *
from pathlib import Path
from utils import paramUtil
import torch
import os
import time

os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'nccl'
from lightning.pytorch.strategies import DDPStrategy
torch.set_float32_matmul_precision('medium')
from utils.plot_script import *

class LitTrainModel(pl.LightningModule):
    def __init__(self, model, train_cfg, model_cfg):
        super().__init__()
        # cfg init
        self.cfg = train_cfg
        self.model_cfg = model_cfg
        self.automatic_optimization = False
        self.save_root = pjoin(self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, 'model')
        self.meta_dir = pjoin(self.save_root, 'meta')
        self.log_dir = pjoin(self.save_root, 'log')
        self.vis_dir = pjoin(self.save_root, 'vis')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        self.model = model
        self.normalizerTorch = MotionNormalizerTorch()
        self.writer = SummaryWriter(self.log_dir)

    def _configure_optim(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=float(self.cfg.TRAIN.LR), weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=10, max_iters=self.cfg.TRAIN.EPOCH, verbose=True)
        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._configure_optim()

    def plot_motion_intergen(self, gt_motion1, gt_motion2, gen_motion1, gen_motion2, length, result_root, caption, mode='train', idx=0):
        # only plot in the main process
        if self.trainer.global_rank != 0:
            return
        
        gt_motion1 = gt_motion1.cpu().numpy()[:length]
        gt_motion2 = gt_motion2.cpu().numpy()[:length]
        gen_motion1 = gen_motion1.cpu().numpy()[:length]
        gen_motion2 = gen_motion2.cpu().numpy()[:length]
        
        mp_data = [gt_motion1, gt_motion2, gen_motion1, gen_motion2]
        mp_joint = []
        
        for data in mp_data:
            joint = data[:,:22*3].reshape(-1,22,3)
            mp_joint.append(joint)

        result_path = Path(result_root) / f"{mode}_{self.current_epoch}_{idx}_combined.mp4"
        plot_3d_motion(str(result_path), paramUtil.t2m_kinematic_chain, mp_joint, title=caption, fps=30)
    
    def text_length_to_motion_torch(self, text, music, length,  spatial=None, body_move=None, rhythm=None):
        # text: 1,*
        # length: 1,
        input_batch = {}
        input_batch["text"] = text
        input_batch["music"] = music
        input_batch["motion_lens"] = length

        if spatial is not None:
            input_batch["spatial"] = spatial
        if body_move is not None:
            input_batch["body_move"] = body_move
        if rhythm is not None:
            input_batch["rhythm"] = rhythm

        # For ReactModel, we need the lead dancer's motion from the test batch
        if self.model_cfg.NAME == "ReactModel":
            # During inference, we'll use ground truth leader motion from the test batch
            # This assumes the lead motion is passed in the batch from sample_text
            if hasattr(self, '_temp_lead_motion'):
                input_batch["lead_motion"] = self._temp_lead_motion
                input_batch["follower_motion"] = self._temp_follower_motion 
                delattr(self, '_temp_lead_motion')  # Clean up after use
                delattr(self, '_temp_follower_motion')  # Clean up after use
        output_batch = self.model.forward_test(input_batch)
        motions_output = output_batch["output"].reshape(output_batch["output"].shape[0], output_batch["output"].shape[1], 2, -1)
        motions_output = self.normalizerTorch.backward(motions_output.detach())
        return motions_output[:,:,0], motions_output[:,:,1]

    def sample_text(self, batch_data, batch_idx, mode):
        motion1, motion2, music, text, motion_lens = batch_data['motion1'], \
            batch_data['motion2'], batch_data['music'], batch_data['text'], batch_data['length']
        
        spatial = batch_data.get('spatial', None)
        body_move = batch_data.get('body_move', None)
        rhythm = batch_data.get('rhythm', None)

        if self.model_cfg.NAME == "DuetModel":
            # Generate both dancer motions
            motion_gen_1, motion_gen_2 = self.text_length_to_motion_torch(text[0:1], music[0:1], motion_lens[0:1],
                spatial=spatial[0:1] if spatial is not None else None,
                body_move=body_move[0:1] if body_move is not None else None,
                rhythm=rhythm[0:1] if rhythm is not None else None
            )

            # Plot both ground truth and generated in one visualization
            self.plot_motion_intergen(
                motion1[0], motion2[0],          # Ground truth motions
                motion_gen_1[0], motion_gen_2[0], # Generated motions
                motion_lens[0], self.vis_dir, text[0],
                mode=mode, idx=batch_idx
            )
        
        elif self.model_cfg.NAME == "ReactModel":
            # Store lead motion for generation
            self._temp_lead_motion = motion1[0:1]
            self._temp_follower_motion = motion2[0:1]
            
            # Generate follower motion
            motion_gen_lead, motion_gen_follower = self.text_length_to_motion_torch(text[0:1], music[0:1], motion_lens[0:1],
                spatial=spatial[0:1] if spatial is not None else None,
                body_move=body_move[0:1] if body_move is not None else None,
                rhythm=rhythm[0:1] if rhythm is not None else None
            )

            # Plot both ground truth and generated in one visualization
            # For ReactModel, lead is the same for both GT and generated
            self.plot_motion_intergen(
                motion1[0], motion2[0],          # Ground truth lead and follower
                motion_gen_lead[0], motion_gen_follower[0], # GT lead and generated follower
                motion_lens[0], self.vis_dir, text[0],
                mode=mode, idx=batch_idx
            )
        
    def forward(self, batch_data):
        motion1, motion2, music, text, motion_lens = batch_data['motion1'], \
            batch_data['motion2'], batch_data['music'], batch_data['text'], batch_data['length']
        
        spatial = batch_data.get('spatial', None)
        body_move = batch_data.get('body_move', None)
        rhythm = batch_data.get('rhythm', None)

        motion1 = motion1.detach().float()  # .to(self.device)
        motion2 = motion2.detach().float()  # .to(self.device)
        motions = torch.cat([motion1, motion2], dim=-1)

        B, T = motion1.shape[:2]

        batch = OrderedDict({})
        batch["text"] = text
        batch["spatial"] = spatial
        batch["body_move"] = body_move
        batch["rhythm"] = rhythm
        batch["music"] = music
        batch["motions"] = motions.reshape(B, T, -1).type(torch.float32)
        batch["motion_lens"] = motion_lens.long()
        # For ReactModel, add leader's motion as input
        if self.model_cfg.NAME == "ReactModel":
            # For reactive dancing, motion1 is considered the leader
            batch["lead_motion"] = motion1
            batch["follower_motion"] = motion2

        loss, loss_logs = self.model(batch)
        return loss, loss_logs

    def on_train_start(self):
        self.rank = self.trainer.global_rank
        self.world_size = self.trainer.world_size
        self.start_time = time.time()
        self.it = self.cfg.TRAIN.LAST_ITER if self.cfg.TRAIN.LAST_ITER else 0
        self.epoch = self.cfg.TRAIN.LAST_EPOCH if self.cfg.TRAIN.LAST_EPOCH else 0
        self.logs = OrderedDict()

    def training_step(self, batch, batch_idx):
        loss, loss_logs = self.forward(batch)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        opt.step()
        if batch_idx < 2 and self.trainer.current_epoch % self.cfg.TRAIN.SAVE_EPOCH==0:
            self.model.eval()
            self.sample_text(batch, batch_idx, 'train')
            self.model.train()

        return {"loss": loss,
            "loss_logs": loss_logs}

    def validation_step(self, batch, batch_idx):
        print('validation step')
        if batch_idx > 2:
            return
        self.sample_text(batch, batch_idx, 'val')

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if outputs.get('skip_batch') or not outputs.get('loss_logs'):
            return
        for k, v in outputs['loss_logs'].items():
            if k not in self.logs:
                self.logs[k] = v.item()
            else:
                self.logs[k] += v.item()

        self.it += 1
        if self.it % self.cfg.TRAIN.LOG_STEPS == 0 and self.trainer.global_rank == 0:
            mean_loss = OrderedDict({})
            for tag, value in self.logs.items():
                mean_loss[tag] = value / self.cfg.TRAIN.LOG_STEPS
                self.writer.add_scalar(tag, mean_loss[tag], self.it)
            self.logs = OrderedDict()
            print_current_loss(self.start_time, self.it, mean_loss,
                               self.trainer.current_epoch,
                               inner_iter=batch_idx,
                               lr=self.trainer.optimizers[0].param_groups[0]['lr'])

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

def build_models(cfg):
    if cfg.NAME == "DuetModel":
        model = DuetModel(cfg)
    elif cfg.NAME == "ReactModel":
        model = ReactModel(cfg)
    else:
        raise NotImplementedError
    return model

def load_checkpoint_weights_only(model, checkpoint_path):
    """Load only model weights from checkpoint, handling DDP prefixes properly"""
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        # Extract state dict from various checkpoint formats
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        
        # Handle DDP prefixes - remove 'model.' prefix if present
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            # Remove various possible prefixes
            key = k
            if key.startswith("model.model."):
                key = key[12:]  # Remove "model.model."
            elif key.startswith("model."):
                key = key[6:]   # Remove "model."
            elif key.startswith("module."):
                key = key[7:]   # Remove "module."
            cleaned_state_dict[key] = v
        
        # Load with strict=False to handle any missing/extra keys
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
            
        print(f"Successfully loaded model weights from {checkpoint_path}")
        return True
        
    except Exception as e:
        print(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return False

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a string input.")
    parser.add_argument("--model_cfg", type=str, default="configs/model_duet_debug.yaml", help="")
    parser.add_argument("--train_cfg", type=str, default="configs/train_duet_debug.yaml", help="")
    parser.add_argument("--data_cfg", type=str, default="configs/datasets_duet.yaml", help="")
    args = parser.parse_args()
    print(args)
    
    model_cfg = get_config(args.model_cfg)
    train_cfg = get_config(args.train_cfg)
    data_cfg = get_config(args.data_cfg).train_set
    val_data_cfg = get_config(args.data_cfg).val_set

    datamodule = DataModule(data_cfg, val_data_cfg, None, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS)
    model = build_models(model_cfg)

    # Handle manual checkpoint loading for weights-only resume
    if train_cfg.TRAIN.RESUME:
        print(f"Loading model weights from: {train_cfg.TRAIN.RESUME}")
        load_checkpoint_weights_only(model, train_cfg.TRAIN.RESUME)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    litmodel = LitTrainModel(model, train_cfg, model_cfg)

    # Setup proper DDP strategy
    if torch.cuda.device_count() > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,  # Set to True only if needed
            gradient_as_bucket_view=True,  # Memory optimization
        )
        devices = torch.cuda.device_count()
    else:
        strategy = "auto"
        devices = 1

    # Lightning checkpoint callback - handles full checkpoint saving/loading
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=litmodel.model_dir,
        filename="epoch_{epoch:04d}",
        every_n_epochs=train_cfg.TRAIN.SAVE_EPOCH,
        save_top_k=-1,
        save_last=True,
        save_on_train_epoch_end=True
    )
    
    trainer = pl.Trainer(
        default_root_dir=litmodel.model_dir,
        devices=devices,
        accelerator='gpu',
        max_epochs=train_cfg.TRAIN.EPOCH,
        strategy=strategy,
        precision=32,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=train_cfg.TRAIN.SAVE_EPOCH,
        num_sanity_val_steps=1,
        enable_progress_bar=True,
        log_every_n_steps=train_cfg.TRAIN.LOG_STEPS
    )
    
    # Check for Lightning checkpoint to resume from
    lightning_ckpt = pjoin(litmodel.model_dir, "last.ckpt")
    resume_ckpt = lightning_ckpt if os.path.exists(lightning_ckpt) else None
    
    if resume_ckpt:
        print(f'Resuming training from Lightning checkpoint: {resume_ckpt}')
        try:
            trainer.fit(model=litmodel, datamodule=datamodule, ckpt_path=resume_ckpt)
        except Exception as e:
            print(f"Failed to resume from Lightning checkpoint: {e}")
            print("Starting fresh training")
            trainer.fit(model=litmodel, datamodule=datamodule)
    else:
        print("Starting fresh training")
        trainer.fit(model=litmodel, datamodule=datamodule)