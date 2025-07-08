# Modified train_text2duet.py with retrieval integration - FIXED VERSION
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
from models.enhanced_duet_model import EnhancedDuetModel, build_enhanced_models, prepare_database_from_dataloader
from pathlib import Path
from utils import paramUtil
import torch
import os
import time
import numpy as np

from models.utils import CosineWarmupScheduler, print_current_loss
from utils.utils import MotionNormalizerTorch
from utils.plot_script import plot_3d_motion

os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'nccl'
from lightning.pytorch.strategies import DDPStrategy
torch.set_float32_matmul_precision('medium')

class LitTrainModelWithRetrieval(pl.LightningModule):
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
        self.database_dir = pjoin(self.save_root, 'database')
        
        # Create directories
        for dir_path in [self.model_dir, self.meta_dir, self.log_dir, self.vis_dir, self.database_dir]:
            os.makedirs(dir_path, exist_ok=True)

        self.model = model
        self.normalizerTorch = MotionNormalizerTorch()  # Now properly imported
        self.writer = SummaryWriter(self.log_dir)
        
        # Retrieval database preparation
        self.database_prepared = False
        self.database_path = pjoin(self.database_dir, 'retrieval_database.npz')
        
        # Training phase control
        self.retrieval_warmup_epochs = getattr(train_cfg.TRAIN, 'RETRIEVAL_WARMUP_EPOCHS', 5)
        self.current_training_phase = "warmup"  # "warmup" -> "retrieval"

    def prepare_retrieval_database_if_needed(self, dataloader):
        """
        Prepare retrieval database from training data if not already done.
        """
        if not self.database_prepared and hasattr(self.model, 'use_retrieval') and self.model.use_retrieval:
            if os.path.exists(self.database_path):
                print(f"Loading existing database from {self.database_path}")
                # Load existing database
                self._load_database()
                self.database_prepared = True
            else:
                print("Preparing retrieval database from training data...")
                prepare_database_from_dataloader(self.model, dataloader, self.database_path)
                self.database_prepared = True
    
    def _load_database(self):
        """Load pre-computed database into the model."""
        if hasattr(self.model.decoder.net, 'retrieval_db'):
            data = np.load(self.database_path)
            db = self.model.decoder.net.retrieval_db
            
            db.text_features = torch.tensor(data['text_features'])
            db.music_features = torch.tensor(data['music_features']) 
            db.captions = data['captions']
            db.duet_motions = data['duet_motions']
            db.m_lengths = data['m_lengths']
            db.interaction_features = data['interaction_features']
            
            print("Database loaded successfully!")

    def _configure_optim(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=float(self.cfg.TRAIN.LR), weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=10, max_iters=self.cfg.TRAIN.EPOCH, verbose=True)  # Now properly imported
        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._configure_optim()
    
    def on_train_start(self):
        self.rank = 0
        self.world_size = 1
        self.start_time = time.time()
        self.it = self.cfg.TRAIN.LAST_ITER if self.cfg.TRAIN.LAST_ITER else 0
        self.epoch = self.cfg.TRAIN.LAST_EPOCH if self.cfg.TRAIN.LAST_EPOCH else 0
        self.logs = OrderedDict()

    def on_train_epoch_start(self):
        """Handle training phase transitions and retrieval settings."""
        # Phase transition logic
        if self.current_epoch < self.retrieval_warmup_epochs:
            if self.current_training_phase != "warmup":
                print(f"Epoch {self.current_epoch}: Switching to warmup phase (no retrieval)")
                self.current_training_phase = "warmup"
                self.model.enable_retrieval(False)
        else:
            if self.current_training_phase != "retrieval":
                print(f"Epoch {self.current_epoch}: Switching to retrieval phase")
                self.current_training_phase = "retrieval"
                self.model.enable_retrieval(True)
                
                # Prepare database if needed
                if not self.database_prepared:
                    # Get dataloader from trainer
                    train_dataloader = self.trainer.datamodule.train_dataloader()
                    self.prepare_retrieval_database_if_needed(train_dataloader)

    def plot_motion_intergen(self, gt_motion1, gt_motion2, gen_motion1, gen_motion2, length, result_root, caption, mode='train', idx=0):
        # only plot in the main process
        if self.device.index != 0:
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
        plot_3d_motion(str(result_path), paramUtil.t2m_kinematic_chain, mp_joint, title=caption, fps=30)  # Now properly imported
    
    def text_length_to_motion_torch(self, text, music, length):
        # text: 1,*
        # length: 1,
        input_batch = {}
        input_batch["text"] = text
        input_batch["music"] = music
        input_batch["motion_lens"] = length
        
        # For ReactModel, we need the lead dancer's motion from the test batch
        if self.model_cfg.NAME == "ReactModel":
            if hasattr(self, '_temp_lead_motion'):
                input_batch["lead_motion"] = self._temp_lead_motion
                input_batch["follower_motion"] = self._temp_follower_motion 
                delattr(self, '_temp_lead_motion')
                delattr(self, '_temp_follower_motion')
                
        output_batch = self.model.forward_test(input_batch)
        motions_output = output_batch["output"].reshape(output_batch["output"].shape[0], output_batch["output"].shape[1], 2, -1)
        motions_output = self.normalizerTorch.backward(motions_output.detach())
        return motions_output[:,:,0], motions_output[:,:,1]

    def sample_text(self, batch_data, batch_idx, mode):
        motion1, motion2, music, text, motion_lens = batch_data['motion1'], \
            batch_data['motion2'], batch_data['music'], batch_data['text'], batch_data['length']
        
        if self.model_cfg.NAME in ["DuetModel", "EnhancedDuetModel"]:
            # Generate both dancer motions
            motion_gen_1, motion_gen_2 = self.text_length_to_motion_torch(text[0:1], music[0:1], motion_lens[0:1])
            
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
            motion_gen_lead, motion_gen_follower = self.text_length_to_motion_torch(text[0:1], music[0:1], motion_lens[0:1])
            
            # Plot both ground truth and generated in one visualization
            self.plot_motion_intergen(
                motion1[0], motion2[0],          # Ground truth lead and follower
                motion_gen_lead[0], motion_gen_follower[0], # GT lead and generated follower
                motion_lens[0], self.vis_dir, text[0],
                mode=mode, idx=batch_idx
            )
        
    def forward(self, batch_data):
        motion1, motion2, music, text, motion_lens = batch_data['motion1'], \
            batch_data['motion2'], batch_data['music'], batch_data['text'], batch_data['length']
        motion1 = motion1.detach().float()
        motion2 = motion2.detach().float()
        motions = torch.cat([motion1, motion2], dim=-1)

        B, T = motion1.shape[:2]

        batch = OrderedDict({})
        batch["text"] = text
        batch["music"] = music
        batch["motions"] = motions.reshape(B, T, -1).type(torch.float32)
        batch["motion_lens"] = motion_lens.long()
        
        # For ReactModel, add leader's motion as input
        if self.model_cfg.NAME == "ReactModel":
            batch["lead_motion"] = motion1
            batch["follower_motion"] = motion2

        loss, loss_logs = self.model(batch)
        return loss, loss_logs

    def training_step(self, batch, batch_idx):
        loss, loss_logs = self.forward(batch)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        opt.step()
        
        # Visualize samples during training
        if batch_idx < 2 and self.trainer.current_epoch % self.cfg.TRAIN.SAVE_EPOCH == 0:
            self.model.eval()
            self.sample_text(batch, batch_idx, 'train')
            self.model.train()

        # Log current training phase
        loss_logs['training_phase'] = 1.0 if self.current_training_phase == "retrieval" else 0.0
        
        return {"loss": loss, "loss_logs": loss_logs}

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
        if self.it % self.cfg.TRAIN.LOG_STEPS == 0 and self.device.index == 0:
            mean_loss = OrderedDict({})
            for tag, value in self.logs.items():
                mean_loss[tag] = value / self.cfg.TRAIN.LOG_STEPS
                self.writer.add_scalar(tag, mean_loss[tag], self.it)
            self.logs = OrderedDict()
            
            # Add phase info to logging
            print_current_loss(self.start_time, self.it, mean_loss,  # Now properly imported
                               self.trainer.current_epoch,
                               inner_iter=batch_idx,
                               lr=self.trainer.optimizers[0].param_groups[0]['lr'])

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

    def save(self, file_name):
        state = {}
        try:
            state['model'] = self.model.module.state_dict()
        except:
            state['model'] = self.model.state_dict()
        state['database_prepared'] = self.database_prepared
        state['training_phase'] = self.current_training_phase
        torch.save(state, file_name, _use_new_zipfile_serialization=False)
        return


# REST OF THE TRAINING SCRIPT CONTINUES...
# (build_models function, argument parsing, main training loop, etc.)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Duet Training with Retrieval")
    parser.add_argument("--model_cfg", type=str, default="configs/model_retrieval_debug.yaml", help="")
    parser.add_argument("--train_cfg", type=str, default="configs/train_retrieval_debug.yaml", help="")
    parser.add_argument("--data_cfg", type=str, default="configs/datasets_duet.yaml", help="")
    parser.add_argument("--prepare_db_only", action="store_true", help="Only prepare database and exit")
    args = parser.parse_args()
    
    model_cfg = get_config(args.model_cfg)
    train_cfg = get_config(args.train_cfg)
    data_cfg = get_config(args.data_cfg).train_set
    val_data_cfg = get_config(args.data_cfg).val_set

    datamodule = DataModule(data_cfg, val_data_cfg, None, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS)
    model = build_enhanced_models(model_cfg)

    # Load checkpoint if resuming
    if train_cfg.TRAIN.RESUME:
        ckpt = torch.load(train_cfg.TRAIN.RESUME, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print("checkpoint state loaded!")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    litmodel = LitTrainModelWithRetrieval(model, train_cfg, model_cfg)

    # Option to only prepare database
    if args.prepare_db_only:
        print("Preparing database only...")
        train_dataloader = datamodule.train_dataloader()
        litmodel.prepare_retrieval_database_if_needed(train_dataloader)
        print("Database preparation complete. Exiting.")
        exit()

    # Set up callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=litmodel.model_dir,
        filename="epoch_{epoch:04d}",
        every_n_epochs=train_cfg.TRAIN.SAVE_EPOCH,
        save_top_k=-1,
        save_last=True
    )
    
    class CustomCheckpointCallback(pl.callbacks.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            if (trainer.current_epoch + 1) % trainer.check_val_every_n_epoch == 0:
                checkpoint_path = pjoin(pl_module.model_dir, 
                                      f"model={model_cfg.NAME}-epoch={trainer.current_epoch}-step={pl_module.it}.ckpt")
                pl_module.save(checkpoint_path)
                print(f"Custom checkpoint saved at {checkpoint_path}")
    
    trainer = pl.Trainer(
        default_root_dir=litmodel.model_dir,
        devices=1,
        accelerator='gpu',
        max_epochs=train_cfg.TRAIN.EPOCH,
        strategy=None,
        precision=32,
        callbacks=[checkpoint_callback, CustomCheckpointCallback()],
        check_val_every_n_epoch=train_cfg.TRAIN.SAVE_EPOCH,
        num_sanity_val_steps=1
    )
    
    ckpt_model = litmodel.model_dir + "/last.ckpt"
    if os.path.exists(ckpt_model):
        print('resume from checkpoint')
        trainer.fit(model=litmodel, datamodule=datamodule, ckpt_path=ckpt_model)
    else:
        trainer.fit(model=litmodel, datamodule=datamodule)