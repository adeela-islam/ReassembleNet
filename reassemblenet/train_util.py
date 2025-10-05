import re
import os
import copy
import functools
import torch
import torch as th
from .utils import *
import blobfile as bf
from .nn import update_ema
from torch.optim import AdamW
from . import logger, dist_util
import torch.distributed as dist
from transformers import Adafactor
from .fp16_util import MixedPrecisionTrainer
from .resample import LossAwareSampler, UniformSampler
from torch.nn.parallel.distributed import DistributedDataParallel as DDP



import warnings
warnings.filterwarnings('ignore')


topK = 20

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        kp_selection_model,
        diffusion,
        data,
        data_sampler,
        is_validate,
        val_data,
        val_data_sampler,
        batch_size,
        epochs,
        local_rank,
        world_rank,
        transfer_learning,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        device='cpu',
        max_num_points = 100,
        mpkdim = 20,
        number_larger_than_pieces_and_points_in_puzzle = 32,
        use_geometry_only = False,
        use_global_texture_only = False,
        images_folder_path = None,
        use_local_texture_only = False,
        use_geometry_and_global_texture = False,
        use_geometry_and_local_texture = False,
        use_local_and_global_texture = False,
        use_geometry_global_local_texture = False,
        use_learnable_kp_selection = False,
        eval_interval = 10,
    

    ):  
        self.use_ddim = False
        self.use_geometry_only = use_geometry_only
        self.use_global_texture_only = use_global_texture_only
        self.images_folder_path = images_folder_path
        self.use_local_texture_only = use_local_texture_only
        self.use_geometry_and_global_texture = use_geometry_and_global_texture
        self.use_geometry_and_local_texture = use_geometry_and_local_texture
        self.use_local_and_global_texture = use_local_and_global_texture
        self.use_geometry_global_local_texture = use_geometry_global_local_texture
        self.eval_interval = eval_interval
        self.mpkdim = mpkdim
        self.ID_COLOR = generate_id_color(self.mpkdim)
        self.number_larger_than_pieces_and_points_in_puzzle = number_larger_than_pieces_and_points_in_puzzle
        self.max_num_points = max_num_points
        self.is_validate = is_validate
        self.val_data = val_data
        self.val_data_sampler = val_data_sampler
        self.data_sampler = data_sampler
        self.device = device
        self.epochs = epochs
        self.local_rank = local_rank
        self.world_rank = world_rank
        self.transfer_learning = transfer_learning

        self.model = model
        self.kp_selection_model = kp_selection_model
        self.use_learnable_kp_selection = use_learnable_kp_selection
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()


        if self.world_rank == 0:
            print(f'World size in TRAIN LOOP: {dist.get_world_size()}')

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters(self.transfer_learning)
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        # self.opt = AdamW(
        #     self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        # )

        if self.use_learnable_kp_selection:
            self.opt = Adafactor(
             list(self.mp_trainer.master_params) + list(self.kp_selection_model.parameters()),
            # self.mp_trainer.master_params, 
            # lr=0.0005, 
            # weight_decay=self.weight_decay,
            # scale_parameter=True,  # default setting, adjust if needed
            # relative_step=False    # set False if using a custom learning rate
        )
        else:
            self.opt = Adafactor(
                self.mp_trainer.master_params, 
                # lr=0.0005, 
                # weight_decay=self.weight_decay,
                # scale_parameter=True,  # default setting, adjust if needed
                # relative_step=False    # set False if using a custom learning rate
            )
        self.best_rmse_rotation = float('inf')  # Initialize with high value
        self.best_rmse_translation = float('inf')  # Initialize with high value


        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.ddp_model = DDP(
                self.model,
                device_ids=[self.device],
                output_device=self.device,
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            assert False


    def process_learnable_kp_selection(self, batch, cond):
        x = torch.cat([cond['poly'], cond['geom_feats']], dim=2).to(dist_util.dev())
        selected_x, nodes, score, edge_index, new_batch, batch_ = self.kp_selection_model(x)
        nodes = nodes.view(cond['poly'].shape[0], -1) % cond['poly'].shape[1]
        cond['poly'] = selected_x[:, :2].view(cond['poly'].shape[0], self.mpkdim*topK, cond['poly'].shape[-1]).to(torch.float64)


        cond['gen_mask'] = torch.where(
            (torch.arange(topK * self.mpkdim, device=cond['gen_mask'].device)[None, :, None] < (cond['num_pieces'] * topK)[:, None, None]) & 
            (torch.arange(topK * self.mpkdim, device=cond['gen_mask'].device)[None, None, :] < (cond['num_pieces'] * topK)[:, None, None]),
            0, 1).to(torch.float64)
        
        cond['self_mask'] = torch.where(
            (torch.arange(topK*self.mpkdim, device=cond['self_mask'].device)[None, :, None] // topK == 
            torch.arange(topK*self.mpkdim, device=cond['self_mask'].device)[None, None, :] // topK) & 
            (torch.arange(topK*self.mpkdim, device=cond['self_mask'].device)[None, :, None] // topK < cond['num_pieces'][:, None, None]),
            0, 1).to(torch.float64)
        
        cond['corner_indices'] = torch.where(
                (torch.arange(topK * self.mpkdim, device=cond['corner_indices'].device)[None, :, None] < (cond['num_pieces'] * topK)[:, None, None])
                &
                (torch.arange(self.number_larger_than_pieces_and_points_in_puzzle, device=cond['corner_indices'].device)[None, None, :] == torch.arange(topK * self.mpkdim, device=cond['corner_indices'].device)[None, :, None] % topK),
                1,
                0
            ).to(torch.float64)
                    
        cond['room_indices'] = torch.where(
                (torch.arange(topK * self.mpkdim, device=cond['room_indices'].device)[None, :, None] // topK < cond['num_pieces'][:, None, None]) &
                (torch.arange(128, device=cond['room_indices'].device)[None, None, :] == (torch.arange(topK * self.mpkdim, device=cond['room_indices'].device)[None, :, None] // topK + 1)),
                1,
                0
            ).to(torch.float64)
        
        cond['src_key_padding_mask'] = torch.where(
            torch.arange(topK * self.mpkdim, device=cond['src_key_padding_mask'].device)[None, :] < (cond['num_pieces'][:, None] * topK),
            0,
            1
        ).to(torch.float64)

        cond['connections'] = torch.where(
        torch.arange(topK * self.mpkdim, device=cond['connections'].device)[None, :, None] < (cond['num_pieces'][:, None, None] * topK),
        torch.stack([
            torch.arange(topK * self.mpkdim, device=cond['connections'].device)[None, :] % topK + (torch.arange(topK * self.mpkdim, device=cond['connections'].device)[None, :] // topK) * topK,
            (torch.arange(topK * self.mpkdim, device=cond['connections'].device)[None, :] + 1) % topK + (torch.arange(topK * self.mpkdim, device=cond['connections'].device)[None, :] // topK) * topK
        ], dim=-1),
        torch.tensor(0.0, device=cond['connections'].device)).to(torch.float64)
        
        cond['rels'] = calc_rels(B=cond['rels'].shape[0], N=topK * self.mpkdim,  K=topK, D=cond['num_pieces'], keypoints=cond['poly']).to(torch.float64)

        cond['trans'] = cond['trans'][torch.arange(cond['trans'].shape[0], device=cond['trans'].device)[:, None], nodes.to(cond['trans'].device)].to(torch.float64)
        cond['rots'] = cond['rots'][torch.arange(cond['rots'].shape[0], device=cond['rots'].device)[:, None], nodes.to(cond['rots'].device)].to(torch.float64)
        cond['trans_rot_gt'] = cond['trans_rot_gt'][torch.arange(cond['trans_rot_gt'].shape[0], device=cond['trans_rot_gt'].device)[:, None], nodes.to(cond['trans_rot_gt'].device)].to(torch.float64)

        if 'geom_feats' in cond.keys():
        
            cond['geom_feats'] = cond['geom_feats'][torch.arange(cond['geom_feats'].shape[0], device=cond['geom_feats'].device)[:, None], nodes.to(cond['geom_feats'].device)].to(torch.float64)
        
        if 'global_texture_feats' in cond.keys():
            cond['global_texture_feats'] = cond['global_texture_feats'][torch.arange(cond['global_texture_feats'].shape[0], device=cond['global_texture_feats'].device)[:, None], nodes.to(cond['global_texture_feats'].device)].to(torch.float64)
        
        if 'local_texture_feats' in cond.keys():
            cond['local_texture_feats'] = cond['local_texture_feats'][torch.arange(cond['local_texture_feats'].shape[0], device=cond['local_texture_feats'].device)[:, None], nodes.to(cond['local_texture_feats'].device)].to(torch.float64)

        batch = cond['trans_rot_gt'].transpose(1, 2)

        return batch, cond

    def _load_and_sync_parameters(self, transfer_learning=False):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            if self.world_rank == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                resume_checkpoint, map_location=self.device
            )
            self.model.load_state_dict(state_dict, strict=not transfer_learning)
            
            if not transfer_learning:
                # Only sync parameters for resuming training
                self.resume_epoch = parse_resume_step_from_filename(resume_checkpoint)
                dist_util.sync_params(self.model.parameters())
            else:
                if self.world_rank == 0:
                    logger.log("Loaded pretrained model for transfer learning.")


    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if self.world_rank == 0:
                    logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                    state_dict = dist_util.load_state_dict(
                        ema_checkpoint, map_location=self.device
                    )
                    ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            if self.world_rank == 0:
                logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=self.device
            )
            self.opt.load_state_dict(state_dict)
    





    def evaluate(self):
        self.model.eval()  
        self.kp_selection_model.eval()
        tmp_count = 0

        
        recals = 0
        press = 0
        f1s = 0
        qposs = 0
        rmse_trans_old = 0
        rmse_rotation_old = 0
        items = 0

        with th.no_grad():
            self.val_data_sampler.set_epoch(0)
            
            for batch_id, data_batch in enumerate(self.val_data): 

                if self.world_rank == 0:
                    print(f'\n Batch: {batch_id + 1} / {len(self.val_data)}')

             
                sample_fn = (
                    self.diffusion.p_sample_loop if not self.use_ddim else self.diffusion.ddim_sample_loop
                )

                batch, cond = data_batch
                batch = batch.to(self.device) 

                for key in cond:
                    cond[key] = cond[key].to(self.device) 
                

                if self.use_learnable_kp_selection:
                    batch, cond = self.process_learnable_kp_selection(batch, cond)
                    
                sample = sample_fn(
                    self.model,
                    batch.shape,
                    clip_denoised=True,
                    model_kwargs=cond,
                )

                sample_gt = batch.unsqueeze(0) 
                sample = sample.permute([0, 1, 3, 2])
                sample_gt = sample_gt.permute([0, 1, 3, 2])

                sample = sample.to(self.device)
                sample_gt = sample_gt.to(self.device)

                gt_translation = sample_gt[-1:][:, :, :, :2] 
                gt_rotation = sample_gt[-1:][:, :, :, 2:]
                
                pred_translation = sample[-1:][:, :, :, :2] 
                pred_rotation = sample[-1:][:, :, :, 2:]


                rmse_trans_old += calculate_rmse(gt_translation, pred_translation)
                rmse_rotation_old += calculate_rotation_angles_and_rmse(pred_rotation, gt_rotation)


                if self.use_learnable_kp_selection:
                    gt = save_samples(sample_gt, 'gt', cond, True, tmp_count, ID_COLOR=self.ID_COLOR, save_svg=False, max_num_points=660, device=self.device)
                    pred = save_samples(sample, 'pred', cond, True, tmp_count, ID_COLOR=self.ID_COLOR, save_svg=False, max_num_points=660, device=self.device)
                
                else:
                    gt = save_samples(sample_gt, 'gt', cond, True, tmp_count, ID_COLOR=self.ID_COLOR, save_svg=False, max_num_points=self.max_num_points, device=self.device)
                    pred = save_samples(sample, 'pred', cond, True, tmp_count, ID_COLOR=self.ID_COLOR, save_svg=False, max_num_points=self.max_num_points, device=self.device)

                qpos_ = get_metric(gt, pred, cond)

                qposs += qpos_
    

                items += 1 
                tmp_count += sample_gt.shape[1]  

            if self.world_rank == 0:
                print("\n--- Evaluation Metrics ---")
                print(f"RMSE Rotation: {rmse_rotation_old/items}")
                print(f"RMSE Translation: {rmse_trans_old/items * 100}")
                print(f"QPos: {qposs/items}")
                print("--- Evaluation complete ---")
        
        return 



    def run_loop(self):
        
        start_epoch = self.resume_epoch if hasattr(self, "resume_epoch") else 0
 
        for epoch in range(start_epoch, self.epochs): 
            
            
            self.data_sampler.set_epoch(epoch)
            
            for batch_idx, data_batch in enumerate(self.data):
                
                if self.world_rank == 0:
                    logger.log(f"Epoch {epoch + 1}/{self.epochs}")


                batch, cond = data_batch
                

                if self.use_learnable_kp_selection:
                    batch, cond = self.process_learnable_kp_selection(batch, cond)

                self.run_step(batch, cond)

                if self.step % 100000 == 0:
                    lr = self.lr * (0.1**(self.step // 100000))
                    for param_group in self.opt.param_groups:
                        param_group["lr"] = lr

                if self.step % self.log_interval == 0:
                    logger.dumpkvs()

                    
                self.step += 1

            
            
            if epoch + 1 == self.epochs:
                self.save(epoch)

                 
            else:
                if self.step % self.save_interval == 0:
                    self.save(epoch)
            
            if self.step + self.resume_step > self.lr_anneal_steps:
                if self.world_rank == 0:
                    logger.log("Stopping condition met. Exiting training loop.")
                break 
        
        





    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
    

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        if self.use_learnable_kp_selection:
            self.kp_selection_model.zero_grad()

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(self.device)
            micro_cond = {
                k: v[i : i + self.microbatch].to(self.device)
                for k, v in cond.items()
            }
            model_kwargs = micro_cond

            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=model_kwargs,
            )
            losses = compute_losses()
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            self.mp_trainer.backward(loss)



    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)



    def _anneal_lr(self):

        self.lr_anneal_steps=500000000000000
        lr=self.lr
        if (self.step>=5000):
            lr =5e-4 
        if (self.step>=70000):
            lr =1e-4 
        if (self.step>=120000):
            lr =8e-5 
        if (self.step>=150000):
            lr =5e-5 
        if (self.step>=200000):
            lr =1e-5 
        if (self.step>=2500000):
            lr =5e-6 


        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
            self.log_lr=lr
       

    def save(self, epoch_num):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if self.world_rank == 0:
                logger.log(f"saving model for epoch: {epoch_num}...")
                filename = f"model_epoch{epoch_num}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)
                
                if self.use_learnable_kp_selection:
                    filename_kp = filename[:-3]+'_kp.pth'
                    with bf.BlobFile(bf.join(get_blob_logdir(), filename_kp), "wb") as f:
                        th.save(self.kp_selection_model.nets.state_dict(), f)

        save_checkpoint(0, self.mp_trainer.master_params)
        # for rate, params in zip(self.ema_rate, self.ema_params):
        #     save_checkpoint(rate, params)

        dist.barrier()



def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form .../model_epoch{epoch_num}_...pt or .../model_epoch{epoch_num}.pt
    and return the epoch number as an integer.
    """
    try:
        basename = os.path.basename(filename)
        if not basename.startswith("model_epoch"):
            return 0
        # Remove the "model_epoch" prefix
        after_prefix = basename[len("model_epoch"):]
        # The epoch is the first number before an underscore or dot
        if "_" in after_prefix:
            epoch_str = after_prefix.split("_")[0]
        else:
            epoch_str = after_prefix.split(".")[0]
        return int(epoch_str)
    except Exception:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    """
    Automatically discover the latest checkpoint in the log directory.
    Assumes checkpoint filenames follow the pattern 'modelNNNNNN.pt'.
    """
    logdir = get_blob_logdir()
    if not bf.exists(logdir):
        return None

    # Find all files that match the checkpoint pattern
    checkpoint_files = [
        file for file in bf.listdir(logdir) if file.startswith("model") and file.endswith(".pt")
    ]

    if not checkpoint_files:
        return None

    # Parse step numbers and find the latest checkpoint
    checkpoints = [(parse_resume_step_from_filename(file), file) for file in checkpoint_files]
    latest_checkpoint = max(checkpoints, key=lambda x: x[0], default=None)

    if latest_checkpoint:
        print(f'Resuming from {latest_checkpoint[1]}...')
        return bf.join(logdir, latest_checkpoint[1])
    return None



def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


