"""
Train a diffusion model on images.
"""

import argparse
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from reassemblenet.k_point_selector import K_Point_Selector
import os
import torch
import numpy as np
from reassemblenet import logger, dist_util
from reassemblenet.resample import create_named_schedule_sampler
from reassemblenet.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    update_arg_parser,
)
from reassemblenet.repair import load_repair_data
from reassemblenet.train_util import TrainLoop

import warnings
warnings.filterwarnings('ignore')



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes', 'y'):
        return True
    elif v.lower() in ('false', '0', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    

def setup():
    dist.init_process_group('nccl', rank=WORLD_RANK, world_size=WORLD_SIZE)

    # seed stuff for reproducibility (same initializations, optimizers steps, sampling, etc...)
    # https://grsahagian.medium.com/what-is-random-state-42-d803402ee76b#:~:text=The%20number%2042%20is%20sort,over%20the%20period%20of%207.5
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

def cleanup():
    """Cleanup function to safely close distributed processes."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser(description="Training Script")

    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Train or test mode")
    parser.add_argument("--kp_path", type=str, default="", help="Path to kp selection checkpoint")


    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--condition_channels", type=int, default=258, help="Conditioning channels")
    parser.add_argument("--diffusion_steps", type=int, default=600, help="Number of diffusion steps")
    parser.add_argument("--rotation", type=str2bool, default=True, help="Enable rotation")
    parser.add_argument("--max_num_points", type=int, default=3300, help="Maximum number of points")
    parser.add_argument("--max_pieces_in_any_puzzle_plus5", type=int, default=49, help="integer")
    parser.add_argument("--number_larger_than_pieces_and_points_in_puzzle", type=int, default=128, help="Number greater than max pieces & points in puzzle")

    parser.add_argument("--exp_name", type=str, default="Exp_RePAIR", help="Experiment name")
    parser.add_argument("--dataset", type=str, default="repair", help="Dataset name")
    parser.add_argument("--dataset_path", type=str, default="/home/aislam/A_PuzzFus_FinishingLine/FinishingLine_Datasets/RePAIR_all_geom_local_global_Texture_harris_FPS_20", help="Dataset path")
    parser.add_argument("--transfer_learning", type=str2bool, default=False, help="Enable transfer learning")
    parser.add_argument("--is_validate", type=str2bool, default=True, help="Enable validation during training")
    parser.add_argument("--set_name", type=str, default="train", help="Dataset split name (train/test/val)")
    
    parser.add_argument("--ema_rate", type=str, default="0.9999", help="EMA rate")
    parser.add_argument("--fp16_scale_growth", type=float, default=0.001, help="FP16 scale growth")
    parser.add_argument("--input_channels", type=int, default=0, help="Number of input channels")
    parser.add_argument("--learn_sigma", type=str2bool, default=False, help="Enable learning sigma")
    parser.add_argument("--log_interval", type=int, default=500, help="Log interval")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--lr_anneal_steps", type=int, default=0, help="Learning rate annealing steps")
    parser.add_argument("--microbatch", type=int, default=-1, help="Microbatch size")
    parser.add_argument("--noise_schedule", type=str, default="cosine", help="Noise schedule type")
    parser.add_argument("--num_channels", type=int, default=128, help="Number of model channels")
    parser.add_argument("--out_channels", type=int, default=0, help="Number of output channels")
    parser.add_argument("--predict_xstart", type=str2bool, default=True, help="Predict x_start in diffusion")
    parser.add_argument("--rescale_learned_sigmas", type=str2bool, default=False, help="Rescale learned sigmas")
    parser.add_argument("--rescale_timesteps", type=str2bool, default=False, help="Rescale timesteps")
    parser.add_argument("--resume_checkpoint", type=str, default="", help="Path to resume checkpoint")
    parser.add_argument("--save_interval", type=int, default=5000, help="Save interval")
    parser.add_argument("--schedule_sampler", type=str, default="uniform", help="Schedule sampler type")
    parser.add_argument("--timestep_respacing", type=str, default="", help="Timestep respacing strategy")
    parser.add_argument("--use_checkpoint", type=str2bool, default=False, help="Enable checkpoint usage")
    parser.add_argument("--use_fp16", type=str2bool, default=False, help="Enable mixed precision (FP16)")
    parser.add_argument("--use_image_features", type=str2bool, default=False, help="Use image features")
    parser.add_argument("--use_kl", type=str2bool, default=False, help="Use KL divergence loss")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--loader_num_workers", type=int, default=32, help="Number of workers for data loading")

    parser.add_argument("--use_geometry_only", type=str2bool, default=False, help="Use geometry features only")
    parser.add_argument("--use_global_texture_only", type=str2bool, default=False, help="Use global texture only")
    parser.add_argument("--images_folder_path", type=str, default="/home/aislam/A_PuzzFus_FinishingLine/FinishingLine_Datasets/Original_2DRePAIR_train_test_split", help="Path to images folder")
    parser.add_argument("--use_local_texture_only", type=str2bool, default=False, help="Use local texture only")
    parser.add_argument("--use_geometry_and_global_texture", type=str2bool, default=False, help="Use geometry and global texture")
    parser.add_argument("--use_geometry_and_local_texture", type=str2bool, default=False, help="Use geometry and local texture")
    parser.add_argument("--use_local_and_global_texture", type=str2bool, default=False, help="Use local and global texture")
    parser.add_argument("--use_geometry_global_local_texture", type=str2bool, default=False, help="Use geometry, global and local texture")
    parser.add_argument("--use_learnable_kp_selection", type=str2bool, default=False, help="Use learnable keypoint selection")
    parser.add_argument("--eval_interval", type=int, default=20, help="Evaluation interval")
    parser.add_argument("--is_kpoint_selection", type=str2bool, default=False, help="Use learnable keypoint selection")

    args = parser.parse_args()
    return args


def main():

    args = get_args()
        
    update_arg_parser(args)

    # dist_util.setup_dist()
    setup()
    
    #   CHANGE MODEL SAVING DIRECTORY HERE
    if WORLD_RANK == 0:
        logger.configure(dir=f'ReassembleNet_ckpts/{args.exp_name}')   


    if WORLD_RANK == 0:
        logger.log("creating model and diffusion...")
        logger.log(f"Running with args: {args}")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # model.to(dist_util.dev())
    model.to(LOCAL_RANK)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)



    
    
    
    

    if WORLD_RANK == 0:
        logger.log("creating data loader...")

    if args.mode == "test":
        kp_selection_model = K_Point_Selector(max_num_pieces=33, 
                                        input_channels=4, hidden_channels=32, output_channels=2, ratio=20, min_score=None, freeze_backbone=True,
                                        backbone_weights_path=args.kp_path).to(dist_util.dev())

        # Load test set instead of training or validation
        test_data, test_data_sampler = load_repair_data(
                batch_size=args.batch_size,
                set_name='test',
                rotation=args.rotation,
                dataset_path = args.dataset_path,
                max_num_points = args.max_num_points,
                maxcount = 33,
                number_larger_than_pieces_and_points_in_puzzle = args.number_larger_than_pieces_and_points_in_puzzle,
                device=f'cuda:{LOCAL_RANK}',
                loader_num_workers=args.loader_num_workers,
                rank = WORLD_RANK,
                world_size = WORLD_SIZE,
                use_geometry_only = args.use_geometry_only,
                use_global_texture_only = args.use_global_texture_only,
                images_folder_path = args.images_folder_path,
                use_local_texture_only = args.use_local_texture_only,
                use_geometry_and_global_texture = args.use_geometry_and_global_texture,
                use_geometry_and_local_texture = args.use_geometry_and_local_texture,
                use_local_and_global_texture = args.use_local_and_global_texture,
                use_geometry_global_local_texture = args.use_geometry_global_local_texture,
                use_learnable_kp_selection = args.use_learnable_kp_selection,
                
        )

        trainer = TrainLoop(
            model=model,
            kp_selection_model=kp_selection_model,
            diffusion=diffusion,
            data=None,
            data_sampler=None,
            is_validate=True,
            val_data=test_data,
            val_data_sampler=test_data_sampler,
            batch_size=args.batch_size,
            epochs=1,
            local_rank=LOCAL_RANK,
            world_rank=WORLD_RANK,
            transfer_learning=args.transfer_learning,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            device=f'cuda:{LOCAL_RANK}',
            max_num_points=args.max_num_points,
            mpkdim=33,
            number_larger_than_pieces_and_points_in_puzzle=args.number_larger_than_pieces_and_points_in_puzzle,
            use_geometry_only=args.use_geometry_only,
            use_global_texture_only=args.use_global_texture_only,
            images_folder_path=args.images_folder_path,
            use_local_texture_only=args.use_local_texture_only,
            use_geometry_and_global_texture=args.use_geometry_and_global_texture,
            use_geometry_and_local_texture=args.use_geometry_and_local_texture,
            use_local_and_global_texture=args.use_local_and_global_texture,
            use_geometry_global_local_texture=args.use_geometry_global_local_texture,
            use_learnable_kp_selection=args.use_learnable_kp_selection,
            eval_interval=args.eval_interval,
        )
        
        trainer.evaluate() 
        return



    kp_selection_model = K_Point_Selector(max_num_pieces=33, 
                                        input_channels=4, hidden_channels=32, output_channels=2, ratio=20, min_score=None, freeze_backbone=False,
                                        backbone_weights_path=args.kp_path).to(dist_util.dev())

    if args.dataset == 'repair':
        
        data, data_sampler = load_repair_data(
            batch_size=args.batch_size,
            set_name=args.set_name,
            rotation=args.rotation,
            dataset_path = args.dataset_path,
            max_num_points = args.max_num_points,
            maxcount = 33,
            number_larger_than_pieces_and_points_in_puzzle = args.number_larger_than_pieces_and_points_in_puzzle,
            device=f'cuda:{LOCAL_RANK}',
            loader_num_workers=args.loader_num_workers,
            rank = WORLD_RANK,
            world_size = WORLD_SIZE,
            use_geometry_only = args.use_geometry_only,
            use_global_texture_only = args.use_global_texture_only,
            images_folder_path = args.images_folder_path,
            use_local_texture_only = args.use_local_texture_only,
            use_geometry_and_global_texture = args.use_geometry_and_global_texture,
            use_geometry_and_local_texture = args.use_geometry_and_local_texture,
            use_local_and_global_texture = args.use_local_and_global_texture,
            use_geometry_global_local_texture = args.use_geometry_global_local_texture,
            use_learnable_kp_selection = args.use_learnable_kp_selection,
        )
        
        if args.is_validate:
            val_data, val_data_sampler = load_repair_data(
                batch_size=args.batch_size,
                set_name='val',
                rotation=args.rotation,
                dataset_path = args.dataset_path,
                max_num_points = args.max_num_points,
                maxcount = 33,
                number_larger_than_pieces_and_points_in_puzzle = args.number_larger_than_pieces_and_points_in_puzzle,
                device=f'cuda:{LOCAL_RANK}',
                loader_num_workers=args.loader_num_workers,
                rank = WORLD_RANK,
                world_size = WORLD_SIZE,
                use_geometry_only = args.use_geometry_only,
                use_global_texture_only = args.use_global_texture_only,
                images_folder_path = args.images_folder_path,
                use_local_texture_only = args.use_local_texture_only,
                use_geometry_and_global_texture = args.use_geometry_and_global_texture,
                use_geometry_and_local_texture = args.use_geometry_and_local_texture,
                use_local_and_global_texture = args.use_local_and_global_texture,
                use_geometry_global_local_texture = args.use_geometry_global_local_texture,
                use_learnable_kp_selection = args.use_learnable_kp_selection,
                )
        else:
            val_data, val_data_sampler = None, None
    else:
        print('dataset does not exist!')
        assert False
    
    if WORLD_RANK == 0:
        logger.log("training...")
    TrainLoop(
        model=model,
        kp_selection_model=kp_selection_model,
        diffusion=diffusion,
        data=data,
        data_sampler = data_sampler,
        is_validate=args.is_validate,
        val_data=val_data,            
        val_data_sampler=val_data_sampler,
        batch_size=args.batch_size,
        epochs = args.epochs,
        local_rank = LOCAL_RANK,
        world_rank = WORLD_RANK,
        transfer_learning = args.transfer_learning,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        device=f'cuda:{LOCAL_RANK}',
        max_num_points = args.max_num_points,
        mpkdim = 33,   
        number_larger_than_pieces_and_points_in_puzzle = args.number_larger_than_pieces_and_points_in_puzzle,
        use_geometry_only = args.use_geometry_only,
        use_global_texture_only = args.use_global_texture_only,
        images_folder_path = args.images_folder_path,
        use_local_texture_only = args.use_local_texture_only,
        use_geometry_and_global_texture = args.use_geometry_and_global_texture,
        use_geometry_and_local_texture = args.use_geometry_and_local_texture,
        use_local_and_global_texture = args.use_local_and_global_texture,
        use_geometry_global_local_texture = args.use_geometry_global_local_texture,
        use_learnable_kp_selection = args.use_learnable_kp_selection,
        eval_interval = args.eval_interval,
        ).run_loop()
    
    # dist_util.cleanup()
    cleanup()




if __name__ == "__main__":
    
    # # Environment Variables created by torchrun
    # LOCAL_RANK = int(os.environ['LOCAL_RANK'])  # different on each process/gpu
    # WORLD_SIZE = int(os.environ['WORLD_SIZE'])  # total number of gpus

    if 'LOCAL_RANK' in os.environ:
        # Environment Variables created by torchrun
        LOCAL_RANK = int(os.environ['LOCAL_RANK'])  # Rank of the GPU on this NODE
        WORLD_SIZE = int(os.environ['WORLD_SIZE'])  # n nodes * n gpus
        WORLD_RANK = int(os.environ['RANK'])        # Rank of the GPU Globally (on all nodes)
    elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        # Environment Variables created by mpirun
        LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])  # Rank of the GPU on this NODE
        WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])        # n nodes * n gpus
        WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])        # Rank of the GPU Globally (on all nodes)
    else:
        raise NotImplementedError("This script must be executed with torchrun or mpirun!")

    # print(f'W Rank: {WORLD_RANK}')
    main()
