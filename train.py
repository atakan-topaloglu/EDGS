import os
from source.trainer import EDGSTrainer
from source.utils_aux import set_seed
import omegaconf
import wandb
import hydra
from argparse import Namespace
from omegaconf import OmegaConf


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(cfg: omegaconf.DictConfig):
    # wandb initialization
    # Using OmegaConf.to_container for logging to handle potential custom resolvers
    wandb_config_log = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    _ = wandb.init(entity=cfg.wandb.entity,
                   project=cfg.wandb.project,
                   config=wandb_config_log, # Log resolved config
                   tags=[cfg.wandb.tag] if cfg.wandb.tag else [], # Handle null tag
                   name = cfg.wandb.name,
                   mode = cfg.wandb.mode)
    
    omegaconf.OmegaConf.resolve(cfg) # Resolve any remaining interpolations in cfg
    set_seed(cfg.seed)

    # Init output folder
    print("Output folder: {}".format(cfg.gs.dataset.model_path))
    os.makedirs(cfg.gs.dataset.model_path, exist_ok=True)
    
    # Save a representation of the config arguments used by the original Scene class
    # This is for compatibility if Scene needs these specific args.
    # Note: hydra's cfg object is more comprehensive.
    scene_args_for_log = {
            "sh_degree": cfg.gs.sh_degree, # Taken from gs config directly
            "source_path": cfg.gs.dataset.source_path,
            "model_path": cfg.gs.dataset.model_path,
            "images": cfg.gs.dataset.images,
            "depths": cfg.gs.dataset.depths if hasattr(cfg.gs.dataset, 'depths') else "", # Optional
            "resolution": cfg.gs.dataset.resolution,
            "white_background": cfg.gs.dataset.white_background, # Corrected: removed underscore
            "train_test_exp": cfg.gs.dataset.train_test_exp if hasattr(cfg.gs.dataset, 'train_test_exp') else False, # Optional
            "data_device": cfg.gs.dataset.data_device,
            "eval": cfg.gs.dataset.eval,
            # Pipe arguments are part of cfg.gs.pipe
            "convert_SHs_python": cfg.gs.pipe.convert_SHs_python,
            "compute_cov3D_python": cfg.gs.pipe.compute_cov3D_python,
            "debug": cfg.gs.pipe.debug,
            "antialiasing": cfg.gs.pipe.antialiasing   
        }
    with open(os.path.join(cfg.gs.dataset.model_path, "scene_cfg_args_for_log.txt"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**scene_args_for_log)))


    # Init Warper3DGS (which includes GaussianModel and Scene)
    gs = hydra.utils.instantiate(cfg.gs) 

    # Init trainer and launch training
    trainer = EDGSTrainer(GS=gs,
        training_config=cfg.gs.opt,
        dataset_white_background=cfg.gs.dataset.white_background, # Pass this explicitly
        device=cfg.device,
        log_wandb=(cfg.wandb.mode != "disabled") # Pass log_wandb status
        )
    
    trainer.load_checkpoints(cfg.load)
    trainer.timer.start()
    # Pass both initialization config blocks
    trainer.init_with_corr(cfg.init_wC, cfg.init_direct_pcd, verbose=cfg.verbose)      
    trainer.train(cfg.train)
    
    # All done
    wandb.finish()
    print("\nTraining complete.")

if __name__ == "__main__":
    main()