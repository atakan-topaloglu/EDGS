import torch
from random import randint
from tqdm.rich import trange
from tqdm import tqdm as tqdm
from source.networks import Warper3DGS
import wandb
import sys
import numpy as np # Added
from plyfile import PlyData # Added
from typing import NamedTuple
# from submodules.gaussian_splatting.utils.graphics_utils import BasicPointCloud # Corrected import style
# from submodules.gaussian_splatting.utils.sh_utils import RGB2SH # RGB2SH is used internally by GaussianModel

sys.path.append('./submodules/gaussian-splatting/')
import lpips
from source.losses import ssim, l1_loss, psnr
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red"
})

from source.corr_init import init_gaussians_with_corr, init_gaussians_with_corr_fast
from source.utils_aux import log_samples

from source.timer import Timer

class EDGSTrainer:
    def __init__(self,
                 GS: Warper3DGS,
                 training_config,
                 dataset_white_background=False,
                 device=torch.device('cuda'),
                 log_wandb=True,
                 ):
        self.GS = GS
        self.scene = GS.scene
        self.viewpoint_stack = GS.viewpoint_stack
        self.gaussians = GS.gaussians

        self.training_config = training_config
        self.GS_optimizer = GS.gaussians.optimizer
        self.dataset_white_background = dataset_white_background

        self.training_step = 1
        self.gs_step = 0
        self.CONSOLE = Console(width=120, theme=custom_theme)
        self.saving_iterations = training_config.save_iterations
        self.evaluate_iterations = None
        self.batch_size = training_config.batch_size
        self.ema_loss_for_log = 0.0

        self.logs_losses = {}
        self.lpips = lpips.LPIPS(net='vgg').to(device)
        self.device = device
        self.timer = Timer()
        self.log_wandb = log_wandb

    def load_checkpoints(self, load_cfg):
        # Load 3DGS checkpoint
        if load_cfg.gs:
            # Corrected: self.GS.gaussians.restore, not self.gs.gaussians.restore
            self.GS.gaussians.restore(
                torch.load(f"{load_cfg.gs}/chkpnt{load_cfg.gs_step}.pth")[0],
                self.training_config)
            self.GS_optimizer = self.GS.gaussians.optimizer
            self.CONSOLE.print(f"3DGS loaded from checkpoint for iteration {load_cfg.gs_step}",
                               style="info")
            self.training_step += load_cfg.gs_step
            self.gs_step += load_cfg.gs_step

    def init_gaussians_from_pcd(self, pcd_path, spatial_lr_scale):
        self.CONSOLE.print(f"Attempting to initialize Gaussians from direct PCD: {pcd_path}", style="info")
        try:
            plydata = PlyData.read(pcd_path)
            elements = plydata.elements[0] # Assuming points are in the first element
        except Exception as e:
            self.CONSOLE.print(f"Failed to read .ply file: {e}", style="danger")
            raise RuntimeError(f"Could not load point cloud from {pcd_path}")

        xyz = np.stack((np.asarray(elements["x"]),
                        np.asarray(elements["y"]),
                        np.asarray(elements["z"])), axis=1).astype(np.float32)
        
        num_points = xyz.shape[0]
        if num_points == 0:
            self.CONSOLE.print(f"Warning: Loaded 0 points from {pcd_path}. Cannot initialize.", style="danger")
            raise RuntimeError(f"Loaded 0 points from {pcd_path}")

        # Colors (RGB 0-1)
        if "red" in elements and "green" in elements and "blue" in elements:
            colors_rgb = np.stack((np.asarray(elements["red"]),
                                   np.asarray(elements["green"]),
                                   np.asarray(elements["blue"])), axis=1)
            if np.max(colors_rgb) > 1.0: # Assuming colors might be 0-255
                colors_rgb = colors_rgb / 255.0
        else:
            self.CONSOLE.print("No color information in PCD, defaulting to gray.", style="info")
            colors_rgb = np.ones_like(xyz) * 0.5  # Default to gray
        colors_rgb = colors_rgb.astype(np.float32)

        # Normals (optional)
        normals = None
        if "nx" in elements and "ny" in elements and "nz" in elements:
            normals = np.stack((np.asarray(elements["nx"]),
                                np.asarray(elements["ny"]),
                                np.asarray(elements["nz"])), axis=1).astype(np.float32)
        
        class BasicPointCloud(NamedTuple):
            points : np.array
            colors : np.array
            normals : np.array
            
        pcd_obj = BasicPointCloud(points=xyz, colors=colors_rgb, normals=normals)
        
        # This will replace any existing Gaussians (e.g., from SfM)
        self.GS.gaussians.create_from_pcd(pcd_obj, spatial_lr_scale)
        self.CONSOLE.print(f"Initialized {self.GS.gaussians.get_xyz.shape[0]} Gaussians from PCD.", style="info")
        return None # No visualization_dict for PCD initialization

    def init_with_corr(self, cfg_init_wC, cfg_init_direct_pcd, verbose=False, roma_model=None):
        visualization_dict = None
        # SfM points are loaded by Scene() if points3D.bin exists, before this method is called.
        initial_sfm_points_count = self.GS.gaussians.get_xyz.shape[0] 
        if initial_sfm_points_count > 0:
             self.CONSOLE.print(f"{initial_sfm_points_count} SfM points were loaded by the Scene object.", style="info")
        else:
             self.CONSOLE.print("No SfM points were loaded by the Scene object.", style="info")


        if cfg_init_direct_pcd.use and cfg_init_direct_pcd.path is not None:
            self.CONSOLE.print(f"Prioritizing direct PCD initialization from: {cfg_init_direct_pcd.path}", style="info")
            
            # Remove pre-existing (SfM) points before loading from PCD.
            # `create_from_pcd` replaces existing points, so an explicit prune before might be redundant
            # but ensures a clean state if `create_from_pcd` behavior changes or if it expects an empty model.
            # For clarity and safety, we ensure the model is "clean" of SfM points if they are not desired.
            if initial_sfm_points_count > 0:
                self.CONSOLE.print(f"Removing {initial_sfm_points_count} pre-existing SfM points before direct PCD loading.", style="info")
                # Create a dummy mask to effectively clear all points, then re-init optimizer
                # Note: create_from_pcd already re-initializes optimizer parameters.
                # So, just ensuring the GaussianModel is empty or that create_from_pcd fully overwrites is key.
                # Let's assume create_from_pcd correctly replaces everything.
                # No explicit prune needed here if create_from_pcd is a full reset.
                # self.GS.gaussians._xyz = torch.empty(0,3).cuda() # Example of clearing, but create_from_pcd should handle it
                pass # create_from_pcd will overwrite.

            visualization_dict = self.init_gaussians_from_pcd(cfg_init_direct_pcd.path, cfg_init_direct_pcd.spatial_lr_scale)
            initial_sfm_points_count = 0 # SfM points are now replaced by PCD points
        
        elif cfg_init_wC.use: # EDGS RoMA initialization
            self.CONSOLE.print("Using EDGS (RoMA) correspondence-based initialization.", style="info")
            
            if cfg_init_wC.nns_per_ref == 1: # This logic is from original EDGS
                init_fn = init_gaussians_with_corr_fast
            else:
                init_fn = init_gaussians_with_corr
            
            # RoMA init function adds points to existing ones (SfM points if any)
            _, _, visualization_dict = init_fn(
                self.GS.gaussians,
                self.scene,
                cfg_init_wC,
                self.device,
                verbose=verbose,
                roma_model=roma_model)

            # Remove SfM points if add_SfM_init is False and SfM points were initially present
            if not cfg_init_wC.add_SfM_init and initial_sfm_points_count > 0:
                current_total_points = self.GS.gaussians.get_xyz.shape[0]
                # Check if RoMA actually added points, otherwise SfM points might be all there is
                if current_total_points > initial_sfm_points_count:
                    # Create a mask to remove the first `initial_sfm_points_count` points (these were the SfM points)
                    mask_remove_sfm = torch.zeros(current_total_points, dtype=torch.bool).to(self.device)
                    mask_remove_sfm[:initial_sfm_points_count] = True
                    
                    self.CONSOLE.print(f"EDGS RoMA: Removing {initial_sfm_points_count} SfM points as per add_SfM_init=False.", style="info")
                    self.GS.gaussians.prune_points(mask_remove_sfm)
                    torch.cuda.empty_cache()
        
        else: # Neither direct PCD nor RoMA initialization is used.
            if initial_sfm_points_count > 0:
                self.CONSOLE.print(f"Using {initial_sfm_points_count} SfM points as primary initialization.", style="info")
            else:
                self.CONSOLE.print("No SfM points found and no other initialization specified. Starting with 0 Gaussians.", style="warning")
                # Consider raising an error or strong warning if train.no_densify is also True,
                # as training might not proceed meaningfully.

        # Common post-processing for scaling, similar to EDGS post-init
        # Apply if EDGS RoMA added new points or if Direct PCD was used.
        # This ensures the initial scales are small, promoting refinement.
        points_were_added_or_replaced_by_edgs_or_pcd = False
        if cfg_init_direct_pcd.use and cfg_init_direct_pcd.path is not None:
            points_were_added_or_replaced_by_edgs_or_pcd = True
        elif cfg_init_wC.use and self.GS.gaussians.get_xyz.shape[0] > initial_sfm_points_count:
             # RoMA was used and it added points beyond any initial SfM points
            points_were_added_or_replaced_by_edgs_or_pcd = True
        
        if points_were_added_or_replaced_by_edgs_or_pcd:
            with torch.no_grad():
                gaussians = self.GS.gaussians
                if gaussians.get_xyz.shape[0] > 0:
                    self.CONSOLE.print("Applying 0.5x scaling adjustment to initialized Gaussians.", style="info")
                    current_param_scales = gaussians._scaling.detach().clone() # These are in logit/log space
                    activated_scales = gaussians.scaling_activation(current_param_scales) # Convert to linear scale
                    adjusted_linear_scales = activated_scales * 0.5 # Reduce linear scale
                    new_parameter_scales = gaussians.scaling_inverse_activation(adjusted_linear_scales) # Convert back to logit/log space
                    gaussians._scaling.data.copy_(new_parameter_scales) # Update parameter in-place
                    
        return visualization_dict

    def train(self, train_cfg):
        # 3DGS training
        self.CONSOLE.print("Train 3DGS for {} iterations".format(train_cfg.gs_epochs), style="info")    
        # Corrected: train_cfg.gs_epochs might be 0, ensure range is valid
        num_training_steps = train_cfg.gs_epochs
        if num_training_steps <=0:
            self.CONSOLE.print("gs_epochs is 0 or less. Skipping training loop.", style="warning")
            # Potentially save model even if no training steps, e.g., after init
            if self.saving_iterations and (0 in self.saving_iterations or self.training_step in self.saving_iterations):
                 self.save_model()
            if self.evaluate_iterations and (0 in self.evaluate_iterations or self.training_step in self.evaluate_iterations):
                 self.evaluate()
            return

        with trange(self.training_step, self.training_step + num_training_steps, desc="[green]Train gaussians") as progress_bar:
            for self.training_step in progress_bar:
                radii = self.train_step_gs(max_lr=train_cfg.max_lr, no_densify=train_cfg.no_densify)
                with torch.no_grad():
                    if train_cfg.no_densify:
                        self.prune(radii)
                    else:
                        self.densify_and_prune(radii)
                    if train_cfg.reduce_opacity:
                        # Slightly reduce opacity every few steps:
                        if self.gs_step < self.training_config.densify_until_iter and self.gs_step % 10 == 0:
                            opacities_new = torch.log(torch.exp(self.GS.gaussians._opacity.data) * 0.99)
                            self.GS.gaussians._opacity.data = opacities_new
                    self.timer.pause()
                    # Progress bar
                    if self.training_step % 10 == 0:
                        progress_bar.set_postfix({"[red]Loss": f"{self.ema_loss_for_log:.{7}f}"}, refresh=True)
                    # Log and save
                    if self.training_step in self.saving_iterations:
                        self.save_model()
                    if self.evaluate_iterations is not None:
                        if self.training_step in self.evaluate_iterations:
                            self.evaluate()
                    else:
                        if (self.training_step <= 3000 and self.training_step % 500 == 0) or \
                            (self.training_step > 3000 and self.training_step % 1000 == 0) : # Changed 228 to 0 for regular interval
                            self.evaluate()

                    self.timer.start()
    # ... (rest of the methods: evaluate, train_step_gs, densify_and_prune, save_model, prune) ...
    # Ensure these methods are correctly defined as part of the class
    # (They are already present in the provided codebase structure)

    def evaluate(self):
        torch.cuda.empty_cache()
        log_gen_images, log_real_images = [], []
        # Ensure TEST_CAM_IDX_TO_LOG is valid
        test_cam_idx_to_log = self.training_config.TEST_CAM_IDX_TO_LOG
        
        validation_configs = []
        test_cameras = self.scene.getTestCameras()
        if test_cameras and len(test_cameras) > 0:
            if test_cam_idx_to_log >= len(test_cameras):
                self.CONSOLE.print(f"Warning: TEST_CAM_IDX_TO_LOG ({test_cam_idx_to_log}) is out of bounds for test cameras ({len(test_cameras)}). Defaulting to 0.", style="warning")
                test_cam_idx_to_log = 0
            validation_configs.append({'name': 'test', 'cameras': test_cameras, 'cam_idx': test_cam_idx_to_log if len(test_cameras) > 0 else -1})

        train_cameras_for_eval = [self.scene.getTrainCameras()[idx % len(self.scene.getTrainCameras())] for idx in range(0, 150, 5)]
        train_cam_idx_to_log = self.training_config.TRAIN_CAM_IDX_TO_LOG # Use the actual train cam idx from config
        if train_cameras_for_eval and len(train_cameras_for_eval) > 0:
            # The train_cam_idx_to_log is an index into the *original* full training camera list,
            # not the subsampled `train_cameras_for_eval`. For simplicity in logging one image,
            # we can just pick one from the subsampled list, e.g., the middle one or one specified.
            # Here, we'll pick one based on `config['cam_idx']` which is set to 10 for train.
            # Let's make sure the 'cam_idx' for train in validation_configs is valid for train_cameras_for_eval
            actual_train_log_idx = 10 # as per original hardcoding
            if actual_train_log_idx >= len(train_cameras_for_eval):
                actual_train_log_idx = 0 if len(train_cameras_for_eval) > 0 else -1

            validation_configs.append({'name': 'train', 'cameras': train_cameras_for_eval, 'cam_idx': actual_train_log_idx})
        
        if self.log_wandb:
            wandb.log({f"Number of Gaussians": len(self.GS.gaussians._xyz)}, step=self.training_step)

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_splat_test = 0.0
                for idx, viewpoint in enumerate(tqdm(config['cameras'], desc=f"Evaluating {config['name']}", leave=False)):
                    image = torch.clamp(self.GS(viewpoint)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to(self.device), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).double()
                    psnr_test += psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).double() # psnr expects batch
                    ssim_test += ssim(image, gt_image).double()
                    lpips_splat_test += self.lpips(image, gt_image).detach().double()
                    if idx == config['cam_idx'] and config['cam_idx'] != -1: # Check if cam_idx is valid
                        log_gen_images.append(image)
                        log_real_images.append(gt_image)
                
                num_eval_cams = len(config['cameras'])
                psnr_test /= num_eval_cams
                l1_test /= num_eval_cams
                ssim_test /= num_eval_cams
                lpips_splat_test /= num_eval_cams
                if self.log_wandb:
                    wandb.log({f"{config['name']}/L1": l1_test.item(), f"{config['name']}/PSNR": psnr_test.item(), \
                            f"{config['name']}/SSIM": ssim_test.item(), f"{config['name']}/LPIPS_splat": lpips_splat_test.item()}, step = self.training_step)
                self.CONSOLE.print("\n[ITER {}], #{} gaussians, Evaluating {}: L1={:.6f},  PSNR={:.6f}, SSIM={:.6f}, LPIPS_splat={:.6f} ".format(
                    self.training_step, len(self.GS.gaussians._xyz), config['name'], l1_test.item(), psnr_test.item(), ssim_test.item(), lpips_splat_test.item()), style="info")
        
        if self.log_wandb and log_gen_images and log_real_images: # Ensure lists are not empty
            with torch.no_grad():
                # Assuming at least one test and one train image was logged if available
                # For simplicity, just log the first pair found (e.g., test image)
                log_samples(torch.stack((log_real_images[0],log_gen_images[0])) , [], self.training_step, caption="Real and Generated Samples (Test/Train)")
                if len(log_gen_images) > 1: # If train image was also logged
                     log_samples(torch.stack((log_real_images[1],log_gen_images[1])) , [], self.training_step, caption="Real and Generated Samples (Train/Test)")

                wandb.log({"time": self.timer.get_elapsed_time()}, step=self.training_step)
        torch.cuda.empty_cache()

    def train_step_gs(self, max_lr = False, no_densify = False):
        self.gs_step += 1
        if max_lr:
            self.GS.gaussians.update_learning_rate(max(self.gs_step, 8_000))
        else:
            self.GS.gaussians.update_learning_rate(self.gs_step)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.gs_step % 1000 == 0:
            self.GS.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self.viewpoint_stack: # Should be populated by Scene.getTrainCameras()
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        
        if not self.viewpoint_stack: # If still empty (e.g. no train cameras)
            self.CONSOLE.print("Viewpoint stack is empty. Cannot train.", style="danger")
            # This can happen if scene loading fails or data is misconfigured
            # To prevent crashing, we can skip this step, but it indicates a deeper problem.
            # For now, let it raise an error if pop is called on an empty list.
            # Or, more gracefully:
            if not self.scene.getTrainCameras():
                self.CONSOLE.print("No training cameras available in the scene.", style="danger")
                return torch.empty(0, device=self.device) # Return empty radii
            self.viewpoint_stack = self.scene.getTrainCameras().copy()


        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
      
        render_pkg = self.GS(viewpoint_cam=viewpoint_cam)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Loss
        gt_image = viewpoint_cam.original_image.to(self.device)
        Ll1 = l1_loss(image, gt_image) # Renamed to avoid conflict with imported l1_loss

        ssim_loss_val = (1.0 - ssim(image, gt_image)) # Renamed
        loss = (1.0 - self.training_config.lambda_dssim) * Ll1 + \
               self.training_config.lambda_dssim * ssim_loss_val
        
        self.timer.pause() 
        current_losses = {"loss": loss.item(), "L1_loss": Ll1.item(), "ssim_loss": ssim_loss_val.item()}
        self.logs_losses[self.training_step] = current_losses
        
        if self.log_wandb:
            for k, v in current_losses.items():
                wandb.log({f"train/{k}": v}, step=self.training_step)
        self.ema_loss_for_log = 0.4 * current_losses["loss"] + 0.6 * self.ema_loss_for_log
        self.timer.start()
        
        self.GS_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        with torch.no_grad():
            if self.gs_step < self.training_config.densify_until_iter and not no_densify:
                self.GS.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.GS.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter])
                self.GS.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        # Optimizer step
        self.GS_optimizer.step()
        self.GS_optimizer.zero_grad(set_to_none=True) # Clear gradients after step
        return radii

    def densify_and_prune(self, radii = None):
        # Densification or pruning
        if self.gs_step < self.training_config.densify_until_iter:
            if (self.gs_step > self.training_config.densify_from_iter) and \
                    (self.gs_step % self.training_config.densification_interval == 0):
                size_threshold = 20 if self.gs_step > self.training_config.opacity_reset_interval else None
                # Pass radii if available, otherwise GaussianModel will use its tmp_radii if set
                current_radii_to_pass = radii if radii is not None else self.GS.gaussians.tmp_radii 
                self.GS.gaussians.densify_and_prune(self.training_config.densify_grad_threshold,
                                                               0.005, # min_opacity for pruning
                                                               self.GS.scene.cameras_extent,
                                                               size_threshold, current_radii_to_pass)
            if self.gs_step % self.training_config.opacity_reset_interval == 0 or (
                    self.dataset_white_background and self.gs_step == self.training_config.densify_from_iter):
                self.GS.gaussians.reset_opacity()             

    def save_model(self):
        if self.GS.gaussians.get_xyz.shape[0] == 0:
            self.CONSOLE.print("\n[ITER {}] No Gaussians to save.".format(self.gs_step), style="warning")
            return
            
        self.CONSOLE.print("\n[ITER {}] Saving Gaussians (scene.save)".format(self.gs_step), style="info")
        self.scene.save(self.gs_step)
        
        # Save checkpoint using capture method
        # capture() returns a Pytorch state_dict like structure
        captured_state = self.GS.gaussians.capture()
        # Check if captured_state is not None and contains actual parameters
        if captured_state and captured_state.get('_xyz') is not None and captured_state['_xyz'].shape[0] > 0:
            self.CONSOLE.print("\n[ITER {}] Saving Checkpoint (capture)".format(self.gs_step), style="info")
            torch.save((captured_state, self.gs_step),
                    self.scene.model_path + "/chkpnt" + str(self.gs_step) + ".pth")
        else:
            self.CONSOLE.print("\n[ITER {}] Capture method returned empty state. Skipping checkpoint saving.".format(self.gs_step), style="warning")


    def prune(self, radii, min_opacity=0.005):
        if self.GS.gaussians.get_xyz.shape[0] == 0: # No points to prune
            return

        self.GS.gaussians.tmp_radii = radii # Set tmp_radii for prune_points if it uses it
        if self.gs_step < self.training_config.densify_until_iter: # Condition from original code
            prune_mask = (self.GS.gaussians.get_opacity < min_opacity).squeeze()
            if torch.any(prune_mask): # Only prune if there's something to prune
                 self.GS.gaussians.prune_points(prune_mask)
                 torch.cuda.empty_cache()
        # self.GS.gaussians.tmp_radii = None # Reset tmp_radii if it's only for this call
                                        # GaussianModel.prune_points actually uses self.tmp_radii if not None.
                                        # So it's better if GaussianModel itself resets it after use, or we ensure it's always passed.
                                        # For now, let's assume GaussianModel handles its tmp_radii state.