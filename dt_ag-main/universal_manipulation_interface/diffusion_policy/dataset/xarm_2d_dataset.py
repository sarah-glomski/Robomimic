# # diffusion_policy/dataset/xarm_2d_dataset.py
# from typing import Dict, List, Tuple, Optional
# import time
# import numpy as np
# import torch
# import copy
# import zarr
# from diffusion_policy.common.pytorch_util import dict_apply
# from diffusion_policy.model.common.normalizer import LinearNormalizer
# from diffusion_policy.common.normalize_util import get_image_range_normalizer
# from diffusion_policy.dataset.base_dataset import BaseImageDataset
# import cv2
# import os

# class XArmImageDataset2D(BaseImageDataset):
#     """
#     Dataset for *2-D* Diffusion Policy on your xArm recordings.
#     Dynamically loads observations based on shape_meta configuration.
#     """

#     def __init__(
#         self,
#         zarr_path: str,
#         shape_meta: Dict[str, Dict] = None,
#         horizon: int = 2,
#         pad_before: int = 0,
#         pad_after: int = 0,
#         seed: int = 42,
#         val_ratio: float = 0.0,
#         max_train_episodes: Optional[int] = None,
#     ):
#         super().__init__()
        
#         # Parse shape_meta configuration
#         self.obs_config = {}
#         self.rgb_keys = []
#         self.non_rgb_keys = []
        
#         if shape_meta is not None and 'obs' in shape_meta:
#             obs_meta = shape_meta['obs']
#             for key, meta in obs_meta.items():
#                 self.obs_config[key] = meta
#                 if meta.get('type') == 'rgb':
#                     self.rgb_keys.append(key)
#                 else:
#                     self.non_rgb_keys.append(key)
#         else:
#             # Default configuration if no shape_meta provided
#             self.rgb_keys = ['rs_side_rgb', 'rs_front_rgb']
#             self.non_rgb_keys = ['pose']
#             self.obs_config = {
#                 'rs_side_rgb': {'shape': [3, 160, 220], 'type': 'rgb'},
#                 'rs_front_rgb': {'shape': [3, 160, 220], 'type': 'rgb'},
#                 'pose': {'shape': [10]}
#             }
        
#         print(f"Loaded observation config:")
#         print(f"  RGB keys: {self.rgb_keys}")
#         print(f"  Non-RGB keys: {self.non_rgb_keys}")
        
#         # Open the Zarr store
#         self.root = zarr.open(zarr_path, mode="r")

#         # List of episodes
#         self.episodes: List[str] = sorted(self.root.group_keys())

#         # Split episodes
#         np.random.seed(seed)
#         n_eps = len(self.episodes)
#         idxs = np.arange(n_eps)
#         np.random.shuffle(idxs)
#         n_val = int(val_ratio * n_eps)
#         val_idxs = set(idxs[:n_val])
#         train_idxs = [i for i in idxs if i not in val_idxs]

#         if max_train_episodes is not None:
#             train_idxs = list(train_idxs)[:max_train_episodes]

#         self.train_eps = {self.episodes[i] for i in train_idxs}
#         self.val_eps = {self.episodes[i] for i in val_idxs}

#         # Build index list: (episode, start_t) for training
#         self.horizon = horizon
#         self.pad_before = pad_before
#         self.pad_after = pad_after
#         self.index: List[Tuple[str,int]] = []
        
#         for epi in self.train_eps:
#             grp = self.root[epi]
#             # Find minimum length across all RGB observations
#             min_T = float('inf')
#             for rgb_key in self.rgb_keys:
#                 if rgb_key in grp:
#                     T = grp[rgb_key].shape[0]
#                     min_T = min(min_T, T)
#                 else:
#                     print(f"Warning: {rgb_key} not found in episode {epi}")
            
#             if min_T > horizon:
#                 for t in range(min_T - horizon):
#                     self.index.append((epi, t))

#     def get_validation_dataset(self) -> "XArmImageDataset2D":
#         # shallow copy, swap to validation episodes
#         val_ds = copy.copy(self)
#         val_ds.index = []
#         for epi in self.val_eps:
#             grp = val_ds.root[epi]
#             # Find minimum length across all RGB observations
#             min_T = float('inf')
#             for rgb_key in val_ds.rgb_keys:
#                 if rgb_key in grp:
#                     T = grp[rgb_key].shape[0]
#                     min_T = min(min_T, T)
            
#             if min_T > val_ds.horizon:
#                 for t in range(min_T - val_ds.horizon):
#                     val_ds.index.append((epi, t))
#         return val_ds

#     def __len__(self) -> int:
#         return len(self.index)

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         epi, t0 = self.index[idx]
#         grp = self.root[epi]
#         sl = slice(t0, t0 + self.horizon)

#         obs_dict = {}
        
#         # Load RGB observations
#         for rgb_key in self.rgb_keys:
#             if rgb_key in grp:
#                 rgb_data = grp[rgb_key][sl]  # (H, h, w, 3) or (H, 3, h, w)
                
#                 # If your Zarr stores HWC, move channels to front
#                 if rgb_data.ndim == 4 and rgb_data.shape[-1] == 3:
#                     # (T, H, W, C) -> (T, C, H, W)
#                     rgb_img = np.moveaxis(rgb_data, -1, 1)
#                 else:
#                     # assume already (T, C, H, W)
#                     rgb_img = rgb_data
                
#                 # Normalize to [0, 1]
#                 rgb_img = rgb_img.astype(np.float32) / 255.0
#                 obs_dict[rgb_key] = rgb_img
        
#         # Load non-RGB observations (e.g., pose)
#         for key in self.non_rgb_keys:
#             if key in grp:
#                 obs_dict[key] = grp[key][sl].astype(np.float32)
        
#         # Load action
#         action = grp["action"][sl].astype(np.float32)
        
#         # # Debug visualization - save images every ~1000 samples
#         # if np.random.randint(0, 1000) == 0:
#         #     save_dir = "debug_images"
#         #     os.makedirs(save_dir, exist_ok=True)
            
#         #     # Save action and non-RGB observations
#         #     np.save(os.path.join(save_dir, "action.npy"), action)
#         #     for key in self.non_rgb_keys:
#         #         if key in obs_dict:
#         #             np.save(os.path.join(save_dir, f"{key}.npy"), obs_dict[key])
            
#         #     # Save the last frame from each RGB camera
#         #     for rgb_key in self.rgb_keys:
#         #         if rgb_key in obs_dict:
#         #             rgb_img_last = obs_dict[rgb_key][-1]
                    
#         #             # Convert from (C, H, W) to (H, W, C) and scale to uint8
#         #             rgb_img_save = (np.transpose(rgb_img_last, (1, 2, 0)) * 255).astype(np.uint8)
                    
#         #             # Convert RGB to BGR for cv2
#         #             rgb_img_save = cv2.cvtColor(rgb_img_save, cv2.COLOR_RGB2BGR)
                    
#         #             # Save the image
#         #             cv2.imwrite(os.path.join(save_dir, f"{rgb_key}.png"), rgb_img_save)
                    
#         #             # print(f"{rgb_key} shape: {obs_dict[rgb_key].shape}")
            
#         #     # print(f"Debug data saved to {save_dir}/")
#         #     # print(f"action shape: {action.shape}")

#         sample = {
#             "obs": obs_dict,
#             "action": action,
#         }
#         return dict_apply(sample, torch.from_numpy)

#     def get_normalizer(self, mode: str = "limits", **kwargs):
#         # Gather data for normalization
#         data_dict = {}
        
#         # Collect non-RGB observations
#         for key in self.non_rgb_keys:
#             data_list = []
#             for epi in self.train_eps:
#                 grp = self.root[epi]
#                 if key in grp:
#                     data_list.append(grp[key][...])
#             if data_list:
#                 data_dict[key] = np.concatenate(data_list, axis=0)
        
#         # Collect actions
#         acts = []
#         for epi in self.train_eps:
#             grp = self.root[epi]
#             acts.append(grp["action"][...])
#         data_dict["action"] = np.concatenate(acts, axis=0)
        
#         # Create normalizer
#         normalizer = LinearNormalizer()
#         normalizer.fit(data=data_dict, last_n_dims=1, mode=mode, **kwargs)
        
#         # Add identity normalizers for RGB images
#         for rgb_key in self.rgb_keys:
#             normalizer[rgb_key] = get_image_range_normalizer()
        
#         return normalizer


# diffusion_policy/dataset/xarm_2d_dataset.py
from typing import Dict, List, Tuple, Optional
import time
import numpy as np
import torch
import copy
import zarr
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
import cv2
import os
import psutil

class XArmImageDataset2D(BaseImageDataset):
    """
    Dataset for *2-D* Diffusion Policy on your xArm recordings.
    Dynamically loads observations based on shape_meta configuration.
    Now supports full RAM preloading for maximum speed.
    """

    def __init__(
        self,
        zarr_path: str,
        shape_meta: Dict[str, Dict] = None,
        horizon: int = 2,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        preload: bool = True,  # NEW: Enable full RAM preloading
    ):
        super().__init__()
        
        self.is_training = True  # Track if we're in training mode
        
        self.preload = preload
        self.preloaded_data = {}  # Will store all episodes if preloading
        self.shape_meta = shape_meta  # Store shape_meta for horizon access
        
        # Parse shape_meta configuration
        self.obs_config = {}
        self.rgb_keys = []
        self.non_rgb_keys = []
        
        if shape_meta is not None and 'obs' in shape_meta:
            obs_meta = shape_meta['obs']
            for key, meta in obs_meta.items():
                self.obs_config[key] = meta
                if meta.get('type') == 'rgb':
                    self.rgb_keys.append(key)
                else:
                    self.non_rgb_keys.append(key)
            
            # Also store action config if present
            if 'action' in shape_meta:
                self.obs_config['action'] = shape_meta['action']
        else:
            # Default configuration if no shape_meta provided
            self.rgb_keys = ['rs_side_rgb', 'rs_front_rgb']
            self.non_rgb_keys = ['pose']
            self.obs_config = {
                'rs_side_rgb': {'shape': [3, 160, 220], 'type': 'rgb'},
                'rs_front_rgb': {'shape': [3, 160, 220], 'type': 'rgb'},
                'pose': {'shape': [10]}
            }
        
        print(f"Loaded observation config:")
        print(f"  RGB keys: {self.rgb_keys}")
        print(f"  Non-RGB keys: {self.non_rgb_keys}")
        
        # Print horizon information for debugging
        for key in self.rgb_keys:
            obs_horizon = self.obs_config[key].get('horizon', horizon)
            print(f"  {key} horizon: {obs_horizon}")
        for key in self.non_rgb_keys:
            obs_horizon = self.obs_config[key].get('horizon', horizon)
            print(f"  {key} horizon: {obs_horizon}")
        if 'action' in self.obs_config:
            action_horizon = self.obs_config['action'].get('horizon', horizon)
            print(f"  action horizon: {action_horizon}")
        
        # Open the Zarr store
        self.root = zarr.open(zarr_path, mode="r")

        # List of episodes
        self.episodes: List[str] = sorted(self.root.group_keys())

        # Split episodes
        np.random.seed(seed)
        n_eps = len(self.episodes)
        idxs = np.arange(n_eps)
        np.random.shuffle(idxs)
        n_val = int(val_ratio * n_eps)
        val_idxs = set(idxs[:n_val])
        train_idxs = [i for i in idxs if i not in val_idxs]

        if max_train_episodes is not None:
            train_idxs = list(train_idxs)[:max_train_episodes]

        self.train_eps = {self.episodes[i] for i in train_idxs}
        self.val_eps = {self.episodes[i] for i in val_idxs}

        # Build index list: (episode, start_t) for training
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.index: List[Tuple[str,int]] = []
        
        # Preload data if requested
        if self.preload:
            print("🚀 Preloading dataset into RAM...")
            self._preload_all_data()
        
        # Build index after preloading
        # Use action horizon as the minimum required timesteps for indexing
        action_horizon = horizon  # default fallback
        if 'action' in self.obs_config:
            action_horizon = self.obs_config['action'].get('horizon', horizon)
        elif self.shape_meta and 'action' in self.shape_meta:
            action_horizon = self.shape_meta['action'].get('horizon', horizon)
        
        print(f"Using action_horizon={action_horizon} for indexing")
        
        for epi in self.train_eps:
            if self.preload:
                min_T = self._get_min_timesteps_preloaded(epi)
            else:
                grp = self.root[epi]
                min_T = self._get_min_timesteps_zarr(grp)
            
            if min_T > action_horizon:  # Use action_horizon instead of horizon
                for t in range(min_T - action_horizon):
                    self.index.append((epi, t))

    def _get_min_timesteps_zarr(self, grp):
        """Get minimum timesteps from zarr group"""
        min_T = float('inf')
        for rgb_key in self.rgb_keys:
            if rgb_key in grp:
                T = grp[rgb_key].shape[0]
                min_T = min(min_T, T)
            else:
                print(f"Warning: {rgb_key} not found in episode")
        return min_T

    def _get_min_timesteps_preloaded(self, epi):
        """Get minimum timesteps from preloaded data"""
        if epi not in self.preloaded_data:
            return 0
        min_T = float('inf')
        for rgb_key in self.rgb_keys:
            if rgb_key in self.preloaded_data[epi]:
                T = self.preloaded_data[epi][rgb_key].shape[0]
                min_T = min(min_T, T)
        return min_T

    def _preload_all_data(self):
        """Load all episodes into RAM"""
        start_time = time.time()
        total_memory_mb = 0
        
        episodes_to_load = self.train_eps | self.val_eps  # Union of train and val
        
        for i, epi in enumerate(episodes_to_load):
            print(f"Loading episode {i+1}/{len(episodes_to_load)}: {epi}")
            grp = self.root[epi]
            
            episode_data = {}
            
            # Load RGB observations
            for rgb_key in self.rgb_keys:
                if rgb_key in grp:
                    rgb_data = grp[rgb_key][...]  # Load entire array
                    
                    # Convert to (T, C, H, W) format and normalize
                    if rgb_data.ndim == 4 and rgb_data.shape[-1] == 3:
                        rgb_data = np.moveaxis(rgb_data, -1, 1)
                    
                    # Convert to float32 and normalize to [0, 1]
                    rgb_data = rgb_data.astype(np.float32) / 255.0
                    episode_data[rgb_key] = rgb_data
                    
                    # Track memory usage
                    memory_mb = rgb_data.nbytes / (1024 * 1024)
                    total_memory_mb += memory_mb
                    print(f"  {rgb_key}: {rgb_data.shape} ({memory_mb:.1f} MB)")
            
            # Load non-RGB observations
            for key in self.non_rgb_keys:
                if key in grp:
                    data = grp[key][...].astype(np.float32)
                    episode_data[key] = data
                    memory_mb = data.nbytes / (1024 * 1024)
                    total_memory_mb += memory_mb
                    print(f"  {key}: {data.shape} ({memory_mb:.1f} MB)")
            
            # Load actions
            action_data = grp["action"][...].astype(np.float32)
            episode_data["action"] = action_data
            memory_mb = action_data.nbytes / (1024 * 1024)
            total_memory_mb += memory_mb
            
            self.preloaded_data[epi] = episode_data
        
        load_time = time.time() - start_time
        print(f"✅ Preloading complete!")
        print(f"   Total data: {total_memory_mb/1024:.2f} GB")
        print(f"   Load time: {load_time:.1f} seconds")
        print(f"   Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")

    def get_validation_dataset(self) -> "XArmImageDataset2D":
        """Create validation dataset - shares preloaded data if available"""
        val_ds = copy.copy(self)
        val_ds.index = []
        val_ds.is_training = False  # Disable noise augmentation for validation
        
        # Use action horizon for indexing (same as training)
        action_horizon = self.horizon  # default fallback
        if 'action' in self.obs_config:
            action_horizon = self.obs_config['action'].get('horizon', self.horizon)
        elif self.shape_meta and 'action' in self.shape_meta:
            action_horizon = self.shape_meta['action'].get('horizon', self.horizon)
        
        for epi in self.val_eps:
            if self.preload:
                min_T = self._get_min_timesteps_preloaded(epi)
            else:
                grp = val_ds.root[epi]
                min_T = self._get_min_timesteps_zarr(grp)
            
            if min_T > action_horizon:  # Use action_horizon instead of val_ds.horizon
                for t in range(min_T - action_horizon):
                    val_ds.index.append((epi, t))
        return val_ds

    def __len__(self) -> int:
        return len(self.index)


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        epi, t0 = self.index[idx]

        if self.preload:
            # Fast path: use preloaded data
            episode_data = self.preloaded_data[epi]
            
            obs_dict = {}
            for rgb_key in self.rgb_keys:
                if rgb_key in episode_data:
                    # Use the horizon from shape_meta for this observation
                    obs_horizon = self.obs_config[rgb_key].get('horizon', self.horizon)
                    sl = slice(t0, t0 + obs_horizon)
                    obs_dict[rgb_key] = episode_data[rgb_key][sl]
            
            for key in self.non_rgb_keys:
                if key in episode_data:
                    # Use the horizon from shape_meta for this observation
                    obs_horizon = self.obs_config[key].get('horizon', self.horizon)
                    sl = slice(t0, t0 + obs_horizon)
                    obs_dict[key] = episode_data[key][sl]
            
            # Action always uses the full horizon (action_horizon from shape_meta)
            action_horizon = self.obs_config.get('action', {}).get('horizon', self.horizon)
            if 'action' in self.obs_config:
                action_horizon = self.obs_config['action'].get('horizon', self.horizon)
            else:
                # Fallback: look in shape_meta for action horizon
                action_horizon = self.shape_meta.get('action', {}).get('horizon', self.horizon)
            sl_action = slice(t0, t0 + action_horizon)
            action = episode_data["action"][sl_action]
            
        else:
            # Slow path: load from zarr (fallback)
            grp = self.root[epi]
            obs_dict = {}
            
            # Load RGB observations with individual horizons
            for rgb_key in self.rgb_keys:
                if rgb_key in grp:
                    obs_horizon = self.obs_config[rgb_key].get('horizon', self.horizon)
                    sl = slice(t0, t0 + obs_horizon)
                    rgb_data = grp[rgb_key][sl]
                    
                    if rgb_data.ndim == 4 and rgb_data.shape[-1] == 3:
                        rgb_img = np.moveaxis(rgb_data, -1, 1)
                    else:
                        rgb_img = rgb_data
                    
                    rgb_img = rgb_img.astype(np.float32) / 255.0
                    obs_dict[rgb_key] = rgb_img
            
            # Load non-RGB observations with individual horizons
            for key in self.non_rgb_keys:
                if key in grp:
                    obs_horizon = self.obs_config[key].get('horizon', self.horizon)
                    sl = slice(t0, t0 + obs_horizon)
                    obs_dict[key] = grp[key][sl].astype(np.float32)
            
            # Load action with action horizon
            action_horizon = self.obs_config.get('action', {}).get('horizon', self.horizon)
            if 'action' in self.obs_config:
                action_horizon = self.obs_config['action'].get('horizon', self.horizon)
            else:
                # Fallback: look in shape_meta for action horizon
                action_horizon = self.shape_meta.get('action', {}).get('horizon', self.horizon)
            sl_action = slice(t0, t0 + action_horizon)
            action = grp["action"][sl_action].astype(np.float32)

        # Apply noise augmentation to pose data during training
        # obs_dict = self._apply_pose_noise(obs_dict)

        sample = {
            "obs": obs_dict,
            "action": action,
        }
        return dict_apply(sample, torch.from_numpy)

    def get_normalizer(self, mode: str = "limits", **kwargs):
        # Gather data for normalization
        data_dict = {}
        
        if self.preload:
            # Fast path: use preloaded data
            for key in self.non_rgb_keys:
                data_list = []
                for epi in self.train_eps:
                    if epi in self.preloaded_data and key in self.preloaded_data[epi]:
                        data_list.append(self.preloaded_data[epi][key])
                if data_list:
                    data_dict[key] = np.concatenate(data_list, axis=0)
            
            # Collect actions
            acts = []
            for epi in self.train_eps:
                if epi in self.preloaded_data:
                    acts.append(self.preloaded_data[epi]["action"])
            data_dict["action"] = np.concatenate(acts, axis=0)
            
        else:
            # Slow path: load from zarr
            for key in self.non_rgb_keys:
                data_list = []
                for epi in self.train_eps:
                    grp = self.root[epi]
                    if key in grp:
                        data_list.append(grp[key][...])
                if data_list:
                    data_dict[key] = np.concatenate(data_list, axis=0)
            
            acts = []
            for epi in self.train_eps:
                grp = self.root[epi]
                acts.append(grp["action"][...])
            data_dict["action"] = np.concatenate(acts, axis=0)
        
        # Create normalizer
        normalizer = LinearNormalizer()
        normalizer.fit(data=data_dict, last_n_dims=1, mode=mode, **kwargs)
        
        # Add identity normalizers for RGB images
        for rgb_key in self.rgb_keys:
            normalizer[rgb_key] = get_image_range_normalizer()
        
        return normalizer