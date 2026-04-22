# Robomimic
Rotation project for Sarah Glomski - training a robot to mimic a human in real-time

Worksheet link: [https://docs.google.com/spreadsheets/d/10szu_UAZG9ofzCyzFNat_vhCXXukv13KucbzM-qtKw4/edit?gid=702452610#gid=702452610](https://docs.google.com/spreadsheets/d/10szu_UAZG9ofzCyzFNat_vhCXXukv13KucbzM-qtKw4/edit?usp=sharing)

# Data Collection 
Data collection pipeline is built on ROS2 backbone and was adapted from Aiden Swann's data collection repo: dt_ag-main

Steps to set up data collection: 
1. Plug all 3 cameras (front view, wrist view, top view) into the computer via USB or USB-c. Ensure the camera serial numbers align with the code and run *view_cameras.py* to ensure all 3 camera streams are up and running with the proper camera names.
2. Camera calibration is optional, but seemed to show marginal improvements. The calibration is for calculating the transforms between each camera, and otherwise this depends on manual inputs in the code to determine the spatial orientation of the scene (which can be visualized with the *launch_sim.py* script). 
3. Power on the robot, release the e-stop, and connect over http in broswer (to ensure robot is reachable). Set the robot in the home configuration if not already there.
4. Activate the *ros2_env* conda environment (if available) to load all modules and dependencies. Otherwise, dependcies will have to be installed manually. 
5. Launch the data collection script with *python3 launch_data_collection.py*. You should see a pygame window pop up plus all 3 camera views. Check that all cameras are online in the pygame window (showing green instead of yellow).
6. Press *s* on the keyboard when you are ready to start. Then move over to the shared workspace and perform your task. Data collection will start when a hand is first registered in the frame.
7. Press *d* to end a data collection. Frames should stop recording as soon as the hand leaves the camera view. When working between episodes, press *p* to pause hand tracking so that the robot does not follow your hand, and *u* to unpause and resume tracking.
8. Upon finishing a trial, it will be saved to the *data_collection/demo_data* folder. The index of the episode will increment by 1 for each demo collected, and if you delete an episode file by moving it to the trash, that index will NOT be repeated, but that is okay. This is still the recommended way for throwing out trials (as opposed to deleting them later), because the training script reads them in ascending order. It's also helpful to keep notes during data collection and testing to keep track of how many trials you've done.

# Data Augmentation
Data augmentation is the step in which datasets were manipulated to have swapped human/robot halves for pairs of trials that have the same end state. These "end states" were "horizonal" (right/left) mug handle and "vertical" (down) mug handle for my experiment. Note that time dilation happens with this method. 

To run augmentation:	python3 augment_data.py \
	--collections Collection4 Collection5 Collection6 Collection7 \
	--horizontal Collection4 Collection6 \
	--vertical Collection5 Collection7 \
	--front-crop-x 250 --head-crop-x 220 \
	--expansion-ratio 2.0 \
	--output-dir demo_data/Augmented 

Use --preview or --dry-run flags at the end of the terminal command to test augmentation crop view first 

Baseline augmentation is the method where the human side of the video is frozen as the first frame for the duration of the video. There is no time dilation here. 

To run baseline augmentation:	python3 augment_data.py \
	--collections Collection4 Collection5 Collection6 Collection7 \
	--horizontal Collection4 Collection6 \
	--vertical Collection5 Collection7 \
	--front-crop-x 250 --head-crop-x 220 \
	--baseline \
	--output-dir demo_data/Baseline

Mirroring augmentation is the method where the human side is mirrored in a small window within its own trial, and does not have time dilation. 

To run mirrored augmentation:	python augment_data.py \
  --collections Collection4 Collection5 Collection6 Collection7 \
  --horizontal Collection4 Collection6 \
  --vertical Collection5 Collection7 \
  --front-crop-x 250 \
  --head-crop-x 220 \
  --mirror \
  --mirror-width 150 \
  --output-dir demo_data/Mirrored

# Training 
To run training (with debug):	python train.py --config-name=train_diffusion_unet_timm_xarm training.debug=True

To run training (without debug): python train.py --config-name=train_diffusion_unet_timm_xarm

# Testing (Inference) 
To run inference:	python launch_inference.py --model /home/sarah/Robomimic/training/data/outputs/2026.02.24/16.03.25_train_diffusion_unet_timm_xarm_xarm_teleop/checkpoints/latest.ckpt

To visualize an episode: python3 visualize_episode.py demo_data/episode_3.hdf5

To convert to zarr: python3 hdf5_to_zarr.py demo_data/Collection1 output.zarr

# Tips
- The data collection script is design to fail loudly if an error occurs, so always keep an eye on the pygame window and/or the terminal logs. If logs start coming out very quickly, something is probably wrong. Most common is the wrist camera unplugs during arm movement.
- Use consistent movements at the start and end of each trial to make sure the data collection cuts off in a similar place for all episodes.
- I would recommend trying the optitrack instead of the realsense cameras with mediapipe, because mediapipe does not do well with hand obstructions. 
