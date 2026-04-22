# Robomimic
Rotation project for Sarah Glomski - training a robot to mimic a human in real-time
Worksheet link: https://docs.google.com/spreadsheets/d/10szu_UAZG9ofzCyzFNat_vhCXXukv13KucbzM-qtKw4/edit?gid=702452610#gid=702452610

# Data Collection 
Data collection pipeline is built on ROS2 backbone and was adapted from Aiden Swann's data collection repo: dt_ag-main

Steps to set up data collection: 
1. Plug all 3 cameras (front view, wrist view, top view) into the computer via USB or USB-c. Ensure the camera serial numbers align with the code and run *view_cameras.py* to ensure all 3 camera streams are up and running with the proper camera names. 
2. Power on the robot, release the e-stop, and connect over http in broswer (to ensure robot is reachable). Set the robot in the home configuration if not already there.
3. Activate the *ros2_env* conda environment (if available) to load all modules and dependencies. Otherwise, dependcies will have to be installed manually. 
4. Launch the data collection script with *python3 launch_data_collection.py*. You should see a pygame window pop up plus all 3 camera views. Check that all cameras are online in the pygame window (showing green instead of yellow).
5. Press *s* on the keyboard when you are ready to start. Then move over to the shared workspace and perform your task. Data collection will start when a hand is first registered in the frame.
6. Press *d* to end a data collection. Frames should stop recording as soon as the hand leaves the camera view. When working between episodes, press *p* to pause hand tracking so that the robot does not follow your hand, and *u* to unpause and resume tracking.
7. Upon finishing a trial, it will be saved to the *data_collection/demo_data* folder. The index of the episode will increment by 1 for each demo collected, and if you delete an episode file by moving it to the trash, that index will NOT be repeated, but that is okay. This is still the recommended way for throwing out trials (as opposed to deleting them later), because the training script reads them in ascending order. It's also helpful to keep notes during data collection and testing to keep track of how many trials you've done.

# Training 


# Testing (Inference) 
To visualize an episode: python3 visualize_episode.py demo_data/episode_3.hdf5
To convert to zarr: python3 hdf5_to_zarr.py demo_data/Collection1 output.zarr

# Tips
- The data collection script is design to fail loudly if an error occurs, so always keep an eye on the pygame window and/or the terminal logs. If logs start coming out very quickly, something is probably wrong. Most common is the wrist camera unplugs during arm movement.
- Use consistent movements at the start and end of each trial to make sure the data collection cuts off in a similar place for all episodes. 
