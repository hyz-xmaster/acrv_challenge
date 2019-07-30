# acrv_challenge
Scripts for playing with and visualizing ACRV challenge data
------------------------------------------------------------

1. Download the demo data from the google drive using the link I shared.

2. To visualize the data, please run the **run_demo.m** in ./matlab folder and change the 'root_dir' of the dataset to yours. The function **findSequencesDifference.m** can be used to find the differences between two sequences and visualize the differences.

3. To play with the data, you can use the **acrv.py**. This python class includes the methods to read all the data, including RGB images, Depth images, Instance images, Segmentation images, Label information, Wheel odometry and Calibration information. In this script, there are also functions to compute the transformation matrices between camera and world, camera and robot, as well as robot and world.

   Please see that script for the usage examples.

4. The ground truth is explained in the **ground_truth_format**.