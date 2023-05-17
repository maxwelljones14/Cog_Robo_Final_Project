# Cog_Robo_Final_Project
 Cognitive Robotics Final Project

This codebase is built on fsm files, which requires a [cozmo](https://www.digitaldreamlabs.com/products/cozmo-robot), a way to run cozmo in SDK mode, and the [cozmo tools library](https://github.com/touretzkyds/cozmo-tools)

From here, you have to go to the requirements.txt file and make sure that you have pip installed all necessary requirements. If you haven't, definitely pip install all of them first. 

Next, in the MiDas/weights folder, put the weights for the model used. This can be found via [download link](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt)

Next, you should be able to just run simple_cli followed by runfsm("Depth") to get it to start working. Afterwards, type tm to start the depth prediction, and you're good to go. 

Placing cubes in front of the camera should update the depth map as well as the matplotlibplot with a marker saying where the pixelwise location is for depth, as well as the terminal with text about how far away the cubes it sees(or has seen) are
Finally, you can click anywhere on the plot and the terminal should output the predicted depth value at that location. 


We've included some pictures as well in the images folder, and our presentation slides are [here](https://docs.google.com/presentation/d/1n0iIeVLtSl2cOxMWcjmNdKemJOXVC5SWlpsthB5d19o/edit?usp=sharing)
