

"""
Do not store:
    Queues
    Frame buffers
    Runtime state
    Detected objects
    Computed homographies
    Thread objects
Config is declarative, not operational.

“Everything that defines 
how the system behaves, but not what it currently sees"

In the future this config file will be edited by the 
game and the commands are sent through the game GUI 
which runs on Unity.  

This way one could tweak game parameters such as
color_masks and see the changes after reloading
immedietly without having to open the source code
and fiddle the variables from there. 

From these config files the capture_layer would 
get it's settings or all the different
detection parameters such as the different 
hsv ranges so one could detect any ball color
they want to choose by just inputting 
the ranges through the GUI. The GUI would then
change this config file and detection_layer
would take all its detection parameters
such as HSV_RANGES from here. 

But all the dynamic positions of all objects
should not be stored here but on the dynamic
world_state_layer instead. 

That's why these config data classes will store:
    - capture configs
    - detection configs
    - tracking configs
    - visualization configs
    - etc...

"""


# ==================================================
# 0. libraries
# ==================================================

import dataclasses 

# ==================================================
# 1. dataclasses
# ==================================================

"""      EXAMPLE:
@dataclasses.dataclass
class GameConfig:
    arena: ArenaConfig
    aruco: ArucoConfig
    detection: DetectionConfig
    tracking: TrackingConfig
"""