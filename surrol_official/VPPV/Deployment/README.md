# To Run the Code (dVRK)
1. Follow [this guide](https://github.com/jhu-dvrk/sawIntuitiveResearchKit/wiki/CatkinBuild) to build dVRK software and check all prerequisites listed [here](https://github.com/jhu-dvrk/sawIntuitiveResearchKit/wiki/FirstSteps#documentation).

2. Install [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) and [IGEV](https://github.com/gangweix/IGEV) and put the checkpoints in the corresponding directories.
3. Change the configurations in player_config.py
4. To excecute a task, please run the following commands ad specify the task, for example:
```
python super_player.py --task needle
```