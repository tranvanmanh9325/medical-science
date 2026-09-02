from gym.envs.registration import register


# PSM Env
register(
    id='TissueApproach-Traj-v0',
    entry_point='surrol.tasks.tissue_approach:TissueApproach',
    max_episode_steps=10,
)

register(
    id='TissueRetract-Traj-v0',
    entry_point='surrol.tasks.tissue_retraction:TissueRetract',
    max_episode_steps=10,
)

register(
    id='NeedleGrasp-Traj-v0',
    entry_point='surrol.tasks.needle_grasping:NeedleGrasp',
    max_episode_steps=10,
)

register(
    id='GauzePick-Traj-v0',
    entry_point='surrol.tasks.gauze_picking:GauzePick',
    max_episode_steps=10,
)

register(
    id='VesselClip-Traj-v0',
    entry_point='surrol.tasks.vessel_clipping:VesselClip',
    max_episode_steps=10,
)