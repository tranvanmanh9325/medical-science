from gym.envs.registration import register


# PSM Env
register(
    id='NeedleReach-v0',
    entry_point='surrol.tasks.needle_reach:NeedleReach',
    max_episode_steps=50,
)

register(
    id='GauzeRetrieve-v0',
    entry_point='surrol.tasks.gauze_retrieve:GauzeRetrieve',
    max_episode_steps=50,
)

register(
    id='NeedlePick-v0',
    entry_point='surrol.tasks.needle_pick:NeedlePick',
    max_episode_steps=50,
)

register(
    id='PegTransfer-v0',
    entry_point='surrol.tasks.peg_transfer:PegTransfer',
    max_episode_steps=50,
)

# Bimanual PSM Env
register(
    id='NeedleRegrasp-v0',
    entry_point='surrol.tasks.needle_regrasp_bimanual:NeedleRegrasp',
    max_episode_steps=50,
)

register(
    id='BiPegTransfer-v0',
    entry_point='surrol.tasks.peg_transfer_bimanual:BiPegTransfer',
    max_episode_steps=50,
)

# ECM Env
register(
    id='ECMReach-v0',
    entry_point='surrol.tasks.ecm_reach:ECMReach',
    max_episode_steps=50,
)

register(
    id='MisOrient-v0',
    entry_point='surrol.tasks.ecm_misorient:MisOrient',
    max_episode_steps=50,
)

register(
    id='StaticTrack-v0',
    entry_point='surrol.tasks.ecm_static_track:StaticTrack',
    max_episode_steps=50,
)

register(
    id='ActiveTrack-v0',
    entry_point='surrol.tasks.ecm_active_track:ActiveTrack',
    max_episode_steps=500,
)

# RL Env

register(
    id='NeedleReachRL-v0',
    entry_point='surrol.tasks.needle_reach_RL:NeedleReach',
    max_episode_steps=50,
)

register(
    id='GauzeRetrieveRL-v0',
    entry_point='surrol.tasks.gauze_retrieve_RL_2:GauzeRetrieve',
    max_episode_steps=50,
)

register(
    id='NeedleRegraspRL-v0',
    entry_point='surrol.tasks.needle_regrasp_bimanual_RL_2:NeedleRegrasp',
    max_episode_steps=50,
)

register(
    id='BiPegTransferRL-v0',
    entry_point='surrol.tasks.peg_transfer_bimanual_RL:BiPegTransfer',
    max_episode_steps=100,
)

register(
    id='NeedlePickRL-v0',
    entry_point='surrol.tasks.needle_pick_RL_2:NeedlePickRL',
    max_episode_steps=50,
)

register(
    id='PegTransferRL-v0',
    entry_point='surrol.tasks.peg_transfer_RL:PegTransferRL',
    max_episode_steps=50,
)

register(
    id='BiPegBoardRL-v0',
    entry_point='surrol.tasks.peg_board_bimanual_RL:BiPegBoard',
    max_episode_steps=100,
)

register(
    id='MatchBoardPanelRL-v0',
    entry_point='surrol.tasks.match_board_panel_RL:MatchBoardPanel',
    max_episode_steps=100,
)

register(
    id='PickAndPlaceRL-v0',
    entry_point='surrol.tasks.pick_and_place_RL:PickAndPlace',
    max_episode_steps=100,
)

register(
    id='MatchBoardRL-v0',
    entry_point='surrol.tasks.match_board_RL:MatchBoard',
    max_episode_steps=100,
)


register(
    id='PegTransferDataRL-v0',
    entry_point='surrol.tasks.peg_transfer_RL_data_collect:PegTransferRL',
    max_episode_steps=50,
)

register(
    id='PickAndPlaceNewDataRL-v0',
    entry_point='surrol.tasks.pick_and_place_RL_new_data_collect:PickAndPlace',
    max_episode_steps=100,
)

register(
    id='NeedlePickDataRL-v0',
    entry_point='surrol.tasks.needle_pick_RL_data_collect:NeedlePickRL',
    max_episode_steps=50,
)

register(
    id='PickAndPlaceNewRL-v0',
    entry_point='surrol.tasks.pick_and_place_RL_new:PickAndPlace',
    max_episode_steps=100,
)