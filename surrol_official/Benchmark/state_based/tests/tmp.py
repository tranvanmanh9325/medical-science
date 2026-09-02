from surrol.tasks.peg_transfer_RL import PegTransferRL
env = PegTransferRL
parameter = {'render_mode': 'human', 'cid': 0}
env = env('human', 0)