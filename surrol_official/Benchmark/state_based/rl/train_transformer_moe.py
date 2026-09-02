import argparse
from trainers.gcdt_moe_trainer import experiment

def train(variant):
    experiment(variant)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', type=str, default='train')
    parser.add_argument('--env', type=str, default='PegTransferRL-v0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--multi_objective', type=bool, default=False)
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--load_epoch', type=int, default=1)
    parser.add_argument('--action_tanh', type=bool, default=True)


    args = parser.parse_args()

    train(vars(args))

