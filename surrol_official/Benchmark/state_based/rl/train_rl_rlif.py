import hydra
from trainers.rl_trainer import RLIFRLTrainer

@hydra.main(version_base=None, config_path="./configs", config_name="train")
def main(cfg):
    exp = RLIFRLTrainer(cfg)
    exp.train()

if __name__ == "__main__":
    main()