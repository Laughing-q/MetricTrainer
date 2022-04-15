from metric_trainer.core import Trainer
from omegaconf import OmegaConf


cfg = OmegaConf.load('configs/test.yaml')


trainer = Trainer(cfg)

trainer.train()
