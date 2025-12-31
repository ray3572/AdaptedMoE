from config.mvtec_SimpleNet_config import SimpleNetConfig

from config.mvtec_AdaptedMoE_config import AdaptedMoEConfig
from models.AdaptedMoE.m_AdaptedMoE_trainer import AdaptedMoETrainer
from models.simplenet.m_simplenet_trainer import SimplenetTrainer

if __name__ == "__main__":
    config = AdaptedMoEConfig()
    config.set_type(config.datasets)
    config.gpus = [1]
    trainer = AdaptedMoETrainer(config)
    trainer.train()
