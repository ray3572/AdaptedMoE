from config.mvtec_SimpleNet_config import SimpleNetConfig

from config.mvtec_AdaptedMoE_config import AdaptedMoEConfig
from models.AdaptedMoE.m_AdaptedMoE_trainer import AdaptedMoETrainer
from models.simplenet.m_simplenet_trainer import SimplenetTrainer

if __name__ == "__main__":
    config = SimpleNetConfig()
    config.data_path = 'mvtec_format/'
    config.datasets = "1 2 3 4"
    trainer = SimplenetTrainer(config)
    trainer.pred_and_save(ckpt_path=r"ckpt_best.pth",
                          dst_path="simplenet/")
