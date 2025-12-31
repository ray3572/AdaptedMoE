

from config.mvtec_AdaptedMoE_config import AdaptedMoEConfig
from models.AdaptedMoE.m_AdaptedMoE_trainer import AdaptedMoETrainer


if __name__ == "__main__":
    config = AdaptedMoEConfig()
    config.data_path = ''
    config.datasets = "1 2 3 4"
    config.topK=1
    trainer = AdaptedMoETrainer(config)
    trainer.pred_and_save(ckpt_path=r"ckpt_best.pth",
                          dst_path="simplenet/")
