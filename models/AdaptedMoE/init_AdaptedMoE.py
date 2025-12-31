import os

from models.AdaptedMoE import AdaptedMoE
from pretrain_backbone import load_timm_backbone
from utils.save_load import create_storage_folder


def init_AdaptedMoE(config, device):
    adapted_moe = AdaptedMoE(config, device)
    backbone = load_timm_backbone(config.backbone_name)
    backbone.name, backbone.seed = config.backbone_name, config.random_seed
    backbone = backbone.to(device)

    adapted_moe.load(
        backbone=backbone,
        layers_to_extract_from=config.layers_to_extract_from,
        device=device,
        input_shape=(3, config.imagesize, config.imagesize),
        pretrain_embed_dimension=config.pretrain_embed_dimension,
        target_embed_dimension=config.target_embed_dimension,
        patchsize=config.patchsize,
        embedding_size=config.embedding_size,
        meta_epochs=config.meta_epochs,
        aed_meta_epochs=config.aed_meta_epochs,
        gan_epochs=config.gan_epochs,
        noise_std=config.noise_std,
        dsc_layers=config.dsc_layers,
        dsc_hidden=config.dsc_hidden,
        dsc_margin=config.dsc_margin,
        dsc_lr=config.dsc_lr,
        auto_noise=config.auto_noise,
        train_backbone=config.train_backbone,
        cos_lr=config.cos_lr,
        pre_proj=config.pre_proj,
        proj_layer_type=config.proj_layer_type,
        mix_noise=config.mix_noise,
    )
    run_save_path = create_storage_folder(
        config.results_path, config.log_project, config.log_group, config.run_name, mode="overwrite"
    )

    models_dir = os.path.join(run_save_path, "models")
    i = 0
    adapted_moe.set_model_dir(os.path.join(models_dir, f"{i}"), config.datasets)
    return adapted_moe
