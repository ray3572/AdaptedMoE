from config.default_config import DefaultConfig


class SimpleNetConfig(DefaultConfig):
    def __init__(self):
        super().__init__()
        self.data_path = ''

        self.datasets = 'bottle'
        # self.datasets=('screw','pill','capsule','carpet','grid',\
        #                'tile', 'wood', 'zipper', 'cable',
        #                'toothbrush', 'transistor', 'metal_nut',
        #                'bottle', 'hazelnut', 'leather')

        self.gpus = [0]
        self.random_seed = 0
        self.log_group = "simplenet_mvtec_wideresnet50_moredata_le1_le2"
        self.log_project = "MVTecAD_Results_cloth_patch"
        self.results_path = "results"
        self.run_name = "run"
        self.save_segmentation_images = True

        self.backbone_name = "wideresnet50"
        self.layers_to_extract_from = ["layer1", "layer2"]
        self.pretrain_embed_dimension = 768
        self.target_embed_dimension = 768
        self.patchsize = 3
        self.patchstride = 1
        self.meta_epochs = 30
        self.embedding_size = 256
        self.gan_epochs = 4
        self.noise_std = 0.015
        self.dsc_hidden = 512
        self.dsc_layers = 2
        self.dsc_margin = 0.5
        self.pre_proj = 1

        self.batch_size = 2
        self.brightness = 0.3
        self.hflip = 0.5
        self.vflip = 0.5
        self.resize = 256
        self.num_workers = 2
        self.imagesize = 256

        self.num_pretrain_feature_process = 1
        self.num_embedding_process = 1

        self.max_cached_noise_num = 30  # cached noise mat num of each shape

        self.min_update_noise_num_once = 2
        self.max_update_noise_num_once = 20
        self.max_cached_update_noise_num = 30  # nums of all the shape of noise mat could be cached in memory
