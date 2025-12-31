from config.default_config import DefaultConfig


class AdaptedMoEConfig(DefaultConfig):
    def __init__(self):
        super().__init__()
        self.device_type="pc" # server or pc

        if self.device_type=="server":
            self.data_path = 'mvtec_anomaly_detection/'
        if self.device_type == "pc":
            self.data_path = r'C:\mvtec_anomaly_detection'

        # self.datasets = "all"
        # self.datasets=('screw','pill','capsule','carpet','grid',\
        #                'tile', 'wood', 'zipper', 'cable',
        #                'toothbrush', 'transistor', 'metal_nut',
        #                'bottle', 'hazelnut', 'leather')
        self.datasets = 'bottle screw'

        self.gpus = [0]
        self.random_seed = 0
        self.log_group = "AdaptedMoE_centerloss_mvtec_wideresnet50_moredata_le1_le2"
        self.log_project = "MVTecAD"
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

        if self.device_type=="server":
            self.gan_epochs = 4  # !!!
        if self.device_type=="pc":
            self.gan_epochs = 2  # !!!

        self.noise_std = 0.015
        self.dsc_hidden = 512
        self.dsc_layers = 2
        self.dsc_margin = 0.5
        self.pre_proj = 1
        self.pre_proj_norm=1

        self.moe_hidden = 128
        self.moe_layers = 1
        self.moe_lr = 0.0002
        self.moe_centerloss = 0.1
        self.moe_center_shift = True
        self.moe_center_sample_pixel_num_per_img=100 # sample 100 pixel per imgs while compute center

        self.topK=1

        if self.datasets == "all":
            self.num_expert = 15
        elif " " not in self.datasets:
            self.num_expert = 1
        else:
            self.num_expert = len(self.datasets.split(" "))



        self.brightness = 0.3
        self.hflip = 0.5
        self.vflip = 0.5
        self.resize = 256
        self.num_workers = 2
        self.imagesize = 256



        # !!!
        if self.device_type=="server":
            self.accumulate_batch = 8  # real batch size=accumulate_batch*batch_size
            self.batch_size = 8  # !!!
            self.cached_databatch=128

            self.num_pretrain_feature_process = 1
            self.num_embedding_process = 1

            self.max_cached_noise_num = 300  # cached noise mat num of each shape

            self.min_update_noise_num_once = 20
            self.max_update_noise_num_once = 200
            self.max_cached_update_noise_num = 300  # nums of all the shape of noise mat could be cached in memory

        if self.device_type == "pc":
            self.accumulate_batch = 8  # real batch size=accumulate_batch*batch_size
            self.batch_size = 2  # !!!
            self.cached_databatch = 8

            self.num_pretrain_feature_process = 1
            self.num_embedding_process = 1

            self.max_cached_noise_num = 30  # cached noise mat num of each shape

            self.min_update_noise_num_once = 2
            self.max_update_noise_num_once = 20
            self.max_cached_update_noise_num = 30  # nums of all the shape of noise mat could be cached in memory

