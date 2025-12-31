import pickle


class DefaultConfig():
    def __init__(self):
        self.results_path = "results"
        self.gpus = [0]
        self.random_seed = 666
        self.log_group = "group"
        self.log_project = "project"
        self.run_name = "test"
        self.test = False
        self.save_segmentation_images = False

        self.backbone_name = "wideresnet50"
        self.layers_to_extract_from = []
        self.pretrain_embed_dimension = 1024
        self.target_embed_dimension = 1024
        self.patchsize = 3
        self.pre_proj = 0
        self.embedding_size = 1024
        self.meta_epochs = 1
        self.aed_meta_epochs = 1
        self.gan_epochs = 1
        self.dsc_layers = 2
        self.dsc_hidden = None
        self.noise_std = 0.05
        self.dsc_margin = 0.8
        self.dsc_lr = 0.0002
        self.auto_noise = 0
        self.train_backbone = False
        self.cos_lr = True
        self.proj_layer_type = 0
        self.mix_noise = 1

        self.data_path = "/data_raid/mvtec_anomaly_detection/"
        self.subdatasets = ""
        self.train_val_split = 1
        self.batch_size = 1
        self.num_workers = 16
        self.resize = 256
        self.imagesize = 224
        self.rotate_degrees = 0
        self.translate = 0
        self.scale = 0.0
        self.brightness = 0.0
        self.contrast = 0.0
        self.saturation = 0.0
        self.gray = 0.0
        self.hflip = 0.0
        self.vflip = 0.0
        self.augment = True

    def write_to_txt(self, save_path):
        with open(save_path, 'w') as f:
            for k, v in self.__dict__.items():
                f.write(f"{k}={v}\n")

    def write_to_pkl(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load_from_pkl(self, load_path):
        with open(load_path, 'rb') as f:
            self.__dict__ = pickle.load(f)


if __name__ == "__main__":
    config = DefaultConfig()
    # config.write_to_txt("config.txt")
    print(config.__dict__)
