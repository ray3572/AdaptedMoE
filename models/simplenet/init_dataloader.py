from dataloader.mvtec import MVTecDataset, DatasetSplit


def init_dataloader(config):
    train_dataset = MVTecDataset(source=config.data_path,
                                 classname=config.datasets,
                                 resize=config.resize,
                                 imagesize=config.imagesize,
                                 split=DatasetSplit.TRAIN,
                                 train_val_split=config.train_val_split,
                                 rotate_degrees=config.rotate_degrees,
                                 translate=config.translate,
                                 brightness_factor=config.brightness,
                                 contrast_factor=config.contrast,
                                 saturation_factor=config.saturation,
                                 gray_p=config.gray,
                                 h_flip_p=config.hflip,
                                 v_flip_p=config.vflip,
                                 scale=config.scale)

    test_dataset = MVTecDataset(source=config.data_path,
                                classname=config.datasets,
                                resize=config.resize,
                                imagesize=config.imagesize,
                                split=DatasetSplit.TEST)
    return train_dataset, test_dataset
