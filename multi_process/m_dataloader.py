import torch


def load_epoch_data_process(config, dataset, data_queue):
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True
    )
    while 1:

        for data in train_dataloader:
            data_queue.put(data)
