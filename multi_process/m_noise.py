import copy

import torch


def noise_update_process(config, device, noise_shape_batchcount, noise_update_queue, pid):
    shape_count = copy.deepcopy(noise_shape_batchcount)
    print("start update noise")
    while 1:
        for shape in shape_count.keys():
            if shape_count[shape] < config.min_update_noise_num_once:
                shape_count[shape] = config.min_update_noise_num_once
            if shape_count[shape] > config.max_update_noise_num_once:
                shape_count[shape] = config.max_update_noise_num_once

            for i in range(shape_count[shape]):
                noise_idxs = torch.randint(0, config.mix_noise, torch.Size([shape[0]]))
                noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=config.mix_noise).to(
                    "cpu")  # (N, K)
                noise = torch.stack([
                    torch.normal(0, config.noise_std * 1.1 ** (k), shape)
                    for k in range(config.mix_noise)], dim=1).to("cpu")  # (N, K, C)
                noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)

                noise_update_queue.put([shape, noise.detach().to("cpu")])
    print("finish noise_update_process process %d" % pid)
