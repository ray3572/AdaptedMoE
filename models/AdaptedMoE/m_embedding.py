import torch

from models.AdaptedMoE.init_AdaptedMoE import init_AdaptedMoE


def embedding_process(config, device, data_queue, embedding_feature_queue, pid):
    adapted_moe = init_AdaptedMoE(config, device)
    while 1:
        data = data_queue.get()
        if data is None:
            data_queue.put(None)
            embedding_feature_queue.put(None)
            break

        if isinstance(data, str):
            if data == "finish epoch":
                embedding_feature_queue.put("finish epoch")
            continue
        image = data["image"]
        label = data["class_label"]
        input_image = image.to(torch.float).to(device)
        features, patch_shapes = adapted_moe._embed(input_image)
        # embedding_feature_queue.put([features.detach().cpu(), patch_shapes])
        embedding_feature_queue.put([features.detach(), patch_shapes, label])
    print("finish embedding process %d" % pid)
