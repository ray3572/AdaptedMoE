import logging
import os
import random as rd
import time
from collections import OrderedDict

import torch
import torch.multiprocessing as mp
import tqdm
from models.simplenet.m_embedding import embedding_process

from models.simplenet.init_dataloader import init_dataloader
from models.simplenet.init_simplenet import init_simplenet
from multi_process.m_dataloader import load_epoch_data_process
from multi_process.m_noise import noise_update_process

LOGGER = logging.getLogger(__name__)
torch.multiprocessing.set_start_method('spawn', force=True)


class SimplenetTrainer():
    def __init__(self, config):
        self.config = config
        if len(config.gpus) == 0:
            self.device = torch.device("cpu")
        elif len(config.gpus) == 1:
            self.device = torch.device("cuda:" + str(config.gpus[0]))
        else:
            gpu_str = ""
            for gpu in config.gpus:
                gpu_str += str(gpu) + ","
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str[:-1]
            self.device = torch.device("cuda")

        self.simplenet = init_simplenet(config, self.device)

        self.start_signal_queue = mp.Queue(maxsize=1)
        self.data_queue = mp.Queue(maxsize=16)  # better to larger than one epoch
        self.embedding_feature_queue = mp.Queue(maxsize=2)

        self.noise_shape_batchcount = {}  # {shape:count}
        self.noise_cache = {}

        self.noise_update_queue = mp.Queue(maxsize=config.max_cached_update_noise_num)
        self.noise_refresh_ind = {}

        self.p_noise_update = None

        self.train_dataset, self.test_dataset = init_dataloader(config)

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
            pin_memory=True,
        )

        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
            pin_memory=True,
        )

    def update_noise_cache(self):
        avaliable_noise_num = self.noise_update_queue.qsize()
        print("[update_noise_dict] avaliable_update_noise_num:", avaliable_noise_num)
        print("[update_noise_dict] noise_refresh_ind:", self.noise_refresh_ind)
        print("[update_noise_dict] current noise cache:")
        for shape in self.noise_cache:
            print(shape, len(self.noise_cache[shape]))

        for i in range(avaliable_noise_num):
            shape, noise = self.noise_update_queue.get()
            if len(self.noise_cache[shape]) < self.config.max_cached_noise_num:  # rolling updates
                self.noise_cache[shape].append(noise)
            else:
                refresh_ind = self.noise_refresh_ind[shape]
                self.noise_cache[shape][refresh_ind] = noise
                self.noise_refresh_ind[shape] += 1
                if self.noise_refresh_ind[shape] == self.config.max_cached_noise_num:
                    self.noise_refresh_ind[shape] = 0
        if not self.p_noise_update:
            # (config,device,noise_shape,noise_update_queue,pid)
            self.p_noise_update = mp.Process(target=noise_update_process,
                                             args=(
                                                 self.config, self.device, self.noise_shape_batchcount,
                                                 self.noise_update_queue, 0))
            self.p_noise_update.start()

    def train_discriminator(self, num_batch, num_samples, warm_up=0):
        """Computes and sets the support features for SPADE."""
        _ = self.simplenet.forward_modules.eval()
        if self.simplenet.pre_proj > 0:
            self.simplenet.pre_projection.train()
        self.simplenet.discriminator.train()

        i_iter = 0
        LOGGER.info(f"Training discriminator...")
        if warm_up:
            print(f"warm_up epoch, maybe a bit slow")
        with tqdm.tqdm(total=self.simplenet.gan_epochs) as pbar:
            for i_epoch in range(self.simplenet.gan_epochs):
                all_loss = []
                all_p_true = []
                all_p_fake = []
                all_p_interp = []
                embeddings_list = []

                for batch_id in range(num_batch):
                    embedding_feature = self.embedding_feature_queue.get()
                    self.simplenet.dsc_opt.zero_grad()
                    if self.simplenet.pre_proj > 0:
                        self.simplenet.proj_opt.zero_grad()

                    i_iter += 1

                    if self.simplenet.pre_proj > 0:
                        true_feats = self.simplenet.pre_projection(embedding_feature[0].to(self.device))
                    else:
                        true_feats = embedding_feature[0].to(self.device)

                    if warm_up:
                        if true_feats.shape in self.noise_shape_batchcount:
                            self.noise_shape_batchcount[true_feats.shape] += 1
                        else:
                            self.noise_shape_batchcount[true_feats.shape] = 1
                            self.noise_cache[true_feats.shape] = []
                            self.noise_refresh_ind[true_feats.shape] = 0

                        noise_idxs = torch.randint(0, self.simplenet.mix_noise, torch.Size([true_feats.shape[0]]))
                        noise_one_hot = torch.nn.functional.one_hot(noise_idxs,
                                                                    num_classes=self.simplenet.mix_noise).to(
                            self.device)  # (N, K)
                        noise = torch.stack([
                            torch.normal(0, self.simplenet.noise_std * 1.1 ** (k), true_feats.shape)
                            for k in range(self.simplenet.mix_noise)], dim=1).to(self.device)  # (N, K, C)
                        noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
                        self.noise_cache[true_feats.shape].append(noise.to('cpu').detach())

                    else:
                        ind = rd.randint(0, len(self.noise_cache[true_feats.shape]) - 1)
                        noise = self.noise_cache[true_feats.shape][ind].to(self.device)

                    fake_feats = true_feats + noise

                    scores = self.simplenet.discriminator(torch.cat([true_feats, fake_feats]))
                    true_scores = scores[:len(true_feats)]
                    fake_scores = scores[len(fake_feats):]

                    th = self.simplenet.dsc_margin
                    p_true = (true_scores.detach() >= th).sum() / len(true_scores)
                    p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
                    true_loss = torch.clip(-true_scores + th, min=0)
                    fake_loss = torch.clip(fake_scores + th, min=0)

                    self.simplenet.logger.logger.add_scalar(f"p_true", p_true, self.simplenet.logger.g_iter)
                    self.simplenet.logger.logger.add_scalar(f"p_fake", p_fake, self.simplenet.logger.g_iter)

                    loss = true_loss.mean() + fake_loss.mean()

                    self.simplenet.logger.logger.add_scalar("loss", loss, self.simplenet.logger.g_iter)
                    self.simplenet.logger.step()

                    loss.backward()
                    if self.simplenet.pre_proj > 0:
                        self.simplenet.proj_opt.step()
                    if self.simplenet.train_backbone:
                        self.simplenet.backbone_opt.step()
                    self.simplenet.dsc_opt.step()

                    loss = loss.detach().cpu()
                    all_loss.append(loss.item())
                    all_p_true.append(p_true.cpu().item())
                    all_p_fake.append(p_fake.cpu().item())

                if len(embeddings_list) > 0:
                    self.simplenet.auto_noise[1] = torch.cat(embeddings_list).std(0).mean(-1)

                if self.simplenet.cos_lr:
                    self.simplenet.dsc_schl.step()

                all_loss = sum(all_loss) / (num_batch)
                all_p_true = sum(all_p_true) / (num_batch)
                all_p_fake = sum(all_p_fake) / (num_batch)
                cur_lr = self.simplenet.dsc_opt.state_dict()['param_groups'][0]['lr']

                if warm_up:
                    pbar_str = f"warm up epoch:{i_epoch} loss:{round(all_loss, 5)} "
                else:
                    pbar_str = f"epoch:{i_epoch} loss:{round(all_loss, 5)} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar_str += f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)}"
                if len(all_p_interp) > 0:
                    pbar_str += f" p_interp:{round(sum(all_p_interp) / num_samples, 3)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)

    def train(self):
        num_batch_per_epoch = len(self.train_dataset) // self.config.batch_size
        if len(self.train_dataset) % self.config.batch_size > 0:
            num_batch_per_epoch += 1

        p_load_data = mp.Process(target=load_epoch_data_process,
                                 args=(self.config, self.train_dataset, self.data_queue))
        p_load_data.start()

        p_embedding = []
        for pid in range(self.config.num_embedding_process):
            p = mp.Process(target=embedding_process,
                           args=(self.config, self.device, self.data_queue, self.embedding_feature_queue, pid))
            p.daemon = True
            p.start()
            p_embedding.append(p)

        ###########################################
        print('ckpt_dir', self.simplenet.ckpt_dir)

        state_dict = {}
        ckpt_path = os.path.join(self.simplenet.ckpt_dir, "ckpt.pth")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.device)
            if 'discriminator' in state_dict:
                self.simplenet.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    self.simplenet.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)

        ckpt_path = os.path.join(self.simplenet.ckpt_dir, "ckpt.pth")

        def update_state_dict(d):
            state_dict["discriminator"] = OrderedDict({
                k: v.detach().cpu()
                for k, v in self.simplenet.discriminator.state_dict().items()})
            if self.simplenet.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k: v.detach().cpu()
                    for k, v in self.simplenet.pre_projection.state_dict().items()})

        best_record = None

        self.train_discriminator(num_batch=num_batch_per_epoch, num_samples=len(self.train_dataset), warm_up=1)
        self.update_noise_cache()

        print("start train")
        for i_mepoch in range(self.config.meta_epochs):

            st = time.time()
            self.train_discriminator(num_batch=num_batch_per_epoch, num_samples=len(self.train_dataset))
            et = time.time()
            print("epoch time:", et - st)

            if i_mepoch % 2 == 0:
                self.update_noise_cache()
                scores, segmentations, features, labels_gt, masks_gt = self.simplenet.predict(self.test_dataloader)
                auroc, full_pixel_auroc, pro = self.simplenet._evaluate(self.test_dataloader, scores, segmentations,
                                                                        features, labels_gt,
                                                                        masks_gt)
                self.simplenet.logger.logger.add_scalar("i-auroc", auroc, i_mepoch)
                self.simplenet.logger.logger.add_scalar("p-auroc", full_pixel_auroc, i_mepoch)
                self.simplenet.logger.logger.add_scalar("pro", pro, i_mepoch)

                if best_record is None:
                    best_record = [auroc, full_pixel_auroc, pro]
                    update_state_dict(state_dict)
                else:

                    if full_pixel_auroc > best_record[1]:
                        best_record = [auroc, full_pixel_auroc, pro]
                        update_state_dict(state_dict)
                        print('[save] save as ' + ckpt_path.replace('.pth', '_best.pth'))
                        torch.save(state_dict, ckpt_path.replace('.pth', '_best.pth'))
                print(f"----- {i_mepoch} I-AUROC:{round(auroc, 4)}(MAX:{round(best_record[0], 4)})"
                      f"  P-AUROC{round(full_pixel_auroc, 4)}(MAX:{round(best_record[1], 4)}) -----"
                      f"  PRO-AUROC{round(pro, 4)}(MAX:{round(best_record[2], 4)}) -----")
                print('[save] save as ' + ckpt_path.replace('.pth', '_last.pth'))
                update_state_dict(state_dict)
                torch.save(state_dict, ckpt_path.replace('.pth', '_last.pth'))

        torch.save(state_dict, ckpt_path)
        p_load_data.kill()
        self.p_noise_update.kill()
        print(self.data_queue.qsize())

        while not self.data_queue.empty():
            self.data_queue.get()
        print(self.data_queue.qsize())
        self.data_queue.put(None)
        self.data_queue.put(None)

        print("finish")

        return best_record

    def pred_and_save(self, ckpt_path,dst_path):
        state_dict = torch.load(ckpt_path, map_location=self.device)
        if 'discriminator' in state_dict:
            self.simplenet.discriminator.load_state_dict(state_dict['discriminator'])
            if "pre_projection" in state_dict:
                self.simplenet.pre_projection.load_state_dict(state_dict["pre_projection"])
        self.simplenet.pred_and_save(self.test_dataloader,dst_path)
