import logging
import os
import random as rd
import time
from collections import OrderedDict

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import tqdm

from models.AdaptedMoE.init_dataloader import init_dataloader
from models.AdaptedMoE.init_AdaptedMoE import init_AdaptedMoE
from models.AdaptedMoE.m_embedding import embedding_process
from multi_process.m_dataloader import load_epoch_data_process
from multi_process.m_noise import noise_update_process
from utils.logger import Logger

LOGGER = logging.getLogger(__name__)
torch.multiprocessing.set_start_method('spawn', force=True)


class AdaptedMoETrainer():
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

        self.adapted_moe = init_AdaptedMoE(config, self.device)

        self.start_signal_queue = mp.Queue(maxsize=1)
        self.data_queue = mp.Queue(maxsize=self.config.cached_databatch)  # better to larger than one epoch
        self.embedding_feature_queue = mp.Queue(maxsize=4)

        self.noise_shape_batchcount = {}  # {shape:count}
        self.noise_cache = {}
        self.noise_update_queue = mp.Queue(maxsize=config.max_cached_update_noise_num)
        self.noise_refresh_ind = {}

        self.p_noise_update = None

        self.train_dataset, self.test_dataset = init_dataloader(config)

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
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
        # print("noise_shape_batchcount",self.noise_shape_batchcount)
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

    def train_discriminator(self, num_batch, num_samples,epoch_ind, warm_up=0):
        """Computes and sets the support features for SPADE."""
        _ = self.adapted_moe.forward_modules.eval()
        if self.adapted_moe.pre_proj > 0:
            self.adapted_moe.pre_projection.train()
        # self.adapted_moe.discriminator.train()
        for i in range(self.config.num_expert):
            self.adapted_moe.experts[i]["model"].train()
        self.adapted_moe.moe.train()

        i_iter = 0
        LOGGER.info(f"Training discriminator...")
        if warm_up:
            print(f"warm_up epoch, maybe a bit slow")
        with (tqdm.tqdm(total=self.adapted_moe.gan_epochs) as pbar):
            for i_epoch in range(self.adapted_moe.gan_epochs):
                update_center=False
                if i_epoch==(self.adapted_moe.gan_epochs-1) and self.config.moe_center_shift:
                    update_center = True
                    expert_center_mat_temp=torch.zeros(self.config.num_expert, self.config.target_embed_dimension).to('cpu')
                    expert_std_mat_temp = torch.zeros(self.config.num_expert, self.config.target_embed_dimension).to('cpu')
                    expert_embedding_list=[]
                    for i in range(self.config.num_expert):
                        expert_embedding_list.append([])


                all_route = []
                all_loss = []
                all_p_true = []
                all_p_fake = []
                all_p_interp = []
                embeddings_list = []
                correct = 0
                feature_num = 0

                num_feature_embedding_persample = 0

                merge_loss = None


                # self.adapted_moe.dsc_opt.zero_grad()
                self.adapted_moe.moe_opt.zero_grad()
                if self.adapted_moe.pre_proj > 0:
                    self.adapted_moe.proj_opt.zero_grad()
                for i in range(self.config.num_expert):
                    self.adapted_moe.experts[i]["opt"].zero_grad()

                for batch_id in range(num_batch):
                    target_loss = {}
                    target_count = {}
                    embedding_feature = self.embedding_feature_queue.get()
                    num_feature_embedding_persample = max(embedding_feature[0].shape[0] // self.config.batch_size,
                                                          num_feature_embedding_persample)

                    i_iter += 1

                    if self.adapted_moe.pre_proj > 0:
                        true_feats = self.adapted_moe.pre_projection(embedding_feature[0].to(self.device),is_norm=False)
                    else:
                        true_feats = embedding_feature[0].to(self.device)

                    if warm_up:
                        if true_feats.shape in self.noise_shape_batchcount:
                            self.noise_shape_batchcount[true_feats.shape] += 1
                        else:
                            self.noise_shape_batchcount[true_feats.shape] = 1
                            self.noise_cache[true_feats.shape] = []
                            self.noise_refresh_ind[true_feats.shape] = 0

                        noise_idxs = torch.randint(0, self.adapted_moe.mix_noise, torch.Size([true_feats.shape[0]]))
                        noise_one_hot = torch.nn.functional.one_hot(noise_idxs,
                                                                    num_classes=self.adapted_moe.mix_noise).to(
                            self.device)  # (N, K)
                        noise = torch.stack([
                            torch.normal(0, self.adapted_moe.noise_std * 1.1 ** (k), true_feats.shape)
                            for k in range(self.adapted_moe.mix_noise)], dim=1).to(self.device)  # (N, K, C)
                        noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
                        self.noise_cache[true_feats.shape].append(noise.to('cpu').detach())

                    else:
                        ind = rd.randint(0, len(self.noise_cache[true_feats.shape]) - 1)
                        noise = self.noise_cache[true_feats.shape][ind].to(self.device)

                    route_target = embedding_feature[2]
                    route_target_loss = route_target.repeat_interleave(num_feature_embedding_persample).to(self.device)
                    
                    norm_true_feats=F.normalize(true_feats, dim=1)
                    route = self.adapted_moe.moe(norm_true_feats)
                    if self.config.moe_centerloss:
                        route_loss = F.nll_loss(route, route_target_loss)
                        +self.config.moe_centerloss * self.adapted_moe.moe_centerloss(route_target_loss, true_feats)
                    else:
                        route_loss = F.nll_loss(route, route_target_loss)

                    pred = route.argmax(dim=1, keepdims=True)
                    correct += pred.eq(route_target_loss.view_as(pred)).sum().item()
                    feature_num += route_target_loss.shape[0]

                    fake_feats = true_feats + noise

                    if self.config.pre_proj_norm:
                        true_feats = F.normalize(true_feats, dim=1)
                        fake_feats = F.normalize(fake_feats, dim=1)

                    th = self.adapted_moe.dsc_margin
                    p_true = None
                    p_fake = None

                    for t in range(route_target.shape[0]):
                        target_id = route_target[t] # expert id
                        sample_true_feats = true_feats[
                                            t * num_feature_embedding_persample:(t + 1) * num_feature_embedding_persample]
                        sample_fake_feats = fake_feats[
                                            t * num_feature_embedding_persample:(t + 1) * num_feature_embedding_persample]
                        if update_center:
                            feat_sampled_indices = torch.randperm(num_feature_embedding_persample)[:self.config.moe_center_sample_pixel_num_per_img]
                            sampled_elements = sample_true_feats[feat_sampled_indices].detach().cpu()  # 随机采样
                            expert_embedding_list[target_id].append(sampled_elements)
                        


                        scores = self.adapted_moe.experts[target_id]["model"](
                            torch.cat([sample_true_feats, sample_fake_feats]))
                        true_scores = scores[:len(sample_true_feats)]
                        fake_scores = scores[len(sample_fake_feats):]
                        if not p_true:
                            p_true = (true_scores.detach() >= th).sum().type(torch.float32)
                            p_fake = (fake_scores.detach() < -th).sum().type(torch.float32)
                        else:
                            p_true += (true_scores.detach() >= th).sum().type(torch.float32)
                            p_fake += (fake_scores.detach() < -th).sum().type(torch.float32)

                        true_loss = torch.clip(-true_scores + th, min=0).type(torch.float32)
                        fake_loss = torch.clip(fake_scores + th, min=0).type(torch.float32)

                        if target_id not in target_loss:
                            target_loss[target_id] = true_loss.mean() + fake_loss.mean()
                            target_count[target_id] = 1
                        else:

                            target_loss[target_id] += (true_loss.mean() + fake_loss.mean())
                            target_count[target_id] += 1

                    p_true /= true_feats.shape[0]
                    p_fake /= fake_feats.shape[0]

                    self.adapted_moe.logger.logger.add_scalar(f"p_true", p_true, self.adapted_moe.logger.g_iter)
                    self.adapted_moe.logger.logger.add_scalar(f"p_fake", p_fake, self.adapted_moe.logger.g_iter)

                    loss = route_loss.mean()
                    for t in target_loss.keys():
                        # expert_loss = (target_loss[t]/target_count[t])*(target_count[t]/self.config.batch_size)
                        expert_loss = (target_loss[t] / self.config.batch_size)
                        loss += expert_loss
                    # loss = true_loss.mean() + fake_loss.mean() + route_loss.mean()

                    self.adapted_moe.logger.logger.add_scalar("loss", loss, self.adapted_moe.logger.g_iter)
                    self.adapted_moe.logger.step()

                    if merge_loss == None:
                        merge_loss = loss
                        # print('merge_loss', type(merge_loss))
                    else:
                        merge_loss += loss.item()
                    if batch_id % self.config.accumulate_batch == 0:
                        # print('merge_loss',type(merge_loss))
                        merge_loss /= (self.config.accumulate_batch*1.0)
                        # print('merge_loss', type(merge_loss))
                        merge_loss.backward()
                        merge_loss = None
                        if self.adapted_moe.pre_proj > 0:
                            self.adapted_moe.proj_opt.step()
                            self.adapted_moe.proj_opt.zero_grad()
                        if self.adapted_moe.train_backbone:
                            self.adapted_moe.backbone_opt.step()

                        # self.adapted_moe.dsc_opt.step()
                        self.adapted_moe.moe_opt.step()
                        self.adapted_moe.moe_opt.zero_grad()
                        for i in range(self.config.num_expert):
                            self.adapted_moe.experts[i]["opt"].step()
                            self.adapted_moe.experts[i]["opt"].zero_grad()

                        if self.adapted_moe.cos_lr:
                            # self.adapted_moe.dsc_schl.step()
                            for t in target_loss.keys():
                                self.adapted_moe.experts[t]["schl"].step()
                            self.adapted_moe.moe_schl.step()

                    loss_tmp = loss.detach().cpu()
                    all_loss.append(loss_tmp.item())
                    all_route.append(route_loss.detach().cpu().item())
                    all_p_true.append(p_true.cpu().item())
                    all_p_fake.append(p_fake.cpu().item())
                if update_center:
                    for i in range(self.config.num_expert):
                        feats=torch.concatenate(expert_embedding_list[i], dim=0)
                        mean=feats.mean(0)
                        std=feats.std(0)
                        self.adapted_moe.expert_center_matrix[i]=mean.to(self.device)
                        self.adapted_moe.expert_std_matrix[i]=std.to(self.device)
                    print("updated center")

            if len(embeddings_list) > 0:
                self.adapted_moe.auto_noise[1] = torch.cat(embeddings_list).std(0).mean(-1)

            all_loss = sum(all_loss) / (num_batch)
            all_p_true = sum(all_p_true) / (num_batch)
            all_p_fake = sum(all_p_fake) / (num_batch)
            all_route = sum(all_route) / (num_batch)
            cur_lr = self.adapted_moe.dsc_opt.state_dict()['param_groups'][0]['lr']

            if warm_up:
                pbar_str = f"warm up epoch:{i_epoch} loss:{round(all_loss, 5)} "
            else:
                pbar_str = f"epoch:{i_epoch} loss:{round(all_loss, 5)} "
            pbar_str += f"lr:{round(cur_lr, 6)}"
            pbar_str += (f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)} "
                         f"route:{round(all_route, 3)} acc:{round(correct / feature_num, 3)} ")
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
        print('ckpt_dir', self.adapted_moe.ckpt_dir)

        state_dict = {}
        ckpt_path = os.path.join(self.adapted_moe.ckpt_dir, "ckpt.pth")
        config_path=os.path.join(self.adapted_moe.ckpt_dir, "config.txt")
        self.logger=Logger(os.path.join(self.adapted_moe.ckpt_dir, "precision_logger.txt"))
        self.config.write_to_txt(config_path)
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.device)
            if 'discriminator' in state_dict:
                self.adapted_moe.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    self.adapted_moe.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)

        ckpt_path = os.path.join(self.adapted_moe.ckpt_dir, "ckpt.pth")

        def update_state_dict(d):
            state_dict["discriminator"] = OrderedDict({
                k: v.detach().cpu()
                for k, v in self.adapted_moe.discriminator.state_dict().items()})

            for i in range(self.config.num_expert):
                state_dict[f"experts.{i}.model"] = OrderedDict({
                    k: v.detach().cpu()
                    for k, v in self.adapted_moe.experts[i]["model"].state_dict().items()})

            state_dict["moe"] = OrderedDict({
                k: v.detach().cpu()
                for k, v in self.adapted_moe.moe.state_dict().items()})
            if self.adapted_moe.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k: v.detach().cpu()
                    for k, v in self.adapted_moe.pre_projection.state_dict().items()})
            if self.config.moe_center_shift:
                state_dict["expert_center_matrix"]=self.adapted_moe.expert_center_matrix.detach().cpu()
                state_dict["expert_std_matrix"] = self.adapted_moe.expert_std_matrix.detach().cpu()

        best_record = None

        self.adapted_moe.gan_epochs = 1
        st = time.time()
        self.train_discriminator(num_batch=num_batch_per_epoch, num_samples=len(self.train_dataset),epoch_ind=-1, warm_up=1)
        et = time.time()
        ###
        scores, segmentations, features, labels_gt, masks_gt = self.adapted_moe.predict(self.test_dataloader)
        auroc, full_pixel_auroc, pro = self.adapted_moe._evaluate(self.test_dataloader, scores, segmentations,
                                                                 features, labels_gt,
                                                                 masks_gt)
        self.logger.add_log("epoch %d i-auroc: %.4f p-auroc: %.4f pro: %.4f"%(-1, auroc, full_pixel_auroc, pro),is_show=True)
        ###

        print("before optimize epoch time cost:", (et - st) * self.config.gan_epochs)
        self.adapted_moe.gan_epochs = self.config.gan_epochs

        self.update_noise_cache()

        print("start train")
        for i_mepoch in range(self.config.meta_epochs):

            st = time.time()
            self.train_discriminator(num_batch=num_batch_per_epoch, num_samples=len(self.train_dataset),epoch_ind=i_mepoch, warm_up=0)
            et = time.time()
            print("epoch time:", et - st)

            if i_mepoch % 4 == 0:
                self.update_noise_cache()
                st = time.time()
                scores, segmentations, features, labels_gt, masks_gt = self.adapted_moe.predict(self.test_dataloader)
                auroc, full_pixel_auroc, pro = self.adapted_moe._evaluate(self.test_dataloader, scores, segmentations,
                                                                         features, labels_gt,
                                                                         masks_gt)
                self.adapted_moe.logger.logger.add_scalar("i-auroc", auroc, i_mepoch)
                self.adapted_moe.logger.logger.add_scalar("p-auroc", full_pixel_auroc, i_mepoch)
                self.adapted_moe.logger.logger.add_scalar("pro", pro, i_mepoch)

                et = time.time()
                print("eval time cost:", et - st)
                update_state_dict(state_dict)

                if best_record is None:
                    best_record = [auroc, full_pixel_auroc, pro]
                else:

                    if full_pixel_auroc > best_record[1]:
                        best_record = [auroc, full_pixel_auroc, pro]
                        print('[save] save as ' + ckpt_path.replace('.pth', '_best.pth'))
                        torch.save(state_dict, ckpt_path.replace('.pth', '_best.pth'))

                self.logger.add_log(
                      f"----- {i_mepoch} I-AUROC:{round(auroc, 4)}(MAX:{round(best_record[0], 4)})"
                      f"  P-AUROC{round(full_pixel_auroc, 4)}(MAX:{round(best_record[1], 4)}) -----"
                      f"  PRO-AUROC{round(pro, 4)}(MAX:{round(best_record[2], 4)}) -----",is_show=True)

                print('[save] save as ' + ckpt_path.replace('.pth', '_last.pth'))
                torch.save(state_dict, ckpt_path.replace('.pth', '_last.pth'))

        torch.save(state_dict, ckpt_path)

        # --- Graceful shutdown of background processes and queues ---
        # 1) Stop the data loading process so it does not keep filling the queue.
        if p_load_data.is_alive():
            p_load_data.kill()
            p_load_data.join()

        # 2) Clear any remaining items in the data queue so that termination
        #    signals can be seen quickly by the embedding workers.
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except Exception:
                break

        # 3) Propagate termination signals to all embedding processes.
        #    Each embedding worker will re-queue the sentinel, but we still
        #    send one per worker to avoid races when reusing the trainer.
        for _ in range(self.config.num_embedding_process):
            self.data_queue.put(None)

        # 4) Give embedding workers a chance to exit.
        for p in p_embedding:
            if p.is_alive():
                p.join(timeout=5.0)

        # 5) Stop the noise update process if it was started.
        if self.p_noise_update is not None and self.p_noise_update.is_alive():
            self.p_noise_update.kill()
            self.p_noise_update.join()

        print("finish")

        return best_record


    def load_state_dict(self,path):
        state_dict = torch.load(path, map_location=self.device)
        for i in range(self.config.num_expert):
            if f"experts.{i}.model" in state_dict:
                self.adapted_moe.experts[i]["model"].load_state_dict(state_dict[f"experts.{i}.model"])
        if "pre_projection" in state_dict:
            self.adapted_moe.pre_projection.load_state_dict(state_dict["pre_projection"])
        if "moe" in state_dict:
            self.adapted_moe.moe.load_state_dict(state_dict["moe"])
        self.adapted_moe.expert_center_matrix = state_dict["expert_center_matrix"].to(self.device)
        self.adapted_moe.expert_std_matrix = state_dict["expert_std_matrix"].to(self.device)

    def pred_and_save(self, ckpt_path,dst_path):
        self.load_state_dict(ckpt_path)

        self.adapted_moe.pred_and_save(self.test_dataloader,dst_path)