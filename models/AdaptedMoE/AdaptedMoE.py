import logging
import math
import os
import pickle
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

import utils.metrics as metrics
from models.AdaptedMoE.MoENet import MoENet
from models.AdaptedMoE.discriminator import Discriminator
from models.AdaptedMoE.init_dataloader import init_dataloader
from models.AdaptedMoE.patch_maker import PatchMaker
from models.AdaptedMoE.preprocessing import Preprocessing, Aggregator, RescaleSegmentor
from models.AdaptedMoE.projection import Projection
from models.AdaptedMoE.loss import CenterLoss
from pretrain_backbone.aggregator import (NetworkFeatureAggregator)
from utils.plot import plot_segmentation_images

LOGGER = logging.getLogger(__name__)


class TBWrapper:

    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1


class AdaptedMoE(torch.nn.Module):
    def __init__(self, config, device):
        """anomaly detection class."""
        super(AdaptedMoE, self).__init__()
        self.config = config
        self.device = device
        self.topK=config.topK

        self.train_dataset, self.test_dataset = init_dataloader(config)

        self.start_signal_queue = mp.Queue(maxsize=1)
        self.data_queue = mp.Queue(maxsize=256)  # better to larger than one epoch
        self.embedding_feature_queue = mp.Queue(maxsize=2)

        self.noise_shape_batchcount = {}  # {shape:count}
        self.noise_cache = {}

        self.noise_update_queue = mp.Queue(maxsize=config.max_cached_update_noise_num)
        self.noise_refresh_ind = {}

        self.p_noise_update = None

    def load(
            self,
            backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,  # 1536
            target_embed_dimension,  # 1536
            patchsize=3,  # 3
            patchstride=1,
            embedding_size=None,  # 256
            meta_epochs=1,  # 40
            aed_meta_epochs=1,
            gan_epochs=1,  # 4
            noise_std=0.05,
            mix_noise=1,
            noise_type="GAU",
            dsc_layers=2,  # 2
            dsc_hidden=None,  # 1024
            dsc_margin=.8,  # .5
            dsc_lr=0.0002,
            train_backbone=False,
            auto_noise=0,
            cos_lr=False,
            lr=1e-3,
            pre_proj=0,  # 1
            proj_layer_type=0,
    ):
        pid = os.getpid()

        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_segmentor = RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.embedding_size = embedding_size if embedding_size is not None else self.target_embed_dimension
        self.meta_epochs = meta_epochs
        self.lr = lr
        self.cos_lr = cos_lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)
        # AED
        self.aed_meta_epochs = aed_meta_epochs

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj,
                                             proj_layer_type)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(self.pre_projection.parameters(), lr * .1)

        # Discriminator
        self.auto_noise = [auto_noise, None]
        self.dsc_lr = dsc_lr
        self.gan_epochs = gan_epochs
        self.mix_noise = mix_noise
        self.noise_type = noise_type
        self.noise_std = noise_std



        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.dsc_lr, weight_decay=1e-5)
        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(self.dsc_opt,
                                                                   (meta_epochs - aed_meta_epochs) * gan_epochs,
                                                                   self.dsc_lr * .4)
        self.dsc_margin = dsc_margin

        self.experts=[]
        print("init %d experts" % self.config.num_expert)
        for i in range(self.config.num_expert):
            expert=Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
            expert.to(self.device)
            opt = torch.optim.Adam(expert.parameters(), lr=self.dsc_lr, weight_decay=1e-5)
            schl = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                                  (meta_epochs - aed_meta_epochs) * gan_epochs,
                                                                  self.dsc_lr * .4)
            self.experts.append({"model":expert,"opt":opt,"schl":schl})
        if self.config.pre_proj_norm:
            print("enable pre_proj_norm")
        if self.config.moe_center_shift:
            print("enable moe_center_shift")
            self.expert_center_matrix=torch.zeros(self.config.num_expert,self.config.target_embed_dimension).to(self.device)
            self.expert_std_matrix = torch.zeros(self.config.num_expert, self.config.target_embed_dimension).to(self.device)


        self.moe = MoENet(in_planes=self.target_embed_dimension, n_expert=self.config.num_expert)
        self.moe.to(self.device)
        self.moe_opt = torch.optim.Adam(self.moe.parameters(), lr=self.config.moe_lr, weight_decay=1e-5)
        self.moe_schl = torch.optim.lr_scheduler.CosineAnnealingLR(self.moe_opt,
                                                                   (meta_epochs - aed_meta_epochs) * gan_epochs,
                                                                   self.config.moe_lr * .4)
        if self.config.moe_centerloss:
            self.moe_centerloss=CenterLoss(self.config.num_expert,self.target_embed_dimension).to(self.device)
            print("enable center loss")

        self.model_dir = ""
        self.dataset_name = ""
        self.tau = 1
        self.logger = None

    def set_model_dir(self, model_dir, dataset_name):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir)

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                    input_image = image.to(torch.float).to(self.device)
                with torch.no_grad():
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings for images."""
        B = len(images)
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](
            features)  # pooling each feature to same channel and stack together
        features = self.forward_modules["preadapt_aggregator"](features)  # further pooling

        return features, patch_shapes

    def test(self, training_data, test_data, save_segmentation_images):
        ckpt_path = os.path.join(self.ckpt_dir, "models.ckpt")
        if os.path.exists(ckpt_path):
            state_dicts = torch.load(ckpt_path, map_location=self.device)
            if "pretrained_enc" in state_dicts:
                self.feature_enc.load_state_dict(state_dicts["pretrained_enc"])
            if "pretrained_dec" in state_dicts:
                self.feature_dec.load_state_dict(state_dicts["pretrained_dec"])

        aggregator = {"scores": [], "segmentations": [], "features": []}
        scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
        aggregator["scores"].append(scores)
        aggregator["segmentations"].append(segmentations)
        aggregator["features"].append(features)

        scores = np.array(aggregator["scores"])
        min_scores = scores.min(axis=-1).reshape(-1, 1)
        max_scores = scores.max(axis=-1).reshape(-1, 1)
        scores = (scores - min_scores) / (max_scores - min_scores)
        scores = np.mean(scores, axis=0)

        segmentations = np.array(aggregator["segmentations"])
        min_scores = (
            segmentations.reshape(len(segmentations), -1)
            .min(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        max_scores = (
            segmentations.reshape(len(segmentations), -1)
            .max(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        segmentations = (segmentations - min_scores) / (max_scores - min_scores)
        segmentations = np.mean(segmentations, axis=0)

        anomaly_labels = [
            x[1] != "good" for x in test_data.dataset.data_to_iterate
        ]

        if save_segmentation_images:
            self.save_segmentation_images(test_data, segmentations, scores)

        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, anomaly_labels
        )["auroc"]

        # Compute PRO score & PW Auroc for all images
        pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
            segmentations, masks_gt
        )
        full_pixel_auroc = pixel_scores["auroc"]

        return auroc, full_pixel_auroc

    def _evaluate(self, test_data, scores, segmentations, features, labels_gt, masks_gt,quick_compute=True):
        scores = np.squeeze(np.array(scores))
        img_min_scores = scores.min(axis=-1)
        img_max_scores = scores.max(axis=-1)
        scores = (scores - img_min_scores) / (img_max_scores - img_min_scores)

        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, labels_gt
        )["auroc"]

        if quick_compute:
            quick_shape=segmentations[0].shape
            quick_shape=[quick_shape[0]//2,quick_shape[1]//2]
            segmentations=np.array(segmentations)
            segmentations=np.expand_dims(segmentations,axis=1)
            masks_gt = F.interpolate(
                torch.tensor(masks_gt), size=quick_shape, mode="bilinear"
            ).numpy()

            masks_gt[masks_gt>0]=1

            segmentations=F.interpolate(
                torch.tensor(segmentations), size=quick_shape, mode="nearest"
            ).numpy()


        # st=time.time()
        if len(masks_gt) > 0:
            segmentations = np.array(segmentations)
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            norm_segmentations = np.zeros_like(segmentations)
            for min_score, max_score in zip(min_scores, max_scores):
                norm_segmentations += (segmentations - min_score) / max(max_score - min_score, 1e-2)
            norm_segmentations = norm_segmentations / len(scores)

            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                norm_segmentations, masks_gt)
            # segmentations, masks_gt
            full_pixel_auroc = pixel_scores["auroc"]

            pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)),
                                      np.squeeze(norm_segmentations))

        else:
            full_pixel_auroc = -1
            pro = -1

        return auroc, full_pixel_auroc, pro

    def predict(self, data, prefix=""):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, prefix)
        return self._predict(data)

    def _predict_dataloader(self, dataloader, prefix):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        img_paths = []
        scores = []
        masks = []
        features = []
        labels_gt = []
        masks_gt = []

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask", None) is not None:
                        masks_gt.extend(data["mask"].numpy().tolist())
                    image = data["image"]
                    img_paths.extend(data['image_path'])
                _scores, _masks, _feats = self._predict(image)
                for score, mask, feat, is_anomaly in zip(_scores, _masks, _feats, data["is_anomaly"].numpy().tolist()):
                    scores.append(score)
                    masks.append(mask)
                    # features.append(feat) # it may cause OOM !

        return scores, masks, features, labels_gt, masks_gt

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        image_scores_list=[]
        masks_list=[]
        features_list=[]
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.eval()
        # self.discriminator.eval()
        for i in range(self.config.num_expert):
            self.experts[i]["model"].eval()

        with torch.no_grad():
            input_batch_size=images.shape[0]
            features, patch_shapes = self._embed(images,
                                                 provide_patch_shapes=True,
                                                 evaluation=True)
            if self.pre_proj > 0:
                features = self.pre_projection(features,is_norm=self.config.pre_proj_norm)

            num_feature_embedding_persample=features.shape[0]//input_batch_size
            for i in range(input_batch_size):
                feature_sample=features[i*num_feature_embedding_persample:(i+1)*num_feature_embedding_persample,:]
                routes = self.moe(feature_sample)
                routes_score = routes.mean(dim=0, keepdim=True)
                routes = routes_score.argsort().cpu().numpy()[0]
                routes_score=routes_score.cpu().numpy()[0]
                #print("routes_score",routes_score)
                #print("routes",routes)

                mean_patch_scores=[]
                mean_image_scores=[]
                weight_sum=0
                weights=[]

                for k in range(self.topK):
                    route = routes[-k-1]
                    weight_sum+=routes_score[route]
                for k in range(self.topK):
                    route = routes[-k-1]
                    weights.append(routes_score[route]/weight_sum)

                for k in range(self.topK):
                    route = routes[-k-1]
                    if self.config.moe_center_shift:
                        center = feature_sample.mean(0)
                        std = feature_sample.std(0)
                        expert_center = self.expert_center_matrix[route]
                        expert_std = self.expert_std_matrix[route]
                        feature_sample = feature_sample-center
                        feature_sample = feature_sample*(expert_std / std)
                        feature_sample = feature_sample+expert_center
                        feature_sample = F.normalize(feature_sample, dim=1)
                        
                        #feature_sample = (feature_sample + (expert_center - center)) * (expert_std / std)

                    patch_scores = image_scores = -self.experts[route]["model"](feature_sample)
                    patch_scores = patch_scores.cpu().numpy()
                    image_scores = image_scores.cpu().numpy()
                    #print('patch_scores',patch_scores.shape)
                    #print('image_scores',image_scores.shape)
                    
                    mean_patch_scores.append(weights[k]*patch_scores)
                    mean_image_scores.append(weights[k]*image_scores)
                    
                mean_patch_scores=np.sum(mean_patch_scores,axis=0)
                mean_image_scores=np.sum(mean_image_scores,axis=0)
                #print('mean_patch_scores',mean_patch_scores.shape)
                #print('mean_image_scores',mean_image_scores.shape)
                    


                patch_scores=mean_patch_scores
                image_scores=mean_image_scores



                image_scores = self.patch_maker.unpatch_scores(
                    image_scores, batchsize=1
                )
                image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
                image_scores = self.patch_maker.score(image_scores)

                patch_scores = self.patch_maker.unpatch_scores(
                    patch_scores, batchsize=1
                )
                scales = patch_shapes[0]
                patch_scores = patch_scores.reshape(1, scales[0], scales[1])
                feature_sample = feature_sample.reshape(1, scales[0], scales[1], -1)
                masks, feature_sample = self.anomaly_segmentor.convert_to_segmentation(patch_scores, feature_sample)
                image_scores_list+=list(image_scores)
                masks_list+=list(masks)
                features_list+=list(feature_sample)

        return image_scores_list, masks_list, features_list

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "params.pkl")

    def save_to_path(self, save_path: str, prepend: str = ""):
        LOGGER.info("Saving data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(params, save_file, pickle.HIGHEST_PROTOCOL)

    def save_segmentation_images(self, data, segmentations, scores):
        image_paths = [
            x[2] for x in data.dataset.data_to_iterate
        ]
        mask_paths = [
            x[3] for x in data.dataset.data_to_iterate
        ]

        def image_transform(image):
            in_std = np.array(
                data.dataset.transform_std
            ).reshape(-1, 1, 1)
            in_mean = np.array(
                data.dataset.transform_mean
            ).reshape(-1, 1, 1)
            image = data.dataset.transform_img(image)
            return np.clip(
                (image.numpy() * in_std + in_mean) * 255, 0, 255
            ).astype(np.uint8)

        def mask_transform(mask):
            return data.dataset.transform_mask(mask).numpy()

        plot_segmentation_images(
            './output',
            image_paths,
            segmentations,
            scores,
            mask_paths,
            image_transform=image_transform,
            mask_transform=mask_transform,
        )

    def print_params(self):
        for k in self.__dict__:
            print(k, self.__dict__[k])

    def pred_and_save(self,dataloader,dst_path):
        _ = self.forward_modules.eval()

        img_paths = []
        scores = []
        masks = []
        features = []
        labels_gt = []
        masks_gt = []

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        vis_result_path=dst_path+'/vis_result/'
        if not os.path.exists(vis_result_path):
            os.makedirs(vis_result_path)
        heatmap_path = dst_path + '/heatmap/'
        if not os.path.exists(heatmap_path):
            os.makedirs(heatmap_path)

        def write_result_npy(dst_path,filepath,cls_name,img,pred,mask,img_score):
            batch_size = len(filepath)
            preds = pred  # B x 1 x H x W
            masks = mask  # B x 1 x H x W
            img_score=img_score
            heights = img.shape[2]
            widths = img.shape[3]
            clsnames = cls_name
            save_dir=dst_path
            for i in range(batch_size):
                file_dir, filename = os.path.split(filepath[i])
                # print("file_dir, filename",file_dir, filename)
                _, subname = os.path.split(file_dir)
                filename = "{}_{}_{}".format(clsnames[i], subname, filename)
                filename, _ = os.path.splitext(filename)
                save_file = os.path.join(save_dir, filename + ".npz")
                # print('preds[i]',preds[i].shape)
                np.savez(
                    save_file,
                    filename=filepath[i],
                    pred=preds[i],
                    mask=masks[i],
                    global_score=img_score[i],
                    height=heights,
                    width=widths,
                    clsname=clsnames[i],
                )


        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask", None) is not None:
                        masks_gt.extend(data["mask"].numpy().tolist())
                    image = data["image"]
                    img_paths.extend(data['image_path'])

                    _scores, pred_masks, _feats = self._predict(image)

                    write_result_npy(dst_path, data['image_path'], data["classname"], data["image"], pred_masks, data["mask"], _scores)


        return scores, masks, features, labels_gt, masks_gt
