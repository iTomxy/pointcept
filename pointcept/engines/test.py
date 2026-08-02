"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import json
from uuid import uuid4
import os
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data

from .defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger, get_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
    vis_confusion_matrix,
    to_dict, calc_stat, nanmean, quiet_nan, to_jsonable, np_smallest_dtype,
)
from pointcept.utils.eval_cm import *

try:
    import pointops
except:
    pointops = None


TESTERS = Registry("testers")


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose and model is None:
            # if model is not none, trigger tester with trainer, no need to print config
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight, weights_only=False)
            self.ckpt_epoch = checkpoint.get("epoch", -1)
            print("ckpt epoch:", self.ckpt_epoch)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@TESTERS.register_module()
class SemSegTester(TesterBase):
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        # create submit folder only on main process
        if (
            self.cfg.data.test.type == "ScanNetDataset"
            or self.cfg.data.test.type == "ScanNet200Dataset"
            or self.cfg.data.test.type == "ScanNetPPDataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
            self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            if os.path.isfile(pred_save_path):
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred = np.load(pred_save_path)
                if "origin_segment" in data_dict.keys():
                    segment = data_dict["origin_segment"]
            else:
                pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
                for i in range(len(fragment_list)):
                    fragment_batch_size = 1
                    s_i, e_i = i * fragment_batch_size, min(
                        (i + 1) * fragment_batch_size, len(fragment_list)
                    )
                    input_dict = collate_fn(fragment_list[s_i:e_i])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]
                    with torch.no_grad():
                        pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                        pred_part = F.softmax(pred_part, -1)
                        if self.cfg.empty_cache:
                            torch.cuda.empty_cache()
                        bs = 0
                        for be in input_dict["offset"]:
                            pred[idx_part[bs:be], :] += pred_part[bs:be]
                            bs = be

                    logger.info(
                        "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                            idx + 1,
                            len(self.test_loader),
                            data_name=data_name,
                            batch_idx=i,
                            batch_num=len(fragment_list),
                        )
                    )
                if self.cfg.data.test.type == "ScanNetPPDataset":
                    pred = pred.topk(3, dim=1)[1].data.cpu().numpy()
                else:
                    pred = pred.max(1)[1].data.cpu().numpy()
                if "origin_segment" in data_dict.keys():
                    assert "inverse" in data_dict.keys()
                    pred = pred[data_dict["inverse"]]
                    segment = data_dict["origin_segment"]
                np.save(pred_save_path, pred)
            if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
            ):
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                    fmt="%d",
                )
            elif self.cfg.data.test.type == "ScanNetPPDataset":
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    pred.astype(np.int32),
                    delimiter=",",
                    fmt="%d",
                )
                pred = pred[:, 0]  # for mIoU, TODO: support top3 mIoU
            elif self.cfg.data.test.type == "SemanticKITTIDataset":
                # 00_000000 -> 00, 000000
                sequence_name, frame_name = data_name.split("_")
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                submit = pred.astype(np.uint32)
                submit = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(submit).astype(np.uint32)
                submit.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "NuScenesDataset":
                np.array(pred + 1).astype(np.uint8).tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "lidarseg",
                        "test",
                        "{}_lidarseg.bin".format(data_name),
                    )
                )

            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - start)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class DINOSemSegTester(TesterBase):
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        # create submit folder only on main process
        if (
            self.cfg.data.test.type == "ScanNetDataset"
            or self.cfg.data.test.type == "ScanNet200Dataset"
            or self.cfg.data.test.type == "ScanNetPPDataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
            self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            dino_coord = data_dict.pop("dino_coord").cuda(non_blocking=True)
            dino_feat = data_dict.pop("dino_feat").cuda(non_blocking=True)
            dino_offset = data_dict.pop("dino_offset").cuda(non_blocking=True)
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            if os.path.isfile(pred_save_path):
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred = np.load(pred_save_path)
                if "origin_segment" in data_dict.keys():
                    segment = data_dict["origin_segment"]
            else:
                pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
                for i in range(len(fragment_list)):
                    fragment_batch_size = 1
                    s_i, e_i = i * fragment_batch_size, min(
                        (i + 1) * fragment_batch_size, len(fragment_list)
                    )
                    input_dict = collate_fn(fragment_list[s_i:e_i])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    input_dict["dino_coord"] = dino_coord
                    input_dict["dino_feat"] = dino_feat
                    input_dict["dino_offset"] = dino_offset
                    idx_part = input_dict["index"]
                    with torch.no_grad():
                        pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                        pred_part = F.softmax(pred_part, -1)
                        if self.cfg.empty_cache:
                            torch.cuda.empty_cache()
                        bs = 0
                        for be in input_dict["offset"]:
                            pred[idx_part[bs:be], :] += pred_part[bs:be]
                            bs = be

                    logger.info(
                        "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                            idx + 1,
                            len(self.test_loader),
                            data_name=data_name,
                            batch_idx=i,
                            batch_num=len(fragment_list),
                        )
                    )
                if self.cfg.data.test.type == "ScanNetPPDataset":
                    pred = pred.topk(3, dim=1)[1].data.cpu().numpy()
                else:
                    pred = pred.max(1)[1].data.cpu().numpy()
                if "origin_segment" in data_dict.keys():
                    assert "inverse" in data_dict.keys()
                    pred = pred[data_dict["inverse"]]
                    segment = data_dict["origin_segment"]
                np.save(pred_save_path, pred)
            if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
            ):
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                    fmt="%d",
                )
            elif self.cfg.data.test.type == "ScanNetPPDataset":
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    pred.astype(np.int32),
                    delimiter=",",
                    fmt="%d",
                )
                pred = pred[:, 0]  # for mIoU, TODO: support top3 mIoU
            elif self.cfg.data.test.type == "SemanticKITTIDataset":
                # 00_000000 -> 00, 000000
                sequence_name, frame_name = data_name.split("_")
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                submit = pred.astype(np.uint32)
                submit = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(submit).astype(np.uint32)
                submit.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "NuScenesDataset":
                np.array(pred + 1).astype(np.uint8).tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "lidarseg",
                        "test",
                        "{}_lidarseg.bin".format(data_name),
                    )
                )

            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class ClsTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        for i, input_dict in enumerate(self.test_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            end = time.time()
            with torch.no_grad():
                output_dict = self.model(input_dict)
            output = output_dict["cls_logits"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred, label, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            intersection_meter.update(intersection), union_meter.update(
                union
            ), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)

            logger.info(
                "Test: [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {accuracy:.4f} ".format(
                    i + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    accuracy=accuracy,
                )
            )

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )

        for i in range(self.cfg.data.num_classes):
            logger.info(
                "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=accuracy_class[i],
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class ClsVotingTester(TesterBase):
    def __init__(
        self,
        num_repeat=100,
        metric="allAcc",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_repeat = num_repeat
        self.metric = metric
        self.best_idx = 0
        self.best_record = None
        self.best_metric = 0

    def test(self):
        for i in range(self.num_repeat):
            logger = get_root_logger()
            logger.info(f">>>>>>>>>>>>>>>> Start Evaluation {i + 1} >>>>>>>>>>>>>>>>")
            record = self.test_once()
            if comm.is_main_process():
                if record[self.metric] > self.best_metric:
                    self.best_record = record
                    self.best_idx = i
                    self.best_metric = record[self.metric]
                info = f"Current best record is Evaluation {i + 1}: "
                for m in self.best_record.keys():
                    info += f"{m}: {self.best_record[m]:.4f} "
                logger.info(info)

    def test_once(self):
        logger = get_root_logger()
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        target_meter = AverageMeter()
        record = {}
        self.model.eval()

        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            voting_list = data_dict.pop("voting_list")
            category = data_dict.pop("category")
            data_name = data_dict.pop("name")
            # pred = torch.zeros([1, self.cfg.data.num_classes]).cuda()
            # for i in range(len(voting_list)):
            #     input_dict = voting_list[i]
            #     for key in input_dict.keys():
            #         if isinstance(input_dict[key], torch.Tensor):
            #             input_dict[key] = input_dict[key].cuda(non_blocking=True)
            #     with torch.no_grad():
            #         pred += F.softmax(self.model(input_dict)["cls_logits"], -1)
            input_dict = collate_fn(voting_list)
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                pred = F.softmax(self.model(input_dict)["cls_logits"], -1).sum(
                    0, keepdim=True
                )
            pred = pred.max(1)[1].cpu().numpy()
            intersection, union, target = intersection_and_union(
                pred, category, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            target_meter.update(target)
            record[data_name] = dict(intersection=intersection, target=target)
            acc = sum(intersection) / (sum(target) + 1e-10)
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))
            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) ".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)
            accuracy_class = intersection / (target + 1e-10)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info("Val result: mAcc/allAcc {:.4f}/{:.4f}".format(mAcc, allAcc))
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        accuracy=accuracy_class[i],
                    )
                )
            return dict(mAcc=mAcc, allAcc=allAcc)

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class PartSegTester(TesterBase):
    def test(self):
        test_dataset = self.test_loader.dataset
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()

        num_categories = len(self.test_loader.dataset.categories)
        iou_category, iou_count = np.zeros(num_categories), np.zeros(num_categories)
        self.model.eval()

        save_path = os.path.join(
            self.cfg.save_path, "result", "test_epoch{}".format(self.cfg.test_epoch)
        )
        make_dirs(save_path)

        for idx in range(len(test_dataset)):
            end = time.time()
            data_name = test_dataset.get_data_name(idx)

            data_dict_list, label = test_dataset[idx]
            pred = torch.zeros((label.size, self.cfg.data.num_classes)).cuda()
            batch_num = int(np.ceil(len(data_dict_list) / self.cfg.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * self.cfg.batch_size_test, min(
                    (i + 1) * self.cfg.batch_size_test, len(data_dict_list)
                )
                input_dict = collate_fn(data_dict_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = self.model(input_dict)["cls_logits"]
                    pred_part = F.softmax(pred_part, -1)
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()
                pred_part = pred_part.reshape(-1, label.size, self.cfg.data.num_classes)
                pred = pred + pred_part.total(dim=0)
                logger.info(
                    "Test: {} {}/{}, Batch: {batch_idx}/{batch_num}".format(
                        data_name,
                        idx + 1,
                        len(test_dataset),
                        batch_idx=i,
                        batch_num=batch_num,
                    )
                )
            pred = pred.max(1)[1].data.cpu().numpy()

            category_index = data_dict_list[0]["cls_token"]
            category = self.test_loader.dataset.categories[category_index]
            parts_idx = self.test_loader.dataset.category2part[category]
            parts_iou = np.zeros(len(parts_idx))
            for j, part in enumerate(parts_idx):
                if (np.sum(label == part) == 0) and (np.sum(pred == part) == 0):
                    parts_iou[j] = 1.0
                else:
                    i = (label == part) & (pred == part)
                    u = (label == part) | (pred == part)
                    parts_iou[j] = np.sum(i) / (np.sum(u) + 1e-10)
            iou_category[category_index] += parts_iou.mean()
            iou_count[category_index] += 1

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} "
                "({batch_time.avg:.3f}) ".format(
                    data_name, idx + 1, len(self.test_loader), batch_time=batch_time
                )
            )

        ins_mIoU = iou_category.sum() / (iou_count.sum() + 1e-10)
        cat_mIoU = (iou_category / (iou_count + 1e-10)).mean()
        logger.info(
            "Val result: ins.mIoU/cat.mIoU {:.4f}/{:.4f}.".format(ins_mIoU, cat_mIoU)
        )
        for i in range(num_categories):
            logger.info(
                "Class_{idx}-{name} Result: iou_cat/num_sample {iou_cat:.4f}/{iou_count:.4f}".format(
                    idx=i,
                    name=self.test_loader.dataset.categories[i],
                    iou_cat=iou_category[i] / (iou_count[i] + 1e-10),
                    iou_count=int(iou_count[i]),
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class InsSegTester(TesterBase):
    def __init__(
        self,
        segment_ignore_index,
        instance_ignore_index,
        **kwargs,
    ):
        self.save_pred = kwargs.pop("save_pred", False)
        super().__init__(**kwargs)
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.valid_class_names = [
            self.cfg.data.names[i]
            for i in range(self.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start InsSegTester Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()

        self.model.eval()
        scenes = []

        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()
            data_name = data_dict.pop("name")
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.model(data_dict)
                segment = data_dict["origin_segment"]
                instance = data_dict["origin_instance"]

                if "origin_coord" in data_dict.keys():
                    # (7 Oct 2025, iTom) BUG: Since `GridSample` uses `index_operator`,
                    # which re-indexes, thus re-orders, points, this 1-nn (closest)
                    # point matching seems aiming at recovering the point order.
                    # However, since `coord` is distorted in data augmentation,
                    # while `origin_coord` is NOT, they are NOT in the same coordinate
                    # space anymore, so this matching does NOT work as expected.
                    reverse, _ = pointops.knn_query(
                        1,
                        data_dict["coord"].float(),
                        data_dict["offset"].int(),
                        data_dict["origin_coord"].float(),
                        data_dict["origin_offset"].int(),
                    )
                    reverse = reverse.cpu().flatten().long()
                    output_dict["pred_masks"] = output_dict["pred_masks"][:, reverse]
                    segment = data_dict["origin_segment"]
                    instance = data_dict["origin_instance"]

                gt_instances, pred_instance = self.associate_instances(
                    output_dict, segment, instance
                )

            scenes.append(dict(gt=gt_instances, pred=pred_instance))
            batch_time.update(time.time() - start)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) ".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                )
            )
            if self.cfg.data.test.type == "ScanNetPPDataset":
                self.write_scannetpp_results(
                    output_dict["pred_scores"],
                    output_dict["pred_masks"],
                    output_dict["pred_classes"],
                    data_name,
                )

            if self.save_pred:
                save_dir = os.path.join(self.cfg.save_path, "result")
                os.makedirs(save_dir, exist_ok=True)
                save_dict = {k: v.cpu().numpy() for k, v in output_dict.items() if isinstance(v, torch.Tensor)}
                save_dict.update({k: v.cpu().numpy() for k, v in data_dict.items() if isinstance(v, torch.Tensor)})
                save_dict.update({"gt_instances": gt_instances, "pred_instance": pred_instance})
                np.savez_compressed(os.path.join(save_dir, "InsSeg-{}.npz".format(data_name)), **save_dict)

        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)
        scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
        ap_scores = self.evaluate_matches(scenes)
        all_ap = ap_scores["all_ap"]
        all_ap_50 = ap_scores["all_ap_50%"]
        all_ap_25 = ap_scores["all_ap_25%"]
        logger.info(
            "Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
                all_ap, all_ap_50, all_ap_25
            )
        )
        for i, label_name in enumerate(self.valid_class_names):
            ap = ap_scores["classes"][label_name]["ap"]
            ap_50 = ap_scores["classes"][label_name]["ap50%"]
            ap_25 = ap_scores["classes"][label_name]["ap25%"]
            logger.info(
                "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                    idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End InsSegTester Evaluation <<<<<<<<<<<<<<<<<")

    def write_scannetpp_results(
        self,
        pred_scores,
        pred_masks,
        pred_classes,
        data_name,
    ):
        pred_scores[pred_scores < 0] = 0
        pred_scores[pred_scores >= 0] = 1

        save_dir = os.path.join(self.cfg.save_path, "result", "submit")
        mask_dir = os.path.join(save_dir, "predicted_masks")
        make_dirs(mask_dir)

        result_path = os.path.join(save_dir, f"{data_name}.txt")
        result_file = open(result_path, "w")
        for i, (score, mask, cls) in enumerate(
            zip(
                pred_scores.cpu().numpy(),
                pred_masks.cpu().numpy(),
                pred_classes.cpu().numpy(),
            )
        ):
            mask = mask.astype(np.uint8)
            length = mask.shape[0]
            mask = np.concatenate([[0], mask, [0]])
            runs = np.where(mask[1:] != mask[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            counts = " ".join(str(x) for x in runs)
            rle = dict(length=length, counts=counts)

            mask_path = os.path.join(mask_dir, f"{data_name}_{i:03d}.json")
            relative_path = os.path.join("predicted_masks", f"{data_name}_{i:03d}.json")
            with open(mask_path, "w") as mask_file:
                json.dump(rle, mask_file, indent=2)
            result_file.write(f"{relative_path} {cls} {score:.3f}\n")
        result_file.close()

    def associate_instances(self, pred, segment, instance):
        """find matched gt & pred (i.e. with positive intersection)"""
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        void_mask = np.in1d(segment, self.segment_ignore_index)

        assert (
            pred["pred_classes"].shape[0]
            == pred["pred_scores"].shape[0]
            == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict() # {class_name: []}
        for i in range(self.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                gt_instances[self.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                    float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
        return ap_scores

    @staticmethod
    def collate_fn(batch):
        # Restrict to bs 1
        return batch[0]


@TESTERS.register_module()
class Ribsegv2VolumeTester(TesterBase):
    """(1 Aug 2026, iTom) volume-wise test for grid-subsampled RibSeg v2.

    One forward pass per volume: the model sees the grid-subsampled cloud
    (~1e5 points), and `GridSample`'s `inverse` scatters that prediction back
    over all ~2e6 original points, so every point is scored at full resolution.
    This needs the test pipeline to `Copy` the label and voxel index into
    `origin_segment` / `origin_voxel_index` before `GridSample` and to collect
    `inverse`; see `Ribsegv2VolumeDataset`.

    Every metric is aggregated ACROSS VOLUMES, never across points or batches:
    each volume contributes one value per class, so a 3e6-point scan and a
    1e6-point scan weigh the same. Classes absent from a volume stay NaN and
    drop out of that volume's average rather than counting as zero.
    """

    DEFAULT_METRICS = ("dice", "iou", "precision", "recall", "specificity", "accuracy")
    N_RIB_CLASSES = 1 + 24 # background + rib1..rib24, what the rib-specific metrics assume

    def __init__(self, cfg, save_pred=False, metrics=None, save_cm=True, rib_metrics=None,
                 model=None, test_loader=None, verbose=False):
        """
        save_pred: bool = False, dump full-resolution predictions as {save_path}/result/{vid}.npz
        metrics: List[str] = None, subset of SemSegEvaluator.METRICS; None -> DEFAULT_METRICS
        save_cm: bool = True, dump & plot the aggregated [C, C] confusion matrix
        rib_metrics: bool = None, also report the rib-index metrics (shift rates,
            label accuracy, off-by-one confusion). They hard-code the 25-class
            rib set, so None auto-enables them only for that class count -- leave
            it alone unless you know better. Stage-1 rib vs. non-rib has nothing
            to shift between, and the class-wise lists would be the wrong length.
        """
        self.save_pred = save_pred
        self.metrics = tuple(metrics) if metrics else self.DEFAULT_METRICS
        self.save_cm = save_cm
        slow = [m for m in self.metrics if m in SemSegEvaluator.DISTANCE_BASED]
        assert not slow, (
            "{} are distance-based: they need the prediction reconstructed as a 3D voxel "
            "grid, not a point list, and are intractable at ~2e6 points/volume. Run them "
            "offline from a save_pred=True dump instead.".format(list(slow))
        )
        self.num_classes = cfg.data.num_classes
        self.bg_class = cfg.data.bg_class
        self.rib_metrics = (
            self.num_classes == self.N_RIB_CLASSES if rib_metrics is None else rib_metrics
        )
        assert not (self.rib_metrics and self.num_classes != self.N_RIB_CLASSES), (
            "rib_metrics needs exactly {} classes, got {}".format(self.N_RIB_CLASSES, self.num_classes)
        )
        super().__init__(cfg, model=model, test_loader=test_loader, verbose=verbose)

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            # pads the volume list to a multiple of world size, so a few volumes
            # are evaluated twice; records are keyed by volume id and deduped.
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, shuffle=False
            )
        else:
            test_sampler = None
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,  # one whole volume per item
            shuffle=False,
            # each worker holds a full ~2e6-point volume, so keep this small
            num_workers=min(2, self.cfg.num_worker_per_gpu),
            pin_memory=False,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        return batch[0]

    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Ribsegv2VolumeTester Evaluation >>>>>>>>>>>>>>>>")
        self.model.eval()
        n_cls = self.num_classes
        if self.save_pred:
            save_path = os.path.join(self.cfg.save_path, "result")
            make_dirs(save_path)

        comm.synchronize()
        records = {}  # volume id -> per-volume metrics
        conf_mats = {}  # volume id -> [n_cls, n_cls]; keyed so duplicates dedupe
        batch_time = AverageMeter()
        end = time.time()
        for i, data_dict in enumerate(self.test_loader):
            name = data_dict.pop("name")
            # full-resolution fields: keep them on CPU, they are ~2e6 long
            inverse = data_dict.pop("inverse").numpy()
            label = data_dict.pop("origin_segment").numpy()
            voxel_index = data_dict.pop("origin_voxel_index").numpy()
            data_dict.pop("segment", None)  # grid-level label; drop it to take the test branch
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)

            with torch.no_grad():
                seg_logits = self.model(data_dict)["seg_logits"]  # [n_grid, n_cls]
            pred_grid = seg_logits.max(1)[1].cpu().numpy()
            pred = pred_grid[inverse]  # [n_orig], back to full resolution
            assert pred.shape == label.shape, \
                "volume {}: pred {} vs label {}".format(name, pred.shape, label.shape)

            records[name] = self.eval_volume(pred, label, n_grid=pred_grid.shape[0])
            conf_mats[name] = np.bincount(
                label.astype(np.int64) * n_cls + pred.astype(np.int64),
                minlength=n_cls * n_cls,
            ).reshape(n_cls, n_cls)
            if self.save_pred:
                np.savez_compressed(
                    os.path.join(save_path, "{}.npz".format(name)),
                    pred=np_smallest_dtype(pred),
                    label=np_smallest_dtype(label),
                    voxel_index=np_smallest_dtype(voxel_index),
                )
            batch_time.update(time.time() - end)
            end = time.time()
            logger.info(
                "Test: [{}/{}] volume {} {} -> {} points {:.3f}s".format(
                    i + 1, len(self.test_loader), name,
                    pred_grid.shape[0], pred.shape[0], batch_time.val,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        gathered = comm.gather(dict(records=records, conf_mats=conf_mats), dst=0)
        if not comm.is_main_process():
            return

        records, conf_mats = {}, {}
        for g in gathered:
            # DistributedSampler pads the volume list, so a few volumes come
            # back twice; keying by volume id drops the duplicates
            records.update(g["records"])
            conf_mats.update(g["conf_mats"])
        conf_mat = np.sum(list(conf_mats.values()), axis=0)

        summary = self.reduce(records, conf_mat)
        prefix = os.path.join(
            self.cfg.save_path, "{}-{}".format(self.cfg.data.test.split, self.__class__.__name__)
        )
        with open(prefix + ".json", 'w') as f:
            json.dump(to_jsonable({
                "time": time.asctime(time.gmtime()),
                "epoch": getattr(self, "ckpt_epoch", -1),
                "n_volumes": len(records),
                "metrics": summary,
                "args": to_dict(self.cfg),
            }), f, indent=1)
        with open(prefix + "-per_volume.json", 'w') as f:
            json.dump(to_jsonable(records), f, indent=1)

        if self.save_cm:
            np.save(prefix + "-confusion_matrix.npy", conf_mat)
            row_norm = conf_mat / np.clip(conf_mat.sum(1, keepdims=True), 1, None)
            try:
                vis_confusion_matrix(
                    np.round(row_norm, 2), list(self.cfg.data.names), prefix + "-confusion_matrix.png",
                    title="Row-normalised confusion matrix ({} volumes)".format(len(records)),
                )
            except Exception as e:
                logger.warning("Failed to plot the confusion matrix: {}".format(e))

        for k, v in summary.items():
            logger.info("{}: {}".format(k, v))
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    def eval_volume(self, pred, label, n_grid):
        """all metrics of a single volume, at full resolution
        Args:
            pred: int numpy.ndarray[n_points], point-wise predicted class id
            label: same shape as `pred`, ground-truth class id
            n_grid: int, how many points the model actually saw
        Returns:
            dict, `<metric>_cw` holds per-class values (NaN where the class is
            absent from this volume) and the rest are volume-level scalars.
        """
        # a throw-away evaluator per volume: its `records[metr][c]` is a 1-element
        # list, which keeps NaN intact. `reduce()` would turn NaN into 0.0 here.
        evaluator = SemSegEvaluator(
            n_classes=self.num_classes, bg_classes=[self.bg_class], select=self.metrics
        )
        with quiet_nan():
            evaluator(pred=pred, y=label)
            row = {
                "{}_cw".format(m): [float(evaluator.records[m][c][0]) for c in range(self.num_classes)]
                for m in evaluator.metrics
            }
            row["n_points"] = int(label.size)
            row["n_grid_points"] = int(n_grid)
            row.update(error_rate(pred, label, self.bg_class))
            integrity = compute_prediction_integrity(pred, label)
            row.update({k: integrity[k] for k in ("mean_integrity", "weighted_mean_integrity", "global_integrity")})
            if self.rib_metrics:
                row.update(error_rate_rib(pred, label, self.bg_class))
                row.update(rib_label_acc(pred, label))
                # left/right and class-wise breakdown of the rib-id shift
                shift = rib_id_shift_rate(pred, label)
                row.update({k: shift[k] for k in ("nsr_left", "nsr_right", "fsr_left", "fsr_right", "nsr_cw", "fsr_cw")})
        return row

    def reduce(self, records, conf_mat):
        """average every per-volume record across volumes"""
        n_cls = self.num_classes
        order = sorted(records, key=int)
        fg = [c for c in range(n_cls) if c != self.bg_class]

        # class-wise overlap metrics, via SemSegEvaluator's own aggregation path
        evaluator = SemSegEvaluator(
            n_classes=n_cls, bg_classes=[self.bg_class], select=self.metrics
        )
        for name in order:
            evaluator.load_from_dict(records[name])
        with quiet_nan():
            summary = evaluator.reduce()

        # `reduce()` averages over every class including background, which the
        # background dominates. Redo the overall number over foreground only,
        # still averaging per volume first.
        for m in self.metrics:
            cls_by_vol = np.asarray(
                [[records[n]["{}_cw".format(m)][c] for n in order] for c in fg], dtype=float
            )  # [n_fg_classes, n_volumes]
            per_volume = nanmean(cls_by_vol, axis=0)
            summary["{}_fg".format(m)] = nanmean(per_volume)
            summary["{}_fg_vol".format(m)] = calc_stat(per_volume, percentages=[5, 25, 50, 75, 95], prec=4)

        # volume-level scalars: error rates, label accuracy, integrity, sizes
        for k, v in records[order[0]].items():
            if k.endswith("_cw") or not isinstance(v, (int, float)):
                continue
            vals = np.asarray([records[n].get(k, np.nan) for n in order], dtype=float)
            summary[k] = nanmean(vals)
            summary[k + "_vol"] = calc_stat(vals, percentages=[5, 25, 50, 75, 95], prec=4)

        if not self.rib_metrics:
            return summary

        # Is the residual rib-vs-rib confusion concentrated on adjacent ribs?
        # A dominant off-by-one band means the model cannot count ribs from the
        # top, i.e. it is missing global context rather than local shape.
        rib = np.arange(1, n_cls)
        fg_cm = conf_mat[1:, 1:].astype(np.float64)
        offset = np.abs(rib[:, None] - rib[None, :])
        cross_side = ((rib[:, None] == 12) & (rib[None, :] == 13)) | \
                     ((rib[:, None] == 13) & (rib[None, :] == 12))
        wrong = fg_cm.sum() - np.trace(fg_cm)
        summary["fg_confusion"] = {
            "n_wrong_rib_points": int(wrong),
            "off_by_1_share": float(fg_cm[(offset == 1) & ~cross_side].sum() / max(wrong, 1.0)),
            "off_by_2_share": float(fg_cm[(offset == 2) & ~cross_side].sum() / max(wrong, 1.0)),
            "cross_side_share": float(fg_cm[cross_side].sum() / max(wrong, 1.0)),
        }
        return summary
