"""Exercise the real volume-tester loop with CPU tensors and a fake predictor.

Load only this class from its source so unrelated GPU tester imports do not
prevent a test of scatter, distributed deduplication, and report persistence.
"""
import ast
import copy
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pointcept.utils.eval_cm import RIBSEG_DEFAULT_METRICS, eval_volume, reduce_records
from pointcept.utils.misc import AverageMeter, natural_sort_key, to_jsonable


def test_scatter_dedup_and_jsonl_confusion_persistence(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "pointcept/engines/test.py"
    node = next(item for item in ast.parse(path.read_text()).body
                if isinstance(item, ast.ClassDef)
                and item.name == "Ribsegv2VolumeTester")
    node.decorator_list = []
    ast_module = ast.Module(body=[node], type_ignores=[])

    class Logger:

        def __init__(self, log_file=None):
            self.path = log_file

        def info(self, message):
            if self.path:
                with open(self.path, "a") as stream:
                    stream.write(message + "\n")

        warning = info

    comm = SimpleNamespace(
        synchronize=lambda: None,
        is_main_process=lambda: True,
        gather=lambda result, dst: [result, copy.deepcopy(result)])
    namespace = dict(TesterBase=object,
                     RIBSEG_DEFAULT_METRICS=RIBSEG_DEFAULT_METRICS,
                     np=np,
                     torch=torch,
                     os=os,
                     time=time,
                     json=json,
                     comm=comm,
                     AverageMeter=AverageMeter,
                     natural_sort_key=natural_sort_key,
                     eval_volume=eval_volume,
                     reduce_records=reduce_records,
                     to_jsonable=to_jsonable,
                     to_dict=lambda cfg: {},
                     get_root_logger=lambda: Logger(),
                     get_logger=lambda name, log_file, fmt: Logger(log_file),
                     vis_confusion_matrix=lambda *args, **kwargs: None)
    exec(compile(ast_module, str(path), "exec"), namespace)
    tester = namespace["Ribsegv2VolumeTester"].__new__(
        namespace["Ribsegv2VolumeTester"])
    tester.num_classes, tester.bg_class, tester.rib_metrics = 25, 0, True
    tester.metrics, tester.save_pred, tester.save_cm = ("dice", ), False, True
    tester.cfg = SimpleNamespace(save_path=str(tmp_path),
                                 data=SimpleNamespace(
                                     test=SimpleNamespace(split="val"),
                                     names=[str(i) for i in range(25)]))

    class Predictor(torch.nn.Module):

        def forward(self, data):
            assert "segment" not in data  # tester must take inference branch
            logits = torch.full((3, 25), -10.)
            logits[torch.arange(3), torch.tensor([1, 2, 0])] = 10.
            return {"seg_logits": logits}

    tester.model = Predictor()
    tester.test_loader = [
        dict(name="one",
             inverse=torch.tensor([0, 1, 1, 2]),
             origin_segment=torch.tensor([1, 1, 1, 2]),
             origin_voxel_index=torch.zeros(4, 3, dtype=torch.long),
             segment=torch.tensor([1, 1, 2]),
             coord=torch.zeros(3, 3),
             offset=torch.tensor([3]))
    ]
    monkeypatch.setattr(torch.Tensor, "cuda", lambda value, **kwargs: value)
    tester.test()
    prefix = tmp_path / "val-Ribsegv2VolumeTester"
    summary = json.loads(prefix.with_suffix(".json").read_text())
    rows = [
        json.loads(line)
        for line in Path(str(prefix) +
                         "-per_volume.jsonl").read_text().splitlines()
    ]
    assert summary["n_volumes"] == 1
    assert len(rows) == 2 and "time" in rows[0]
    record = rows[1]["metrics"]
    cm = np.asarray(record["confusion"])
    assert cm.sum() == 4  # original resolution, and duplicate rank not counted
    assert cm[1, 1] == 1 and cm[1, 2] == 2 and cm[2, 0] == 1
    assert record["fg_purity_cw"][1] == 2 / 3
    assert record["fg_recall_cw"][1] == 1 / 3
    assert record["fg_purity_cw"][2] is None
    assert summary["metrics"]["fg_scored_pair_count"] == 1
    np.testing.assert_array_equal(
        cm, np.load(str(prefix) + "-confusion_matrix.npy"))
