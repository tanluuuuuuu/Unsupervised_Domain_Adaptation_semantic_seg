# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Modification of config and checkpoint to support legacy models
# - Add inference mode and HRDA output flag

import argparse
import os
from PIL import Image

import mmseg
import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
import numpy as np
import zipfile
from tqdm import tqdm
import cv2

CLASSES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

PALETTE = [
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]

id_to_trainid = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}
trainid_to_id = {v: k for (k, v) in id_to_trainid.items()}

CITYSCAPES_COLORS = {k: v for (k, v) in zip(CLASSES, PALETTE)}

def save_prediction_as_png(output_dir, img_metas, pred_result):
    """Save the prediction as a PNG image."""
    for i, img_meta in enumerate(img_metas):
        img_name = img_meta["ori_filename"]
        pred = pred_result
        new_pred = pred.copy()
        for id1, id2 in enumerate(trainid_to_id.items()):
            print("OLD id: ", id1)
            new_pred[pred == id1] = id2
            print("NEW id: ", id2)

        # Extract the directory from the image name (e.g., berlin)
        city_name = img_name.split("/")[0]

        # Prepare the output directory path
        city_output_dir = os.path.join(output_dir, city_name)

        # Create the city-specific directory if it doesn't exist
        os.makedirs(city_output_dir, exist_ok=True)

        # Prepare the output file path
        submission_img_name = img_name.replace("leftImg8bit.png", "_pred.png")
        submission_img_path = os.path.join(output_dir, submission_img_name)

        # Save the color-encoded prediction as PNG
        cv2.imwrite(submission_img_path, new_pred)


def update_legacy_cfg(cfg):
    # The saved json config does not differentiate between list and tuple
    cfg.data.test.pipeline[1]["img_scale"] = tuple(
        cfg.data.test.pipeline[1]["img_scale"]
    )
    cfg.data.val.pipeline[1]["img_scale"] = tuple(cfg.data.val.pipeline[1]["img_scale"])
    # Support legacy checkpoints
    if cfg.model.decode_head.type == "UniHead":
        cfg.model.decode_head.type = "DAFormerHead"
        cfg.model.decode_head.decoder_params.fusion_cfg.pop("fusion", None)
    if cfg.model.type == "MultiResEncoderDecoder":
        cfg.model.type = "HRDAEncoderDecoder"
    if cfg.model.decode_head.type == "MultiResAttentionWrapper":
        cfg.model.decode_head.type = "HRDAHead"
    cfg.model.backbone.pop("ema_drop_path_rate", None)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="mmseg test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--output-dir", help="output directory")
    parser.add_argument("--zip-name", help="file zip name")
    parser.add_argument(
        "--aug-test", action="store_true", help="Use Flip and Multi scale aug"
    )
    parser.add_argument(
        "--inference-mode",
        choices=[
            "same",
            "whole",
            "slide",
        ],
        default="same",
        help="Inference mode.",
    )
    parser.add_argument(
        "--test-set", action="store_true", help="Run inference on the test set"
    )
    parser.add_argument(
        "--hrda-out",
        choices=["", "LR", "HR", "ATT"],
        default="",
        help="Extract LR and HR predictions from HRDA architecture.",
    )
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument(
        "--show-dir", help="directory where painted images will be saved"
    )
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu_collect is not specified",
    )
    parser.add_argument(
        "--options", nargs="+", action=DictAction, help="custom options"
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.5,
        help="Opacity of painted segmentation map. In (0, 1] range.",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg = update_legacy_cfg(cfg)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.inference_mode == "same":
        # Use pre-defined inference mode
        pass
    elif args.inference_mode == "whole":
        print("Force whole inference.")
        cfg.model.test_cfg.mode = "whole"
    elif args.inference_mode == "slide":
        print("Force slide inference.")
        cfg.model.test_cfg.mode = "slide"
        crsize = cfg.data.train.get("sync_crop_size", cfg.crop_size)
        cfg.model.test_cfg.crop_size = crsize
        cfg.model.test_cfg.stride = [int(e / 2) for e in crsize]
        cfg.model.test_cfg.batched_slide = True
    else:
        raise NotImplementedError(args.inference_mode)

    if args.hrda_out == "LR":
        cfg["model"]["decode_head"]["fixed_attention"] = 0.0
    elif args.hrda_out == "HR":
        cfg["model"]["decode_head"]["fixed_attention"] = 1.0
    elif args.hrda_out == "ATT":
        cfg["model"]["decode_head"]["debug_output_attention"] = True
    elif args.hrda_out == "":
        pass
    else:
        raise NotImplementedError(args.hrda_out)

    if args.test_set:
        for k in cfg.data.test:
            if isinstance(cfg.data.test[k], str):
                print("OLD: ", cfg.data.test[k])
                cfg.data.test[k] = cfg.data.test[k].replace("val", "test")
                print("NEW: ", cfg.data.test[k])

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        map_location="cpu",
        revise_keys=[(r"^module\.", ""), ("model.", "")],
    )
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if "PALETTE" in checkpoint.get("meta", {}):
        model.PALETTE = checkpoint["meta"]["PALETTE"]
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get("efficient_test", False)
    print("efficient_test: ", efficient_test)

    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(
        model, data_loader, args.show, args.show_dir, efficient_test, args.opacity
    )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for i, data in tqdm(enumerate(data_loader)):
        img_metas = data["img_metas"][0].data[0]
        save_prediction_as_png(output_dir, img_metas, outputs[i])

    # Zip the output files for submission
    with zipfile.ZipFile(args.zip_name, "w") as submission_zip:
        for class_name in os.listdir(output_dir):
            class_path = os.path.join(output_dir, class_name)
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                submission_zip.write(img_path, img_file)

    print(f"Submission file created: {args.zip_name}")


if __name__ == "__main__":
    main()
