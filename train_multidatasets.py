#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os
import sys
import time
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    # default_setup,
    hooks,
    launch,
)
from detectron2.engine.defaults import _try_get_key, setup_logger, collect_env_info, _highlight, CfgNode
from detectron2.utils.env import seed_all_rng
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.events import (
    CommonMetricPrinter, 
    JSONWriter, 
    TensorboardXWriter
)
from detectron2.checkpoint import DetectionCheckpointer

from groundingdino.util.events import WandbWriter
from groundingdino.util import ema

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from detectron2.data import DatasetCatalog, MetadataCatalog
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import glob
import json

def multi_datasets_setup_logger(args):
    output_dir = args.output_dir
    rank = comm.get_rank()
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)
    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())
    logger.info("Command line arguments: " + str(args))
    return logger

def default_setup(cfg, args, logger):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = _try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        if isinstance(cfg, CfgNode):
            with PathManager.open(path, "w") as f:
                f.write(cfg.dump())
        else:
            LazyConfig.save(cfg, path)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = _try_get_key(
            cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False
        )

    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                _highlight(PathManager.open(args.config_file, "r").read(), args.config_file),
            )
        )

class Trainer(SimpleTrainer):
    """
    We've combine Simple and AMP Trainer together.
    """

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        amp=False,
        clip_grad_params=None,
        grad_scaler=None,
        batch_size_scale=1,
        categories_names=[],
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)

        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        if amp:
            if grad_scaler is None:
                from torch.cuda.amp import GradScaler

                grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler
        
        # set True to use amp training
        self.amp = amp

        # gradient clip hyper-params
        self.clip_grad_params = clip_grad_params
        
        # batch_size_scale
        self.batch_size_scale = batch_size_scale
        
        # 
        self.categories_names = categories_names

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.amp import autocast

        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        with autocast("cuda", enabled=self.amp):
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """

        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())           
            if self.iter % self.batch_size_scale == 0:
                # print(self.iter)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            if self.iter % self.batch_size_scale == 0:
                # print(self.iter)
                self.optimizer.step()
                self.optimizer.zero_grad()

        self._write_metrics(loss_dict, data_time)

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )

    def state_dict(self):
        ret = super().state_dict()
        if self.grad_scaler and self.amp:
            ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.grad_scaler and self.amp:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])

    def after_train(self):
        if isinstance(self.model, DistributedDataParallel):
            if hasattr(self.model.module, "add_cls_prompt"):
                self.model.module.add_cls_prompt(self.categories_names)
                
        else:
            if hasattr(self.model, "add_cls_prompt"):
                self.model.add_cls_prompt(self.categories_names)

        if isinstance(self.model, DistributedDataParallel):
            if hasattr(self.model.module, "after_train"):
                self.model.module.after_train()
        else:
            if hasattr(self.model, "after_train"):
                self.model.after_train()
                
        return super().after_train()
    
    def before_train(self):
        if isinstance(self.model, DistributedDataParallel):
            if hasattr(self.model.module, "before_train"):
                self.model.module.before_train()
        else:
            if hasattr(self.model, "before_train"):
                self.model.before_train()
        return super().before_train()

    # def before_train(self):
    #     if isinstance(self.model, DistributedDataParallel):
    #         if self.model.module.use_prompt_tuning:
    #             self.model.module.add_cls_prompt(self.categories_names)
    #     else:
    #         if self.model.use_prompt_tuning:
    #             self.model.add_cls_prompt(self.categories_names)
    #     return super().before_train()

class MemoryReplayer(Trainer):
    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        # data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        with autocast(enabled=self.amp):
            if isinstance(self.model, DistributedDataParallel):
                loss_dict = self.model.module.replay_memory()
            else:
                loss_dict = self.model.replay_memory()
                
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """

        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())           
            if self.iter % self.batch_size_scale == 0:
                # print(self.iter)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            if self.iter % self.batch_size_scale == 0:
                # print(self.iter)
                self.optimizer.step()
                self.optimizer.zero_grad()

        self._write_metrics(loss_dict, data_time)

    def after_train(self):
        self.storage.iter = self.iter
        for h in self._hooks:
            h.after_train()

class PeriodicCheckpointer(hooks.PeriodicCheckpointer):
    def after_train(self):
        self.step(self.max_iter)
        return super().after_train()

def load_model(model_config_path, model_checkpoint_path):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    # _ = model.eval()
    return model

def do_test(cfg, model, eval_only=False):
    logger = logging.getLogger("detectron2")

    if eval_only:
        logger.info("Run evaluation under eval-only mode")
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            logger.info("Run evaluation with EMA.")
        else:
            logger.info("Run evaluation without EMA.")
        if "evaluator" in cfg.dataloader:
            ret = inference_on_dataset(
                model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
            )
            print_csv_format(ret)
        return ret
    
    logger.info("Run evaluation without EMA.")
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)

        if cfg.train.model_ema.enabled:
            logger.info("Run evaluation with EMA.")
            with ema.apply_model_ema_and_restore(model):
                if "evaluator" in cfg.dataloader:
                    ema_ret = inference_on_dataset(
                        model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
                    )
                    print_csv_format(ema_ret)
                    ret.update(ema_ret)
        return ret

def do_train(args, cfg, replay=False):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    # model = instantiate(cfg.model)
    config_file = args.model_config_file  # change the path of the model config file
    checkpoint_path = args.model_checkpoint_path  # change the path of the model
    model = load_model(config_file, checkpoint_path)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)
    if model.use_prompt_tuning:
        model.add_cls_prompt(cfg.dataloader.train.mapper.categories_names)
    
    # instantiate optimizer
    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    # build training loader
    train_loader = None
    if not replay:
        train_loader = instantiate(cfg.dataloader.train)
    
    # create ddp model
    model = create_ddp_model(model, **cfg.train.ddp)

    # build model ema
    ema.may_build_model_ema(cfg, model)
    
    if replay:
        trainer_class = MemoryReplayer
    else:
        trainer_class = Trainer
    trainer = trainer_class(
        model=model,
        dataloader=train_loader,
        optimizer=optim,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
        categories_names=cfg.dataloader.train.mapper.categories_names,
    )
    
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
        # save model ema
        **ema.may_get_ema_checkpointer(cfg, model)
    )

    if comm.is_main_process():
        # writers = default_writers(cfg.train.output_dir, cfg.train.max_iter)
        output_dir = cfg.train.output_dir
        PathManager.mkdirs(output_dir)
        writers = [
            CommonMetricPrinter(cfg.train.max_iter),
            JSONWriter(os.path.join(output_dir, "metrics.json")),
            TensorboardXWriter(output_dir),
        ]
        if cfg.train.wandb.enabled:
            PathManager.mkdirs(cfg.train.wandb.params.dir)
            writers.append(WandbWriter(cfg))

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            ema.EMAHook(cfg, model) if cfg.train.model_ema.enabled else None,
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                writers,
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
        
    trainer.train(start_iter, cfg.train.max_iter)

def main(args):
    import random
    seed_all_rng(args.seed) 
    logger = multi_datasets_setup_logger(args)
    if args.zero_shot:
        pre_trained_model_path = args.model_checkpoint_path
    
    # glob tasks and shuffle tasks
    config_dirs = args.config_file
    ow_config_files = glob.glob(os.path.join(config_dirs, "for_train", "*.py"))
    if args.shuffle_tasks:
        random.shuffle(ow_config_files)
    
    for tid, ow_config_file in enumerate(ow_config_files):
        torch.cuda.empty_cache()
        args.config_file = ow_config_file
        cfg = LazyConfig.load(ow_config_file)
        cfg = LazyConfig.apply_overrides(cfg, args.opts)
        default_setup(cfg, args, logger)
        if not args.eval_only:
            do_train(args, cfg)
        args.model_checkpoint_path = os.path.join(cfg.train.output_dir, "model_final.pth")
    
    # replaying memory
    coco_config_file = os.path.join(config_dirs, "test_zero_shot_coco.py")
    if os.path.exists(coco_config_file):
        if (not args.eval_only) and args.replay:
            torch.cuda.empty_cache()
            args.config_file = coco_config_file
            cfg = LazyConfig.load(coco_config_file)
            cfg = LazyConfig.apply_overrides(cfg, args.opts)
            default_setup(cfg, args, logger)
            do_train(args, cfg, replay=True)
            args.model_checkpoint_path = os.path.join(cfg.train.output_dir, "model_final.pth")
        os.system("cp {} {}".format(args.model_checkpoint_path, os.path.join(args.output_dir, "model_final.pth")))

    # eval
    args.eval_only = True
    if args.zero_shot:
        args.model_checkpoint_path = pre_trained_model_path
        
    if os.path.exists(coco_config_file):
        ow_config_files = ow_config_files + [coco_config_file]
        # raise ValueError("coco config file exists")
    json_paths = {}
        
    for ow_config_file in ow_config_files:
        torch.cuda.empty_cache()
        args.config_file = ow_config_file
        cfg = LazyConfig.load(args.config_file)
        cfg = LazyConfig.apply_overrides(cfg, args.opts)
        default_setup(cfg, args, logger)

        config_file = args.model_config_file  # change the path of the model config file
        checkpoint_path = args.model_checkpoint_path  # change the path of the model
        model = load_model(config_file, checkpoint_path)
        model.unfreeze_module(model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        # using ema for evaluation
        ema.may_build_model_ema(cfg, model)
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            ema.apply_model_ema(model)
        json_path = os.path.join(cfg.train.output_dir, "result.json")
        json_paths[ow_config_file] = json_path
        # res = do_test(cfg, model, eval_only=True)
        # if ow_config_file == coco_config_file:
        #     print(res)
        # with open(json_path, "w") as jf:
        #     json.dump(res, jf)
    
    avg_res = {}
    for k, v in json_paths.items():
        with open(v, "r") as jf:
            res = json.load(jf)
            if 'bbox' in res:
                avg_res[k] = res['bbox']['AP']

    # avg_res is a dict, key is the config file, value is the AP, logger the result
    result_str = "AP results: {}".format(avg_res)
    logger.info(result_str)
    sum = 0
    coco_count = 0
    for k, v in avg_res.items():
        if k != coco_config_file:
            sum += v
        else:
            coco_count += 1
    logger.info("average AP: {}".format(sum / (len(avg_res) - coco_count)))
    if coco_config_file in avg_res:
        logger.info("AP on COCO: {}".format(avg_res[coco_config_file]))

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--model-config-file", "-c", type=str, required=True, help="path to model config file")
    parser.add_argument("--model-checkpoint-path", "-p", type=str, required=True, help="path to model checkpoint file")
    parser.add_argument("--seed", type=int, default=None, required=False, help="path to model checkpoint file")
    parser.add_argument("--output-dir", type=str, default="output/odinw", required=False, help="path to model checkpoint file")
    parser.add_argument("--shuffle-tasks", action="store_true", help="perform shuffle tasks only")
    parser.add_argument("--replay", action="store_true", help="perform shuffle tasks only")
    parser.add_argument("--zero-shot", action="store_true", help="perform shuffle tasks only")
    args = parser.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
