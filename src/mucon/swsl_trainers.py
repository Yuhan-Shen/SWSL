from pathlib import Path
from typing import Optional, Iterable, Any, Dict, List, Union

import torch
from fandak import Trainer, Evaluator, Model, Dataset
from fandak.core.trainers import Scheduler
from torch import optim
from torch.nn import Parameter
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from yacs.config import CfgNode

from mucon.evaluators import MuConEvaluatorResult
from core.datasets.general_dataset import ExtendBatch
from core.datasets.utils import create_tf_input, create_tf_target
from fandak.utils.misc import print_with_time
from tqdm import tqdm
import numpy as np
from mucon.swsl_models import MuConSWSLoss

import torch.nn.functional as F
from .pseudo_label_generator import get_enhanced_predict

from itertools import groupby
def remove_continuous_duplicates(a):
    return np.array([k for k,g in groupby(a)])

MODEL_FILE_NAME = "model"
PSUEDOLABEL_FILE_NAME = "pseudo_labels"
ALL_PSUEDOLABEL_FILE_NAME = "all_pseudo_labels"
OPTIMIZER_FILE_NAME = "optimizer"
SCHEDULER_FILE_NAME = "scheduler"

def create_optimizer(cfg: CfgNode, parameters: Iterable[Parameter]) -> Optimizer:
    learning_rate = cfg.trainer.learning_rate
    momentum = cfg.trainer.momentum
    optimizer_name = cfg.trainer.optimizer
    weight_decay = cfg.trainer.weight_decay

    if optimizer_name == "SGD":
        return optim.SGD(
            params=parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    elif optimizer_name == "Adam":
        return optim.Adam(
            params=parameters, lr=learning_rate, weight_decay=weight_decay, amsgrad=True
        )
    else:
        raise Exception("Invalid optimizer name (%s)" % optimizer_name)


def create_scheduler(cfg: CfgNode, optimizer: Optimizer) -> Optional[Scheduler]:
    scheduler_name = cfg.trainer.scheduler.name
    if scheduler_name == "none":
        return None
    elif scheduler_name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=cfg.trainer.scheduler.plateau.mode,
            factor=cfg.trainer.scheduler.plateau.factor,
            verbose=cfg.trainer.scheduler.plateau.verbose,
            patience=cfg.trainer.scheduler.plateau.patience,
        )
    elif scheduler_name == "step":
        milestones = cfg.trainer.scheduler.step.milestones
        gamma = cfg.trainer.scheduler.step.gamma
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    else:
        raise Exception("Invalid scheduler name (%s)" % scheduler_name)


class SimpleTrainer(Trainer):
    def update_trainer_using_config(self):
        self.save_every = self.cfg.trainer.save_every
        self.eval_every = self.cfg.trainer.eval_every
        self.later_save_every = self.cfg.add_params.later_save_every
        self.later_eval_every = self.cfg.add_params.later_eval_every

    def on_start_epoch(self, epoch_num: int):
        self.model.set_teacher_forcing(self.cfg.model.teacher_forcing)

    # noinspection PyUnresolvedReferences
    def on_finish_epoch(self, epoch_num: int):
        if ((epoch_num + 1) % self.eval_every == 0) or (
                epoch_num > self.warmup_epoch and (epoch_num + 1) % self.later_eval_every == 0):
            # Saving stuff for streamlit visualization.
            for evaluator in self.evaluators:
                evaluator.set_checkpointing_folder(self._get_checkpointing_folder())
                evaluator.save_stuff()

    def figure_root(self) -> Path:
        return Path(self.cfg.trainer.root)

    def figure_optimizer(self) -> Optimizer:
        original_lr = self.cfg.trainer.learning_rate
        return create_optimizer(self.cfg, self.model.get_params(original_lr))

    def figure_scheduler(self, optimizer: Optimizer) -> Optional[Scheduler]:
        return create_scheduler(self.cfg, optimizer)

    def figure_clip_grad_norm(self) -> Optional[float]:
        if self.cfg.trainer.clip_grad_norm:
            return self.cfg.trainer.clip_grad_norm_value
        else:
            return None

    def figure_accumulate_grad(self) -> int:
        return self.cfg.trainer.accumulate_grad_every

    def figure_num_epochs(self) -> int:
        return self.cfg.trainer.num_epochs

    def create_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_db,
            batch_size=1,
            shuffle=True,
            num_workers=self.cfg.system.num_workers,
            collate_fn=self.train_db.collate_fn,
            pin_memory=True,
        )


    def figure_scheduler_input(
        self, eval_results: List[MuConEvaluatorResult]
    ) -> Dict[str, Any]:
        if self.cfg.trainer.scheduler.name == "plateau":
            return {"metrics": eval_results[0].s_mof_nbg}
        else:
            return {}

class SWSLTrainer(SimpleTrainer):
    def __init__(
        self,
        cfg: CfgNode,
        exp_name: str,
        train_db: Dataset,
        model: Model,
        device: torch.device = torch.device("cpu"),
        evaluators: Optional[Union[Iterable[Evaluator], Evaluator]] = None,
        train_db_all: Dataset = None,
    ):
        super().__init__(
            cfg=cfg,
            exp_name=exp_name,
            train_db=train_db,
            model=model,
            device=device,
            evaluators=evaluators,
        )
        self.train_db_all = train_db_all
        self.warmup_epoch = cfg.add_params.warmup_epoch
        #self.cons_warmup_epoch = cfg.add_params.cons_warmup_epoch
        self.alpha = cfg.add_params.alpha

    def create_train_dataloader(self) -> DataLoader:
        if self.epoch_num < self.warmup_epoch:
            train_db = self.train_db
        else:
            train_db = self.train_db_all
        return DataLoader(
            train_db,
            batch_size=1,
            shuffle=True,
            num_workers=self.cfg.system.num_workers,
            collate_fn=self.train_db.collate_fn,
            pin_memory=True,
        )

    def on_start_epoch(self, epoch_num: int):
        self.model.set_teacher_forcing(self.cfg.model.teacher_forcing)

    def save_add_dict(self, add_dict, file_name=PSUEDOLABEL_FILE_NAME):
        self.checkpoint_this(file_name, add_dict, torch_save=False)

    # noinspection PyUnusedLocal
    def _train_1_batch(self, iter_num: int, batch):
        # callback
        self.on_start_batch(self.iter_num, batch)

        # FIXME: this might be slow depending on the config system
        accumulate_grad_every = self.figure_accumulate_grad()
        if accumulate_grad_every is None:
            accumulate_grad_every = 1

        # TODO: move to the end.
        # TODO: move to infinitely flexible callbacks. like fastai v2.
        # initial setup
        if iter_num % accumulate_grad_every == 0:
            self.optimizer.zero_grad()
        batch.to(self.device)

        # forward pass
        if batch.is_weakly:
            self.model.set_teacher_forcing(True)
            forward_out = self.model.forward(batch)
            # if the video is weakly labeled, then we only use the original MuCon Loss for training
            loss = self.model.loss(batch, forward_out)
            extend_loss = MuConSWSLoss(
                main=loss.main,
                transcript_loss=loss.transcript_loss,
                length_loss=loss.length_loss,
                mucon_loss=loss.mucon_loss,
                smoothing_loss=loss.smoothing_loss,
                ftp_loss=0,
            )
            the_loss = extend_loss.main / accumulate_grad_every
        else:
            self.model.set_teacher_forcing(False)
            forward_out = self.model.forward(batch)
            # if the video is unlabeled, then we use ftp loss
            ftp_loss = self.model.ftp_loss(batch, forward_out, self.train_db.training_transcripts_per_task, self.train_db.training_actions_per_task, self.train_db.training_avg_n_steps_per_task)

            main_loss = self.cfg.add_params.ftp.loss_weight * ftp_loss
            extend_loss = MuConSWSLoss(
                main=main_loss,
                transcript_loss=0,
                length_loss=0,
                mucon_loss=0,
                smoothing_loss=0,
                ftp_loss=ftp_loss,
            )
            the_loss = extend_loss.main / accumulate_grad_every

        # backward pass
        the_loss.backward()

        # TODO: refactor fandak with this as a function for easier implementation.
        # optional gradient clipping
        if self.clip_grad_norm is not None:
            if self.cfg.trainer.clip_grad_norm_separate:
                clip_grad_norm_(self.model.encode_params, max_norm=self.clip_grad_norm)
                clip_grad_norm_(self.model.decode_params, max_norm=self.clip_grad_norm)
            else:
                if self.cfg.trainer.clip_grad_norm_every_param:
                    for p in self.model.parameters():
                        clip_grad_norm_(p, self.clip_grad_norm)
                else:
                    clip_grad_norm_(
                        self.model.parameters(), max_norm=self.clip_grad_norm
                    )

        # optimizer step
        if iter_num % accumulate_grad_every == (accumulate_grad_every - 1):
            self.optimizer.step()

        # callback
        self.on_finish_batch(self.iter_num, batch, forward_out, extend_loss)

        return extend_loss, forward_out

    def loss_given_transcript(self, extend_batch, transcript):
        # compute the MuCon Loss for a video given a transcript
        self.model.set_teacher_forcing(True)

        transcript = torch.tensor(transcript).long()  # N
        transcript_tf_input = torch.tensor(
            create_tf_input(transcript, sos_i=self.train_db.sos_token_id)
        ).long()  # [N + 1]
        transcript_tf_target = torch.tensor(
            create_tf_target(transcript, eos_i=self.train_db.eos_token_id)
        ).long()  # [N + 1]
        extend_batch.transcript = transcript
        extend_batch.transcript_tf_target = transcript_tf_target
        extend_batch.transcript_tf_input = transcript_tf_input

        extend_batch.to(self.device)
        # forward pass
        forward_out = self.model.forward(extend_batch)
        loss = self.model.loss(extend_batch, forward_out)
        return loss

    def generate_pseudo_transcripts_unlabeled(self, select_num, use_action_mask=True):
        # generate pseudo transcripts for all unlabeled videos in the current dataset,
        # and select the samples with the hightest confidences (lowest loss values)
        # select_num: int, the number of samples to select
        # use_action_mask: bool, whether use action mask given the task label for FTP
        dataloader = DataLoader(
                    self.train_db_all,
                    batch_size=1,
                    shuffle=True,
                    num_workers=self.cfg.system.num_workers,
                    collate_fn=self.train_db.collate_fn,
                    pin_memory=True,
                    )

        self.model.to(self.device)
        self.model.eval()

        ### use the combination of weak transcripts based on L_weak
        videos_list = []
        losses_list = []
        transcripts_list = []
        task_labels_list = []

        for batch in tqdm(dataloader):
            if batch.is_weakly:
                continue
            extend_batch = ExtendBatch(
            feats=batch.feats,
            video_name=batch.video_name,
            task_label=batch.task_label,
            )
            possible_transcripts = self.train_db.training_transcripts_per_task[batch.task_label.item()].copy()

            batch.to(self.device)
            self.model.set_teacher_forcing(False)
            forward_out = self.model.forward(batch)


            transcripts_onehot = [F.one_hot(torch.tensor(transcript), self.model.EOS_token_id).float() for transcript in possible_transcripts]
            loss_list = []
            # find the best matched transcript in the weakly labeled set
            for transcript in possible_transcripts:
                loss = self.loss_given_transcript(extend_batch, transcript)
                if self.cfg.add_params.ftp.use_only_mucon_loss:
                    loss_list.append(loss.mucon_loss.item())
                else:
                    loss_list.append(loss.main.item())
            loss_list = np.array(loss_list)
            best_ind = np.argmin(loss_list)
            best_transcript_onehot = transcripts_onehot[best_ind]

            possible_actions = self.train_db.training_actions_per_task[batch.task_label.item()]
            avg_n_steps = self.train_db.training_avg_n_steps_per_task[batch.task_label.item()]
            # get the prototypes (flexible transcripts) of unlabeled videos
            if use_action_mask:
                prototypes = self.model.ftp(batch, forward_out, avg_n_steps, given_actions=torch.tensor(list(possible_actions)))
            else:
                prototypes = self.model.ftp(batch, forward_out, avg_n_steps, given_actions=None)

            swap_threshold = self.cfg.add_params.sre.swap_threshold
            # if swap_threshold is not set, then set it the same as threshold
            if swap_threshold == 0:
                swap_threshold = self.cfg.add_params.sre.threshold

            # get the final pseudo transcript
            pseudo_transcript = get_enhanced_predict(best_transcript_onehot, prototypes, bg_cost=self.cfg.add_params.sre.threshold, swap_cost=swap_threshold)

            predict_transcript = pseudo_transcript.argmax(1)
            prob_transcript = pseudo_transcript.max(1)
            if np.where(prob_transcript>0)[0].shape[0] > 0:
                predict_transcript = predict_transcript[prob_transcript>0]
            predict_transcript = remove_continuous_duplicates(predict_transcript)

            ### use predicted transcript to compute L_weak and select confident samples
            loss = self.loss_given_transcript(extend_batch, predict_transcript)
            videos_list.append(batch.video_name)
            task_labels_list.append(batch.task_label)
            if self.cfg.add_params.ftp.use_only_mucon_loss:
                losses_list.append(loss.mucon_loss.item())
            else:
                losses_list.append(loss.main.item())
            transcripts_list.append(predict_transcript)

        select_num = min(select_num, len(videos_list))

        sort_ind = np.argsort(losses_list)
        select_dict = dict()
        all_dict  = dict()
        for rank, i in enumerate(sort_ind):
            all_dict[videos_list[i]] = (task_labels_list[i], transcripts_list[i], losses_list[i])
            if rank < select_num:
                select_dict[videos_list[i]] = (task_labels_list[i], transcripts_list[i], losses_list[i])

        return select_dict, all_dict

    def save_training(self):
        print_with_time("Saving model ...")
        self.checkpoint_this(MODEL_FILE_NAME, self.model.state_dict(), torch_save=True)
        self.checkpoint_this(
            OPTIMIZER_FILE_NAME, self.optimizer.state_dict(), torch_save=True
        )
        if self.scheduler is not None:
            self.checkpoint_this(SCHEDULER_FILE_NAME, self.scheduler, torch_save=True)

    def train(self):
        num_epochs = self.figure_num_epochs()
        self._mark_the_run()
        print_with_time("Training for run number: {:d}".format(self.run_number))
        epoch_range_start = self.epoch_num
        epoch_range = range(epoch_range_start, epoch_range_start + num_epochs)

        # callback
        self.on_start_training(num_epochs)

        assert len(self.cfg.add_params.update_epochs) == len(self.cfg.add_params.update_ratios)
        num_unlabeled_videos = self.train_db_all.num_all - self.train_db_all.num_labeled

        update_num_dict = dict()
        for i, update_epoch in enumerate(self.cfg.add_params.update_epochs):
            num_select = int(num_unlabeled_videos * self.cfg.add_params.update_ratios[i])
            update_num_dict[update_epoch] = num_select
            num_unlabeled_videos -= num_select

        for epoch_num in epoch_range:
            self.epoch_num = epoch_num

            # callback
            self.on_start_epoch(epoch_num)

            # resetting metrics
            for n, m in self.metrics.items():
                if self.metrics[n].report_average:
                    m.reset_values()

            # training for 1 epoch
            dataloader = self.create_train_dataloader()
            self.train_1_epoch(epoch_num, dataloader)

            # saving
            if ((epoch_num + 1) % self.save_every == 0) or (
                    epoch_num > self.warmup_epoch and (epoch_num + 1) % self.later_save_every == 0):
                self.save_training()

            # evaluation
            eval_results = []
            if ((epoch_num + 1) % self.eval_every == 0) or (
                    epoch_num > self.warmup_epoch and (epoch_num + 1) % self.later_eval_every == 0):
                for evaluator in self.evaluators:
                    eval_results.append(evaluator.evaluate())
                    evaluator.reset_storage()
                self.track_end_of_epoch_metrics(eval_results, epoch_num)

            # scheduler
            if self.scheduler is not None:
                # noinspection PyArgumentList
                self.scheduler.step(**self.figure_scheduler_input(eval_results))

            # callback
            self.on_finish_epoch(epoch_num)

            # update unlabeled videos into weakly-labeled set
            if (epoch_num + 1) in update_num_dict:
                add_dict, all_dict = self.generate_pseudo_transcripts_unlabeled(select_num=update_num_dict[epoch_num+1], use_action_mask=self.cfg.add_params.ftp.use_given_actions)
                self.save_add_dict(all_dict, file_name=ALL_PSUEDOLABEL_FILE_NAME)
                self.train_db_all.update_pseudo_transcripts(add_dict)
                self.save_add_dict(add_dict)
                if self.cfg.add_dataset.rebalance_ratio:
                    self.train_db_all.rebalance_dataset(self.cfg.add_params.balance_ratio, len(add_dict))

        # callback
        self.on_finish_training(num_epochs)


class TrainerForTFExperiments(SimpleTrainer):
    def __init__(
        self,
        cfg: CfgNode,
        exp_name: str,
        train_db: Dataset,
        model: Model,
        device: torch.device,
        evaluators: Optional[Union[Iterable[Evaluator], Evaluator]] = None,
        turnoff_tf_after_epoch: int = 1000,
    ):
        super().__init__(
            cfg=cfg,
            exp_name=exp_name,
            train_db=train_db,
            model=model,
            device=device,
            evaluators=evaluators,
        )
        self.turnoff_tf_after_epoch = turnoff_tf_after_epoch

    def on_start_epoch(self, epoch_num: int):
        if epoch_num >= self.turnoff_tf_after_epoch:
            self.model.set_teacher_forcing(teacher_forcing=False)
        else:
            self.model.set_teacher_forcing(teacher_forcing=True)

