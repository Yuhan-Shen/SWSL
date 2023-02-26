import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from fandak import Dataset
from fandak.core.datasets import GeneralBatch
from torch import Tensor
from yacs.config import CfgNode

from core.datasets.utils import create_tf_input, create_tf_target
from core.viterbi.grammar import ModifiedPathGrammar


@dataclass(repr=False)
class Batch(GeneralBatch):
    """
    T: the video length
    D: the feat dim
    M: number of actions in the video.
    N: length of the transcript
    """

    feats: Tensor  # [1 x T x D] float
    gt_label: Tensor  # [T] long
    transcript: Tensor  # [N] long, 0 <= each item < M
    transcript_tf_input: Tensor  # [N + 1] long: equal to BOS + transcript
    # 0 <= each item < M + 2, +2 because of EOS and BOS in decoder dictionary
    transcript_tf_target: Tensor  # [N + 1] long: equal to transcript + EOS
    # 0 <= each item < M + 2, +2 because of EOS and BOS in decoder dictionary
    video_name: str  # the name of the video

@dataclass(repr=False)
class UnLabeledBatch(GeneralBatch):
    """
    T: the video length
    D: the feat dim
    M: number of actions in the video.
    N: length of the transcript
    """

    feats: Tensor  # [1 x T x D] float
    video_name: str  # the name of the video
    task_label: Tensor # the task label of the un-labeled video


@dataclass(repr=False)
class ExtendBatch(UnLabeledBatch):
    transcript: Tensor = None  # [N] long, 0 <= each item < M
    transcript_tf_input: Tensor = None  # [N + 1] long: equal to BOS + transcript
    # 0 <= each item < M + 2, +2 because of EOS and BOS in decoder dictionary
    transcript_tf_target: Tensor = None  # [N + 1] long: equal to transcript + EOS


@dataclass(repr=False)
class FullySupervisedBatch(Batch):
    absolute_lengths: Tensor  # [N]  float


@dataclass(repr=False)
class MixedSupervisionBatch(FullySupervisedBatch):
    fully_supervised: bool  # whether full supervision should be used or not.

@dataclass(repr=False)
class SemiSupervisionBatch(Batch):
    task_label: Tensor
    is_weakly: bool  # whether full supervision should be used or not.

@dataclass(repr=False)
class TaskLabeledBatch(Batch):
    task_label: Tensor # the task label of the weakly-labeled video
    is_weakly: bool



class GeneralDataset(Dataset):
    def __init__(
        self,
        cfg: CfgNode,
        root: Path,
        relative_path_to_list: Path = "split1.train",
        relative_path_to_mapping: Path = "mapping.txt",
        feat_dim: int = -1,
        relative_path_to_train_list: Path = None,
    ):
        """
            relative_path_to_train_list is added because sometimes we need to do full decoding,
            which means we need access to the full set of training transcripts.
        """
        super().__init__(cfg)
        self.root = root
        self.file_list = root / relative_path_to_list
        if relative_path_to_train_list is not None:
            train_file_list = root / relative_path_to_train_list
        else:
            train_file_list = None
        self.mapping_file = root / relative_path_to_mapping
        self.end_class_id = 0
        self.mof_eval_ignore_classes = []
        self.background_class_ids = [0]

        # following are defaults, should be set
        self.feat_dim = feat_dim
        self.convenient_name = None
        self.split = -1
        self.max_transcript_length = 100

        with open(self.file_list) as f:
            self.file_names = [x.strip() for x in f if len(x.strip()) > 0]
        task_list = sorted(list(set([file_name.split('_')[-1] for file_name in self.file_names])))
        self.task_id_to_name = {i: task for i, task in enumerate(task_list)}
        self.task_name_to_id = {task: i for i, task in enumerate(task_list)}

        self.action_id_to_name = {}
        self.action_name_to_id = {}
        if self.mapping_file is not None:
            with open(self.mapping_file) as f:
                the_mapping = [tuple(x.strip().split()) for x in f]

                for (i, l) in the_mapping:
                    self.action_id_to_name[int(i)] = l
                    self.action_name_to_id[l] = int(i)

        self.num_actions = len(self.action_id_to_name)

        self.feat_file_paths = [
            self.root / "features" / f"{x}.npy" for x in self.file_names
        ]
        self.gt_file_paths = [
            self.root / "labels" / f"{x}.npy" for x in self.file_names
        ]
        self.tr_file_paths = [
            self.root / "transcripts" / f"{x}.npy" for x in self.file_names
        ]
        self.task_labels = [
            self.task_name_to_id[x.split('_')[-1]] for x in self.file_names 
        ]
        self.is_weakly = [True for x in self.file_names]

        self.eos_token = "_EOS_"  # end of sentence
        self.sos_token = "_SOS_"  # start of sentence
        self.eos_token_id = self.num_actions  # = M, 48 for breakfast
        self.sos_token_id = self.num_actions + 1  # = M + 1, 49 for breakfast
        self.action_id_to_name[self.eos_token_id] = self.eos_token
        self.action_name_to_id[self.eos_token] = self.eos_token_id
        self.action_id_to_name[self.sos_token_id] = self.sos_token
        self.action_name_to_id[self.sos_token] = self.sos_token_id

        # loading the training transcripts
        training_transcripts_per_task = dict()
        if train_file_list is not None:
            with open(train_file_list) as f:
                train_file_names = [x.strip() for x in f if len(x.strip()) > 0]
            tr_train_file_paths = [
                self.root / "transcripts" / f"{x}.npy" for x in train_file_names
            ]

            train_task_labels = [
                self.task_name_to_id[x.split('_')[-1]] for x in train_file_names 
            ]
            training_transcripts = set()
            for i, tr_file_path in enumerate(tr_train_file_paths):
                transcript = tuple(np.load(str(tr_file_path)))
                training_transcripts.add(transcript)
                task_label = train_task_labels[i]
                training_transcripts_this_task = training_transcripts_per_task.get(task_label, set())
                training_transcripts_this_task.add(transcript)
                training_transcripts_per_task[task_label] = training_transcripts_this_task

            self.training_transcripts_list = []
            for t in training_transcripts:
                self.training_transcripts_list.append(list(t))

            self.training_transcripts_per_task = {task: [] for task in training_transcripts_per_task}
            for task in training_transcripts_per_task:
                for t in training_transcripts_per_task[task]:
                    self.training_transcripts_per_task[task].append(list(t))

            # the set of actions for each task
            self.training_actions_per_task = {task: set() for task in training_transcripts_per_task}
            # the average length of transcripts for each task
            self.training_avg_n_steps_per_task = {}
            for task in training_transcripts_per_task:
                n_steps_list = []
                for transcript in self.training_transcripts_per_task[task]:
                    self.training_actions_per_task[task].update(set(transcript))
                    n_steps_list.append(len(transcript))
                self.training_avg_n_steps_per_task[task] = np.mean(n_steps_list)

            self.training_path_grammar = ModifiedPathGrammar(
                transcripts=self.training_transcripts_list,
                num_classes=self.num_actions
            )
    
    def add_videos(self, add_dict):
        for video in add_dict:
            task_label, transcript = add_dict[video][:2]
            self.file_names.append(video)
            self.feat_file_paths.append(self.root / "features" / "{}.npy".format(video))
            self.gt_file_paths.append(self.root / "labels" / "{}.npy".format(video))
            self.tr_file_paths.append(transcript)
            self.is_weakly.append(True)
            self.task_labels.append(task_label)

    def get_num_classes(self) -> int:
        return self.num_actions

    def __len__(self) -> int:
        return len(self.feat_file_paths)

    def __getitem__(self, item: int) -> TaskLabeledBatch:
        feat_file_path = str(self.feat_file_paths[item])
        task_label = self.task_labels[item]

        numpy_feats = np.load(feat_file_path)  # T x D
        vid_feats = torch.tensor(numpy_feats).float()  # T x D
        vid_feats.unsqueeze_(0)  # 1 x T x D

        if self.is_weakly[item]:
            gt_file_path = str(self.gt_file_paths[item])
            if isinstance(self.tr_file_paths[item], type(self.feat_file_paths[item])):
                tr_file_path = str(self.tr_file_paths[item])
                transcript = np.load(tr_file_path)  # N
            else:
                transcript = np.array(self.tr_file_paths[item])
            gt_labels = np.load(gt_file_path)  # T

            gt_labels = torch.tensor(gt_labels).long()  # T
            transcript = torch.tensor(transcript).long()  # N
        else:
            raise NotImplementedError

        transcript_tf_input = torch.tensor(
            create_tf_input(transcript, sos_i=self.sos_token_id)
        ).long()  # [N + 1]
        transcript_tf_target = torch.tensor(
            create_tf_target(transcript, eos_i=self.eos_token_id)
        ).long()  # [N + 1]

        task_label = torch.tensor([task_label]).long()

        return TaskLabeledBatch(
            feats=vid_feats,
            gt_label=gt_labels,
            transcript=transcript,
            transcript_tf_input=transcript_tf_input,
            transcript_tf_target=transcript_tf_target,
            video_name=self.file_names[item],
            task_label=task_label,
            is_weakly=self.is_weakly[item],
        )

    def collate_fn(self, items: List[Batch]) -> Batch:
        # We assume batch_size = 1
        assert len(items) == 1

        return items[0]


class GeneralFullySupervisedDataset(GeneralDataset):
    def __init__(
        self,
        cfg: CfgNode,
        root: Path,
        relative_path_to_list: Path = "split1.train",
        relative_path_to_mapping: Path = "mapping.txt",
        feat_dim: int = -1,
    ):
        super().__init__(
            cfg, root, relative_path_to_list, relative_path_to_mapping, feat_dim
        )
        self.len_file_paths = [
            self.root / "lengths" / f"{x}.npy" for x in self.file_names
        ]

    def __getitem__(self, item: int) -> FullySupervisedBatch:
        the_item = super().__getitem__(item)

        len_file_path = str(self.len_file_paths[item])
        absolute_lengths = np.load(len_file_path)  # [N]
        absolute_lengths = torch.tensor(absolute_lengths, dtype=torch.float32)

        return FullySupervisedBatch(
            feats=the_item.feats,
            gt_label=the_item.gt_label,
            transcript=the_item.transcript,
            transcript_tf_target=the_item.transcript_tf_target,
            transcript_tf_input=the_item.transcript_tf_input,
            video_name=the_item.video_name,
            absolute_lengths=absolute_lengths,
        )


class GeneralMixedSupervisionDataset(GeneralFullySupervisedDataset):
    def __init__(
        self,
        cfg: CfgNode,
        root: Path,
        full_supervision_percentage: float,
        relative_path_to_list: Path = "split1.train",
        relative_path_to_mapping: Path = "mapping.txt",
        feat_dim: int = -1,
    ):
        super().__init__(
            cfg, root, relative_path_to_list, relative_path_to_mapping, feat_dim
        )
        assert 0.0 < full_supervision_percentage < 100.0
        self.full_supervision_percentage = full_supervision_percentage

        self.number_of_full_supervision_examples = min(
            len(self.feat_file_paths),
            max(
                1,
                int(
                    round(
                        (
                            len(self.feat_file_paths)
                            * self.full_supervision_percentage
                            / 100.0
                        )
                    )
                ),
            ),
        )
        self.is_it_supervised = [False] * len(self.feat_file_paths)
        self.is_it_supervised[: self.number_of_full_supervision_examples] = [
            True for _ in range(self.number_of_full_supervision_examples)
        ]
        random.seed(
            f"{self.cfg.system.seed}-{self.number_of_full_supervision_examples}"
        )
        random.shuffle(self.is_it_supervised)

    def __getitem__(self, item: int) -> MixedSupervisionBatch:
        fully_supervised_batch = super().__getitem__(item)
        is_this_supervised = self.is_it_supervised[item]

        return MixedSupervisionBatch(
            feats=fully_supervised_batch.feats,
            gt_label=fully_supervised_batch.gt_label,
            transcript=fully_supervised_batch.transcript,
            transcript_tf_input=fully_supervised_batch.transcript_tf_input,
            transcript_tf_target=fully_supervised_batch.transcript_tf_target,
            video_name=fully_supervised_batch.video_name,
            absolute_lengths=fully_supervised_batch.absolute_lengths,
            fully_supervised=is_this_supervised,
        )

class GeneralSemiSupervisionDataset(Dataset):
    def __init__(
        self,
        cfg: CfgNode,
        root: Path,
        relative_path_to_list: Path = "split1.train",
        relative_path_to_mapping: Path = "mapping.txt",
        feat_dim: int = -1,
        relative_path_to_train_list: Path = None,
        relative_path_to_all_list: Path = "split1.train",
    ):
        """
            relative_path_to_train_list is added because sometimes we need to do full decoding,
            which means we need access to the full set of training transcripts.
            relative_path_to_all_list is added because we need the full list to get the filenames of unlabeled videos
        """
        super().__init__(cfg)
        self.root = root
        self.file_list = root / relative_path_to_list
        if relative_path_to_train_list is not None:
            train_file_list = root / relative_path_to_train_list
        else:
            train_file_list = None
        if relative_path_to_all_list is not None:
            all_file_list = root / relative_path_to_all_list
        else:
            all_file_list = None
        self.all_file_list = all_file_list
        self.mapping_file = root / relative_path_to_mapping
        self.end_class_id = 0
        self.mof_eval_ignore_classes = []
        self.background_class_ids = [0]

        # following are defaults, should be set
        self.feat_dim = feat_dim
        self.convenient_name = None
        self.split = -1
        self.max_transcript_length = 100

        with open(self.file_list) as f:
            self.file_names = [x.strip() for x in f if len(x.strip()) > 0]
        with open(self.all_file_list) as f:
            self.all_file_names = [x.strip() for x in f if len(x.strip()) > 0]

        task_list = sorted(list(set([file_name.split('_')[-1] for file_name in self.file_names])))
        self.task_id_to_name = {i: task for i, task in enumerate(task_list)}
        self.task_name_to_id = {task: i for i, task in enumerate(task_list)}

        self.action_id_to_name = {}
        self.action_name_to_id = {}
        if self.mapping_file is not None:
            with open(self.mapping_file) as f:
                the_mapping = [tuple(x.strip().split()) for x in f]

                for (i, l) in the_mapping:
                    self.action_id_to_name[int(i)] = l
                    self.action_name_to_id[l] = int(i)

        self.num_actions = len(self.action_id_to_name)

        self.feat_file_paths = [
            self.root / "features" / f"{x}.npy" for x in self.all_file_names
        ]
        self.gt_file_paths = [
            self.root / "labels" / f"{x}.npy" for x in self.all_file_names
        ]
        self.tr_file_paths = [
            self.root / "transcripts" / f"{x}.npy" for x in self.all_file_names
        ]
        self.task_labels = [
            self.task_name_to_id[x.split('_')[-1]] for x in self.all_file_names 
        ]
        self.is_it_weakly_labeled = [
            x in self.file_names for x in self.all_file_names 
        ]


        self.eos_token = "_EOS_"  # end of sentence
        self.sos_token = "_SOS_"  # start of sentence
        self.eos_token_id = self.num_actions  # = M, 48 for breakfast
        self.sos_token_id = self.num_actions + 1  # = M + 1, 49 for breakfast
        self.action_id_to_name[self.eos_token_id] = self.eos_token
        self.action_name_to_id[self.eos_token] = self.eos_token_id
        self.action_id_to_name[self.sos_token_id] = self.sos_token
        self.action_name_to_id[self.sos_token] = self.sos_token_id

        # loading the training transcripts
        training_transcripts_per_task = dict()
        if train_file_list is not None:
            with open(train_file_list) as f:
                train_file_names = [x.strip() for x in f if len(x.strip()) > 0]
            tr_train_file_paths = [
                self.root / "transcripts" / f"{x}.npy" for x in train_file_names
            ]

            train_task_labels = [
                self.task_name_to_id[x.split('_')[-1]] for x in train_file_names 
            ]
            training_transcripts = set()
            for i, tr_file_path in enumerate(tr_train_file_paths):
                transcript = tuple(np.load(str(tr_file_path)))
                training_transcripts.add(transcript)
                task_label = train_task_labels[i]
                training_transcripts_this_task = training_transcripts_per_task.get(task_label, set())
                training_transcripts_this_task.add(transcript)
                training_transcripts_per_task[task_label] = training_transcripts_this_task

            self.training_transcripts_list = []
            for t in training_transcripts:
                self.training_transcripts_list.append(list(t))

            self.training_transcripts_per_task = {task: [] for task in training_transcripts_per_task}
            for task in training_transcripts_per_task:
                for t in training_transcripts_per_task[task]:
                    self.training_transcripts_per_task[task].append(list(t))

            self.training_actions_per_task = {task: set() for task in training_transcripts_per_task}
            for task in training_transcripts_per_task:
                for transcript in self.training_transcripts_per_task[task]:
                    self.training_actions_per_task[task].update(set(transcript))

            self.training_path_grammar = ModifiedPathGrammar(
                transcripts=self.training_transcripts_list,
                num_classes=self.num_actions
            )
        num_labeled = len(self.file_names)
        num_all = len(self.all_file_names)
        self.num_labeled = num_labeled
        self.num_all = num_all
        self.num_added = 0


    def balance_dataset(self, target_ratio):
        # balance the number of labeled and unlabeled videos to make the ratio of labeled videos be target_ratio
        num_labeled = len(self.file_names)
        num_all = len(self.all_file_names)
        self.num_labeled = num_labeled
        self.num_all = num_all
        num_to_add = int((target_ratio * num_all - num_labeled) / (1 - target_ratio))
        print('Num. of Labeled Videos:{}, Num. of All Videos:{}, Num. of Videos to Add:{}'.format(num_labeled, num_all, num_to_add))
        num_added = 0
        while num_added < num_to_add:
            for file_name in self.file_names:
                self.feat_file_paths.append(
                    self.root / "features" / "{}.npy".format(file_name)) 
                self.gt_file_paths.append(
                    self.root / "labels" / "{}.npy".format(file_name))
                self.tr_file_paths.append(
                    self.root / "transcripts" / "{}.npy".format(file_name))
                self.task_labels.append(
                    self.task_name_to_id[file_name.split('_')[-1]])
                self.is_it_weakly_labeled.append(True)
                self.all_file_names.append(file_name)
                num_added += 1
                if num_added >= num_to_add:
                    break
        self.num_added = num_added

    def rebalance_dataset(self, target_ratio, num_pseudo):
        # rebalance the number of labeled and unlabeled videos to make the ratio of labeled videos be target_ratio
        assert len(set(self.all_file_names)) == self.num_all
        num_to_delete = min(int(num_pseudo /  ( 1 - target_ratio)), self.num_added)
        if num_to_delete <= 0:
            return
        self.feat_file_paths = self.feat_file_paths[:-num_to_delete]
        self.gt_file_paths = self.gt_file_paths[:-num_to_delete]
        self.tr_file_paths = self.tr_file_paths[:-num_to_delete]
        self.task_labels = self.task_labels[:-num_to_delete]
        self.is_it_weakly_labeled = self.is_it_weakly_labeled[:-num_to_delete]
        self.all_file_names = self.all_file_names[:-num_to_delete]
        self.num_added -= num_to_delete
        assert len(set(self.all_file_names)) == self.num_all

    def update_pseudo_transcripts(self, add_dict):
        for video in add_dict:
            task_label, transcript = add_dict[video][:2]
            video_id = self.all_file_names.index(video)
            self.tr_file_paths[video_id] = transcript
            self.is_it_weakly_labeled[video_id] = True

    def get_num_classes(self) -> int:
        return self.num_actions

    def __len__(self) -> int:
        return len(self.feat_file_paths)

    def __getitem__(self, item: int) -> SemiSupervisionBatch:
        feat_file_path = str(self.feat_file_paths[item])
        if isinstance(self.tr_file_paths[item], type(self.feat_file_paths[item])):
            tr_file_path = str(self.tr_file_paths[item])
            transcript = np.load(tr_file_path)  # N
        else:
            transcript = np.array(self.tr_file_paths[item])

        gt_file_path = str(self.gt_file_paths[item])
        task_label = self.task_labels[item]
        is_this_weakly_labeled = self.is_it_weakly_labeled[item]

        numpy_feats = np.load(feat_file_path)  # T x D
        vid_feats = torch.tensor(numpy_feats).float()  # T x D
        vid_feats.unsqueeze_(0)  # 1 x T x D

        gt_labels = np.load(gt_file_path)  # T

        gt_labels = torch.tensor(gt_labels).long()  # T
        transcript = torch.tensor(transcript).long()  # N

        transcript_tf_input = torch.tensor(
            create_tf_input(transcript, sos_i=self.sos_token_id)
        ).long()  # [N + 1]
        transcript_tf_target = torch.tensor(
            create_tf_target(transcript, eos_i=self.eos_token_id)
        ).long()  # [N + 1]

        task_label = torch.tensor([task_label]).long()

        return SemiSupervisionBatch(
            feats=vid_feats,
            gt_label=gt_labels,
            transcript=transcript,
            transcript_tf_input=transcript_tf_input,
            transcript_tf_target=transcript_tf_target,
            video_name=self.all_file_names[item],
            task_label=task_label,
            is_weakly=is_this_weakly_labeled,
        )

    def collate_fn(self, items: List[Batch]) -> Batch:
        # We assume batch_size = 1
        assert len(items) == 1

        return items[0]

