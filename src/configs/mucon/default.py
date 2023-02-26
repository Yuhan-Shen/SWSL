import os

from yacs.config import CfgNode as CN

from core.config import dataset_cfg, system_cfg

_C = CN()
_C.experiment_name = "mucon_swsl"

_C.system = system_cfg
_C.dataset = dataset_cfg

### add configs for SWSL
_C.add_dataset = CN()
#_C.add_dataset.sample = 1
_C.add_dataset.ratio = 0.1 # the ratio of weakly labeled videos
_C.add_dataset.rebalance_ratio = True

### add params for SWSL
_C.add_params = CN()
_C.add_params.use_unlabeled = True # use unlabeled videos for training
_C.add_params.unlabel_teacher_forcing = False
_C.add_params.warmup_epoch = 60
_C.add_params.update_epochs = [80, 100, 120] # the epochs to update unlabeled to weakly-labeled
_C.add_params.update_ratios = [0.333, 0.5, 1] # the percentages of the remaining unlabeled data to be updated to weakly-labeled
_C.add_params.alpha = 0.999
_C.add_params.balance_ratio = 0.5 # the target ratio of labeled videos after dataset balance
_C.add_params.model_file_name = ''

# add config for distance between unlabeled and labeled videos
_C.add_params.sre = CN()
_C.add_params.sre.score_type = 'log' # 'prob', 'log'
_C.add_params.sre.alpha = 0.01
_C.add_params.sre.center_norm = False # whether normalize the centers when computing pairwise distance
_C.add_params.sre.sim = 'cos' # similarity metric, 'cos', 'exp', 'euc'
_C.add_params.sre.threshold = 5.0 # the threshold for insertion or deletion
_C.add_params.sre.swap_threshold = 0.0 # the threshold for swapping (transposition), set as threshold if 0
_C.add_params.sre.softmax = 'no' # 'no', 'row', 'col', 'all', whether apply softmax on the pairwise distance matrix 
_C.add_params.sre.norm_by_len = False # whether normalize the distance by length

_C.add_params.later_save_every = 10 # save_every after warmup
_C.add_params.later_eval_every = 10 # eval_every after warmup

# add config for flexible transcript prediction
_C.add_params.ftp = CN()
_C.add_params.ftp.penalty_weight = 0.001 # penalty of segment length, lambda in Algo. 1
_C.add_params.ftp.loss_weight = 0.1
_C.add_params.ftp.pooling_stride = 8
_C.add_params.ftp.n_seg_ratio = [0.8, 1.5]
_C.add_params.ftp.use_given_actions = False
_C.add_params.ftp.given_actions_train = False
_C.add_params.ftp.use_avg_prob = False
_C.add_params.ftp.use_only_mucon_loss = False
_C.add_params.ftp.match_reduction = 'min' ## 'min' or 'mean'

_C.trainer = CN()
_C.trainer.root = os.path.expanduser("~/work/MuCon/root")
_C.trainer.num_epochs = 180
_C.trainer.clip_grad_norm = True
# separate clipping for encoder and decoder
_C.trainer.clip_grad_norm_separate = True
_C.trainer.clip_grad_norm_every_param = False
_C.trainer.clip_grad_norm_value = 100.0
_C.trainer.optimizer = "SGD"  # ["SGD", "Adam", "RAdam"]
_C.trainer.learning_rate = 0.01
_C.trainer.momentum = 0.0
_C.trainer.weight_decay = 0.005
_C.trainer.accumulate_grad_every = 1
_C.trainer.scheduler = CN()
_C.trainer.scheduler.name = "step"  # can be 'none', 'plateau', 'step'
# below are the settings for plateau lr scheduler.
_C.trainer.scheduler.plateau = CN()
_C.trainer.scheduler.plateau.mode = "max"
_C.trainer.scheduler.plateau.factor = 0.1
_C.trainer.scheduler.plateau.verbose = True
_C.trainer.scheduler.plateau.patience = 20
_C.trainer.scheduler.step = CN()
_C.trainer.scheduler.step.milestones = [70]
_C.trainer.scheduler.step.gamma = 0.1
_C.trainer.save_every = 10
_C.trainer.eval_every = 10

_C.evaluator = CN()
_C.evaluator.viterbi = CN()
_C.evaluator.viterbi.multi_length = False

_C.model = CN()
_C.model.teacher_forcing = True
_C.model.name = "mucon"  # "mucon", "mucon-lnlstm", "mucon-glm"
_C.model.first_gru_hidden_size = 128
_C.model.loss = CN()
_C.model.loss.mul_mucon = 1.0
_C.model.loss.mul_transcript = 1.0
_C.model.loss.mul_smoothing = 0.1
_C.model.loss.mul_length = 0.1
_C.model.loss.length_width = 2.0
# whether or not to average the transcript loss for each word in the transcript
_C.model.loss.transcript_average = False

_C.model.loss.mucon_weight_background = False
_C.model.loss.mucon_weight_background_value = 0.5
_C.model.loss.mucon_weight_background_index = 0

_C.model.loss.transcript_weight_background = False
_C.model.loss.transcript_weight_background_value = 0.5
_C.model.loss.transcript_weight_background_index = 0

_C.model.loss.fully_supervised = CN()
_C.model.loss.fully_supervised.mul_classification = 1.0
_C.model.loss.fully_supervised.mul_supervised_length = 1.0

_C.model.loss.smoothing = CN()
_C.model.loss.smoothing.log_softmax_before = True
_C.model.loss.smoothing.clamp = True
_C.model.loss.smoothing.clamp_min = 0
_C.model.loss.smoothing.clamp_max = 16

_C.model.loss.mucon = CN()
_C.model.loss.mucon.type = "flint"  # "flint", "arithmetic"
_C.model.loss.mucon.template = "box"  # "box", "gaussian", "trapezoid"
_C.model.loss.mucon.overlap = 0.0


_C.model.ft = CN()
_C.model.ft.type = "wavenet"  # "wavenet", "mstcnpp", "noft"
_C.model.ft.stages = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
_C.model.ft.pooling = True
_C.model.ft.pooling_type = "max"
_C.model.ft.pooling_layers = [1, 2, 4, 8]
_C.model.ft.hidden_size = 128
_C.model.ft.dropout_rate = 0.25
_C.model.ft.leaky_relu = False

_C.model.ft.last_gn = True
_C.model.ft.last_gn_num_groups = 32

_C.model.ft.last_relu = True
_C.model.ft.last_dropout = True
_C.model.ft.last_dropout_rate = 0.25

_C.model.fs = CN()
_C.model.fs.jit_no_reverse = True
_C.model.fs.encoder = CN()
_C.model.fs.encoder.hidden_size = 128
_C.model.fs.encoder.bidirectional = True
# if num layers is 1, then dropout has no effect.
_C.model.fs.encoder.dropout = 0.0

_C.model.fs.decoder = CN()
_C.model.fs.decoder.embedding_dim = 128
_C.model.fs.decoder.embedding_dropout = 0.25
_C.model.fs.decoder.hidden_size = (
    128  # todo: potentially remove? should be equal to encoder hidden?
)
_C.model.fs.decoder.num_layers = 1
# if num layers is 1, then dropout has no effect.
_C.model.fs.decoder.dropout = 0.0

_C.model.fc = CN()


def get_cfg_defaults():
    return _C.clone()
