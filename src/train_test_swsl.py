from typing import List

import click
from fandak.utils import common_config
from fandak.utils.config import update_config

from configs.mucon.default import get_cfg_defaults
from core.datasets import handel_dataset, handel_semi_supervision_dataset #, handel_unsupervised_dataset
from mucon.evaluators import MuConEvaluator
from mucon.swsl_models import create_model #, create_semi_supervision_model
from mucon.swsl_trainers import SWSLTrainer

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

@click.command()
@common_config
@click.option("--exp-name", default="")
def main(file_configs: List[str], set_configs: List[str], exp_name: str):

    cfg = update_config(
        default_config=get_cfg_defaults(),
        file_configs=file_configs,
        set_configs=set_configs,
    )
    if exp_name != "":
        cfg.defrost()
        cfg.experiment_name = exp_name
        cfg.freeze()

    print('config:', cfg)
    """
    train_db: dataset for weakly-labeled videos
    train_db_all: dataset for both weakly-labeled and unlabeled videos
    """
    train_db = handel_dataset(cfg, train=True)
    train_db_all = handel_semi_supervision_dataset(cfg, train=True)
    train_db_all.balance_dataset(cfg.add_params.balance_ratio)
    test_db = handel_dataset(cfg, train=False)

    model = create_model(
        cfg=cfg,
        num_classes=train_db.get_num_classes(),
        max_decoding_steps=train_db.max_transcript_length + 1,
        # plus one is because of EOS
        input_feature_size=train_db.feat_dim,
    )

    test_evaluator = MuConEvaluator(
        cfg=cfg, test_db=test_db, model=model, device=cfg.system.device
    )

    test_evaluator.set_name("test_eval")

    evaluators = [test_evaluator]
    trainer = SWSLTrainer(
        cfg=cfg,
        exp_name=cfg.experiment_name,
        train_db=train_db,
        model=model,
        device=cfg.system.device,
        evaluators=evaluators,
        train_db_all=train_db_all,
    )

    trainer.train()
    trainer.save_training()

    # full evaluation with viterbi
    test_evaluator.viterbi_mode(True)
    evaluator_result = test_evaluator.evaluate()
    print(evaluator_result)

    # saving inside epoch folder results.
    test_evaluator.set_checkpointing_folder(trainer._get_checkpointing_folder())
    test_evaluator.save_stuff()

    # full evaluation with viterbi

    # resetting the value of the metric and saving.
    trainer.metrics[trainer.eval_metric_name_format.format(1)].set_value(
        evaluator_result, trainer.epoch_num
    )
    trainer.metrics[trainer.eval_metric_name_format.format(1)].save()


if __name__ == "__main__":
    main()
