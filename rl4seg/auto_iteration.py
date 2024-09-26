import pickle
from datetime import datetime
from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from runner import main as runner_main

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="config", config_name="auto_iteration")
def main(cfg):
    GlobalHydra.instance().clear()
    initialize(version_base=None, config_path='config')

    iterations = cfg.num_iter
    output_path = cfg.output_path
    Path(output_path + "/0/").mkdir(parents=True, exist_ok=True)
    main_overrides = [f"logger.save_dir={output_path}"]
    target_experiment = cfg.target
    source_experiment = cfg.source

    timestamp = datetime.now().timestamp()
    experiment_split_column = f"split_{timestamp}"
    experiment_gt_column = f"Gt_{timestamp}"

    # train supervised network for initial actor
    overrides = main_overrides + [f"trainer.max_epochs={cfg.sup_num_epochs}",
                                  f'model.predict_save_dir={None}',  # no predictions here
                                  f"model.ckpt_path={output_path}/{0}/actor.ckpt",
                                  f"model.loss.label_smoothing={cfg.sup_loss_label_smoothing}",
                                  f"experiment=supervised_{source_experiment}"]
    sub_cfg = compose(config_name=f"supervised_runner.yaml", overrides=overrides)
    print(OmegaConf.to_yaml(sub_cfg))

    # prepare dataset with custom split and gt column
    if experiment_split_column != sub_cfg.datamodule.splits_column:
        df = pd.read_csv(sub_cfg.datamodule.data_dir + sub_cfg.datamodule.csv_file, index_col=0)
        df[experiment_split_column] = df.loc[:, sub_cfg.datamodule.splits_column]
        df[experiment_gt_column] = df.loc[:, sub_cfg.datamodule.gt_column]
        df.to_csv(sub_cfg.datamodule.data_dir + sub_cfg.datamodule.csv_file)
    sub_cfg.datamodule.splits_column = experiment_split_column
    sub_cfg.datamodule.gt_column = experiment_gt_column
    #runner_main(sub_cfg)

    # Predict and test (baseline) on target domain
    overrides = main_overrides + cfg.rl_overrides + [f"trainer.max_epochs=0",
                                                     f"predict_subset_frac={cfg.rl_num_predict}",
                                                     f"model.actor.actor.pretrain_ckpt={output_path}/{0}/actor.ckpt",
                                                     f"model.actor.actor.ref_ckpt={output_path}/{0}/actor.ckpt",
                                                     "reward@model.reward=pixelwise_accuracy",  # will not be used
                                                     f"model.actor_save_path={output_path}/{0}/actor.ckpt",  # no need
                                                     f"model.critic_save_path=null",  # no need
                                                     f'model.predict_save_dir={output_path}',
                                                     f"experiment=ppo_{target_experiment}"
                                                     ]
    sub_cfg = compose(config_name=f"RL_runner.yaml", overrides=overrides)
    # prepare dataset with custom split and gt column
    if experiment_split_column != sub_cfg.datamodule.splits_column:
        df = pd.read_csv(sub_cfg.datamodule.data_dir + sub_cfg.datamodule.csv_file, index_col=0)
        df[experiment_split_column] = df.loc[:, sub_cfg.datamodule.splits_column]
        df[experiment_gt_column] = df.loc[:, sub_cfg.datamodule.gt_column]
        df.to_csv(sub_cfg.datamodule.data_dir + sub_cfg.datamodule.csv_file)
    sub_cfg.datamodule.splits_column = experiment_split_column
    sub_cfg.datamodule.gt_column = experiment_gt_column
    #runner_main(sub_cfg)

    for i in range(1, iterations + 1):
        # train reward net
        overrides = main_overrides + [f"trainer.max_epochs={cfg.rn_num_epochs}",
                                      f"datamodule.data_path={output_path}/",
                                      f"model.save_model_path={output_path}/{i - 1}/rewardnet.ckpt",
                                      f"+model.var_file={cfg.var_file}"]
        sub_cfg = compose(config_name=f"reward_runner.yaml", overrides=overrides)
        print(OmegaConf.to_yaml(sub_cfg))
        runner_main(sub_cfg)

        next_output_path = f'{output_path}/{i}/'
        Path(next_output_path).mkdir(parents=True, exist_ok=True)

        # TODO: MAYBE RETHINK THIS WAY OF PASSING THE VARIABLE(S)
        # load temporary variable file
        saved_vars = pickle.load(open(cfg.var_file, "rb"))

        # train PPO model with fresh reward net
        overrides = main_overrides + cfg.rl_overrides + \
                    [f"trainer.max_epochs={cfg.rl_num_epochs}",
                     f"predict_subset_frac={cfg.rl_num_predict}",
                     f"datamodule.splits_column={experiment_split_column}",
                     f"datamodule.gt_column={experiment_gt_column}",
                     f"+datamodule.train_batch_size={8 * i}",
                     f"model.actor.actor.pretrain_ckpt={output_path}/{i - 1}/actor_s-p.ckpt",
                     f"model.actor.actor.ref_ckpt={output_path}/{0}/actor.ckpt",  # always supervised
                     f"model.reward.state_dict_path={output_path}/{i - 1}/rewardnet.ckpt",
                     # f"model.reward.temp_factor={float(saved_vars['Temperature_factor'])}",
                     f"model.actor_save_path={output_path}/{i}/actor.ckpt",
                     f"model.critic_save_path={output_path}/{i}/critic.ckpt",
                     f'model.predict_save_dir={output_path if iterations > i else None}',
                     f"model.entropy_coeff={max(0.3 / (i * 2), 0)}",
                     f"model.divergence_coeff={0.1 / (i * 2)}",
                     f"experiment=ppo_{target_experiment}"
                     ]
        if Path(f"{output_path}/{i - 1}/critic.ckpt").exists():
            overrides += [f"model.actor.critic.pretrain_ckpt={output_path}/{i - 1}/critic_s-p.ckpt"]
        sub_cfg = compose(config_name=f"RL_runner.yaml", overrides=overrides)
        print(OmegaConf.to_yaml(sub_cfg))
        runner_main(sub_cfg)


if __name__ == '__main__':
    # Load any available `.env` file
    load_dotenv()
    main()
