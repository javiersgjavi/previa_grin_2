import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from tsl import logger
from tsl.data import ImputationDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import MetrLA, PemsBay, AirQuality
from electric_data import ElectricData
from tsl.engines import Imputer
from tsl.experiment import Experiment
from tsl.metrics import torch as torch_metrics, numpy as numpy_metrics
from tsl.nn.models import RNNImputerModel, BiRNNImputerModel, GRINModel
from tsl.ops.imputation import add_missing_values
from tsl.transforms import MaskInput
from tsl.utils.casting import torch_to_numpy


def get_model_class(model_str):
    if model_str == 'rnni':
        model = RNNImputerModel
    elif model_str == 'birnni':
        model = BiRNNImputerModel
    elif model_str == 'grin':
        model = GRINModel
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def get_dataset(dataset_name: str, p_fault=0., p_noise=0.):
    if dataset_name.startswith('air'):
        return AirQuality(impute_nans=True, small=dataset_name[3:] == '36')
    if dataset_name.endswith('_point'):
        p_fault, p_noise = 0., 0.25
        dataset_name = dataset_name[:-6]
    if dataset_name.endswith('_block'):
        p_fault, p_noise = 0.0015, 0.05
        dataset_name = dataset_name[:-6]
    if dataset_name == 'la':
        return add_missing_values(MetrLA(), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=9101112)
    if dataset_name == 'bay':
        return add_missing_values(PemsBay(), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=56789)
    if dataset_name == 'electric':
        return add_missing_values(ElectricData(normalized=True), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=56789)

    raise ValueError(f"Dataset {dataset_name} not available in this setting.")


def run_imputation(cfg: DictConfig):
    ########################################
    # data module                          #
    ########################################

    dataset = get_dataset(cfg.dataset.name,
                          p_fault=cfg.get('p_fault'),
                          p_noise=cfg.get('p_noise'))
    print(f'\n\n{dataset}')
    print(f"Sampling period: {dataset.freq}\n"
          f"Has missing values: {dataset.has_mask}\n"
          f"Percentage of missing values: {(1 - dataset.mask.mean()) * 100:.2f}%\n"
          f"Percentage of missing values val: {(dataset.eval_mask.mean()) * 100:.2f}%\n"
          f"Has dataset exogenous variables: {dataset.has_covariates}\n"
          f"Relevant attributes: {', '.join(dataset.attributes.keys())}\n\n")

    # get adjacency matrix
    adj = dataset.get_connectivity(**cfg.dataset.connectivity)

    # instantiate dataset
    torch_dataset = ImputationDataset(target=dataset.dataframe(),
                                      eval_mask=dataset.eval_mask,
                                      input_mask=dataset.training_mask,
                                      transform=MaskInput(),
                                      connectivity=adj,
                                      window=cfg.window,
                                      stride=cfg.stride)

    scalers = {
        'target': StandardScaler(axis=(0, 1))
    }

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        #scalers=scalers,
        splitter=dataset.get_splitter(**cfg.dataset.splitting),
        batch_size=cfg.batch_size,
        workers=cfg.workers
    )
    dm.setup()

    if cfg.get('in_sample', False):
        dm.trainset = list(range(len(torch_dataset)))

    ########################################
    # imputer                              #
    ########################################

    model_cls = get_model_class(cfg.model.name)

    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=torch_dataset.n_channels)

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    loss_fn = torch_metrics.MaskedMAE(compute_on_step=True)

    log_metrics = {'mae': torch_metrics.MaskedMAE(),
                   'mse': torch_metrics.MaskedMSE(),
                   #'mre': torch_metrics.MaskedMRE(),
                   #'mape': torch_metrics.MaskedMAPE()
    }

    if cfg.lr_scheduler is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    # setup imputer
    imputer = Imputer(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        loss_fn=loss_fn,
        metrics=log_metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        whiten_prob=cfg.whiten_prob,
        prediction_loss_weight=cfg.prediction_loss_weight,
        impute_only_missing=cfg.impute_only_missing,
        warm_up_steps=cfg.warm_up_steps
    )

    ########################################
    # logging options                      #
    ########################################

    if 'wandb' in cfg:
        exp_logger = WandbLogger(name=cfg.run.name,
                                 save_dir=cfg.run.dir,
                                 offline=cfg.wandb.offline,
                                 project=cfg.wandb.project)
    else:
        exp_logger = TensorBoardLogger(save_dir=cfg.run.dir, name='tensorboard')

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=cfg.patience,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    trainer = Trainer(max_epochs=cfg.epochs,
                      default_root_dir=cfg.run.dir,
                      logger=exp_logger,
                      gpus=1 if torch.cuda.is_available() else None,
                      gradient_clip_val=cfg.grad_clip_val,
                      callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(imputer, datamodule=dm)

    ########################################
    # testing                              #
    ########################################

    imputer.load_model(checkpoint_callback.best_model_path)

    imputer.freeze()
    trainer.test(imputer, datamodule=dm)

    output = trainer.predict(imputer, dataloaders=dm.test_dataloader())
    output = torch_to_numpy(output)
    y_hat, y_true, mask = output['y_hat'], \
        output['y'], \
        output.get('mask', None)
    res = dict(test_mae=numpy_metrics.mae(y_hat, y_true, mask),
               test_mre=numpy_metrics.mre(y_hat, y_true, mask),
               test_mape=numpy_metrics.mape(y_hat, y_true, mask))

    output = trainer.predict(imputer, dataloaders=dm.val_dataloader())
    output = torch_to_numpy(output)
    y_hat, y_true, mask = output['y_hat'], \
        output['y'], \
        output.get('mask', None)
    res.update(dict(val_mae=numpy_metrics.mae(y_hat, y_true, mask),
                    val_rmse=numpy_metrics.rmse(y_hat, y_true, mask),
                    val_mape=numpy_metrics.mape(y_hat, y_true, mask)))

    return res


if __name__ == '__main__':
    exp = Experiment(run_fn=run_imputation, config_path='config', config_name='default')
    res = exp.run()
    logger.info(res)
