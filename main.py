import os
from datetime import datetime

import dask.array as da
import hydra
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import xarray as xr
from hydra.utils import to_absolute_path
from lightning.pytorch import LightningDataModule
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import PowerTransformer
from dask.array import coarsen

torch.set_float32_matmul_precision('medium') 

try:
    import wandb  # Optional, for logging to Weights & Biases
except ImportError:
    wandb = None

from src.models import get_model
from src.utils import (
    Normalizer,
    calculate_weighted_metric,
    convert_predictions_to_kaggle_format,
    create_climate_data_array,
    create_comparison_plots,
    get_lat_weights,
    get_logger,
    get_trainer_config,
)


# Setup logging
log = get_logger(__name__)


# --- Data Handling ---


# Dataset to precompute all tensors during initialization
class ClimateDataset(Dataset):
    def __init__(self, inputs_norm_dask, outputs_dask, output_is_normalized=True):
        # Store dataset size
        self.size = inputs_norm_dask.shape[0]

        # Log once with basic information
        log.info(
            f"Creating dataset: {self.size} samples, input shape: {inputs_norm_dask.shape}, normalized output: {output_is_normalized}"
        )

        # Precompute all tensors in one go
        inputs_np = inputs_norm_dask.compute()
        outputs_np = outputs_dask.compute()

        # Convert to PyTorch tensors
        self.input_tensors = torch.from_numpy(inputs_np).float()
        self.output_tensors = torch.from_numpy(outputs_np).float()

        # Handle NaN values (should not occur)
        if torch.isnan(self.input_tensors).any() or torch.isnan(self.output_tensors).any():
            raise ValueError("NaN values detected in dataset tensors")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.input_tensors[idx], self.output_tensors[idx]


def _load_process_ssp_data(ds, ssp, input_variables, output_variables, member_id, spatial_template):
    """
    Loads and processes input and output variables for a single SSP using Dask.

    Args:
        ds (xr.Dataset): The opened xarray dataset.
        ssp (str): The SSP identifier (e.g., 'ssp126').
        input_variables (list): List of input variable names.
        output_variables (list): List of output variable names.
        member_id (int): The member ID to select.
        spatial_template (xr.DataArray): A template DataArray with ('y', 'x') dimensions
                                          for broadcasting global variables.

    Returns:
        tuple: (input_dask_array, output_dask_array)
               - input_dask_array: Stacked dask array of inputs (time, channels, y, x).
               - output_dask_array: Stacked dask array of outputs (time, channels, y, x).
    """
    ssp_input_dasks = []
    for var in input_variables:
        da_var = ds[var].sel(ssp=ssp)
        # Rename spatial dims if needed
        if "latitude" in da_var.dims:
            da_var = da_var.rename({"latitude": "y", "longitude": "x"})
        # Select member if applicable
        if "member_id" in da_var.dims:
            da_var = da_var.sel(member_id=member_id)

        # Process based on dimensions
        if set(da_var.dims) == {"time"}:  # Global variable, broadcast to spatial dims:
            # Broadcast like template, then transpose to ensure ('time', 'y', 'x')
            da_var_expanded = da_var.broadcast_like(spatial_template).transpose("time", "y", "x")
            ssp_input_dasks.append(da_var_expanded.data)
        elif set(da_var.dims) == {"time", "y", "x"}:  # Spatially resolved
            ssp_input_dasks.append(da_var.data)
        else:
            raise ValueError(f"Unexpected dimensions for variable {var} in SSP {ssp}: {da_var.dims}")

    # Stack inputs along channel dimension -> dask array (time, channels, y, x)
    stacked_input_dask = da.stack(ssp_input_dasks, axis=1)

    # Prepare output dask arrays for each output variable
    output_dasks = []
    for var in output_variables:
        da_output = ds[var].sel(ssp=ssp, member_id=member_id)
        # Ensure output also uses y, x if necessary
        if "latitude" in da_output.dims:
            da_output = da_output.rename({"latitude": "y", "longitude": "x"})

        # Add time, y, x dimensions as a dask array
        output_dasks.append(da_output.data)

    # Stack outputs along channel dimension -> dask array (time, channels, y, x)
    stacked_output_dask = da.stack(output_dasks, axis=1)
    return stacked_input_dask, stacked_output_dask


class ClimateEmulationDataModule(LightningDataModule):
    def __init__(
        self,
        path: str,
        input_vars: list,
        output_vars: list,
        train_ssps: list,
        test_ssp: str,
        target_member_id: int,
        member_ids: list = None,
        test_months: int = 360,
        batch_size: int = 32,
        eval_batch_size: int = None,
        num_workers: int = 0,
        seed: int = 42,
        # new flags
        pr_transform: bool = False,
        month_encoding: bool = False,
        derived_features: bool = False,
        spatial_smoothing: bool = False,
        robust_scaling: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.path = to_absolute_path(path)
        # flags
        self.pr_transform     = pr_transform
        self.month_encoding   = month_encoding
        self.derived_features = derived_features
        self.spatial_smoothing = spatial_smoothing
        self.robust_scaling   = robust_scaling
        self.normalizer       = Normalizer()

        if eval_batch_size is None:
            self.hparams.eval_batch_size = batch_size
        if self.hparams.member_ids is None:
            self.hparams.member_ids = [self.hparams.target_member_id]

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.lat_coords, self.lon_coords, self._lat_weights_da = None, None, None

    def prepare_data(self):
        if not os.path.exists(self.hparams.path):
            raise FileNotFoundError(f"Data path not found: {self.hparams.path}")
        log.info(f"Data found at: {self.hparams.path}")

    def setup(self, stage: str | None = None):
        log.info(f"Setting up data module for stage: {stage} from {self.hparams.path}")

        with xr.open_zarr(
            self.hparams.path,
            consolidated=True,
            chunks={"time": 24},
        ) as ds:
            log.info(f"data_vars: {list(ds.data_vars.keys())}")

            # spatial template
            spatial_template_da = ds["rsdt"].isel(time=0, ssp=0, drop=True)
            ny, nx = spatial_template_da.shape

            # --- Month encoding ---
            if self.month_encoding:
                month_vals = ds["time"].dt.month.data  # (T,)
                angles    = 2 * np.pi * (month_vals - 1) / 12
                base_sin  = da.from_array(np.sin(angles)[:,None,None,None], chunks=(24,1,1,1))
                base_cos  = da.from_array(np.cos(angles)[:,None,None,None], chunks=(24,1,1,1))
                ones      = da.ones((1,1,ny,nx), chunks=(1,1,ny,nx))
                sin_da    = base_sin * ones  # (T,1,ny,nx)
                cos_da    = base_cos * ones
                log.info("Applied month sine/cosine encoding.")
            else:
                log.info("Skipping month encoding.")
                sin_da = cos_da = None

            # prepare lists
            train_inputs_dask_list, train_outputs_dask_list = [], []
            train_sin_list, train_cos_list = [], []
            val_ssp, val_months = "ssp370", 120
            val_input_list, val_output_list = [], []
            val_sin_list, val_cos_list = [], []

            for ssp in self.hparams.train_ssps:
                for member in self.hparams.member_ids:
                    log.info(f"Loading ssp: '{ssp}' from member_id '{member}'")
                    ssp_in, ssp_out = _load_process_ssp_data(
                        ds,
                        ssp,
                        self.hparams.input_vars,
                        self.hparams.output_vars,
                        member,
                        spatial_template_da,
                    )

                    # --- Derived features ---
                    if self.derived_features:
                        delta = da.concatenate([ssp_in[0:1], ssp_in[1:] - ssp_in[:-1]], axis=0)
                        cumsum = da.cumsum(ssp_in, axis=0)
                        ssp_in = da.concatenate([ssp_in, delta, cumsum], axis=1)
                        log.info("Appended derived features (Δ, cumsum).")
                    else:
                        log.info("Skipping derived features.")

                    # split train/val
                    if ssp == val_ssp:
                        # val
                        val_input_list.append(ssp_in[-val_months:])
                        if self.month_encoding:
                            val_sin_list.append(sin_da[-val_months:])
                            val_cos_list.append(cos_da[-val_months:])
                        val_output_list.append(ssp_out[-val_months:])
                        # train
                        train_inputs_dask_list.append(ssp_in[:-val_months])
                        if self.month_encoding:
                            train_sin_list.append(sin_da[:-val_months])
                            train_cos_list.append(cos_da[:-val_months])
                        train_outputs_dask_list.append(ssp_out[:-val_months])
                    else:
                        train_inputs_dask_list.append(ssp_in)
                        if self.month_encoding:
                            train_sin_list.append(sin_da)
                            train_cos_list.append(cos_da)
                        train_outputs_dask_list.append(ssp_out)

            # concat train/val inputs
            train_input_dask = da.concatenate(train_inputs_dask_list, axis=0)
            val_input_dask   = da.concatenate(val_input_list,         axis=0)
            if self.month_encoding:
                train_input_dask = da.concatenate([train_input_dask,
                                                  da.concatenate(train_sin_list,axis=0),
                                                  da.concatenate(train_cos_list,axis=0)], axis=1)
                val_input_dask   = da.concatenate([val_input_dask,
                                                  da.concatenate(val_sin_list,axis=0),
                                                  da.concatenate(val_cos_list,axis=0)], axis=1)

            train_output_dask = da.concatenate(train_outputs_dask_list, axis=0)
            val_output_dask   = da.concatenate(val_output_list,         axis=0)

            # --- PR transform ---
            if "pr" in self.hparams.output_vars and self.pr_transform:
                log.info("Log-transforming 'pr' on train and val.")
                pr_idx = self.hparams.output_vars.index("pr")
                pr_train = train_output_dask[:, pr_idx:pr_idx+1]
                train_output_dask = da.concatenate([
                    train_output_dask[:, :pr_idx],
                    da.log1p(pr_train),
                    train_output_dask[:, pr_idx+1:]
                ], axis=1)
                pr_val = val_output_dask[:, pr_idx:pr_idx+1]
                val_output_dask = da.concatenate([
                    val_output_dask[:, :pr_idx],
                    da.log1p(pr_val),
                    val_output_dask[:, pr_idx+1:]
                ], axis=1)
            else:
                log.info("Skipping log-transform of 'pr'.")

            # --- Spatial smoothing ---
            if self.spatial_smoothing:
                log.info("Applying spatial smoothing.")
                coarse_train = coarsen(np.mean, train_input_dask, {2:2,3:2}, trim_excess=True)
                train_input_dask = da.concatenate([train_input_dask,
                                                  coarse_train.repeat(2,axis=2).repeat(2,axis=3)], axis=1)
                coarse_val = coarsen(np.mean, val_input_dask, {2:2,3:2}, trim_excess=True)
                val_input_dask = da.concatenate([val_input_dask,
                                                coarse_val.repeat(2,axis=2).repeat(2,axis=3)], axis=1)
            else:
                log.info("Skipping spatial smoothing.")

            # --- Input scaling ---
            arr_np = train_input_dask.compute()
            if self.robust_scaling:
                log.info("Using median+IQR for input scaling.")
                med = np.nanmedian(arr_np, axis=0, keepdims=True)
                q1  = np.nanpercentile(arr_np, 25, axis=0, keepdims=True)
                q3  = np.nanpercentile(arr_np, 75, axis=0, keepdims=True)
                iqr = np.where(q3 - q1 > 0, q3 - q1, 1.0)
                self.normalizer.set_input_statistics(mean=med, std=iqr)
            else:
                log.info("Using mean+std for input scaling.")
                m   = np.nanmean(arr_np, axis=0, keepdims=True)
                s   = np.nanstd(arr_np, axis=0, keepdims=True)
                s[s==0] = 1.0
                self.normalizer.set_input_statistics(mean=m, std=s)

            # --- Output scaling ---
            out_mean = da.nanmean(train_output_dask, axis=(0,2,3), keepdims=True).compute()
            out_std  = da.nanstd (train_output_dask, axis=(0,2,3), keepdims=True).compute()
            self.normalizer.set_output_statistics(mean=out_mean, std=out_std)

            # normalize
            train_input_norm_dask  = self.normalizer.normalize(train_input_dask, data_type="input")
            train_output_norm_dask = self.normalizer.normalize(train_output_dask, data_type="output")
            val_input_norm_dask    = self.normalizer.normalize(val_input_dask,    data_type="input")
            val_output_norm_dask   = self.normalizer.normalize(val_output_dask,   data_type="output")

            # --- Prepare test ---
            full_in, full_out = _load_process_ssp_data(
                ds, self.hparams.test_ssp, self.hparams.input_vars,
                self.hparams.output_vars, self.hparams.target_member_id,
                spatial_template_da)
            slice_t = slice(-self.hparams.test_months, None)
            test_in = full_in[slice_t]; test_out = full_out[slice_t]

            # derived
            if self.derived_features:
                d = da.concatenate([test_in[0:1], test_in[1:] - test_in[:-1]], axis=0)
                c = da.cumsum(test_in, axis=0)
                test_in = da.concatenate([test_in, d, c], axis=1)
            # month encoding
            if self.month_encoding:
                test_in = da.concatenate([test_in,
                                          sin_da[slice_t],
                                          cos_da[slice_t]], axis=1)
            # smoothing
            if self.spatial_smoothing:
                c_t = coarsen(np.mean, test_in, {2:2,3:2}, trim_excess=True)
                test_in = da.concatenate([test_in,
                                          c_t.repeat(2,2).repeat(2,3)], axis=1)

            test_input_norm_dask = self.normalizer.normalize(test_in, data_type="input")
            test_output_raw_dask = test_out

        self.train_dataset = ClimateDataset(train_input_norm_dask, train_output_norm_dask, True)
        self.val_dataset   = ClimateDataset(val_input_norm_dask,   val_output_norm_dask,   True)
        self.test_dataset  = ClimateDataset(test_input_norm_dask,  test_output_raw_dask,   False)

        log.info(f"Datasets created. Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")


    # Common DataLoader configuration
    def _get_dataloader_kwargs(self, is_train=False):
        """Return common DataLoader configuration as a dictionary"""
        return {
            "batch_size": self.hparams.batch_size if is_train else self.hparams.eval_batch_size,
            "shuffle": is_train,  # Only shuffle training data
            "num_workers": self.hparams.num_workers,
            "persistent_workers": self.hparams.num_workers > 0,
            "pin_memory": True,
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self._get_dataloader_kwargs(is_train=True))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self._get_dataloader_kwargs(is_train=False))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self._get_dataloader_kwargs(is_train=False))

    def get_lat_weights(self):
        """
        Returns area weights for the latitude dimension as an xarray DataArray.
        The weights can be used with xarray's weighted method for proper spatial averaging.
        """
        if self._lat_weights_da is None:
            with xr.open_zarr(self.hparams.path, consolidated=True) as ds:
                template = ds["rsdt"].isel(time=0, ssp=0)
                y_coords = template.y.values

                # Calculate weights based on cosine of latitude
                weights = get_lat_weights(y_coords)

                # Create DataArray with proper dimensions
                self._lat_weights_da = xr.DataArray(weights, dims=["y"], coords={"y": y_coords}, name="area_weights")

        return self._lat_weights_da

    def get_coords(self):
        """
        Returns the y and x coordinates (representing latitude and longitude).

        Returns:
            tuple: (y array, x array)
        """
        if self.lat_coords is None or self.lon_coords is None:
            # Get coordinates if they haven't been stored yet
            with xr.open_zarr(self.hparams.path, consolidated=True) as ds:
                template = ds["rsdt"].isel(time=0, ssp=0, drop=True)
                self.lat_coords = template.y.values
                self.lon_coords = template.x.values

        return self.lat_coords, self.lon_coords

# --- PyTorch Lightning Module ---
class ClimateEmulationModule(pl.LightningModule):
    def __init__(
        self, 
        model: nn.Module, 
        loss: str = "MSE",
        loss_params = None,
        optimizer: str = "",
        learning_rate: float = 1e-4, 
        scheduler: str = "", 
        scheduler_params = None,
        edge_weight: float = 0.1
    ):
        super().__init__()
        self.model = model
        # Access hyperparams via self.hparams object after saving, e.g., self.hparams.learning_rate
        self.save_hyperparameters(ignore=["model"])
        if loss == "L1":
            log.info("Using L1 loss")
            self.criterion = nn.L1Loss()
        elif loss == "SmoothL1":
            if loss_params:
                beta = loss_params.get("beta", 0.5)
            else:
                beta =  0.5
            log.info(f"Using Smooth L1 loss w/ beta {beta}")
            self.criterion = nn.SmoothL1Loss(beta)
        else: 
            log.info("Defaulted to MSE loss")
            self.criterion = nn.MSELoss()

        self.normalizer = None
        # Store evaluation outputs for time-mean calculation
        self.test_step_outputs = []
        self.validation_step_outputs = []
        if scheduler and scheduler_params: 
            self.scheduler = scheduler
            self.scheduler_params = scheduler_params

        # Save Edge Weight 
        self.edge_weight = edge_weight
        
        # Prepare Sobel filters as buffers
        sobel = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel.unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', sobel.t().unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self) -> None:
        self.normalizer = self.trainer.datamodule.normalizer  # Access the normalizer from the datamodule

    def gradient_loss(self, pred, tgt):
        # apply Sobel per-channel via grouped convolution
        C = pred.shape[1]
        sx = self.sobel_x.repeat(C, 1, 1, 1)
        sy = self.sobel_y.repeat(C, 1, 1, 1)
        gx_pred = F.conv2d(pred, sx, padding=1, groups=C)
        gx_tgt  = F.conv2d(tgt,  sx, padding=1, groups=C)
        gy_pred = F.conv2d(pred, sy, padding=1, groups=C)
        gy_tgt  = F.conv2d(tgt,  sy, padding=1, groups=C)
        return ((gx_pred - gx_tgt).abs() + (gy_pred - gy_tgt).abs()).mean()

    def training_step(self, batch, batch_idx):
        # report loss in transformed space
        x, y_true_norm = batch
        y_pred_norm = self(x)

        main_loss = self.criterion(y_pred_norm, y_true_norm)
        edge_loss = self.gradient_loss(y_pred_norm, y_true_norm)  / (torch.mean(torch.abs(y_true_norm)) + 1e-6)
        loss = main_loss + (self.edge_weight * edge_loss)

        self.log("train/main_loss", main_loss, prog_bar=True, batch_size=x.size(0))
        self.log("train/edge_loss", edge_loss, prog_bar=True, batch_size=x.size(0))
        self.log("train/loss", loss, prog_bar=True, batch_size=x.size(0))

        # loss = self.criterion(y_pred_norm, y_true_norm)
        # self.log("train/loss", loss, prog_bar=True, batch_size=x.size(0))

        return loss

    def validation_step(self, batch, batch_idx):
        # report loss in transformed space
        x, y_true_log_norm = batch
        y_pred_log_norm = self(x)
        loss = self.criterion(y_pred_log_norm, y_true_log_norm)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0), sync_dist=True)

        # save outputs for decadal mean/stddev calculation in validation_epoch_end
        # 1) de-norm
        y_pred_log = self.normalizer.inverse_transform_output(y_pred_log_norm.cpu().numpy())
        y_true_log  = self.normalizer.inverse_transform_output(y_true_log_norm.cpu().numpy())

        # 2) de-log
        if "pr" in self.trainer.datamodule.hparams.output_vars:
            pr_idx = self.trainer.datamodule.hparams.output_vars.index("pr")
            y_pred_log[:, pr_idx] = np.expm1(y_pred_log[:, pr_idx])
            y_true_log[:, pr_idx] = np.expm1(y_true_log[:, pr_idx])

        # 3) evaluate in un-transformed space
        self.validation_step_outputs.append((y_pred_log, y_true_log))
        return loss

    def _evaluate_predictions(self, predictions, targets, is_test=False):
        """
        Helper method to evaluate predictions against targets with climate metrics.

        Args:
            predictions (np.ndarray): Prediction array with shape (time, channels, y, x)
            targets (np.ndarray): Target array with shape (time, channels, y, x)
            is_test (bool): Whether this is being called from test phase (vs validation)
        """
        phase = "test" if is_test else "val"
        log_kwargs = {"prog_bar": not is_test, "sync_dist": not is_test}
        current_epoch = self.current_epoch

        # Get number of evaluation timesteps
        n_timesteps = predictions.shape[0]

        # Get area weights for proper spatial averaging
        area_weights = self.trainer.datamodule.get_lat_weights()

        # Get coordinates
        lat_coords, lon_coords = self.trainer.datamodule.get_coords()
        time_coords = np.arange(n_timesteps)
        output_vars = self.trainer.datamodule.hparams.output_vars

        # Process each output variable
        for i, var_name in enumerate(output_vars):
            # Extract channel data
            preds_var = predictions[:, i, :, :]
            trues_var = targets[:, i, :, :]

            var_unit = "mm/day" if var_name == "pr" else "K" if var_name == "tas" else "unknown"

            # Create xarray objects for weighted calculations
            preds_xr = create_climate_data_array(
                preds_var, time_coords, lat_coords, lon_coords, var_name=var_name, var_unit=var_unit
            )
            trues_xr = create_climate_data_array(
                trues_var, time_coords, lat_coords, lon_coords, var_name=var_name, var_unit=var_unit
            )

            # 1. Calculate weighted month-by-month RMSE over all samples
            diff_squared = (preds_xr - trues_xr) ** 2
            overall_rmse = calculate_weighted_metric(diff_squared, area_weights, ("time", "y", "x"), "rmse")
            self.log(f"{phase}/{var_name}/avg/monthly_rmse", float(overall_rmse), **log_kwargs)

            # 2. Calculate time-mean (i.e. decadal, 120 months average) and calculate area-weighted RMSE for time means
            pred_time_mean = preds_xr.mean(dim="time")
            true_time_mean = trues_xr.mean(dim="time")
            mean_diff_squared = (pred_time_mean - true_time_mean) ** 2
            time_mean_rmse = calculate_weighted_metric(mean_diff_squared, area_weights, ("y", "x"), "rmse")
            self.log(f"{phase}/{var_name}/time_mean_rmse", float(time_mean_rmse), **log_kwargs)

            # 3. Calculate time-stddev (temporal variability) and calculate area-weighted MAE for time stddevs
            pred_time_std = preds_xr.std(dim="time")
            true_time_std = trues_xr.std(dim="time")
            std_abs_diff = np.abs(pred_time_std - true_time_std)
            time_std_mae = calculate_weighted_metric(std_abs_diff, area_weights, ("y", "x"), "mae")
            self.log(f"{phase}/{var_name}/time_stddev_mae", float(time_std_mae), **log_kwargs)

            # Extra logging of sample predictions/images to wandb for test phase (feel free to use this for validation)
            if is_test:
                # Generate visualizations for test phase when using wandb
                if isinstance(self.logger, WandbLogger):
                    print("Logging everything in wandb")
                    # Time mean visualization
                    fig = create_comparison_plots(
                        true_time_mean,
                        pred_time_mean,
                        title_prefix=f"{var_name} Mean",
                        metric_value=time_mean_rmse,
                        metric_name="Weighted RMSE",
                    )
                    self.logger.experiment.log({f"img/{var_name}/time_mean": wandb.Image(fig)})
                    plt.close(fig)

                    # Time standard deviation visualization
                    fig = create_comparison_plots(
                        true_time_std,
                        pred_time_std,
                        title_prefix=f"{var_name} Stddev",
                        metric_value=time_std_mae,
                        metric_name="Weighted MAE",
                        cmap="plasma",
                    )
                    self.logger.experiment.log({f"img/{var_name}/time_Stddev": wandb.Image(fig)})
                    plt.close(fig)

                    # Sample timesteps visualization
                    if n_timesteps > 3:
                        timesteps = np.random.choice(n_timesteps, 3, replace=False)
                        for t in timesteps:
                            true_t = trues_xr.isel(time=t)
                            pred_t = preds_xr.isel(time=t)
                            fig = create_comparison_plots(true_t, pred_t, title_prefix=f"{var_name} Timestep {t}")
                            self.logger.experiment.log({f"img/{var_name}/month_idx_{t}": wandb.Image(fig)})
                            plt.close(fig)
            elif (not is_test) and ((current_epoch == 0) or (current_epoch == self.trainer.max_epochs - 1)):
                print("Logging validation stuff")
                prefix = f"val_img_{current_epoch}/{var_name}"
                # 1) time‐mean
                fig = create_comparison_plots(
                    true_time_mean,
                    pred_time_mean,
                    title_prefix=f"{var_name} Val Mean (Epoch {current_epoch})",
                    metric_value=time_mean_rmse,
                    metric_name="Weighted RMSE",
                )
                try:
                    if isinstance(self.logger, WandbLogger):
                        self.logger.experiment.log({f"{prefix}/time_mean": wandb.Image(fig)})
                except Exception as e:
                    fig.savefig(f"{prefix.replace('/', '_')}_time_mean.png")
                plt.close(fig)

                # 2) time‐stddev
                fig = create_comparison_plots(
                    true_time_std,
                    pred_time_std,
                    title_prefix=f"{var_name} Val Stddev (Epoch {current_epoch})",
                    metric_value=time_std_mae,
                    metric_name="Weighted MAE",
                    cmap="plasma",
                )
                try:
                    if isinstance(self.logger, WandbLogger):
                        self.logger.experiment.log({f"{prefix}/time_stddev": wandb.Image(fig)})
                except Exception as e:
                    fig.savefig(f"{prefix.replace('/', '_')}_time_stddev.png")
                plt.close(fig)

                # 3) three random timesteps
                if n_timesteps > 3:
                    steps = np.random.choice(n_timesteps, 3, replace=False)
                    for t in steps:
                        true_t = trues_xr.isel(time=t)
                        pred_t = preds_xr.isel(time=t)
                        fig = create_comparison_plots(
                            true_t,
                            pred_t,
                            title_prefix=f"{var_name} Val Timestep {t} (Ep {current_epoch})",
                        )
                        try:
                            if isinstance(self.logger, WandbLogger):
                                self.logger.experiment.log({f"{prefix}/timestep_{t}": wandb.Image(fig)})
                        except Exception:
                            fig.savefig(f"{prefix.replace('/', '_')}_t{t}.png")
                        plt.close(fig)

    def on_validation_epoch_end(self):
        # Compute time-mean and time-stddev errors using all validation months
        if not self.validation_step_outputs:
            return

        # Stack all predictions and ground truths
        all_preds_np = np.concatenate([pred for pred, _ in self.validation_step_outputs], axis=0)
        all_trues_np = np.concatenate([true for _, true in self.validation_step_outputs], axis=0)

        # Use the helper method to evaluate predictions
        self._evaluate_predictions(all_preds_np, all_trues_np, is_test=False)

        self.validation_step_outputs.clear()  # Clear the outputs list for next epoch

    def test_step(self, batch, batch_idx):
        x, y_true_denorm = batch
        y_pred_log_norm = self(x)

        # 1) de-norm predictions
        y_pred_log = self.normalizer.inverse_transform_output(y_pred_log_norm.cpu().numpy())

        # 2) de-log predictions
        if (
            "pr" in self.trainer.datamodule.hparams.output_vars 
            and self.trainer.datamodule.hparams.pr_transform
        ):          
            pr_idx = self.trainer.datamodule.hparams.output_vars.index("pr")
            y_pred_log[:, pr_idx] = np.expm1(y_pred_log[:, pr_idx])

        # record testing metrics and submission in untransformed space
        self.test_step_outputs.append((y_pred_log, y_true_denorm.cpu().numpy()))

    def on_test_epoch_end(self):
        # Concatenate all predictions and ground truths from each test step/batch into one array
        all_preds_denorm = np.concatenate([pred for pred, true in self.test_step_outputs], axis=0)
        all_trues_denorm = np.concatenate([true for pred, true in self.test_step_outputs], axis=0)

        # Use the helper method to evaluate predictions
        self._evaluate_predictions(all_preds_denorm, all_trues_denorm, is_test=True)

        # Save predictions for Kaggle submission. This is the file that should be uploaded to Kaggle.
        log.info("Saving Kaggle submission...")
        self._save_kaggle_submission(all_preds_denorm)

        self.test_step_outputs.clear()  # Clear the outputs list

    def _save_kaggle_submission(self, predictions, suffix=""):
        """
        Create a Kaggle submission file from the model predictions.

        Args:
            predictions (np.ndarray): Predicted values with shape (time, channels, y, x)
        """
        # Get coordinates
        lat_coords, lon_coords = self.trainer.datamodule.get_coords()
        output_vars = self.trainer.datamodule.hparams.output_vars
        n_times = predictions.shape[0]
        time_coords = np.arange(n_times)

        # Convert predictions to Kaggle format
        submission_df = convert_predictions_to_kaggle_format(
            predictions, time_coords, lat_coords, lon_coords, output_vars
        )

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = to_absolute_path(f"submissions/kaggle_submission{suffix}_{timestamp}.csv")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure directory exists
        submission_df.to_csv(filepath, index=False)

        if wandb is not None and isinstance(self.logger, WandbLogger):
            pass
            # Optionally, uncomment the following line to save the submission to the wandb cloud
            # self.logger.experiment.log_artifact(filepath)  # Log to wandb if available

        log.info(f"Kaggle submission saved to {filepath}")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if self.scheduler_params:
            log.info(f"scheduler_params: {self.scheduler_params}")
            if self.scheduler == "cosine":
                log.info("Using CosineAnnealingLR scheduler")
                sched = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.scheduler_params.T_max,
                    eta_min=self.scheduler_params.eta_min,
                )
                return [optimizer], [sched]
            else:
                log.info("No scheduler provided.")
        else:
            log.info("No scheduler_params provided")
        log.info("Using Adam optimizer")
        return optimizer
        
# --- Temporal Versions ---
# ----------------------------------------------------------------------
# Helper: turn a 4‑D ClimateDataset into a temporal window on‑the‑fly
# ----------------------------------------------------------------------
class WindowWrapperDataset(torch.utils.data.Dataset):
    """Wrap an *existing* ClimateDataset so __getitem__ returns [T,C,H,W]."""
    def __init__(self, base_ds: ClimateDataset, window: int = 3):
        self.base   = base_ds
        self.window = window

    def __len__(self):
        return len(self.base)

    @staticmethod
    def _indices(idx, k):
        left = max(0, idx - k + 1)
        seq  = list(range(left, idx + 1))
        while len(seq) < k:
            seq.insert(0, seq[0])
        return seq[-k:]

    def __getitem__(self, idx):
        if self.window == 1:
            x, y = self.base[idx]            # x:[C,H,W]
            return x.unsqueeze(0), y         # → [1,C,H,W]
        seq = self._indices(idx, self.window)
        frames = [self.base[i][0] for i in seq]   # collect frames
        _, y   = self.base[idx]                   # target only at idx
        x = torch.stack(frames, dim=0)            # [T,C,H,W]
        return x, y

# ---------------------------------------------------------------------
# Temporal variants  (keep originals unchanged)
# ---------------------------------------------------------------------
class TemporalClimateDataset(ClimateDataset):
    r"""Return a temporal window shaped [T, C, H, W] instead of [C, H, W]."""
    def __init__(self, *args, window_length: int = 3, **kw):
        super().__init__(*args, **kw)
        self.window_length = window_length

    # helper: indices for [t‑(k‑1), …, t] with left‑padding
    @staticmethod
    def _window(idx, k):
        left = max(0, idx - k + 1)
        seq  = list(range(left, idx + 1))
        while len(seq) < k:
            seq.insert(0, seq[0])
        return seq[-k:]

    def __getitem__(self, idx):
        if self.window_length == 1:
            x, y = super().__getitem__(idx)      # original [C,H,W]
            return x.unsqueeze(0), y             # -> [1,C,H,W]
        seq = self._window(idx, self.window_length)
        frames = [super().__getitem__(i)[0] for i in seq]   # list of [C,H,W]
        # (targets only from centre idx)
        _, y = super().__getitem__(idx)
        x = torch.stack(frames, dim=0)           # [T,C,H,W]
        return x, y


class TemporalClimateEmulationDataModule(ClimateEmulationDataModule):
    """Re‑uses parent setup() then wraps datasets in WindowWrapperDataset."""
    def __init__(self, *args, window_length: int = 3, **kw):
        super().__init__(*args, **kw)
        self.window_length = window_length

    def setup(self, stage=None):
        super().setup(stage)  # builds self.train/val/test_dataset (4‑D)

        # Wrap each dataset so it yields temporal windows
        self.train_dataset = WindowWrapperDataset(self.train_dataset,
                                                  window=self.window_length)
        self.val_dataset   = WindowWrapperDataset(self.val_dataset,
                                                  window=self.window_length)
        self.test_dataset  = WindowWrapperDataset(self.test_dataset,
                                                  window=self.window_length)


# --- Main Execution with Hydra ---
@hydra.main(version_base=None, config_path="configs", config_name="main_config.yaml")
def main(cfg: DictConfig):
    # Print resolved configs
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Create data module with parameters from configs
    # datamodule = ClimateEmulationDataModule(seed=cfg.seed, **cfg.data)

    # TO-DO: Fix unexpected keyword window_length when using SimpleCNN
    temporal_models = {"temporal_cnn", "st_vit"}
    dm_kwargs   = dict(cfg.data) # convert ΩConf to plain dict
    win_length  = dm_kwargs.pop("window_length", 1)
    
    datamodule = None
    if cfg.model.type in temporal_models:
        datamodule = TemporalClimateEmulationDataModule(seed=cfg.seed, window_length=win_length, **dm_kwargs)
    else:
        datamodule = ClimateEmulationDataModule(seed=cfg.seed, **dm_kwargs)

    model = get_model(cfg)

    # Create lightning module
    lightning_module = ClimateEmulationModule(
        model, 
        loss=cfg.training.loss,
        learning_rate=cfg.training.lr, 
        optimizer=cfg.training.optimizer,
        scheduler=cfg.training.scheduler,
        scheduler_params=cfg.training.scheduler_params,
        edge_weight=cfg.training.edge_weight
    )

    # Create lightning trainer
    trainer_config = get_trainer_config(cfg, model=model)
    trainer = pl.Trainer(**trainer_config)

    if cfg.ckpt_path and isinstance(cfg.ckpt_path, str):
        cfg.ckpt_path = to_absolute_path(cfg.ckpt_path)

    # Train model
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    log.info("Training finished.")

    # Log all checkpoints as W&B artifacts  ⬇⬇⬇
    if cfg.use_wandb and isinstance(trainer_config.get("logger"), WandbLogger):
        trainer.logger.experiment.save("checkpoints/**/*.ckpt")

    # Test model
    # IMPORTANT: Please note that the test metrics will be bad because the test targets have been corrupted on the public Kaggle dataset.
    # The purpose of testing below is to generate the Kaggle submission file based on your model's predictions.
    trainer_config["devices"] = 1  # Make sure you test on 1 GPU only to avoid synchronization issues with DDP
    eval_trainer = pl.Trainer(**trainer_config)
    eval_trainer.test(lightning_module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    if cfg.use_wandb and isinstance(trainer_config.get("logger"), WandbLogger):
        wandb.finish()  # Finish the run if using wandb


if __name__ == "__main__":
    main()
