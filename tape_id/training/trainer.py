"""
Training loop for tape parameter identification.
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import traceback
import logging


class TapeIdentificationTrainer:
    """Trainer supporting classification and regression, single/multi/triple-param.

    Detects mode automatically from controller attributes (triple_param, multi_param, regression).

    Args:
        patience: Early stopping patience (epochs without improvement).
        min_delta: Minimum loss improvement to reset patience counter.
        grad_clip_norm: Maximum gradient norm for clipping.
    """

    def __init__(
        self,
        encoder: nn.Module,
        controller: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = "cuda",
        output_dir: str = "outputs/checkpoints",
        log_dir: str = "outputs/logs",
        patience: int = 15,
        min_delta: float = 0.001,
        grad_clip_norm: float = 1.0,
        forward_model: nn.Module = None,
        signal_loss_weight: float = 0.0,
        param_loss_weight: float = 1.0,
        signal_loss_fn: nn.Module = None,
        min_param: float = 0.0,
        max_param: float = 1.0,
        min_depth: float = 0.0,
        max_depth: float = 1.0,
        min_rate: float = 0.0,
        max_rate: float = 1.0,
    ):
        self.encoder = encoder.to(device)
        self.controller = controller.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.grad_clip_norm = grad_clip_norm

        # Detect mode from controller
        self.triple_param = getattr(controller, 'triple_param', False)
        self.multi_param = getattr(controller, 'multi_param', False)
        self.regression = getattr(controller, 'regression', False)

        # Param ranges for scaled-space regression (sigmoid → scale → MSE)
        self.min_param = min_param
        self.max_param = max_param
        self.param_range = max_param - min_param
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_range = max_depth - min_depth
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.rate_range = max_rate - min_rate

        # Loss function
        self.loss_fn = nn.MSELoss() if self.regression else nn.CrossEntropyLoss()

        # Signal loss
        self.forward_model = forward_model.to(device) if forward_model is not None else None
        self.signal_loss_weight = signal_loss_weight
        self.param_loss_weight = param_loss_weight
        self.signal_loss_fn = signal_loss_fn
        self.use_signal_loss = (
            forward_model is not None
            and signal_loss_weight > 0.0
            and signal_loss_fn is not None
            and self.regression
        )

        # TensorBoard
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.global_step = 0

        # Early stopping
        self.patience = patience
        self.patience_counter = 0
        self.min_delta = min_delta

        # Logging
        log_file = self.output_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    # ------------------------------------------------------------------
    # Core forward step (shared between train and validation)
    # ------------------------------------------------------------------

    def _forward_step(self, batch):
        """Run encoder + controller on batch, compute loss and per-param metrics.

        Returns:
            loss: Scalar loss tensor
            metrics: dict of metric_name -> (sum_value, count)
        """
        # When signal loss is enabled, x_clean is prepended to the batch tuple
        if self.use_signal_loss:
            x_clean = batch[0].to(self.device)
            batch = batch[1:]
        else:
            x_clean = None

        if self.triple_param:
            y, target_ja, target_d, target_r = batch
            y = y.to(self.device)
            target_ja = target_ja.to(self.device).unsqueeze(1)
            target_d = target_d.to(self.device).unsqueeze(1)
            target_r = target_r.to(self.device).unsqueeze(1)

            pred = self.controller(self.encoder(y))
            pred_ja_norm = torch.sigmoid(pred["ja"])
            pred_d_norm = torch.sigmoid(pred["depth"])
            pred_r_norm = torch.sigmoid(pred["rate"])

            # Scale to physical units
            pred_ja_sc = pred_ja_norm * self.param_range + self.min_param
            pred_d_sc = pred_d_norm * self.depth_range + self.min_depth
            pred_r_sc = pred_r_norm * self.rate_range + self.min_rate
            target_ja_sc = target_ja * self.param_range + self.min_param
            target_d_sc = target_d * self.depth_range + self.min_depth
            target_r_sc = target_r * self.rate_range + self.min_rate

            loss = (self.loss_fn(pred_ja_sc, target_ja_sc)
                    + self.loss_fn(pred_d_sc, target_d_sc)
                    + self.loss_fn(pred_r_sc, target_r_sc))

            if self.use_signal_loss and x_clean is not None:
                pred_norm = {
                    "ja":    pred_ja_norm,
                    "depth": pred_d_norm,
                    "rate":  pred_r_norm,
                }
                y_rec = self.forward_model(x_clean, pred_norm)
                sig_loss = self.signal_loss_fn(y_rec.squeeze(1), y.squeeze(1))
                loss = self.param_loss_weight * loss + self.signal_loss_weight * sig_loss

            n = target_ja.size(0)
            diff_ja = pred_ja_sc - target_ja_sc
            diff_d = pred_d_sc - target_d_sc
            diff_r = pred_r_sc - target_r_sc
            return loss, {
                "ae_ja": (diff_ja.abs().sum().item(), n),
                "ae_depth": (diff_d.abs().sum().item(), n),
                "ae_rate": (diff_r.abs().sum().item(), n),
                "se_ja": ((diff_ja ** 2).sum().item(), n),
                "se_depth": ((diff_d ** 2).sum().item(), n),
                "se_rate": ((diff_r ** 2).sum().item(), n),
            }

        elif self.regression and self.multi_param:
            y, target_d, target_r = batch
            y = y.to(self.device)
            target_d = target_d.to(self.device).unsqueeze(1)
            target_r = target_r.to(self.device).unsqueeze(1)

            pred = self.controller(self.encoder(y))
            pred_d_norm = torch.sigmoid(pred["depth"])
            pred_r_norm = torch.sigmoid(pred["rate"])

            pred_d_sc = pred_d_norm * self.depth_range + self.min_depth
            pred_r_sc = pred_r_norm * self.rate_range + self.min_rate
            target_d_sc = target_d * self.depth_range + self.min_depth
            target_r_sc = target_r * self.rate_range + self.min_rate

            loss = self.loss_fn(pred_d_sc, target_d_sc) + self.loss_fn(pred_r_sc, target_r_sc)

            if self.use_signal_loss and x_clean is not None:
                pred_norm = {
                    "depth": pred_d_norm,
                    "rate":  pred_r_norm,
                }
                y_rec = self.forward_model(x_clean, pred_norm)
                sig_loss = self.signal_loss_fn(y_rec.squeeze(1), y.squeeze(1))
                loss = self.param_loss_weight * loss + self.signal_loss_weight * sig_loss

            n = target_d.size(0)
            diff_d = pred_d_sc - target_d_sc
            diff_r = pred_r_sc - target_r_sc
            return loss, {
                "ae_depth": (diff_d.abs().sum().item(), n),
                "ae_rate": (diff_r.abs().sum().item(), n),
                "se_depth": ((diff_d ** 2).sum().item(), n),
                "se_rate": ((diff_r ** 2).sum().item(), n),
            }

        elif self.regression:
            y, target = batch
            y = y.to(self.device)
            target = target.to(self.device).unsqueeze(1)

            raw = self.controller(self.encoder(y))
            pred_norm = torch.sigmoid(raw)

            # Predict and compute loss in scaled (physical) space
            pred_scaled = pred_norm * self.param_range + self.min_param
            target_scaled = target * self.param_range + self.min_param
            loss = self.loss_fn(pred_scaled, target_scaled)

            if self.use_signal_loss and x_clean is not None:
                y_rec = self.forward_model(x_clean, pred_norm)
                sig_loss = self.signal_loss_fn(y_rec.squeeze(1), y.squeeze(1))
                loss = self.param_loss_weight * loss + self.signal_loss_weight * sig_loss

            n = target.size(0)
            diff = pred_scaled - target_scaled
            return loss, {
                "ae": (diff.abs().sum().item(), n),
                "se": ((diff ** 2).sum().item(), n),
            }

        elif self.multi_param:
            y, depth_idx, rate_idx = batch
            y = y.to(self.device)
            depth_idx = depth_idx.to(self.device)
            rate_idx = rate_idx.to(self.device)

            logits = self.controller(self.encoder(y))
            loss = self.loss_fn(logits["depth"], depth_idx) + self.loss_fn(logits["rate"], rate_idx)

            n = depth_idx.size(0)
            preds_d = torch.argmax(logits["depth"], dim=-1)
            preds_r = torch.argmax(logits["rate"], dim=-1)
            return loss, {
                "correct_depth": ((preds_d == depth_idx).sum().item(), n),
                "correct_rate": ((preds_r == rate_idx).sum().item(), n),
                "correct_both": (((preds_d == depth_idx) & (preds_r == rate_idx)).sum().item(), n),
            }

        else:
            y, class_idx = batch
            y = y.to(self.device)
            class_idx = class_idx.to(self.device)

            logits = self.controller(self.encoder(y))
            loss = self.loss_fn(logits, class_idx)

            n = class_idx.size(0)
            preds = torch.argmax(logits, dim=-1)
            return loss, {
                "correct": ((preds == class_idx).sum().item(), n),
            }

    # ------------------------------------------------------------------
    # Metric accumulation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _accumulate_metrics(accum, step_metrics):
        """Add step metrics into accumulator."""
        for key, (val, count) in step_metrics.items():
            if key not in accum:
                accum[key] = [0.0, 0]
            accum[key][0] += val
            accum[key][1] += count

    def _compute_ratios(self, accum):
        """Compute ratio (sum / count) for each accumulated metric."""
        return {k: v[0] / v[1] if v[1] > 0 else 0.0 for k, v in accum.items()}

    def _format_pbar(self, loss_val, ratios):
        """Format progress bar postfix from metric ratios."""
        postfix = {"loss": f"{loss_val:.4f}"}
        for k, v in ratios.items():
            short = k.replace("correct_", "").replace("ae_", "").replace("se_", "")
            if k.startswith("ae_") or k.startswith("correct"):
                postfix[short] = f"{v:.3f}"
        return postfix

    def _log_epoch_metrics(self, prefix, ratios, epoch):
        """Write metrics to TensorBoard."""
        for k, v in ratios.items():
            if k.startswith("ae_"):
                self.writer.add_scalar(f"{prefix}/mae_{k[3:]}", v, epoch)
            elif k.startswith("se_"):
                self.writer.add_scalar(f"{prefix}/rmse_{k[3:]}", math.sqrt(v), epoch)
            elif k.startswith("correct"):
                name = k.replace("correct_", "accuracy_").replace("correct", "accuracy")
                self.writer.add_scalar(f"{prefix}/{name}", v, epoch)

    # ------------------------------------------------------------------
    # Training and validation loops
    # ------------------------------------------------------------------

    def train_epoch(self) -> float:
        """Run one training epoch."""
        self.encoder.train()
        self.controller.train()

        total_loss = 0.0
        accum = {}
        all_params = list(self.encoder.parameters()) + list(self.controller.parameters())
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            loss, step_metrics = self._forward_step(batch)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=self.grad_clip_norm)
            self.optimizer.step()

            total_loss += loss.item()
            self._accumulate_metrics(accum, step_metrics)
            self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)
            self.global_step += 1

            ratios = self._compute_ratios(accum)
            pbar.set_postfix(self._format_pbar(loss.item(), ratios))

        ratios = self._compute_ratios(accum)
        self._log_epoch_metrics("train", ratios, self.current_epoch)
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation."""
        self.encoder.eval()
        self.controller.eval()

        total_loss = 0.0
        accum = {}

        for batch in tqdm(self.val_loader, desc="Validation"):
            loss, step_metrics = self._forward_step(batch)
            total_loss += loss.item()
            self._accumulate_metrics(accum, step_metrics)

        avg_loss = total_loss / len(self.val_loader)
        ratios = self._compute_ratios(accum)
        self.writer.add_scalar("val/loss", avg_loss, self.current_epoch)
        self._log_epoch_metrics("val", ratios, self.current_epoch)

        # Print validation summary
        parts = []
        for k, v in ratios.items():
            if k.startswith("ae_"):
                parts.append(f"MAE_{k[3:]}={v:.4f}")
            elif k.startswith("se_"):
                parts.append(f"RMSE_{k[3:]}={math.sqrt(v):.4f}")
            elif "correct" in k:
                name = k.replace("correct_", "acc_").replace("correct", "acc")
                parts.append(f"{name}={v:.3f}")
        if parts:
            print(f"  val: {' | '.join(parts)}")

        return avg_loss

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def save_checkpoint(self, is_best=False, is_last=False, emergency=False):
        """Save checkpoint (only best, last, or emergency)."""
        checkpoint = {
            "epoch": self.current_epoch,
            "encoder_state": self.encoder.state_dict(),
            "controller_state": self.controller.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
        }

        if emergency:
            path = self.output_dir / f"checkpoint_epoch{self.current_epoch}_emergency.pt"
            torch.save(checkpoint, path)
            logging.warning(f"Saved emergency checkpoint to {path}")

        if is_best:
            path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, path)
            print(f"Saved best model (epoch {self.current_epoch}, val_loss={self.best_val_loss:.4f})")

        if is_last:
            path = self.output_dir / "last_model.pt"
            torch.save(checkpoint, path)
            print(f"Saved last model (epoch {self.current_epoch})")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint["encoder_state"])
        self.controller.load_state_dict(checkpoint["controller_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        self.current_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint["best_val_loss"]
        self.global_step = checkpoint.get("global_step", 0)

        logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        logging.info(f"Resuming from epoch {self.current_epoch}")
        logging.info(f"Best val loss so far: {self.best_val_loss:.4f}")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """Run full training loop."""
        if self.triple_param:
            mode = "regression triple-param (ja + depth + rate)"
        elif self.regression:
            mode = "regression multi-param" if self.multi_param else "regression"
        elif self.multi_param:
            mode = "classification multi-param"
        else:
            mode = "classification"
        print(f"Starting training for {num_epochs} epochs [{mode}]")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"TensorBoard logs: {self.log_dir}")

        start_epoch = 0
        if resume_from:
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch
        else:
            pass  # Fresh start: best_val_loss stays at float("inf") → first epoch always saves

        max_consecutive_errors = 3

        try:
            consecutive_errors = 0
            for epoch in range(start_epoch, num_epochs):
                self.current_epoch = epoch

                try:
                    train_loss = self.train_epoch()
                    val_loss = self.validate()
                    consecutive_errors = 0

                    self.writer.add_scalar("train/loss_epoch", train_loss, epoch)
                    self.writer.add_scalars("loss_comparison", {
                        "train": train_loss, "val": val_loss,
                    }, epoch)

                    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

                    if self.scheduler:
                        self.scheduler.step(val_loss)

                    # Best model + early stopping
                    if val_loss < self.best_val_loss - self.min_delta:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        self.save_checkpoint(is_best=True)
                        print(f"New best model! val_loss={val_loss:.4f}")
                    else:
                        self.patience_counter += 1
                        print(f"No improvement (best: {self.best_val_loss:.4f}, "
                              f"patience: {self.patience_counter}/{self.patience})")
                        if self.patience_counter >= self.patience:
                            print(f"Early stopping after {self.patience} epochs without improvement")
                            break

                except Exception as e:
                    consecutive_errors += 1
                    logging.error(f"Error in epoch {epoch}: {str(e)}")
                    logging.error(traceback.format_exc())
                    if consecutive_errors == 1:
                        self.save_checkpoint(emergency=True)
                    if consecutive_errors >= max_consecutive_errors:
                        logging.error(f"Aborting: {max_consecutive_errors} consecutive errors")
                        break
                    logging.warning(f"Attempting to continue ({consecutive_errors}/{max_consecutive_errors})...")
                    continue

        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            self.save_checkpoint(emergency=True)

        except Exception as e:
            logging.error(f"Fatal error: {str(e)}")
            logging.error(traceback.format_exc())
            self.save_checkpoint(emergency=True)
            raise

        finally:
            self.save_checkpoint(is_last=True)
            print("\nTraining finished!")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"\nTo view training logs, run:")
            print(f"  tensorboard --logdir={self.log_dir}")
            self.writer.close()
