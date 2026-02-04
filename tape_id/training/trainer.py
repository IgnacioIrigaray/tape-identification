"""
Training loop para identificación de parámetros de tape.
"""

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
    """
    Trainer para modelo de identificación de parámetros.

    Args:
        encoder: Encoder model
        controller: Controller model
        processor: Tape processor model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        loss_fn: Loss function
        optimizer: Optimizer
        l1_weight: Weight for the L1 regularization loss for model parameters
        device: Device (cuda/cpu)
        output_dir: Directorio para guardar checkpoints
        log_dir: Directorio para logs de TensorBoard
    """

    def __init__(
        self,
        encoder: nn.Module,
        controller: nn.Module,
        processor: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        params_reg_weight: float = 0.1,
        params_reg_loss=None,
        scheduler=None,
        device: str = "cuda",
        output_dir: str = "outputs/checkpoints",
        log_dir: str = "outputs/logs",
    ):
        self.encoder = encoder.to(device)
        self.controller = controller.to(device)
        self.processor = processor.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.params_reg_weight = params_reg_weight
        self.params_reg_loss = params_reg_loss if params_reg_loss else nn.L1Loss()
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.global_step = 0

        # Early stopping
        self.patience = 15
        self.patience_counter = 0
        self.min_delta = 0.001

        # Configurar logging
        log_file = self.output_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def train_epoch(self) -> float:
        """Ejecuta una época de entrenamiento."""
        self.encoder.train()
        self.controller.train()
        self.processor.train()

        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (x, y, params) in enumerate(pbar):
            # Mover a device (el dataset ya retorna [batch, 1, samples])
            x = x.to(self.device)
            y = y.to(self.device)

            # Forward pass
            # 1. Obtener embeddings
            e_x = self.encoder(x)
            e_y = self.encoder(y)

            # 2. Predecir logits de clasificación
            logits = self.controller(e_x, e_y)

            # 3. Aplicar procesador (usa softmax + expected value en training)
            y_pred = self.processor(x, logits, use_argmax=False)

            # 4. Calcular loss
            if self.params_reg_weight > 0 and params is not None:
                
                loss_params = self.params_reg_loss(logits, params.to(self.device))

            loss = self.loss_fn(y_pred.squeeze(1), y.squeeze(1)) + self.params_reg_weight * loss_params

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping para estabilidad
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) +
                list(self.controller.parameters()) +
                list(self.processor.parameters()),
                max_norm=1.0
            )

            self.optimizer.step()

            total_loss += loss.item()

            # TensorBoard logging
            self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)

            # Loggear estadísticas de gain predicho (expected value)
            gain_pred = self.processor.get_gain_from_logits(logits, use_argmax=False)
            self.writer.add_scalar("train/gain_mean", gain_pred.mean().item(), self.global_step)
            self.writer.add_scalar("train/gain_std", gain_pred.std().item(), self.global_step)
            self.writer.add_scalar("train/gain_min", gain_pred.min().item(), self.global_step)
            self.writer.add_scalar("train/gain_max", gain_pred.max().item(), self.global_step)

            # Loggear learning rate
            current_lr = self.optimizer.params_groups[0]['lr']
            self.writer.add_scalar("train/learning_rate", current_lr, self.global_step)

            self.global_step += 1
            pbar.set_postfix({"loss": loss.item(), "gain_mean": gain_pred.mean().item()})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> float:
        """Ejecuta validación."""
        self.encoder.eval()
        self.controller.eval()
        self.processor.eval()

        total_loss = 0.0
        all_gain_preds = []

        for x, y in tqdm(self.val_loader, desc="Validation"):
            x = x.to(self.device)
            y = y.to(self.device)

            e_x = self.encoder(x)
            e_y = self.encoder(y)
            logits = self.controller(e_x, e_y)
            y_pred = self.processor(x, logits, use_argmax=False)

            loss = self.loss_fn(y_pred.squeeze(1), y.squeeze(1))
            total_loss += loss.item()

            # Acumular predicciones para estadísticas
            gain_pred = self.processor.get_gain_from_logits(logits, use_argmax=False)
            all_gain_preds.append(gain_pred.cpu())

        # Calcular estadísticas de validación
        all_gain_preds = torch.cat(all_gain_preds, dim=0)
        avg_loss = total_loss / len(self.val_loader)

        # TensorBoard logging
        self.writer.add_scalar("val/loss", avg_loss, self.current_epoch)
        self.writer.add_scalar("val/gain_mean", all_gain_preds.mean().item(), self.current_epoch)
        self.writer.add_scalar("val/gain_std", all_gain_preds.std().item(), self.current_epoch)
        self.writer.add_scalar("val/gain_min", all_gain_preds.min().item(), self.current_epoch)
        self.writer.add_scalar("val/gain_max", all_gain_preds.max().item(), self.current_epoch)

        # Loggear histograma de predicciones (comentado por incompatibilidad numpy/tensorboard)
        # self.writer.add_histogram("val/gain_distribution", all_gain_preds.flatten().numpy(), self.current_epoch)

        return avg_loss

    def save_checkpoint(self, is_best: bool = False, is_last: bool = False, emergency: bool = False):
        """Guarda checkpoint (solo best, last o emergency)."""
        checkpoint = {
            "epoch": self.current_epoch,
            "encoder_state": self.encoder.state_dict(),
            "controller_state": self.controller.state_dict(),
            "processor_state": self.processor.state_dict(),
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
            print(f"✓ Saved best model (epoch {self.current_epoch}, val_loss={self.best_val_loss:.4f})")

        if is_last:
            path = self.output_dir / "last_model.pt"
            torch.save(checkpoint, path)
            print(f"✓ Saved last model (epoch {self.current_epoch})")

    def load_checkpoint(self, checkpoint_path: str):
        """Carga checkpoint para continuar entrenamiento."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint["encoder_state"])
        self.controller.load_state_dict(checkpoint["controller_state"])
        self.processor.load_state_dict(checkpoint["processor_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        self.current_epoch = checkpoint["epoch"] + 1  # Siguiente época
        self.best_val_loss = checkpoint["best_val_loss"]
        self.global_step = checkpoint.get("global_step", 0)

        logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        logging.info(f"Resuming from epoch {self.current_epoch}")
        logging.info(f"Best val loss so far: {self.best_val_loss:.4f}")

    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """Ejecuta loop de entrenamiento completo."""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"TensorBoard logs: {self.log_dir}")

        # Cargar checkpoint si se especifica
        start_epoch = 0
        if resume_from:
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch

        try:
            for epoch in range(start_epoch, num_epochs):
                self.current_epoch = epoch

                try:
                    # Train
                    train_loss = self.train_epoch()

                    # Validate
                    val_loss = self.validate()

                    # Loggear métricas por época
                    self.writer.add_scalar("train/loss_epoch", train_loss, epoch)
                    self.writer.add_scalars("loss_comparison", {
                        "train": train_loss,
                        "val": val_loss,
                    }, epoch)

                    print(
                        f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                    )

                    # Learning rate scheduling
                    if self.scheduler:
                        self.scheduler.step(val_loss)

                    # Check for best model (no early stopping)
                    if val_loss < self.best_val_loss - self.min_delta:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(is_best=True)
                        print(f"✓ New best model! val_loss={val_loss:.4f}")
                    else:
                        print(f"No improvement (best: {self.best_val_loss:.4f})")

                except Exception as e:
                    logging.error(f"Error in epoch {epoch}: {str(e)}")
                    logging.error(traceback.format_exc())

                    # Guardar checkpoint de emergencia
                    self.save_checkpoint(emergency=True)

                    # Decidir si continuar o abortar
                    logging.warning("Attempting to continue training...")
                    continue

        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            self.save_checkpoint(emergency=True)

        except Exception as e:
            logging.error(f"Fatal error during training: {str(e)}")
            logging.error(traceback.format_exc())
            self.save_checkpoint(emergency=True)
            raise

        finally:
            # Guardar último modelo
            self.save_checkpoint(is_last=True)

            print("\nTraining finished!")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"\nTo view training logs, run:")
            print(f"  tensorboard --logdir={self.log_dir}")

            self.writer.close()
