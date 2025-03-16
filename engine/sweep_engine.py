import wandb

from engine.train_engine import Engine


class SweepEngine(Engine):
    def setup_training(self):
        super().setup_training()
        self.accelerator.wait_for_everyone()

        self.cfg.training.batch_size = wandb.config.batch_size
        self.cfg.training.lr = wandb.config.learning_rate
