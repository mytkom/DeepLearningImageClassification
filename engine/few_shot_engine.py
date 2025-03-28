import os
import time
import torch

from engine.train_engine import Engine


def _remap_labels(support_label, query_label):
    unique_classes = torch.unique(support_label)
    class_mapping = {cls.item(): idx for idx, cls in enumerate(unique_classes)}

    support_remapped = torch.tensor(
        [class_mapping[cls.item()] for cls in support_label],
        device=support_label.device
    )
    query_remapped = torch.tensor(
        [class_mapping[cls.item()] for cls in query_label],
        device=query_label.device
    )
    return support_remapped, query_remapped


class FewShotEngine(Engine):
    def _train_one_epoch(self):
        epoch_progress = self.sub_task_progress.add_task("loader", total=len(self.train_loader))
        self.model.train()
        step_loss = 0
        start = time.time()
        for loader_idx, (img, label) in enumerate(self.train_loader, 1):
            current_step = (self.current_epoch - 1) * len(self.train_loader) + loader_idx
            self.data_time.update(time.time() - start)
            with self.accelerator.accumulate(self.model):
                n_support = self.cfg.few_shot.n_way * self.cfg.few_shot.n_shot

                support_img, query_img = img[:n_support], img[n_support:]
                support_label, query_label = label[:n_support], label[n_support:]

                support_label_remapped, query_label_remapped = _remap_labels(support_label, query_label)

                support_embeddings = self.model(support_img)
                prototypes = torch.stack([
                    support_embeddings[support_label_remapped == i].mean(0)
                    for i in range(self.cfg.few_shot.n_way)
                ])

                query_embeddings = self.model(query_img)
                distances = torch.cdist(query_embeddings, prototypes)
                loss = self.loss_fn(-distances, query_label_remapped)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

                loss = self.accelerator.gather(loss.detach().cpu().clone()).mean()
                step_loss += loss.item() / self.cfg.training.accum_iter
            self.iter_time.update(time.time() - start)

            if self.accelerator.is_main_process and self.accelerator.sync_gradients:
                self.accelerator.log(
                    {
                        "loss/train": step_loss,
                    },
                    step=current_step,
                )
                step_loss = 0

            self.accelerator.log(
                {
                    "time/iter": self.iter_time.val,
                    "time/data": self.data_time.val,
                },
                step=current_step,
            )
            self.sub_task_progress.update(epoch_progress, advance=1)

            start = time.time()
        self.sub_task_progress.remove_task(epoch_progress)

    def validate(self):
        valid_progress = self.sub_task_progress.add_task("validate", total=len(self.val_loader))
        all_preds = []
        all_labels = []
        all_losses = []
        self.model.eval()
        with torch.no_grad():
            for img, label in self.val_loader:
                n_support = self.cfg.few_shot.n_way * self.cfg.few_shot.n_shot
                support_img, query_img = img[:n_support], img[n_support:]
                support_label, query_label = label[:n_support], label[n_support:]

                support_label_remapped, query_label_remapped = _remap_labels(support_label, query_label)

                support_embeddings = self.model(support_img)
                prototypes = torch.stack([
                    support_embeddings[support_label_remapped == i].mean(0)
                    for i in range(self.cfg.few_shot.n_way)
                ])

                query_embeddings = self.model(query_img)
                distances = torch.cdist(query_embeddings, prototypes)

                loss = self.loss_fn(-distances, query_label_remapped)
                all_losses.append(loss.item())

                all_preds.append(-distances.detach().cpu())
                all_labels.append(query_label_remapped.cpu())

                self.sub_task_progress.update(valid_progress, advance=1)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        metric_results = self.metrics.compute(all_preds, all_labels, all_losses)

        if self.accelerator.is_main_process:
            self.accelerator.print(
                f"val. acc.: {metric_results['accuracy']:.3f}, loss: {metric_results['loss']:.3f}, "
                f"precision: {metric_results['precision']:.3f}, recall: {metric_results['recall']:.3f}, f1: {metric_results['f1']:.3f}"
            )
            self.accelerator.log(
                {
                    "acc/val": metric_results['accuracy'],
                    "loss/val": metric_results['loss'],
                    "precision/val": metric_results['precision'],
                    "recall/val": metric_results['recall'],
                    "f1/val": metric_results['f1']
                },
                step=self.current_epoch * len(self.train_loader),  # Use train steps
            )
        if self.accelerator.is_main_process and metric_results['accuracy'] > self.max_acc:
            save_path = os.path.join(self.base_dir, "checkpoint")
            self.accelerator.print(f"new best found with: {metric_results['accuracy']:.3f}, save to {save_path}")
            self.max_acc = metric_results['accuracy']
            self.save_checkpoint(os.path.join(save_path, f"epoch_{self.current_epoch}"))

        self.sub_task_progress.remove_task(valid_progress)

