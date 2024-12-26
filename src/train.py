import torch

from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from torch import nn

from . import accelerator
from .config import Config
from .data import FaceDataset


def train(
        model: nn.Module,
        config: Config
):
    dataset = FaceDataset(image_size=config.dataset.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        pin_memory=True
    )

    opt = torch.optim.Adam(
        model.parameters(), lr=config.lr
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=int(config.warmup * len(dataloader)),
        num_training_steps=config.num_epochs * len(dataloader)
    )

    model, dataloader, opt, lr_scheduler = accelerator.prepare(
        model, dataloader, opt, lr_scheduler
    )

    model.train()
    for epoch in range(config.num_epochs):
        if accelerator.is_main_process:
            print(f"EPOCH {epoch + 1} / {config.num_epochs}")

        total_metrics = [0 for _ in range(len(config.metrics))]
        for i, image in enumerate(dataloader):
            opt.zero_grad()

            with accelerator.autocast():
                loss = model(image)

            accelerator.backward(loss["loss"])
            if config.clip_grad:
                accelerator.clip_grad_norm_(model.parameters(), config.clip_grad)

            opt.step()
            lr_scheduler.step()

            for j, m in enumerate(loss["metrics"]):
                total_metrics[j] += m

            if accelerator.is_main_process and i % config.log_interval == 0:
                print(f"{i} / {len(dataloader)} iters.\t{"\t".join([f'{k}: {v:.4f}' for k, v in zip(config.metrics, loss['metrics'])])}")

        config.log(model, *[m / len(dataloader) for m in total_metrics])
