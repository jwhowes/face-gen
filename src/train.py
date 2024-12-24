import torch

from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from . import accelerator
from .model import FlowModel
from .config import Config


def train(
        model: FlowModel,
        dataloader: DataLoader,
        config: Config
):
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

        total_loss = 0
        for i, image in enumerate(dataloader):
            opt.zero_grad()

            with accelerator.autocast():
                loss = model(image)

            accelerator.backward(loss)
            if config.clip_grad:
                accelerator.clip_grad_norm_(model.parameters(), config.clip_grad)

            opt.step()
            lr_scheduler.step()

            total_loss += loss.item()

            if accelerator.is_main_process and i % config.log_interval == 0:
                print(f"{i} / {len(dataloader)} iters.\tLoss: {loss.item():.6f}")

        config.log(model, total_loss / len(dataloader))
