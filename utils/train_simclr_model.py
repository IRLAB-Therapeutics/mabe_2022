import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.keypoint_dataset import get_keypoints_map
from math import sqrt
import torchvision
from simclr import SimCLR
from simclr.modules import NT_Xent
from simclr.modules import LARS
from tqdm import tqdm

class MouseSimCLR(pl.LightningModule):
    def __init__(self, batch_size, embedding_size: int = 128, weight_decay: float = 1e-6):
        super().__init__()

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.weight_decay = weight_decay

        resnet_encoder = torchvision.models.resnet50(pretrained=True)
        n_features = resnet_encoder.fc.in_features
        self.simclr = SimCLR(resnet_encoder, self.embedding_size, n_features)
        world_size = 1
        temperature = 0.5
        self.criterion = NT_Xent(batch_size, temperature, world_size)

    def forward(self, batch):
        if len(batch) == 2:
            x_i = batch[0]
            x_j = batch[1]
            h_i, h_j, z_i, z_j = self.simclr(x_i, x_j)
            return h_i, h_j, z_i, z_j
        else:
            x = batch
            h = self.simclr.encoder(x)
            z = self.simclr.projector(h)
            return h, z

    def training_step(self, batch, _):
        x_i = batch[0]
        x_j = batch[1]
        h_i, h_j, z_i, z_j = self.simclr(x_i, x_j)
        loss = self.criterion(z_i, z_j)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        # learning_rate = 0.3 * self.batch_size / 256
        # optimizer = LARS(
        #     self.parameters(),
        #     lr=learning_rate,
        #     weight_decay=self.weight_decay,
        #     exclude_from_weight_decay=["batch_normalization", "bias"],
        # )
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

        # "decay the learning rate with the cosine decay schedule without restarts"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs, eta_min=0, last_epoch=-1
        )
        return [optimizer], [scheduler]

if __name__ == "__main__":
    from utils.augmentation import TransformsSimCLR
    from utils.video_dataset import SimCLRDataset, create_bounding_box
    import os
    from pytorch_lightning.callbacks import ModelCheckpoint

    data_dir = '/data/behavior-representation'
    video_size = 'full_size'
    video_set = 'submission'

    kp = create_bounding_box(data_dir)['sequences']


    video_data_dir = os.path.join(data_dir, 'videos', video_size, video_set)
    train_transform = TransformsSimCLR((224, 224), pretrained=False)
    train_dataset = SimCLRDataset(video_data_dir, kp, transform=train_transform)

    val_transform = TransformsSimCLR((224, 224), pretrained=False, validation=True)
    val_dataset = SimCLRDataset(video_data_dir, kp, transform=val_transform)

    batch_size = 76

    weight = np.load("cache/average_motion.npy")
    # weight = np.sqrt(weight)
    p = weight / weight.sum()
    num_samples = 100000
    sampler = torch.utils.data.WeightedRandomSampler(p, num_samples)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=False,
        num_workers=8,
        sampler=sampler
    )
    predict_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        num_workers=8
    )
    model = MouseSimCLR(batch_size=batch_size)

    checkpoint_callback = ModelCheckpoint(monitor="train_loss", save_top_k=1, save_last=True)
    # trainer = pl.Trainer(gpus=[1], max_epochs=1, precision=16, callbacks=[checkpoint_callback], limit_predict_batches=1/60)
    trainer = pl.Trainer(gpus=[0], max_epochs=25, precision=16, callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=train_loader)    

    predictions = trainer.predict(model, predict_loader)
    cache_path = 'cache/simclr_embeddings.pt'
    print(f"Saving embeddings to {cache_path}")
    torch.save(predictions, cache_path)