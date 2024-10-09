import datetime
import os
from datetime import datetime

import pandas as pd
import torch


class DiffiTTrainer:
    def __init__(self, train_dataloader, valid_dataloader, model, optimizer, loss_function, device, save_folder,
                 batch_size, num_epochs, noise_scheduler):
        self.train_dataloader = train_dataloader
        self.val_dataloader = valid_dataloader
        self.model = model
        self.optimizer = optimizer
        self.loss = loss_function
        self.device = device
        self.save_folder = save_folder
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.noise_scheduler = noise_scheduler

        self.history = {
            "NUM_EPOCHS": num_epochs,
            "BATCH_SIZE": batch_size,
            "LEARNING_RATE": optimizer.param_groups[0]['lr'],
            "train_loss": [],
            "valid_loss": []
        }

    def train_and_validate(self):
        t = datetime.now()
        minute = str(t.minute).zfill(2)
        daytime = f"{t.year}-{t.month}-{t.day}-{t.hour}-{minute}"

        save_path = os.path.join(self.save_folder, daytime + ".ckpt")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for epoch in range(1, self.num_epochs + 1):
            print(f'\nEpoch {epoch:02d}:')

            # TRAINING STEP
            train_total_batches = len(self.train_dataloader)
            train_epoch_loss = 0

            self.model.train()
            self.model.to(self.device)

            for x_train, y_train in self.train_dataloader:
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)
                t = torch.randint(0, self.noise_scheduler.T, (x_train.shape[0],), device=self.device).long()

                noise, x_train_noisy = self.noise_scheduler.noisify(x_train, t)
                self.optimizer.zero_grad()
                pred = self.model(x_train_noisy, t, y_train)
                loss = self.loss(pred, noise)
                loss.backward()
                self.optimizer.step()
                train_epoch_loss += loss

            avg_train_loss = train_epoch_loss / train_total_batches
            avg_train_loss_np = avg_train_loss.cpu().detach().numpy()
            print(f'train Loss: {avg_train_loss:.4f}')

            # VALIDATION STEP
            valid_total_batches = len(self.val_dataloader)
            valid_epoch_loss = 0

            self.model.eval()

            with torch.no_grad():
                for x_valid, y_valid in self.val_dataloader:
                    x_valid = x_valid.to(self.device)
                    y_valid = y_valid.to(self.device)
                    t = torch.randint(0, self.noise_scheduler.T, (x_valid.shape[0],), device=self.device).long()
                    noise, x_valid_noisy = self.noise_scheduler.noisify(x_valid, t)
                    pred = self.model(x_valid, t, y_valid)
                    loss = self.loss(pred, noise)
                    valid_epoch_loss += loss

            avg_valid_loss = valid_epoch_loss / valid_total_batches
            avg_valid_loss_np = avg_valid_loss.cpu().detach().numpy()
            print(f'validation Loss: {avg_valid_loss:.4f}')

            # SAVE STEP
            self.history["train_loss"].append(avg_train_loss_np)
            self.history["valid_loss"].append(avg_valid_loss_np)

            model_path = os.path.join(save_path, f'model_{epoch}.pth')
            torch.save(self.model.state_dict(), model_path)

        history_df = pd.DataFrame(self.history)
        history_path = os.path.join(save_path, 'history.csv')
        history_df.to_csv(history_path, index=False)
        return self.history
