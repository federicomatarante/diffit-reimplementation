import torch
import math
import torch.nn as nn
import datetime
from datetime import datetime
import torch.optim as optim
import os
import pandas as pd

from latent_diffit import LatentDiffiT

T = 1000
BETA_START = 1e-4
BETA_END = 0.02
LEARNING_RATE = ...
NUM_EPOCHS = ...
BATCH_SIZE = ...


def beta_comp(t, beta_start, beta_end):
    # linear interpolation
    return BETA_START + (t / T) * (BETA_END - BETA_START)


def alpha_comp(t):
    return 1 - beta_comp(t, BETA_START, BETA_END)


def alpha_hat_comp(t):
    return math.prod([alpha_comp(j) for j in range(t)])


def noisify(sample, timestep):
    # Generation of the Gaussian noise with the same shape of the sample
    noise = torch.randn(sample.shape)
    # Apply the noise to the sample
    noise_sample = []
    for i in range(len(timestep)):
        alpha_hat = alpha_hat_comp(timestep[i])
        noise_sample.append(
            (math.sqrt(alpha_hat) * sample[i]) + (math.sqrt(1 - alpha_hat) * noise[i])
        )
    noise_sample = torch.stack(noise_sample, dim=0)
    return noise, noise_sample


class DiffiTTrainer:
    def __init__(self, train_dataloader, valid_dataloader, model, optimizer, loss, device, save_folder, num_features,
                 batch_size):

        self.train_dataloader = train_dataloader
        self.val_dataloader = valid_dataloader
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.save_folder = save_folder

        self.history = {
            "NUM_EPOCHS": NUM_EPOCHS,
            "NUM_FEATURES": num_features,
            "BATCH_SIZE": batch_size,
            "LEARNING_RATE": LEARNING_RATE,
            "train_loss": [],
            "valid_loss": []
        }

    def train_and_validate(self):

        num_epochs = NUM_EPOCHS

        t = datetime.datetime.now()
        minute = str(t.minute)
        if t.minute < 10:
            minute = '0' + minute
        daytime = str(t.year) + ':' + str(t.month) + ':' + str(t.day) + '-' + str(t.hour) + ':' + minute

        save_path = self.save_folder + '/' + daytime

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        best_avg_valid_f1_weighted = -100

        for epoch in range(1, num_epochs + 1):

            if epoch < 10:
                print(f'\nEpoch 0{epoch}:')
            if epoch >= 10:
                print(f'\nEpoch {epoch}:')

            # TRAINING STEP

            # parameters useful to compute accuracy
            # train_total = len(self.train_dataloader.dataset)
            train_total_batches = len(
                self.train_dataloader)  # number of batches in the training set (self.train_dataloader.dataset / BATCH_SIZE)
            # compute and save loss and f1-score
            train_epoch_loss = 0

            self.model.train()

            for x_train in self.train_dataloader:
                # Forward Pass
                x_train = x_train.to(self.device)
                # Generate random time indices
                t = torch.randint(0, T, (x_train.shape[0],), device=self.device).long()
                # Generate noisy data for training
                noise, x_train_noisy = noisify(x_train, T, t)
                self.optimizer.zero_grad()
                pred = self.model(x_train_noisy, t)
                loss = self.loss(pred, noise)
                # BackPropagation
                loss.backward()
                # Optimization
                self.optimizer.step()
                # Update total train epoch loss
                train_epoch_loss += loss
            # compute epoch average loss, accuracy, f1
            avg_train_loss = train_epoch_loss / train_total_batches  # we compute the avg loss for each batch, because otherwise we would depend on the size of the dataset
            avg_train_loss_np = avg_train_loss.cpu().detach().numpy()  # to put it into self.history
            # print them
            print('{} Loss: {:.4f} '.format('train', avg_train_loss))

            # VALIDATION STEP
            valid_total_batches = len(
                self.val_dataloader)  # number of batches in the validation set (self.val_dataloader.dataset / BATCH_SIZE)
            # compute and save loss and f1-score
            valid_epoch_loss = 0

            self.model.eval()

            with torch.no_grad():
                for x_valid in self.val_dataloader:
                    # Forward Pass
                    x_valid = x_valid.to(self.device)
                    # Generate random time indices
                    t = torch.randint(0, T, (x_valid.shape[0],), device=self.device).long()
                    # Generate noisy data for training
                    noise, x_valid_noisy = noisify(x_valid, T, t)
                    pred = self.model(x_valid)
                    loss = self.loss(pred, noise)
                    valid_epoch_loss += loss
            # epoch average loss, accuracy, f1
            avg_valid_loss = valid_epoch_loss / valid_total_batches
            avg_valid_loss_np = avg_valid_loss.cpu().detach().numpy()
            # print them
            print('{} Loss: {:.4f}'.format('validation', avg_valid_loss))

            # SAVE STEP
            self.history["train_loss"].append(avg_train_loss_np)
            self.history["valid_loss"].append(avg_valid_loss_np)

            # for each epoch, verify that the model obtained by training+validating is the best yet => saves only the best
            model_path = os.path.join(save_path, 'model_' + epoch + '_.pth')
            torch.save(self.model.state_dict(), model_path)

        # for each training, save history and return it
        history_df = pd.DataFrame(self.history)
        history_path = save_path + '/history.csv'
        history_df.to_csv(history_path, index=False)
        return self.history


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))

    # create folder to save best model and history
    save_folder = '...'
    try:
        os.makedirs(save_folder)
    except OSError as e1:
        print("Creation of the directory %s failed" % save_folder)
        print("Error:", e1)

    # initialize parameters needed by the trainer
    model = LatentDiffiT().to(device)
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE)

    # initialize a "TennisShotsTrainer" object
    trainer = DiffiTTrainer(train_dataloader, val_dataloader, model, optimizer, loss, device, save_folder, BATCH_SIZE)


if __name__ == '__main__':
    main()
