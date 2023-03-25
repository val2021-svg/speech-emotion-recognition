import matplotlib.pyplot as plt
import os
import torch

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state. Also saves images of results.
    """
    def __init__(self, weights_path, images_path, feature, session_to_test, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        self.weights_path = weights_path
        self.images_path = images_path
        self.feature = feature
        self.session = session_to_test

    def save_model(self, current_valid_loss, epoch, acc, f1, model): 
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
           
            torch.save(model.state_dict(), f'{self.weights_path}/best-model-parameters-{self.feature}-{self.session_to_test}.pt')

            with open(f'{self.weights_path}/history-{self.feature}-{self.session_to_test}.txt', 'a') as f:
                f.write(f'New model saved: Epoch = {epoch}, Loss = {current_valid_loss}, F1 = {f1}, Accuracy = {acc}.\n\n')
                
    def plot_results(self, epoch_table, output_train, output_test):
        
        save_path = f'{self.images_path}/{self.feature}/{self.session_to_test}'
        if not os.path.exists(save_path)
        os.makedirs(save_path)

        plt.plot(epoch_table, output_train["loss"], label="Train")
        plt.plot(epoch_table, output_test["loss"], label="Test")
        plt.title("Loss")
        plt.legend(loc='upper right')
        plt.savefig(f'{save_path}/loss__epoch_{epoch_table[-1]}.png')
        plt.show()

        plt.plot(epoch_table, output_train["f1"], label="Train")
        plt.plot(epoch_table, output_test["f1"], label="Test")
        plt.title("F1")
        plt.savefig(f'{save_path}/f1_epoch_{epoch_table[-1]}.png')
        plt.legend(loc='upper left')
        plt.show()

        plt.plot(epoch_table, output_train["acc"], label="Train")
        plt.plot(epoch_table, output_test["acc"], label="Test")
        plt.title("Accuracy")
        plt.legend(loc='upper left')
        plt.savefig(f'{save_path}/accuracy_epoch_{epoch_table[-1]}.png')
        plt.show()

