import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from NetWork import ResNet
from ImageUtils import parse_record

""" This script defines the training, validation and testing process.
"""

class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        self.network = ResNet(
            self.config.resnet_version,
            self.config.resnet_size,
            self.config.num_classes,
            self.config.first_num_filters,
        )
        ### YOUR CODE HERE
        # define cross entropy loss and optimizer
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=1e-4,
            # using weight decay
            weight_decay = self.config.weight_decay
        )
        ### YOUR CODE HERE
    
    def train(self, x_train, y_train, max_epoch):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size
        
        print('### Training... ###')
        for epoch in range(1, max_epoch+1):

            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            new_learning_rate = 1e-4
            if(epoch % 10 == 0):
                new_learning_rate /= 10

            # setting the new learning rate for the param group the new learning rate
            self.optimizer.param_groups[0]['lr'] =  new_learning_rate


            ### YOUR CODE HERE
            # loss for one epoch
            epoch_loss = 0
            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay

                x_for_current_batch = curr_x_train[i * self.config.batch_size : (i + 1) * self.config.batch_size]
                y_for_current_batch = curr_y_train[i * self.config.batch_size : (i + 1) * self.config.batch_size]
                
                # print('size:', curr_x_train[0].shape)

                training_curr_batch = []
                for index in range(i * self.config.batch_size , (i + 1) * self.config.batch_size):
                  temp = parse_record(curr_x_train[index], True)
                  training_curr_batch.append(temp)
                
                # converted the list training_curr_batch into the np.array to increase the speed of computation of creating tensor
                output = self.network(torch.tensor(np.array(training_curr_batch),dtype=torch.float).to('cuda'))
                loss = self.cross_entropy_loss(output,torch.tensor(y_for_current_batch,dtype=torch.long).to('cuda'))

                l2_regularization_lambda=0.0001
                l2_regularization_value = 0
                if isinstance(self.network.parameters(), list):
                    for weights in self.model.named_parameters():
                        if "weight" in weights[0]:
                            layer_name = weights[0].replace(".weight", "")
                            if layer_name in self.network.params():
                                l2_regularization_value = l2_regularization_value + W[1].norm(2)
                loss = loss + l2_regularization_value * l2_regularization_lambda
               
                ### YOUR CODE HERE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
                epoch_loss = epoch_loss + loss
             
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, epoch_loss/(num_batches if num_batches>= 1 else 1), duration))

            if epoch % self.config.save_interval == 0:
                self.save(epoch)


    def test_or_validate(self, x, y, checkpoint_num_list):
        self.network.eval()
        print('### Test or Validation ###')
        #for checkpoint_num in checkpoint_num_list:
         #   if not os.path.exists(self.config.modeldir):
          #      os.makedirs(self.config.modeldir)
           
            # print(f"contents of {self.config.modeldir}: {os.listdir(self.config.modeldir)}")
           
        for checkpoint_num in checkpoint_num_list:
            # creating a directory first
            checkpoint_dir = os.path.join(self.config.modeldir, f'model_{checkpoint_num}')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

                # creating the checkpoint for the file
                checkpointfile = os.path.join(checkpoint_dir, f'model-{checkpoint_num}.ckpt')

                # cgeck if the checkpoint file exixts
                if not os.path.exists(checkpointfile):
                    torch.save(self.network.state_dict(), checkpointfile)
                else:
                    self.load(checkpointfile)

            preds = []
            # print('Shape of preds 1:', len(preds), x.shape[0])
            for i in tqdm(range(x.shape[0])):
                ### YOUR CODE HERE
                
                with torch.no_grad():
                    input_data = parse_record(x[i], False).reshape(1,3,32,32)
                    input_data_tensor = torch.tensor(input_data,dtype=torch.float).to('cuda')
                    output_data = self.network(input_data_tensor)
                    predicted_values = torch.argmax(output_data, dim=1)
                    # print ('predicted_values shape:', predicted_values.shape)
                    preds.append(predicted_values)
                
                ### END CODE HERE
            print(y.shape)
            y = torch.tensor(y)
            preds = torch.tensor(preds)
            # print('Shape of preds:', preds.shape)
            # print('Shape of y:', y.shape)
            #y_on_hot_encoded = torch.argmax(y, dim=0)
            #y = y_on_hot_encoded
            print('torch size:', torch.sum(preds==y), 'shape:', y, y.shape[0])
            #print('Test accuracy: {:.4f}'.format(torch.sum(preds==y)).item()/y.shape[0])
            print('Test accuracy: {:.4f}'.format(torch.sum(preds == y) / y.shape[0]))
            
    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))
