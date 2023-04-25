################################################################################
# CSE 151B: Programming Assignment 3
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin, Yash, James
# Fall 2022
################################################################################

import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import math
from tqdm import tqdm
from copy import deepcopy
from nltk.tokenize import word_tokenize
from caption_utils import *

ROOT_STATS_DIR = './experiment_data'
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model
import torch.nn as nn
from PIL import Image


# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco, self.__coco_test, self.__vocab, self.__train_loader, \
        self.__val_loader, self.__test_loader = get_datasets(config_data)

        # Setup Experiment
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__early_stop = config_data['experiment']['early_stop']
        self.__patience = config_data['experiment']['patience']
        self.__batch_size = config_data['dataset']['batch_size']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Init Model
        self.__model = get_model(config_data, self.__vocab)
        self.__best_model = deepcopy(self.__model.state_dict())

        # criterion ; automatically combines both softmax + NLLLoss.
        self.__criterion = nn.CrossEntropyLoss()

        # optimizer
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=config_data['experiment']['learning_rate'])

        # LR Scheduler
        self.__lr_scheduler = torch.optim.lr_scheduler.StepLR(self.__optimizer, config_data['experiment']['step'])

        self.__init_model()

        self.__load_experiment()


    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])
        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()
        # for mac
        try: 
            if torch.has_mps:
                self.__model = self.__model.mps().float()
                self.__criterion = self.__criterion.mps()
        except:
            pass

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        patience_count = 0
        min_loss = 100
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            print(f'Epoch {epoch + 1}')
            print('--------')
            start_time = datetime.now()
            self.__current_epoch = epoch
            print('Training...')
            print('-----------')
            train_loss = self.__train()
            print('Validating...')
            print('-------------')
            val_loss = self.__val()
            # save best model
            if val_loss < min_loss:
                min_loss = val_loss
                self.__best_model = deepcopy(self.__model.state_dict())

            # early stop if model starts overfitting
            if self.__early_stop:
                if epoch > 0 and val_loss > self.__val_losses[epoch - 1]:
                    patience_count += 1
                if patience_count >= self.__patience:
                    print('\nEarly stopping!')
                    break

            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
            if self.__lr_scheduler is not None:
                self.__lr_scheduler.step()
            #break
        self.__model.load_state_dict(self.__best_model)

    def __compute_loss(self, images, captions):
        """
        Computes the loss after a forward pass through the model
        """
        batch_size = images.shape[0]
        
        logits = self.__model(images, captions, teacher_forcing=True)
#         print(logits.shape)
        logits = torch.flatten(logits, 0, 1)
        captions = torch.flatten(captions)
        loss = self.__criterion(logits, captions)
        return loss

    def __train(self):
        """
        Trains the model for one epoch using teacher forcing and minibatch stochastic gradient descent
        """
        # for_testing_only = 0
        self.__model.train()
        loss = 0
        n = len(self.__train_loader)
        tl = tqdm(self.__train_loader, total = n)
        #for_test_only = 0
        for images, captions, image_id in tl:
            
#             if for_testing_only > 50:
#                 break
            images, captions = images.to(self.device), captions.to(self.device)
            
            self.__optimizer.zero_grad()
            
#             outputs = self.__model(images, captions, teacher_forcing = True)
#             print(outputs.shape, captions.shape)
            l = self.__compute_loss(images, captions)
            l.backward()
            self.__optimizer.step()
            tl.set_description("traning batch loss: %.5f"%l.item())
            loss += l.item()
            #break
            #for_testing_only += 1
        avg_loss = loss / n
        return avg_loss

    def __generate_captions(self, img_id, outputs, testing):
        """
        Generate captions without teacher forcing
        Params:
            img_id: Image Id for which caption is being generated
            outputs: output from the forward pass for this img_id
            testing: whether the image_id comes from the validation or test set
        Returns:
            tuple (list of original captions, predicted caption)
        """
        if testing:
            coco = self.__coco_test
        else:
            coco = self.__coco
        actual = coco.imgToAnns[img_id]
        # TODO:
        # map index (which is outputs) -> words using  self.__vocab.idx2word


        #idx = torch.argmax(outputs,1)
        #print(idx.shape)
        predicted = []
        captions = []
        for x in actual:
            captions.append(word_tokenize(x['caption'].lower()))
        for i in outputs:
            word = self.__vocab.idx2word[i]

            if word == '<end>':
                break
            if word != '<start>':
                predicted.append(word.lower())
 
        sentence = ' '.join(predicted)
        
        bleu_1 = bleu1(captions, predicted)
        



        bleu_4 = bleu4(captions, predicted)

        
        return actual, sentence, bleu_1, bleu_4


    def __str_captions(self, img_id, original_captions, predicted_caption):
        """
            !OPTIONAL UTILITY FUNCTION!
            Create a string for logging ground truth and predicted captions for given img_id
        """
        result_str = "Captions: Img ID: {},\nActual: {},\nPredicted: {}\n".format(
            img_id, original_captions, predicted_caption)
        return result_str

    def __val(self):
        """
        Validate the model for one epoch using teacher forcing
        """
        self.__model.eval()
        loss = 0
        n = len(self.__val_loader)
        tl = tqdm(self.__val_loader, total = n)
        #for_testing_only = 0
        with torch.no_grad():
            for images, captions, image_id in tl:
                #if for_testing_only > 50: break
                images, captions = images.to(self.device), captions.to(self.device)
                #id is a list?
                #self.__generate_captions(image_id[0], outputs, False)
                l = self.__compute_loss(images, captions)
                b_tem1 = 0
                b_tem4 = 0
                
#                 for i in range(len(image_id)):
#                     actual, sentence, bleu1, bleu4 = self.__generate_captions(image_id[i], logits[i], False)
#                     b_tem1 += bleu1
#                     b_tem4 += bleu4
                   
#                 b1 += b_tem1/len(image_id)
#                 b4 += b_tem4/len(image_id)
            
                tl.set_description("val batch loss: %.5f"%l.item())

                loss += l.item()
                #for_testing_only += 1
                #break
                
#             print('bleu1 ', b1/n)
#             print('bleu4 ', b4/n)
        return loss / n

    def test(self):
        """
        Test the best model on test data. Generate captions and calculate bleu scores
        """
        #self.__best_model.eval()
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model = torch.load(root_model_path)
        self.__model.load_state_dict(model["model"])
        self.__optimizer.load_state_dict(model["optimizer"])
        self.__model.eval()
        loss = 0
        n = len(self.__test_loader)
        tl = tqdm(self.__test_loader, total = n)
        b1 = 0
        b4 = 0
        count_1 = []
        count_4 = []
        count_de = {}
        count_5 = {}
        count_0001 = {}
        with torch.no_grad():
            for images, captions, image_id in tl:
                b_tem1 = 0
                b_tem4 = 0
#                 if for_test_only > 50: break
                images, captions = images.to(self.device), captions.to(self.device)
                #print(captions)
                #self.__optimizer.zero_grad()

                #id is a list?
                #self.__generate_captions(image_id[0], captions[0], True)
                self.__model.to(self.device)

                l = self.__compute_loss(images, captions)
                outputs_de = self.__model(images, captions, False, True)
                self.__model.load_state_dict(model["model"])
                outputs_04 = self.__model(images, captions, False, False, 0.4)
                self.__model.load_state_dict(model["model"])
                outputs_5 = self.__model(images, captions, False, False, 5)
                self.__model.load_state_dict(model["model"])
                outputs_0001 = self.__model(images, captions, False, False, 0.0001)
#                 print(len(image_id))
               
                for i in range(len(image_id)):
                    predict = []
                    for x in outputs_04:
                        predict.append(x[i].item())
                   
                    actual, sentence, bleu1, bleu4 = self.__generate_captions(image_id[i], predict, True)
                   
                    count_1.append((bleu1, image_id[i], actual, sentence, image_id[i]))
                    count_4.append((bleu4, image_id[i], actual, sentence, image_id[i]))

                    b_tem1 += bleu1
                    b_tem4 += bleu4
                    
                    predict_de = []
                    for x in outputs_de:
                        predict_de.append(x[i].item())
                   
                    actual_de, sentence_de, bleu1_de, bleu4_de = self.__generate_captions(image_id[i], predict_de, True)
                    count_de[image_id[i]] = [actual_de, sentence_de, bleu1_de, bleu4_de]
                    
                    predict_5 = []
                    for x in outputs_5:
                        predict_5.append(x[i].item())
                   
                    actual_5, sentence_5, bleu1_5, bleu4_5 = self.__generate_captions(image_id[i], predict_5, True)
                    count_5[image_id[i]] = [actual_5, sentence_5, bleu1_5, bleu4_5]

                    predict_0001 = []
                    for x in outputs_0001:
                        predict_0001.append(x[i].item())
                   
                    actual_0001, sentence_0001, bleu1_0001, bleu4_0001 = self.__generate_captions(image_id[i], predict_0001, True)
                    count_0001[image_id[i]] = [actual_0001, sentence_0001, bleu1_0001, bleu4_0001]
                    
#                     print(actual, sentence)

                b1 += b_tem1/len(image_id)
                b4 += b_tem4/len(image_id)
                loss += l.item()
               
                count_1 = sorted(count_1)
                count_4 = sorted(count_4)
               
            for i in range(3):
                file_1 = './data/images/test/' + self.__coco_test.loadImgs(count_1[i][1])[0]['file_name']
                img_1 = Image.open(file_1)
                img_1.convert('RGB')
                img_1.save(self.__experiment_dir+'/worst_b1_' + str(i)+ '.pdf')

                captions = ''
                for x in count_1[i][2]:
                    captions += x['caption'].lower()
                    captions += '\n'

                log_to_file(self.__experiment_dir+'/worst_b1_' + str(i)+ '.txt', captions)
                log_to_file(self.__experiment_dir+'/worst_b1_' + str(i)+ '.txt', count_1[i][3])
                
                id = count_1[i][-1]
                log_to_file(self.__experiment_dir+'/worst_b1_' + str(i)+ '.txt', 'deterministic:')
                log_to_file(self.__experiment_dir+'/worst_b1_' + str(i)+ '.txt', count_de[id][1])
                log_to_file(self.__experiment_dir+'/worst_b1_' + str(i)+ '.txt', 'bleu1 %.4f'%count_de[id][2])
                log_to_file(self.__experiment_dir+'/worst_b1_' + str(i)+ '.txt', 'bleu4 %.4f'%count_de[id][3])

                log_to_file(self.__experiment_dir+'/worst_b1_' + str(i)+ '.txt', 'temperature = 5:')
                log_to_file(self.__experiment_dir+'/worst_b1_' + str(i)+ '.txt', count_5[id][1])
                log_to_file(self.__experiment_dir+'/worst_b1_' + str(i)+ '.txt', 'bleu1 %.4f'%count_5[id][2])
                log_to_file(self.__experiment_dir+'/worst_b1_' + str(i)+ '.txt', 'bleu4 %.4f'%count_5[id][3])

                log_to_file(self.__experiment_dir+'/worst_b1_' + str(i)+ '.txt', 'temperature = 0.0001:')
                log_to_file(self.__experiment_dir+'/worst_b1_' + str(i)+ '.txt', count_0001[id][1])
                log_to_file(self.__experiment_dir+'/worst_b1_' + str(i)+ '.txt', 'bleu1 %.4f'%count_0001[id][2])
                log_to_file(self.__experiment_dir+'/worst_b1_' + str(i)+ '.txt', 'bleu4 %.4f'%count_0001[id][3])




                file_2 = './data/images/test/' + self.__coco_test.loadImgs(count_4[i][1])[0]['file_name']
                img_2 = Image.open(file_2)
                img_2.convert('RGB')
                img_2.save(self.__experiment_dir+'/worst_b4_' + str(i)+ '.pdf')

                captions = ''
                for x in count_4[i][2]:
                    captions += x['caption'].lower()
                    captions += '\n'
                    
                log_to_file(self.__experiment_dir+'/worst_b4_' + str(i)+ '.txt', captions)
                log_to_file(self.__experiment_dir+'/worst_b4_' + str(i)+ '.txt', count_4[i][3])
                
                id = count_4[i][-1]
                log_to_file(self.__experiment_dir+'/worst_b4_' + str(i)+ '.txt', 'deterministic:')
                log_to_file(self.__experiment_dir+'/worst_b4_' + str(i)+ '.txt', count_de[id][1])
                log_to_file(self.__experiment_dir+'/worst_b4_' + str(i)+ '.txt', 'bleu1 %.4f'%count_de[id][2])
                log_to_file(self.__experiment_dir+'/worst_b4_' + str(i)+ '.txt', 'bleu4 %.4f'%count_de[id][3])

                log_to_file(self.__experiment_dir+'/worst_b4_' + str(i)+ '.txt', 'temperature = 5:')
                log_to_file(self.__experiment_dir+'/worst_b4_' + str(i)+ '.txt', count_5[id][1])
                log_to_file(self.__experiment_dir+'/worst_b4_' + str(i)+ '.txt', 'bleu1 %.4f'%count_5[id][2])
                log_to_file(self.__experiment_dir+'/worst_b4_' + str(i)+ '.txt', 'bleu4 %.4f'%count_5[id][3])

                log_to_file(self.__experiment_dir+'/worst_b4_' + str(i)+ '.txt', 'temperature = 0.0001:')
                log_to_file(self.__experiment_dir+'/worst_b4_' + str(i)+ '.txt', count_0001[id][1])
                log_to_file(self.__experiment_dir+'/worst_b4_' + str(i)+ '.txt', 'bleu1 %.4f'%count_0001[id][2])
                log_to_file(self.__experiment_dir+'/worst_b4_' + str(i)+ '.txt', 'bleu4 %.4f'%count_0001[id][3])    
                
            


                file_3 = './data/images/test/' + self.__coco_test.loadImgs(count_1[-i-1][1])[0]['file_name']
                img_3 = Image.open(file_3)
                img_3.convert('RGB')
                img_3.save(self.__experiment_dir+'/best_b1_' + str(i)+ '.pdf')

                captions = ''
                for x in count_1[-i-1][2]:
                    captions += x['caption'].lower()
                    captions += '\n'


                log_to_file(self.__experiment_dir+'/best_b1_' + str(i)+ '.txt', captions)
                log_to_file(self.__experiment_dir+'/best_b1_' + str(i)+ '.txt', count_1[-i-1][3])
                
                id = count_1[-i-1][-1]
                log_to_file(self.__experiment_dir+'/best_b1_' + str(i)+ '.txt', 'deterministic:')
                log_to_file(self.__experiment_dir+'/best_b1_' + str(i)+ '.txt', count_de[id][1])
                log_to_file(self.__experiment_dir+'/best_b1_' + str(i)+ '.txt', 'bleu1 %.4f'%count_de[id][2])
                log_to_file(self.__experiment_dir+'/best_b1_' + str(i)+ '.txt', 'bleu4 %.4f'%count_de[id][3])

                log_to_file(self.__experiment_dir+'/best_b1_' + str(i)+ '.txt', 'temperature = 5:')
                log_to_file(self.__experiment_dir+'/best_b1_' + str(i)+ '.txt', count_5[id][1])
                log_to_file(self.__experiment_dir+'/best_b1_' + str(i)+ '.txt', 'bleu1 %.4f'%count_5[id][2])
                log_to_file(self.__experiment_dir+'/best_b1_' + str(i)+ '.txt', 'bleu4 %.4f'%count_5[id][3])

                log_to_file(self.__experiment_dir+'/best_b1_' + str(i)+ '.txt', 'temperature = 0.0001:')
                log_to_file(self.__experiment_dir+'/best_b1_' + str(i)+ '.txt', count_0001[id][1])
                log_to_file(self.__experiment_dir+'/best_b1_' + str(i)+ '.txt', 'bleu1 %.4f'%count_0001[id][2])
                log_to_file(self.__experiment_dir+'/best_b1_' + str(i)+ '.txt', 'bleu4 %.4f'%count_0001[id][3]) 





                file_4 = './data/images/test/' + self.__coco_test.loadImgs(count_4[-i-1][1])[0]['file_name']
                img_4 = Image.open(file_4)
                img_4.convert('RGB')
                img_4.save(self.__experiment_dir+'/best_b4_' + str(i)+ '.pdf')

                captions = ''
                for x in count_4[-i-1][2]:
                    captions += x['caption'].lower()
                    captions += '\n'
            
                log_to_file(self.__experiment_dir+'/best_b4_' + str(i)+ '.txt', captions)
                log_to_file(self.__experiment_dir+'/best_b4_' + str(i)+ '.txt', count_4[-i-1][3])
                
                id = count_4[-i-1][-1]
                log_to_file(self.__experiment_dir+'/best_b4_' + str(i)+ '.txt', 'deterministic:')
                log_to_file(self.__experiment_dir+'/best_b4_' + str(i)+ '.txt', count_de[id][1])
                log_to_file(self.__experiment_dir+'/best_b4_' + str(i)+ '.txt', 'bleu1 %.4f'%count_de[id][2])
                log_to_file(self.__experiment_dir+'/best_b4_' + str(i)+ '.txt', 'bleu4 %.4f'%count_de[id][3])

                log_to_file(self.__experiment_dir+'/best_b4_' + str(i)+ '.txt', 'temperature = 5:')
                log_to_file(self.__experiment_dir+'/best_b4_' + str(i)+ '.txt', count_5[id][1])
                log_to_file(self.__experiment_dir+'/best_b4_' + str(i)+ '.txt', 'bleu1 %.4f'%count_5[id][2])
                log_to_file(self.__experiment_dir+'/best_b4_' + str(i)+ '.txt', 'bleu4 %.4f'%count_5[id][3])

                log_to_file(self.__experiment_dir+'/best_b4_' + str(i)+ '.txt', 'temperature = 0.0001:')
                log_to_file(self.__experiment_dir+'/best_b4_' + str(i)+ '.txt', count_0001[id][1])
                log_to_file(self.__experiment_dir+'/best_b4_' + str(i)+ '.txt', 'bleu1 %.4f'%count_0001[id][2])
                log_to_file(self.__experiment_dir+'/best_b4_' + str(i)+ '.txt', 'bleu4 %.4f'%count_0001[id][3])     
                                
            print('bleu1 ', b1/n)
            print('bleu4 ', b4/n)
            print('test loss', loss/n)
            self.test_loss = loss / n
            self.b1 = b1/n
            self.b4 = b4/n
            self.__record_test()
        return loss / n

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)
        

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')
    
    def __record_test(self):
        with open(os.path.join(self.__experiment_dir, 'test_log.txt'), "w") as f:
            f. write("test loss: %f, bleu1: %f, bleu4: %f"%(self.test_loss, self.b1, self.b4))
    
    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
