from sklearn import metrics
from sklearn.preprocessing import label_binarize

import numpy as np
import time
from flcore.clients.clientbase import Client
from flcore.utils_core.ALA import ALA
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
class clientALA(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)

        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx

        train_data = self.train_data
        self.ALA = ALA(self.id, self.loss, train_data, self.batch_size, 
                    self.rand_percent, self.layer_idx, self.eta, self.device)
    def eval_train_loss(self, task=None):
        trainloader = self.load_train_data(task=task)
        self.model.to(self.device).eval()
        total_loss, total_num = 0.0, 0

        with torch.no_grad():
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.loss(out, y)
                total_loss += loss.item() * y.size(0)
                total_num += y.size(0)

        return total_loss / total_num
    def train(self, task):
        #print("Client", self.id, "model device:", next(self.model.parameters()).device)
        trainloader = self.load_train_data(task=task)
        # self.model.to(self.device)
        self.model.to(self.device).train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                init_loss = self.loss(output, y)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.trained_local_loss = self.eval_train_loss(task=task)
        if self.args.teval:
            self.grad_eval(old_model=self.model)

        if self.args.pca_eval:
            self.proto_eval(model = self.model)

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        

    def local_initialization(self, received_global_model):
        received_global_model.to(self.device)
        self.model.to(self.device)
        self.ALA.adaptive_local_aggregation(received_global_model, self.model)
        self.init_local_loss = self.eval_train_loss(self.current_task)