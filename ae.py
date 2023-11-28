import torch
import torch.nn as nn

from utils import *
device = get_default_device()

class Encoder1(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, int(in_size/2))
        self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
        self.linear3 = nn.Linear(int(in_size/4), latent_size)
        self.relu = nn.ReLU(True)
        
    def forward(self, w):
        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        z = self.relu(out)
        return z
    
class Decoder1(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(out_size/4))
        self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
        self.linear3 = nn.Linear(int(out_size/2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        w = self.sigmoid(out)
        return w
    
class AE(nn.Module):
    def __init__(self, w_size, z_size):
        super().__init__()
        self.encoder = Encoder1(w_size, z_size)
        self.decoder = Decoder1(z_size, w_size)

  
    def training_step_ae(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder(z)

        loss = 1/n*torch.mean((batch-w1)**2)**2
        return loss

    def validation_step_ae(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder(z)

        loss = 1/n*torch.mean((batch-w1)**2)**2
        return {'val_loss': loss}
    
    def validation_epoch_end_ae(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end_ae(self, epoch, result):
        print("Epoch[{}], val_loss:{:.4f}".format(epoch, result['val_loss']))
    
def evaluate_ae(model, val_loader, n):
    outputs = [model.validation_step_ae(to_device(batch,device), n) for [batch] in val_loader]
    return model.validation_epoch_end_ae(outputs)

def training_ae(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(list(model.encoder.parameters())+list(model.decoder.parameters()))
        
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch=to_device(batch,device)
            
            # Train AE
            loss = model.training_step_ae(batch,epoch+1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


        result = evaluate_ae(model, val_loader, epoch+1)
        model.epoch_end_ae(epoch, result)
        history.append(result)
    return history
    
def testing_ae(model, test_loader):
    results=[]
    for [batch] in test_loader:
        batch=to_device(batch,device)
        w=model.decoder(model.encoder(batch))
        results.append(torch.mean((batch-w)**2,axis=1)**2)
    return results