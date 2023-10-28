import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class LogisticRegression:
    
    class _LogisticModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim, bias=False)
        
        def forward(self, x):
            return self.linear(x)
    
    def __init__(self, random_state=None, warm_start=False, max_iter=1000, C=1.0):
        torch.manual_seed(random_state if random_state is not None else 42)
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.C = C
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
    
    def fit(self, X, Y):
        if not self.warm_start or self.model is None:
            input_dim = X.size(1)
            output_dim = Y.max() + 1
            self.model = self._LogisticModel(input_dim, output_dim).cuda()
        
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, weight_decay=1/self.C)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.1, weight_decay=1/self.C)
        
        # for epoch in range(self.max_iter):
        #     outputs = self.model(X)
        #     loss = self.criterion(outputs.squeeze(), Y.long())
            
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     self.optimizer.step()

        pbar = tqdm(range(self.max_iter))
        for epoch in pbar:
            outputs = self.model(X)
            loss = self.criterion(outputs.squeeze(), Y.long())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.set_description(f"Epoch {epoch+1}/{self.max_iter}, Loss: {loss.item():.4f}")


    def predict(self, X):
        with torch.no_grad():
            outputs = self.model(X)
            predictions = outputs.argmax(dim=1)
        return predictions
    
    def predict_proba(self, X):
        with torch.no_grad():
            probas = self.model(X)
        return torch.cat((1 - probas, probas), dim=1)

    def score(self, X, Y):
        predictions = self.predict(X)
        accuracy = (predictions == Y).float().mean().item()
        return accuracy
