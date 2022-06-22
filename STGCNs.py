import torch as tc
import torch_geometric as tg
import torch.nn.functional as F
import torch_geometric.transforms as T

#===========================================================
# GNN usando recorrência espacial e temporal com módulos GRU
#	- inicializa os valores de S
#   - usa dropout
#===========================================================
class DoubleGatedGCNv1(tc.nn.Module):
    
    def __init__(self, params, tau, loop=1, dropout=0):
        super(DoubleGatedGCNv1, self).__init__()
        
        self.device = ('cuda:0' if tc.cuda.is_available() else 'cpu')

        self.delay = int (tau if tau > 1 else 1)
        self.samples = params['municipios']
        self.features = params['dim_estat'] + params['dim_dinam']
        self.hidden = self.features * self.delay
        self.output = params['dim_dinam']
        
        self.init = tc.nn.Linear(self.features, self.features)
        self.conv = tg.nn.GatedGraphConv(out_channels=self.features, num_layers=loop)
        self.rec = tc.nn.GRUCell(self.hidden, self.hidden)
        self.mlp = tc.nn.Sequential(
            tc.nn.Linear(self.hidden, self.hidden),
            tc.nn.ReLU(),
            tc.nn.Linear(self.hidden, self.hidden),
            tc.nn.ReLU(),
            tc.nn.Linear(self.hidden, self.output),
            tc.nn.Dropout(p=dropout),
            tc.nn.Tanhshrink()
        )
    
    def forward(self, Xd, Xs, A, W, mask, H=None):
        H = tc.zeros([self.samples, self.hidden]).to(self.device) if H is None else H
        S = [ self.init(tc.zeros([self.samples, self.features]).to(self.device)) for t in range(self.delay-1) ]
        W = (W * mask.float()).sum(0)
        A = A[ :, mask.sum(0) ]
        Y = []
        
        for t in range(Xd.size(0)):
            Z = tc.cat([Xd[t], Xs], dim=1)
            Z = F.relu(self.conv(Z, A, W))
            S.append(Z)
            Z = tc.cat(S, dim=1)
            H = self.rec(Z, H)
            Z = self.mlp(H)
            Y.append(Z)
            S.pop(0)
        
        while len(S):
            S.pop(0)
        return tc.stack(Y), H

    def forecast(self, Xd, Xs, A, W, mask, H=None, qt_days=1, drop_past=False):
        H = tc.zeros([self.samples, self.hidden]).to(self.device) if H is None else H
        S = [ self.init(tc.zeros([self.samples, self.features]).to(self.device)) for t in range(self.delay-1) ]
        W = (W * mask.float()).sum(0)
        A = A[ :, mask.sum(0) ]
        Y = []
        
        # série temporal informada
        for t in range(Xd.size(0)):
            Z = tc.cat([Xd[t], Xs], dim=1)
            Z = F.relu(self.conv(Z, A, W))
            S.append(Z)
            Z = tc.cat(S, dim=1)
            H = self.rec(Z, H)
            Z = self.mlp(H)
            Y.append(Z)
            S.pop(0)
        
        if drop_past:
            Y = [ Xd[Xd.size(0) - 1] ]
        
        # série temporal prevista
        for t in range(qt_days):
            Z = tc.cat([Y[-1], Xs], dim=1)
            Z = F.relu(self.conv(Z, A, W))
            S.append(Z)
            Z = tc.cat(S, dim=1)
            H = self.rec(Z, H)
            Z = self.mlp(H)
            Y.append(Z)
            S.pop(0)
        
        while len(S):
            del S[0]
        return tc.stack(Y), H

#===========================================================
# GNN usando recorrência espacial e temporal com módulos GRU
#	- inicializa os valores de S
#   - usa batch normalization
#===========================================================
class DoubleGatedGCNv2(tc.nn.Module):
    
    def __init__(self, params, tau, loop=1):
        super(DoubleGatedGCNv2, self).__init__()
        
        self.device = ('cuda:0' if tc.cuda.is_available() else 'cpu')

        self.delay = int (tau if tau > 1 else 1)
        self.samples = params['municipios']
        self.features = params['dim_estat'] + params['dim_dinam']
        self.hidden = self.features * self.delay
        self.output = params['dim_dinam']
        
        self.init = tc.nn.Linear(self.features, self.features)
        self.conv = tg.nn.GatedGraphConv(out_channels=self.features, num_layers=loop)
        self.norm = tc.nn.BatchNorm1d(self.features)
        self.rec = tc.nn.GRUCell(self.hidden, self.hidden)
        self.mlp = tc.nn.Sequential(
            tc.nn.Linear(self.hidden, self.hidden),
            tc.nn.ReLU(),
            tc.nn.Linear(self.hidden, self.hidden),
            tc.nn.ReLU(),
            tc.nn.Linear(self.hidden, self.output),
            tc.nn.Tanhshrink()
        )
    
    def forward(self, Xd, Xs, A, W, mask, H=None):
        H = tc.zeros([self.samples, self.hidden]).to(self.device) if H is None else H
        S = [ self.init(tc.zeros([self.samples, self.features]).to(self.device)) for t in range(self.delay-1) ]
        W = (W * mask.float()).sum(0)
        A = A[ :, mask.sum(0) ]
        Y = []
        
        for t in range(Xd.size(0)):
            Z = tc.cat([Xd[t], Xs], dim=1)
            Z = F.relu(self.norm(self.conv(Z, A, W)))
            S.append(Z)
            Z = tc.cat(S, dim=1)
            H = F.relu(self.rec(Z, H))
            Z = self.mlp(H)
            Y.append(Z)
            S.pop(0)
        
        while len(S):
            S.pop(0)
        return tc.stack(Y), H

    def forecast(self, Xd, Xs, A, W, mask, H=None, qt_days=1, drop_past=False):
        H = tc.zeros([self.samples, self.hidden]).to(self.device) if H is None else H
        S = [ self.init(tc.zeros([self.samples, self.features]).to(self.device)) for t in range(self.delay-1) ]
        W = (W * mask.float()).sum(0)
        A = A[ :, mask.sum(0) ]
        Y = []
        
        # série temporal informada
        for t in range(Xd.size(0)):
            Z = tc.cat([Xd[t], Xs], dim=1)
            Z = F.relu(self.norm(self.conv(Z, A, W)))
            S.append(Z)
            Z = tc.cat(S, dim=1)
            H = F.relu(self.rec(Z, H))
            Z = self.mlp(H)
            Y.append(Z)
            S.pop(0)
        
        if drop_past:
            Y = [ Xd[Xd.size(0) - 1] ]
        
        # série temporal prevista
        for t in range(qt_days):
            Z = tc.cat([Y[-1], Xs], dim=1)
            Z = F.relu(self.norm(self.conv(Z, A, W)))
            S.append(Z)
            Z = tc.cat(S, dim=1)
            H = self.rec(Z, H)
            Z = self.mlp(H)
            Y.append(Z)
            S.pop(0)
        
        while len(S):
            del S[0]
        return tc.stack(Y), H

# GNN usando recorrência espacial e temporal com módulos GRU
class DoubleGatedGCNv3(tc.nn.Module):
    
    def __init__(self, params, tau, loop=1, channels=4):
        super(DoubleGatedGCNv3, self).__init__()

        self.delay = int (tau if tau > 1 else 1) - 1
        self.samples = params['municipios']
        self.features = params['dim_estat'] + params['dim_dinam']
        self.hidden = self.features * (self.delay + 1)
        self.output = params['dim_dinam']
        
        self.conv = tg.nn.GatedGraphConv(out_channels=self.features, num_layers=loop)
        self.lin = tc.nn.Linear(channels, 1, bias=False)
        self.rec = tc.nn.GRUCell(self.hidden, self.hidden)
        self.mlp = tc.nn.Sequential(
            tc.nn.Linear(self.hidden, self.hidden),
            tc.nn.ReLU(),
            tc.nn.Linear(self.hidden, self.hidden),
            tc.nn.ReLU(),
            tc.nn.Linear(self.hidden, self.output),
            tc.nn.Tanhshrink()
        )
    
    def forward(self, Xd, Xs, A, W, mask, H=None):
        H = tc.zeros([self.samples, self.hidden]).to(self.device) if H is None else H
        S = [ tc.zeros([self.samples, self.features]).to(self.device) for t in range(self.delay) ]
        W = (W * mask.float()).t()
        W = (self.lin(W)).t()
        A = A[ :, mask.sum(0) ]
        Y = []
        
        for t in range(Xd.size(0)):
            Z = tc.cat([Xd[t], Xs], dim=1)
            Z = F.relu(self.conv(Z, A, W))
            S.append(Z)
            Z = tc.cat(S, dim=1)
            H = self.rec(Z, H)
            Z = self.mlp(H)
            Y.append(Z)
            del S[0]
        
        while len(S):
            del S[0]
        return tc.stack(Y), H

    def forecast(self, Xd, Xs, A, W, mask, H=None, qt_days=1):
        H = tc.zeros([self.samples, self.hidden]).to(self.device) if H is None else H
        S = [ tc.zeros([self.samples, self.features]).to(self.device) for t in range(self.delay) ]
        W = (W * mask.float()).t()
        W = (self.lin(W)).t()
        A = A[ :, mask.sum(0) ]
        Y = []
        
        # série temporal informada
        for t in range(Xd.size(0)):
            Z = tc.cat([Xd[t], Xs], dim=1)
            Z = F.relu(self.conv(Z, A, W))
            S.append(Z)
            Z = tc.cat(S, dim=1)
            H = self.rec(Z, H)
            Z = self.mlp(H)
            Y.append(Z)
            del S[0]
        
        # série temporal prevista
        for t in range(qt_days):
            Z = tc.cat([Y[-1], Xs], dim=1)
            Z = F.relu(self.conv(Z, A, W))
            S.append(Z)
            Z = tc.cat(S, dim=1)
            H = self.rec(Z, H)
            Z = self.mlp(H)
            Y.append(Z)
            del S[0]
        
        while len(S):
            del S[0]
        return tc.stack(Y), H

# GNN usando recorrência espacial e temporal com módulos GRU
class DoubleGatedGCNv4(tc.nn.Module):
    
    def __init__(self, params, tau, loop=1, channels=4):
        super(DoubleGatedGCNv4, self).__init__()

        self.delay = int (tau if tau > 1 else 1) - 1
        self.samples = params['municipios']
        self.features = params['dim_estat'] + params['dim_dinam']
        self.hidden = self.features * (self.delay + 1)
        self.output = params['dim_dinam']
        
        self.device = ('cuda:0' if tc.cuda.is_available() else 'cpu')
        
        self.inferState = tc.nn.Sequential(
            tc.nn.Linear(self.output, self.features),
            tc.nn.ReLU(),
            tc.nn.Linear(self.features, self.features),
            tc.nn.Tanhshrink()
        )
        self.inferContext = tc.nn.Sequential(
            tc.nn.Linear(self.features, self.hidden),
            tc.nn.ReLU(),
            tc.nn.Linear(self.hidden, self.hidden),
            tc.nn.Tanhshrink()
        )
        self.gnn = [ 
        	tc.nn.Sequential(
        		tg.nn.GatedGraphConv(out_channels=self.features, num_layers=loop),
	        	tc.nn.BatchNorm1d(self.features),
        	).to(self.device)
        	for i in range(channels) 
        ]
        self.cnn = tc.nn.Conv1d(channels, 1, kernel_size=1, bias=False)
        self.rec = tc.nn.GRUCell(self.hidden, self.hidden)
        self.mlp = tc.nn.Sequential(
            tc.nn.Linear(self.hidden, self.hidden),
            tc.nn.ReLU(),
            tc.nn.Linear(self.hidden, self.hidden),
            tc.nn.ReLU(),
            tc.nn.Linear(self.hidden, self.output),
            tc.nn.Tanhshrink()
        )
    
    def forward(self, Xd, Xs, A, W, mask, H=None):
        S_init = self.inferState(Xd[0])
        H = self.inferContext(S_init) if H is None else H
        S = [ S_init.clone() for t in range(self.delay) ]
        Y = []

        for t in range(Xd.size(0)):
            X = tc.cat([Xd[t], Xs], dim=1)
            C = tc.stack([ F.relu(g(X, A[:, m], W[m])) for m, g in zip(mask, self.gnn) ], dim=1)
            Z = F.relu(self.cnn(C)).view(-1, self.features)
            S.append(Z)
            U = tc.cat(S, dim=1)
            H = self.rec(U, H)
            Y.append(self.mlp(H))
            del S[0]
        
        while len(S):
            del S[0]
        return tc.stack(Y), H

    def forecast(self, Xd, Xs, A, W, mask, H=None, qt_days=1):
        S_init = self.inferState(Xd[0])
        H = self.inferContext(S_init) if H is None else H
        S = [ S_init.clone() for t in range(self.delay) ]
        Y = []
        
        # série temporal informada
        for t in range(Xd.size(0)):
            X = tc.cat([Xd[t], Xs], dim=1)
            C = tc.stack([ F.relu(self.gnn(X, A[:, m], W[m])) for m in mask ], dim=1)
            # torch.Size([5570, 4, 5]) torch.Size([5570, 5])
            Z = F.relu(self.cnn(C)).view(-1, self.features)
            S.append(Z)
            U = tc.cat(S, dim=1)
            H = self.rec(U, H)
            Y.append(self.mlp(H))
            del S[0]
        
        # série temporal prevista
        for t in range(qt_days):
            X = tc.cat([Y[-1], Xs], dim=1)
            C = tc.stack([ F.relu(self.gnn(X, A[:, m], W[m])) for m in mask ], dim=1)
            # torch.Size([5570, 4, 5]) torch.Size([5570, 5])
            Z = F.relu(self.cnn(C)).view(-1, self.features)
            S.append(Z)
            U = tc.cat(S, dim=1)
            H = self.rec(U, H)
            Y.append(self.mlp(H))
            del S[0]
        
        while len(S):
            del S[0]
        return tc.stack(Y), H
