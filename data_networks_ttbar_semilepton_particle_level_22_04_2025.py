
from torch.utils.data import Dataset, IterableDataset
import torch.nn as nn
import glob 
import torch 
import uproot 
import tables 
import pandas as pd 
import numpy as np
class dataset( Dataset ):
    def __init__( self, path,  device):
        self.files=glob.glob(path)
        self.device=device

        dfs=[]
        for f in self.files:
            thedata=pd.read_hdf(f, 'df')
            dfs.append(thedata)
        big_df=pd.concat( dfs )
        self.length=len(big_df)
        

        self.control_vars = torch.Tensor(big_df[['control_cnr_crn', 'control_cnk_ckn', 'control_crk_ckr']].values).to(self.device)

        self.weights  =torch.Tensor(big_df[["weight_sm", "weight_lin"]].values).to(self.device)

        self.variables = torch.Tensor(big_df[['l_px', 'l_py', 'l_pz',
                                        'b1_px', 'b1_py', 'b1_pz',
                                        'b2_px', 'b2_py', 'b2_pz',
                                        'q1_px', 'q1_py', 'q1_pz',
                                        'q2_px', 'q2_py', 'q2_pz',
                                        'met_px', 'met_py']].values).to(self.device)

        

        
   
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.weights[i,:], self.control_vars[i,:], self.variables[i,:]
    
    def name_cvaris(self,index):
        if index == 0:
            return  "$c_{rn} - c_{nr}$"
        elif index == 1:
            return  "$c_{kn} - c_{nk}$"
        elif index == 2:
            return  "$c_{rk} - c_{kr}$"
               
class Network(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.main_module = nn.Sequential( 
            nn.Linear(17,128), 
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1 ),
        )
        self.main_module.to(device)
        self.device=device

    def forward(self, x):
        cpx= torch.stack([-x[:,0], -x[:,1] , -x[:,2],    # -lep
                          -x[:,6], -x[:,7] , -x[:,8],    # -b2 
                          -x[:,3], -x[:,4], -x[:,5],     # -b1 # this shouldnt add information but ok 
                          -x[:,12], -x[:,13] , -x[:,14], # -q2
                          -x[:,9], -x[:,10] , -x[:,11],  # -q1
                          -x[:,15], -x[:,16]          ],  # -met
                         dim=1).to(self.device)
        return self.main_module(x)-self.main_module(cpx)
    

def network(device):
    return Network(device)

class network_noeq(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.main_module = nn.Sequential( 
            nn.Linear(17,128), 
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1 ),
        )
        self.main_module.to(device)

    def forward(self, x):
        return self.main_module(x)
