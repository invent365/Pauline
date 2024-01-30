import torch, torch.nn as nn,torch.nn.functional as F

def yield_repr(seq_len, dim):
    positions = torch.arange(0,seq_len).unsqueeze(1) #specify the position board - a 2D object  
    dimensions = torch.arange(0,dim//2) #reference points that would hold future angles.
    theta = 1/(10000**(2*dimensions/dim)) #given by the paper
    snapshot = positions * theta #generate a 2-dimensional product of the positions and theta which act as our snapshot/frams
    sin_theta, cos_theta = torch.sin(snapshot),torch.cos(snapshot) #retrieve the sin and cos values of the products
    return sin_theta,cos_theta,dimensions
"""
We enforce different representations of simple positional values according to a specified function by generating a position-sensitive
scaffold (theta) which is broadcasted along the position board. This mesh solidifies relative position, boosted by using trigonometric functions
to create even more impressionistic patterns by virtue of position. We inject this into a three-dimensional tensor of shape (seq_len,dim,dim). The dim-dim
face holds the rotary "secret" by which positions can be man-oeuvred.
"""

def inject(batch_seq, seq_len, dim):
    sin_vector,cos_vector,dimensions = yield_repr(seq_len,dim)
    x = torch.zeros((batch_seq,seq_len, dim, dim)) #cache
    coses = dimensions*2 #we are leaving a space in between two cos values for sin values.
    sins = coses+1 #we are filling up the spaces between coses with sin values.
    x[:, :, coses, coses] = cos_vector #coses=[0,2,4..dim//2], coses,coses = [[0,0],[2,2]...[dim//2-1, dim//2-1]]
    x[:,:, coses, sins] = -sin_vector #coses, sins=[1,3,5...dim//2+1] = [[0,1], [2,3],[4,5]...dim//2-1, (dim//2)]
    x[:,:, sins, coses] = sin_vector#sins,coses=[[1,0],[3,2],[5,4]...(dim//2), (dim//2)-1]
    x[:, :, sins, sins] = cos_vector #sins, sins = [[1,1], [3,3], [5,5]...(dim//2), (dim//2)]
    return x

def yield_template(input_:torch.Tensor): #input should be of size(seq_len,dim) at the very least.
    batch_seq,seq_len,dim=input_.shape
    template = inject(batch_seq,seq_len,dim) #(seq_len,dim,dim) ->rotary matrix
    return  torch.einsum("ijkk, ijk->ijk", template, input_)
