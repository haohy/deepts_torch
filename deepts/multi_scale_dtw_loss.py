import torch
from deepts.soft_dtw import pairwise_distances, SoftDTWBatch
from deepts.path_soft_dtw import PathDTWBatch 


def multi_scale_dtw_loss(outputs, targets, scale_list, alpha=0.5, gamma=0.01, device='cpu'):
	# outputs, targets: shape (batch_size, N_output, 1)
	batch_size, N_output = outputs.shape
	loss_shape = 0
	softdtw_batch = SoftDTWBatch.apply
	D = torch.zeros((batch_size, N_output,N_output )).to(device)
	for k in range(batch_size):
		Dk = pairwise_distances(targets[k,:].view(-1,1),outputs[k,:].view(-1,1))
		D[k:k+1,:,:] = Dk     
		
	loss_shape = 0.0
	for scale in scale_list:
		loss_shape = loss_shape + softdtw_batch(D,gamma,scale)
	loss_shape = loss_shape/len(scale_list)

	path_dtw = PathDTWBatch.apply
	path = path_dtw(D,gamma)           
	Omega =  pairwise_distances(torch.arange(1,N_output+1).view(N_output,1)).to(device)
	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) 
	loss_temporal = loss_temporal
	loss = alpha*loss_shape+ (1-alpha)*loss_temporal
	
	return loss
	# return softdtw_batch(D,gamma,scale_list)