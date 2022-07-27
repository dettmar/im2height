import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from im2height import Im2Height
from data import NpyPredictionDataset


load_config = {
	"batch_size": 32,
	"pin_memory": True,
	"num_workers": 32
}

def run(input, output, weights):

	# load weights
	model = Im2Height.load_from_checkpoint(weights)
	device = torch.device("cuda")
	model.to(device)

	data_loader = torch.utils.data.DataLoader(NpyPredictionDataset(input), **load_config)

	# predict and store
	for filenames, tensors in data_loader:
		
		with torch.no_grad():
			tensors = tensors.to(device)
			predictions = model(tensors)
		
		for filename, img in zip(filenames, predictions.cpu().detach().numpy()):
			np.save(f"{output}/{os.path.basename(filename)}", img[0])



if __name__ == '__main__':

	DESCRIPTION = """
	Command line interface for batch compatible generic model prediction.

	Usage:
		$ python predict.py -i path/to/my/files/*.npy -o my/output/path -w pth/to/weight.ckpt

	Performs predictions for all .npy files obtained through shell globbing
	and serialises the outputs as specified in the main routine below.
	"""

	parser = argparse.ArgumentParser(description=DESCRIPTION)
	parser.add_argument("-i", "--input", type=str, help="Input file paths", required=True, nargs="+")
	parser.add_argument("-o", "--output", type=str, help="Output directory", required=True)
	parser.add_argument("-w", "--weights", type=str, help="Weights path", required=True)
	args = parser.parse_args()
	run(**vars(args))
