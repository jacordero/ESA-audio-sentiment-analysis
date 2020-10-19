import requests
import numpy as np


def download_resource(url, output_filename):
	r = requests.get(url)

	with open(output_filename, 'wb') as f:
		f.write(r.content)


if __name__ == "__main__":

	#Download dataset from ESARepo/Audio/Datasets/Development
	#url = "https://surfdrive.surf.nl/files/index.php/s/KlU39g1GB9gq8ZN/download"
	#output_filename = "../data/raw/example_dataset.npz"
	#download_resource(url, output_filename)

	#Download model from ESARepo/Audio/Models/Development
	#url = "https://surfdrive.surf.nl/files/index.php/s/PRb56AcTxTmPrKp/download"
	#output_filename = "../models/example_model.h5"
	#download_resource(url, output_filename)

	