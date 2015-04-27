This document explains the what each directory contains

cifar10_deconv_examples
	- This model contains example pictures for cifar 10 (original code)

Notation: 
	- model + number: This model is trained with original data
	- model + r + number: This model is trained with augmented data (rotated 12 angles)
	- The number corresponds to the same model
		- 0: normal setup in plankton_conv_visualize_model.yaml
		- 1: wide1 setup plankton_conv_visualize_rotated_model_wide1.yaml
		- 2: 5+2=7 layer model to be specified (on 40 by 40)
			- yaml specification: 
		- 3: 8+2=10 layer model to be specified (on 95 by 95)
			- yaml specification: 
