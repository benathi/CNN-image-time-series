Plankton Project Notes
======================

April 12
	- For each class, plot  of average feature map values (they're non negative - check) (for a given layer)
	- also display sample images for each class

Should do before end of day:
1. Fix the b and do visualization again
2. Make it automatically shows kernels in each layer
	- design this : either call multiple times or one time but layer it differently

3. Make it automatically shows all layers (optional)


Next:
	Can we use feature map as distance for hierachical clustering?
	- yes. 
-----------------------
ToDo:


	- For each feature in a given layer, by selecting the top 10% (by activation value),
	show the histogram of classes that excites this feature: does this make sense?


April 18
	- Finished HAC to cluster classes based on features in layer3
	- Should we do it for other layer? (like layer0, layer1?)

April 19
	- 