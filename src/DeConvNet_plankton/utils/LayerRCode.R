channels = c(32,48,64)
kernel_shape = c(5,5,3)
pool_size = c(2,2,2)
numpixels = 40

Outputlayer2(channels,kernel_shape,pool_size, 40)
Outputlayer2(channels,kernel_shape,pool_size, 40, 1)

#Choose between toggle = 0 (print outputMatrix directly), versus any other toggle.



Outputlayer2<-function(channels,kernel_shape,pool_size,num_pixels, toggle = 0){
  
  n = length(channels)
  p = 3
  
  outputMatrix = matrix(0,nrow = n, ncol = p)
  
  outputMatrix[1,1] = num_pixels
  outputMatrix[1,2] = num_pixels - kernel_shape[1] + 1
  outputMatrix[1,3] = outputMatrix[1,2] / pool_size[1]
  
  for ( i in 2:n){
    outputMatrix[i,1] = outputMatrix[(i-1),3]     
    outputMatrix[i,2] = outputMatrix[i,1] - kernel_shape[i] + 1
    outputMatrix[i,3] = outputMatrix[i,2]/ pool_size[i]
  }
  
  if ( (toggle != 0) ){
    ## Print out params we want
    for( j in 1:2){
      
      if ( (j == 1)){
        readline("Now showing layers")
        tmp = ""
      } else {
        readline("Now showing up layers")
        tmp = "Up"
      }
      for (i in 1:n){
        if( (i == 1)){
          
          print(paste(tmp, "Layer", (i-1)))
          mystr = paste("Image Shape:", 1, "NUM_C", outputMatrix[i,1],outputMatrix[i,1])
          print(mystr)
          mystr = paste("Filter Shape:", channels[i], "NUM_C", kernel_shape[i], kernel_shape[i])
          print(mystr) 
          print(paste("Pool size is", pool_size[i]))
          
          readline("Pause to fill in")
        } else {
          
          print(paste("Layer", (i-1)))
          mystr = paste("Image Shape:", 1, channels[i-1], outputMatrix[i,1],outputMatrix[i,1])
          print(mystr)
          mystr = paste("Filter Shape:", channels[i], channels[i-1], kernel_shape[i], kernel_shape[i])
          print(mystr) 
          print(paste("Pool size is", pool_size[i]))
          readline("Pause to fill in")    
        }
        
      }
    }
    ## Now print out the down layers
    
    for (i in n:2){
      print(paste("Down Layer", (i-1)))  
      mystr = paste("Image Shape: 1", channels[i], outputMatrix[i,2], outputMatrix[i,2])
      print(mystr)
      mystr = paste("Filter Shape:", channels[i-1], channels[i], outputMatrix[i,2], outputMatrix[i,2], kernel_shape[i])
      print(mystr) 
      print(paste("Pool size is", pool_size[i]))
      readline("Pause to fill in")  
    } 
    i = 1
    print(paste("Down Layer", 0))
    mystr = paste("Image Shape: 1", channels[i], outputMatrix[i,2], outputMatrix[i,2])
    print(mystr)    
    mystr = paste("Filter Shape: NUM_C", channels[i], outputMatrix[i,2], outputMatrix[i,2], kernel_shape[i])
    print(mystr) 
    print(paste("Pool size is", pool_size[i]))
    readline("Pause to fill in")    
    
  } else {
    print(outputMatrix)
  }
}