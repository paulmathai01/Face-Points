# My initial work on a simple CNN to detect facial key points on tensorflow
   
This network contains 2 convolutional layers and 2 fully connected layers, and the output layer is linear. Each convolutional layer is followed by a max-pooling layer, and use relu as activation function.

The loss of this system is squared errors(l2 loss), and use AdamOptimizer to minimize the loss.


See that there is a big gap between train loss and validation loss, for the reason of dropout and regularization?

  
batch size | initial learning rate | early stop patience | validation size | l2 regularization coeffient | learning rate decay rate | optimizer | best loss
---|---|---|---|---|---|---|---
64 | 1.00E-03 | 50 | 100 | 2.00E-07 | 0.95 | AdamOptimizer | 0.015985    
64 | 5.00E-04 | 50 | 100 | 2.00E-07 | 0.98 | AdamOptimizer | 0.016929  
64 | 5.00E-04 | 50 | 100 | 2.00E-07 | 0.99 | AdamOptimizer | 0.021319  

Some things to pay attention to are:  
 
* The initial learning rate should not be too large, or the system will fail to converge   
* Large batch size will generate more stable result, but large batch size will consume more memory   
* Learning rate decay rate should not be too small, or the step will shrink too fast, and it's hard for the loss to get to minimum value   

Now the final score is not very good, a lot can be down to promote it, like training one model for each keypoint, dealing with null value, and flipping the image.   





