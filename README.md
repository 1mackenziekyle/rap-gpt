![](rap-gpt.gif)
# Rap-GPT

A Generative Pre-Trained Model used to generate new rap lyrics

## Training

Trained for ~1 hour on dataset of rap lyrics from many different rappers


## Training Checkpoints

### **no-training.pt** (No training)

### No training:

Example output:

*>FAbr1V½¤U¯­P¡j!`S5xê?(ë@'Sä:sM»83¥JD;»¹Ä,`.½²G£"	u¯20ðy'.F&ìâ¿¥Tq.°Hµ*

*¡s±«$B*


### **1hr-rap.pt**: 

### After ~1hr of training on rap dataset, the model makes decent sentences and is sometimes rhyming.

Validation Loss: 1.220

Example output:

*Bracelete*

*For two baby, real friend!*

*As the prometer bring me, right through the second through*

*I know one time away master, friends try to pay you*

*Perform these things on my fucking mind tonguer*

*You probably see me, 'Bout this lucky's deminite without the grain*

*I always keep a man and a club and on they'll bucking*

*Pinnies to a maseration, to a man and and on a maseration*

*But not just hand a stranger than Island, I'mpearation*

### **Reduce learning rate**

lr 3e-4 -> 1e-5
