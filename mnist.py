import numpy as np

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))
    
#read in data from train.csv
print "reading in data"
x = [] #input list
y = [] #output list

data_file = open('train.csv', 'r') #open file
lineIn = data_file.readline() #get rid of the header in the file
while True: #read until null
    lineIn = data_file.readline()
    if lineIn == '':
        break
    #parse the line
    lineIn = lineIn.rstrip()
    lineInSplit = lineIn.split(',')

    #build the output array, set the index of which digit 
    #it is labeled to true, the rest are false
    toY = [0,0,0,0,0,0,0,0,0,0]
    toY[int(lineInSplit[0])]=1
    y.append(toY) #add to output list

    #build input list
    toX = []
    for i in range(1, 785):
        toX.append(int(lineInSplit[i]))
    x.append(toX)


x = np.array([x])
                
y = np.array([y])

np.random.seed(1)

# randomly initialize weights with mean 0
syn0 = 2*np.random.random((len(x[0]),len(x))) - 1
syn1 = 2*np.random.random((len(x),10)) - 1

for j in xrange(60000):

	# Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # error
    l2_error = y - l2
    
    if (j% 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    # mutes confident error changes 
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    # mutes confident error changes 
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

l0 = X
l1 = nonlin(np.dot(l0,syn0))
l2 = nonlin(np.dot(l1,syn1))

for i in l2:
	print str(round(i))