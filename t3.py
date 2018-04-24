import numpy as np
import time

start = []
end = []
diff = []

fo = open("open2.csv", "r")
for line in fo:
    start.append(float(line[:len(line)-1]))

fc = open("close2.csv", "r")
for line in fc:
    end.append(float(line[:len(line)-1]))

for i in range(0, len(start)):
    if i == 0:
       pass
    else:
        diff.append(end[i]-end[i-1])
    # print(diff)
    # time.sleep(2)

# print(diff[len(diff)-1])

# ndiff = []
# for i in range(0, len(diff)):
#     ndiff.append(i)

# print(diff)

import torch
import torch.nn as nn
from torch.autograd import Variable

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return category_i

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

n_hidden = 256
rnn = RNN(1, n_hidden, 2)

def toTensor(diff):
    tensor = torch.zeros(1, 1)
    tensor[0][0] = diff
    return tensor

# input = Variable(toTensor(diff[0]))
# hidden = Variable(torch.zeros(1, n_hidden))
# #print(input)

# output, next_hidden = rnn(input, hidden)
# print(output)
# guess = categoryFromOutput(output)
# correct = '✓' if guess == categoryFromOutput(output) else '✗ (%s)' % categoryFromOutput(output)
# print(correct)

criterion = nn.NLLLoss()

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(diff):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    # print(diff)
    # time.sleep(1)

    for i in range(len(diff)-1):
        input = Variable(toTensor(diff[i]))
        output, hidden = rnn(input, hidden)
    
    # target = Variable(torch.LongTensor([int(diff[len(diff)-1])]))
    # target += 500
    target = Variable(torch.LongTensor([0]))
    if diff[len(diff) - 1] > 0:
        target = Variable(torch.LongTensor([1]))
    loss = criterion(output, target)
    # print("loss:{0}".format(loss))
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0], target

# # Just return an output given a line
def evaluate(diff):
    hidden = rnn.initHidden()

    for i in range(len(diff)):
        input = Variable(toTensor(diff[i]))
        output, hidden = rnn(input, hidden)

    return output

import time
import math

n_iters = len(diff)
print_every = 50

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def lastEle(diff):
    if diff[len(diff) - 1] > 0:
        return 1
    return 0

# print()

test = diff
# test.append(-125)

start = time.time()
count = 0
for step in range(0, 10000):
    for iter in range(2, len(diff)):
        category = lastEle(diff[:iter+1])
        output, loss, target = train(diff[:iter+1])
        guess = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        if correct == '✓':
            count += 1
        # if step > 5:
        #     output1 = evaluate(test)
        #     correct = '✓' if guess == categoryFromOutput(output1) else '✗ (%s)' % categoryFromOutput(output1)
        #     if correct == '✓':
        #         count += 1

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d %d%% (%s) %.4f %s %s' % (step, iter, iter / n_iters * 100, timeSince(start), loss, guess, correct))
            print('%.2f' % ((count/print_every) * 100))
            count = 0
            # if step > 5:
            #     print('%.2f' % ((count/print_every) * 100))

            
           
           
            