import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

start = []
end = []
diff = []

starta = []
enda = []
diffa = []

fo1 = open("test_set_open1.csv", "r")
for line in fo1:
    starta.append(float(line[:len(line)-1]))

fc1 = open("test_set_close1.csv", "r")
for line in fc1:
    enda.append(float(line[:len(line)-1]))

for i in range(0, len(starta)):
    if i == 0:
       pass
    else:
        diffa.append(enda[i]-enda[i-1])

fo = open("train_set_open1.csv", "r")
for line in fo:
    start.append(float(line[:len(line)-1]))

fc = open("train_set_close1.csv", "r")
for line in fc:
    end.append(float(line[:len(line)-1]))

for i in range(0, len(start)):
    if i == 0:
       pass
    else:
        diff.append(end[i]-end[i-1])

testset = []
temp = copy.deepcopy(diff)
for i in range(0, len(diffa)):
    temp.append(diffa[i])
    testset.append(temp)
    temp = copy.deepcopy(testset[i])

def toTensor(diff):
    tensor = torch.zeros(1, 1)
    tensor[0][0] = diff
    return tensor

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return category_i

def lastEle(diff):
    if diff[len(diff) - 1] > 0:
        return 1
    return 0

def evaluate(diff):
    model.hidden = model.init_hidden()
    tag_scores = model(diff)
    return tag_scores

class LSTMTagger(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, output_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, inputs):
        for i in range(len(inputs)-1):
            iv = autograd.Variable(toTensor(inputs[i]))
            lstm_out, self.hidden = self.lstm(
                iv.view(1, 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(1, -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


model = LSTMTagger(1, 126, 2)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# input = autograd.Variable(toTensor(diff[0]))
# print(diff[:10])
# tag_scores = model(diff[:10])
# print(categoryFromOutput(tag_scores))
# print(diff[9])

print_every = 50
n_iters = len(diff)
count  = 0
count1 = 0
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    # for iter in range(0, len(diff)-1, 1):
        #print(diff[iter:iter+5])
        # target = autograd.Variable(torch.LongTensor([lastEle(diff[iter:iter+5])]))
        # # # Step 1. Remember that Pytorch accumulates gradients.
        # # # We need to clear them out before each instance
        # model.zero_grad()

        # # # Also, we need to clear out the hidden state of the LSTM,
        # # # detaching it from its history on the last instance.
        # model.hidden = model.init_hidden()

        # # # Step 2. Get our inputs ready for the network, that is, turn them into
        # # # Variables of word indices.

        # if len(diff[iter:iter+5]) < 2:
        #     continue
        # # # Step 3. Run our forward pass.
        # tag_scores = model(diff[iter:iter+5])
        # # # print(tag_scores)
        # # # print(target)

        # # Step 4. Compute the loss, gradients, and update the parameters by
        # #  calling optimizer.step()
        # loss = loss_function(tag_scores, target)
        # loss.backward()
        # optimizer.step()
    target = autograd.Variable(torch.LongTensor([lastEle(diff[:len(diff)-1])]))
    # # Step 1. Remember that Pytorch accumulates gradients.
    # # We need to clear them out before each instance
    model.zero_grad()

    # # Also, we need to clear out the hidden state of the LSTM,
    # # detaching it from its history on the last instance.
    model.hidden = model.init_hidden()

    # # Step 2. Get our inputs ready for the network, that is, turn them into
    # # Variables of word indices.

    if len(diff[:len(diff)-1]) < 2:
        continue
    # # Step 3. Run our forward pass.
    tag_scores = model(diff[:len(diff)-1])


    # Step 4. Compute the loss, gradients, and update the parameters by
    #  calling optimizer.step()
    loss = loss_function(tag_scores, target)
    loss.backward()
    optimizer.step()

    # for iter in range(2, len(diff)-1):
    #     target = autograd.Variable(torch.LongTensor([lastEle(diff[iter:])]))
    #     # # Step 1. Remember that Pytorch accumulates gradients.
    #     # # We need to clear them out before each instance
    #     model.zero_grad()

    #     # # Also, we need to clear out the hidden state of the LSTM,
    #     # # detaching it from its history on the last instance.
    #     model.hidden = model.init_hidden()

    #     # # Step 2. Get our inputs ready for the network, that is, turn them into
    #     # # Variables of word indices.

    #     if len(diff[iter:]) < 2:
    #         continue
    #     # # Step 3. Run our forward pass.
    #     tag_scores = model(diff[iter:])
    #     # # print(tag_scores)
    #     # # print(target)

    #     # Step 4. Compute the loss, gradients, and update the parameters by
    #     #  calling optimizer.step()
    #     loss = loss_function(tag_scores, target)
    #     loss.backward()
    #     optimizer.step()

        # target = autograd.Variable(torch.LongTensor([lastEle(diff[:iter+1])]))
        # # Step 1. Remember that Pytorch accumulates gradients.
        # # We need to clear them out before each instance
        # model.zero_grad()
        # # print(target)

        # # Also, we need to clear out the hidden state of the LSTM,
        # # detaching it from its history on the last instance.
        # model.hidden = model.init_hidden()

        # # Step 2. Get our inputs ready for the network, that is, turn them into
        # # Variables of word indices.

        # # Step 3. Run our forward pass.
        # tag_scores = model(diff[:iter+1])
        # tag = categoryFromOutput(tag_scores)
        # if tag == lastEle(diff[:iter+1]):
        #     count += 1

        # # Step 4. Compute the loss, gradients, and update the parameters by
        # #  calling optimizer.step()
        # loss = loss_function(tag_scores, target)
        # loss.backward()
        # optimizer.step()

        # if iter % print_every == 0:
        #     print('%d %d %d%% %.4f %.2f' % (epoch, iter, iter / n_iters * 100, loss, count/print_every*100))
        #     count = 0
        #     # for iter in range(0, len(testset)):
        #     #     tag_scores = evaluate(testset[iter])
        #     #     tag = categoryFromOutput(tag_scores)
        #     #     if tag == lastEle(testset[iter]):
        #     #         count1 += 1
        #     # print('%.2f' % ((count1/(len(testset)-1))*100))
        #     # count1 = 0
    print('%.4f' % (loss))
    for iter in range(0, len(testset)):
        tag_scores = evaluate(testset[iter])
        tag = categoryFromOutput(tag_scores)
        if tag == lastEle(testset[iter]):
            count1 += 1
    print('%.2f' % ((count1/(len(testset)-1))*100))
    count1 = 0