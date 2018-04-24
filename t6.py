import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

end = []
diff = []
for i in range(0, 18):
    fc = open("up{0}.csv".format(i), "r")
    temp = []
    for line in fc:
        temp.append(float(line))
    end.append(temp)

for i in range(0, len(end)):
    if i == 0:
       pass
    else:
        temp = []
        for j in range(1, len(end[i])-1):
            temp.append(end[i][j]-end[i][j-1])
        diff.append(temp)

testset = []
fa = open("test_up0.csv", "r")
loadtest = []
for line in fa:
    loadtest.append(float(line))

for i in range(1, len(loadtest)-1):
    testset.append(loadtest[i]-loadtest[i-1])
# print(testset）

testset1 = []
fa = open("test_up1.csv", "r")
loadtest = []
for line in fa:
    loadtest.append(float(line))

for i in range(1, len(loadtest)):
    testset1.append(loadtest[i]-loadtest[i-1])

# print(testset1)

def toTensor(diff):
    tensor = torch.zeros(1, 1)
    tensor[0][0] = diff
    return tensor

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return category_i

def evaluate(diff):
    model.hidden = model.init_hidden()
    tag_scores = model(diff)
    return tag_scores

class LSTMTagger(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

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
        for i in range(len(inputs)):
            iv = autograd.Variable(toTensor(inputs[i]))
            lstm_out, self.hidden = self.lstm(
                iv.view(1, 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(1, -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


model = LSTMTagger(1, 126, 2)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005)

print_every = 50
n_iters = len(diff)
count  = 0
count1 = 0
for epoch in range(1000):
    for iter in range(0, len(diff)):
        if (len(diff[iter]) < 2):
            continue
        for j in range(1, len(diff[iter])):
            if j == len(diff[iter])-1:
                target = autograd.Variable(torch.LongTensor([1]))
            else:
                target = autograd.Variable(torch.LongTensor([0]))

            # # Step 1. Remember that Pytorch accumulates gradients.
            # # We need to clear them out before each instance
            model.zero_grad()

            # # Also, we need to clear out the hidden state of the LSTM,
            # # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # # Step 2. Run our forward pass.
            tag_scores = model(diff[iter][:j+1])

            # Step 3. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, target)
            loss.backward()
            optimizer.step()

    idx1 = []
    for iter in range(0, len(diff)):
        if (len(diff[iter]) < 2):
            continue
        count_local = 0
        for j in range(1, len(diff[iter])):
            tag_scores = evaluate(diff[iter][:j+1])
            tag = categoryFromOutput(tag_scores)
            if tag == 1:
                count_local += 1
        idx1.append(count_local)

    print('loss %.4f train_set_count %d len %s' % (loss, len(idx1), len(diff)))
    print(idx1)
    idx = []
    count1 = 0
    for iter in range(0, len(testset)):
        tag_scores = evaluate(testset[:iter+1])
        tag = categoryFromOutput(tag_scores)
        if tag == 1:
            idx.append(iter)
    print('found %d epoch %d len %d' % (len(idx), epoch, len(testset)))
    for i in idx:
        print('%d' % (i+1))
    if len(idx) > 0:
        print('\n')

    idx = []
    for iter in range(0, len(testset1)):
        tag_scores = evaluate(testset1[:iter+1])
        tag = categoryFromOutput(tag_scores)
        if tag == 1:
            idx.append(iter)
    print('found %d epoch %d len %d' % (len(idx), epoch, len(testset1)))
    for i in idx:
        print('%d' % (i+1))
    if len(idx) > 0:
        print('\n')