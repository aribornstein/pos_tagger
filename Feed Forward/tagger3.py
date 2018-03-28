"""
Written by Ari Bornstein
"""

import torch
from torch.autograd import Variable
import utils
from sklearn.metrics import accuracy_score, classification_report

class Tagger_Net(torch.nn.Module):
    """
    A Feed forward network for tagging
    """
    def __init__(self, V, E, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Tagger_Net, self).__init__()
        self.embeddings = torch.nn.Embedding(V, E)
        self.embeddings.weight.data.copy_(torch.from_numpy(utils.VECS))
        self.linear1 = torch.nn.Linear(D_in * E, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        tanh_func = torch.nn.Tanh()
        embeds = self.embeddings(x)
        w_size, ixs_size =[5, 3]
        ixs = embeds.view(x.size()[0], w_size, ixs_size, -1)# break embed rows into ix tensor for each word
        sum_ixs = ixs.view(x.size()[0], ixs_size, -1).sum(dim=1) # sum ixs together to get rows of word embed vectors
        sum_fix = sum_ixs.view(x.size()[0], -1)
        tanh = tanh_func(self.linear1(sum_fix))
        return self.linear2(tanh)

def train_model(train_file, dev_file, outpath, epochs):
    """
    trains a tagging model
    """
    tags, trainloader, dev_x, dev_y = utils.generate_input_data(train_file, dev_file,
                                                                pretrained=True, ixs=3)

    V = len(utils.VOCAB)
    E = 50 # Embedding dimensions
    D_in = 5 # pp_w, p_w, words[i], n_w, nn_w 
    H = 128 # hidden layers
    D_out = len(tags) # out layer

    # Construct our model by instantiating the class defined above
    model = Tagger_Net(V, E, D_in, H, D_out)
    criterion = torch.nn.CrossEntropyLoss() # cross entropy loss
    optimizer = torch.optim.Adam(model.parameters()) # ADAM

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            if torch.cuda.is_available():
                model.cuda()
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:    # print every 2000 mini-batches
                labels, predicted = utils.get_metrics_data(model, dev_x, dev_y)
                print('[%d, %5d] loss: %.3f dev: %.3f' % (epoch + 1, i + 1, running_loss / 2000,
                                                          utils.accuracy(labels, predicted, tags)))
                running_loss = 0.0
        # Checkpoint every epoch
        torch.save(model, outpath)
    return [model, tags]

if __name__ == "__main__":
    POS_MODEL, POS_TAGS = train_model(r'./pos/train', r'./pos/dev',
                                      './pos/train_ixs.model', utils.EPOCHS)    

    utils.write_model_results(POS_MODEL, r'./pos/test', POS_TAGS, 'test4.pos', ixs=3)


    NER_MODEL, NER_TAGS = train_model(r'./ner/train', r'./ner/dev',
                                      'NER_tagger_ixs.model', utils.EPOCHS)    
    utils.write_model_results(NER_MODEL, r'./ner/test', NER_TAGS, 'test4.ner', ixs=3)
