"""
Written by Ari Bornstein
"""
import torch
from torch.autograd import Variable
import utils

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
        embeds = self.embeddings(x).view(x.size()[0], -1)
        tanh = tanh_func(self.linear1(embeds))
        return self.linear2(tanh)

def train_model(train_file, dev_file, outpath, epochs):
    """
    trains a tagging model
    """
    tags, trainloader, dev_x, dev_y = utils.generate_input_data(train_file, dev_file)

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
                                      './pos/train.model', utils.EPOCHS)
    utils.write_model_results(POS_MODEL, r'./pos/test', POS_TAGS, 'test1.pos')
    NER_MODEL, NER_TAGS = train_model(r'./ner/train', r'./ner/dev',
                                      './ner/train.model', utils.EPOCHS)    
    utils.write_model_results(NER_MODEL, r'./ner/test', NER_TAGS, 'test1.ner')


    # POS_TAGS, _, _, _ = utils.generate_input_data(r'./pos/train', r'./pos/dev')
    # POS_MODEL = torch.load('./pos/train.model') 
