import torch
import torch.nn as nn
import torch.nn.functional as F


class RestrictedBoltzmanMachine(nn.Module):

    def __init__(self, n_vis=784, n_hid=128, k=1):
        super(RestrictedBoltzmanMachine, self).__init__()
        self.visibleNode = nn.Parameter(torch.randn(1, n_vis))
        self.hiddenNode = nn.Parameter(torch.randn(1, n_hid))
        self.weights = nn.Parameter(torch.randn(n_hid, n_vis))
        self.gibsSampling = k

    def visible2hidden(self, visibleNode):
        probability = torch.sigmoid(
            F.linear(visibleNode, self.weights, self.hiddenNode))
        return probability, probability.bernoulli()

    def hidden2visible(self, hiddenNode):
        probability = torch.sigmoid(
            F.linear(hiddenNode, self.weights.t(), self.visibleNode))
        return probability, probability.bernoulli()

    def free_energy(self, visibleNode):
        visible_term = torch.matmul(visibleNode, self.visibleNode.t())
        weighted_x_h = F.linear(visibleNode, self.weights, self.hiddenNode)
        hidden_term = torch.sum(F.softplus(weighted_x_h), dim=1)
        return torch.mean(-hidden_term - visible_term)

    def forward(self, visibleNode):
        hiddenNode = self.visible2hidden(visibleNode)
        for _ in range(self.gibsSampling):
            visibleNode_gibs = self.hidden2visible(hiddenNode)
            hiddenNode = self.visible2hidden(visibleNode_gibs)
        return visibleNode, visibleNode_gibs
