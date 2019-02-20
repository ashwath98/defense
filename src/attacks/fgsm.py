from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from abc import ABC, abstractmethod


class Attack(ABC):

  def __init__(self, model, device, min_value=0, max_value=1):
    self.model = model
    self.device = device
    self.min_value = min_value
    self.max_value = max_value

  @abstractmethod
  def perturb(self, data, epsilon, y=None, y_target=None):
    raise NotImplementedError

  def generate(self, perturbed_data):

    output = self.model(perturbed_data)
    final_pred = output.max(1, keepdim=True)[1]
    return final_pred


class FGSM(Attack):

  def __init__(self, model, device, min_value=0, max_value=1):
    """
        parameters:-
        model :-The model under attack
        device :- CPU or GPU acc to usage
        data :- input image
        epsilon :- value of the perturbation
        y :- target /output labels
        targeted :- targeted version of attack

        4 Cases are possible according to the combination of targeted and y variables
        Case 1 :-y is specified and targeted is False .. then y is treated as the real output labels
        Case 2 :-y is specified and targeted is True ... then the targeted version of the attack takes place and y is the target label
        Case 3 :-y is None and targeted is False ... then the predicted outputs of the model are treated as the real outputs and the attack takes place
        Case 4 :-y is None and targeted is True .. Invalid Input"""

    super().__init__(model, device, min_value, max_value)

  def perturb(self, data, epsilon, y=None, y_target=None):

    if y_target is not None and type(y_target) != torch.Tensor:

      y = torch.Tensor([y_target]).type(torch.int64)

    if y is not None and type(y) != torch.Tensor:

      y = torch.Tensor([y]).type(torch.int64)

    data = data.to(self.device)
    data.requires_grad = True
    output = self.model(data)
    init_pred = output.max(1, keepdim=True)[1]
    if y is None:
      # if no y is specified use predictions as the label for the attack
      target = init_pred.view(1)
    else:
      target = y  # use y itself as the target
    target = target.to(self.device)
    loss = F.nll_loss(output, target)
    if y_target is not None:

      loss = -loss
    self.model.zero_grad()

    loss.backward()
    data_grad = data.grad.data

    sign_data_grad = data_grad.sign()
    perturbed_image = data + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, self.min_value,
                                  self.max_value)
    return init_pred, perturbed_image
