from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from abc import ABC, abstractmethod


class Attack(ABC):

  def __init__(self,
               model,
               device,
               data,
               epsilon,
               y=None,
               targeted=False,
               min_value=0,
               max_value=1):
    self.model = model
    self.device = device
    self.data = data
    self.epsilon = epsilon
    self.y = y
    self.min_value = min_value
    self.max_value = max_value
    self.targeted = targeted

  @abstractmethod
  def generate(self):

    pass


class FGSM(Attack):

  def __init__(self,
               model,
               device,
               data,
               epsilon,
               y=None,
               targeted=False,
               min_value=0,
               max_value=1):
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

    super().__init__(model, device, data, epsilon, y, targeted, min_value,
                     max_value)

  def fgsm_update(self, image, epsilon, data_grad):
    """Update the image with gradients of input"""

    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, self.min_value,
                                  self.max_value)
    return perturbed_image

  def generate(self):

    if self.targeted == True and self.y is None:

      print("label should be specified when targeted is true")
      return -1, -1, -1

    if self.y is not None and type(self.y) != torch.Tensor:

      self.y = torch.Tensor([self.y]).type(torch.int64)

    data = self.data.to(self.device)
    data.requires_grad = True
    output = self.model(data)
    init_pred = output.max(1, keepdim=True)[1]
    if self.y is None:
      # if no y is specified use predictions as the label for the attack
      target = init_pred.view(1)
    else:
      target = self.y  # use y itself as the target
    target = target.to(self.device)
    loss = F.nll_loss(output, target)
    if self.targeted:
      loss = -loss
    self.model.zero_grad()

    loss.backward()
    data_grad = data.grad.data
    perturbed_data = self.fgsm_update(data, self.epsilon, data_grad)
    output = self.model(perturbed_data)
    final_pred = output.max(1, keepdim=True)[1]
    return init_pred.item(), perturbed_data, final_pred.item()
