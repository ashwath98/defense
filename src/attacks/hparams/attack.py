from .registry import register


class HParams():

  def __init__(self):
    self.model = "ResNet18"
    self.targeted = False


@register
def mnist_temp_attack_fgsm():
  hps = HParams()
  hps.epsilons = [0, .05, .2, .25, .3]
  hps.model_path = "lenet_mnist_model.pth"
  hps.batch_size = 20
  hps.attack = "FGSM"
  return hps


@register
def mnist_temp_attack_pgd():
  hps = HParams()
  hps.epsilons = [0, .05, .2, .25, .3]
  hps.model_path = "lenet_mnist_model.pth"
  hps.batch_size = 20
  hps.attack = "PGD"
  return hps