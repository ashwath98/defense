class Test_Attack:

  def __init__(self, attack, test_data, device, epsilons):
    """
    parameters:-
    attack:-attack to be tested
    test_data:-test loader object to load the test data
    device:-torch.device object 
    epsilons:-list of epsilon values to test"""

    self.attack = attack
    self.test_data = test_data
    self.device = device
    self.epsilons = epsilons

  def test(self):
    """ returns accuracies and a sample of adversarial examples for each epsilon value"""
    accuracies = []
    examples = []  #Run test for each epsilon
    for eps in self.epsilons:

      acc, ex = self.test_eps_value(eps)
      accuracies.append(acc)
      examples.append(ex)
    return accuracies, examples

  def test_eps_value(self, epsilon):
    """ generates adversarial examples and calculates accuracy of predicting adversary for a particular epsilon value"""

    correct = 0
    adv_examples = []
    for data, target in self.test_data:
      data, target = data.to(self.device), target.to(self.device)

      init_pred, perturbed_data, final_pred = self.attack.generate(
          data, target, epsilon)

      if init_pred != target:

        continue
      if final_pred == target:

        correct += 1
        # Special case for saving 0 epsilon examples
        if (epsilon == 0) and (len(adv_examples) < 5):

          adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
          adv_examples.append((init_pred, final_pred, adv_ex))

        else:

          # Save some adv examples for visualization later
          if len(adv_examples) < 5:

            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((init_pred, final_pred, adv_ex))
    final_acc = correct / float(len(self.test_data))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(
        epsilon, correct, len(self.test_data), final_acc))

    return final_acc, adv_examples
