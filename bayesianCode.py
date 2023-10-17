def posterior(prior, likelihood, observation):
    false_postier = 1 * (1 - prior)
    for i in range(len(observation)):
        if observation[i] == True:
            false_postier = false_postier * likelihood[i][0]
        else:
            false_postier = false_postier * (1 - likelihood[i][0])
    true_postier = 1 * prior
    for i in range(len(observation)):
        if observation[i] == True:
            true_postier = true_postier * likelihood[i][1]
        else:
            true_postier = true_postier * (1 - likelihood[i][1])
    return true_postier/(true_postier+false_postier)

#test 
prior = 0.05
likelihood = ((0.001, 0.3),(0.05,0.9),(0.7,0.99))

observation = (True, True, True)

class_posterior_true = posterior(prior, likelihood, observation)
print("P(C=False|observation) is approximately {:.5f}"
      .format(1 - class_posterior_true))
print("P(C=True |observation) is approximately {:.5f}"
      .format(class_posterior_true))

prior = 0.05
likelihood = ((0.001, 0.3),(0.05,0.9),(0.7,0.99))

observation = (True, False, True)

class_posterior_true = posterior(prior, likelihood, observation)
print("P(C=False|observation) is approximately {:.5f}"
      .format(1 - class_posterior_true))
print("P(C=True |observation) is approximately {:.5f}"
      .format(class_posterior_true))

prior = 0.05
likelihood = ((0.001, 0.3),(0.05,0.9),(0.7,0.99))

observation = (False, False, False)

class_posterior_true = posterior(prior, likelihood, observation)
print("P(C=False|observation) is approximately {:.5f}"
      .format(1 - class_posterior_true))
print("P(C=True |observation) is approximately {:.5f}"
      .format(class_posterior_true))

import csv

def learn_prior(file_name, pseudo_count=0):
    with open(file_name) as in_file:
        training_examples = [tuple(row) for row in csv.reader(in_file)] 
    sum1 = 0
    sum2 = 0
    for i in range(1, len(training_examples)):
        if training_examples[i][-1] == "1":
            sum1 += 1
        else:
            sum2 += 1

    return (sum1 + pseudo_count) / ((sum1 + pseudo_count) + (sum2 + pseudo_count))

#test

prior = learn_prior("spam-labelled.csv")
print("Prior probability of spam is {:.5f}.".format(prior))


prior = learn_prior("spam-labelled.csv")
print("Prior probability of not spam is {:.5f}.".format(1 - prior))

prior = learn_prior("spam-labelled.csv", pseudo_count = 1)
print(format(prior, ".5f"))

import csv

def learn_likelihood(file_name, pseudo_count=0):
    with open(file_name) as in_file:
        training_examples = [tuple(row) for row in csv.reader(in_file)] 
    like = []
    for i in range(12):
        tsum = 0
        fsum = 0
        t_sum = 0
        f_sum = 0
        for k in range(1, len(training_examples)):
            if training_examples[k][-1] != "1":
                f_sum += 1
                if training_examples[k][i] == "1":
                    fsum += 1
            else:
                t_sum += 1
                if training_examples[k][i] == "1":
                    tsum += 1
        like.append(((fsum + pseudo_count)/(f_sum + (2 * pseudo_count)), (tsum + pseudo_count)/(t_sum + (2 * pseudo_count))))
    return like

#test 

likelihood = learn_likelihood("spam-labelled.csv")
print(len(likelihood))
print([len(item) for item in likelihood])

likelihood = learn_likelihood("spam-labelled.csv")

print("P(X1=True | Spam=False) = {:.5f}".format(likelihood[0][False]))
print("P(X1=False| Spam=False) = {:.5f}".format(1 - likelihood[0][False]))
print("P(X1=True | Spam=True ) = {:.5f}".format(likelihood[0][True]))
print("P(X1=False| Spam=True ) = {:.5f}".format(1 - likelihood[0][True]))



likelihood = learn_likelihood("spam-labelled.csv", pseudo_count=1)

print("With Laplacian smoothing:")
print("P(X1=True | Spam=False) = {:.5f}".format(likelihood[0][False]))
print("P(X1=False| Spam=False) = {:.5f}".format(1 - likelihood[0][False]))
print("P(X1=True | Spam=True ) = {:.5f}".format(likelihood[0][True]))
print("P(X1=False| Spam=True ) = {:.5f}".format(1 - likelihood[0][True]))


def nb_classify(prior, likelihood, input_vector):
    ammount = posterior(prior, likelihood, input_vector)
    if ammount > 0.5:
        return ("Spam", ammount)
    
    return ("Not Spam", 1-ammount)
