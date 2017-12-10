import csv
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

training_data = []
testing_data = []
training_label = []
testing_label = []


def load_file(fname):
    with open(fname, "rb") as f:
        reader = csv.DictReader(f)

        arr = []
        labels = []
        # read in values from file
        for row in reader:
            # c_charge_degree
            if row['c_charge_degree'] == 'M':
                c_charge_degree = 0
            if row['c_charge_degree'] == 'F':
                c_charge_degree = 1

            # race
            if row['race'] == 'Caucasian':
                race = 0
            if row['race'] == 'African-American':
                race = 1
            if row['race'] == 'Hispanic':
                race = 2
            if row['race'] == 'Asian':
                race = 3
            if row['race'] == 'Native American':
                race = 4
            if row['race'] == 'Other':
                race = 5

            # age_cat
            if row['age_cat'] == 'Less than 25':
                age_cat = 0
            if row['age_cat'] == '25 - 45':
                age_cat = 1
            if row['age_cat'] == 'Greater than 45':
                age_cat = 2

            # sex
            if row['sex'] == 'Male':
                sex = 0
            if row['sex'] == 'Female':
                sex = 1

            arr.append([c_charge_degree, race, age_cat, sex, int(row['priors_count'])])
            labels.append(row['two_year_recid'])

    return arr, labels


def build_logistic_model():
    clf = LogisticRegression()
    clf.fit(training_data, training_label)
    # print 'Built Logistic Regression'
    # print clf.coef_
    return clf


def build_mlp_model():
    clf = MLPClassifier(
        hidden_layer_sizes=(100,), activation='logistic', solver='adam', alpha=0.001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=200, shuffle=True,
        random_state=0, tol=0.001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.1, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
    nn_mlp = clf.fit(training_data, training_label)
    # print "Built MLP"
    return nn_mlp


def test_gen_accuracy(model):
    if model == 'nn_mlp':
        clf = build_mlp_model()
    elif model == 'log_model':
        clf = build_logistic_model()
    else:
        print 'Invalid input'

    num_correct = 0
    for i in range(len(testing_data)):
        if int(clf.predict([testing_data[i]])[0]) == int(testing_label[i]):
            num_correct += 1

    print model + ' accuracy: ' + str(num_correct/float(len(testing_data)))


def test_race_accuracy(model, subgroup):
    if model == 'nn_mlp':
        clf = build_mlp_model()
    elif model == 'log_model':
        clf = build_logistic_model()
    else:
        print 'Invalid input'

    # race encoding
    if subgroup == 'Caucasian':
        race = 0
    if subgroup == 'African-American':
        race = 1
    if subgroup == 'Hispanic':
        race = 2
    if subgroup == 'Asian':
        race = 3
    if subgroup == 'Native American':
        race = 4
    if subgroup == 'Other':
        race = 5

    # accuracy
    acc_count = 0
    acc_denom = 0
    fp = 0
    fn = 0
    for i in range(len(testing_data)):
        if testing_data[i][1] == race:
            acc_denom += 1
            if int(clf.predict([testing_data[i]])[0]) == int(testing_label[i]):
                acc_count += 1
            elif int(clf.predict([testing_data[i]])[0]) == 0 and int(testing_label[i]) == 1:
                fn += 1
            elif int(clf.predict([testing_data[i]])[0]) == 1 and int(testing_label[i]) == 0:
                fp += 1

    print model + ' accuracy for ' + subgroup + ': ' + str(acc_count / float(acc_denom))
    print model + ' FPR for ' + subgroup + ': ' + str(fp / float(fp + fn))
    print model + ' FNR for ' + subgroup + ': ' + str(fn / float(fp + fn))
    print '\n'


training_data, training_label = load_file('COMPAS_training_data.csv')
testing_data, testing_label = load_file('COMPAS_testing_data.csv')

test_gen_accuracy('log_model')
test_gen_accuracy('nn_mlp')

test_race_accuracy('log_model', 'African-American')
test_race_accuracy('nn_mlp', 'African-American')
test_race_accuracy('log_model', 'Caucasian')
test_race_accuracy('nn_mlp', 'Caucasian')
test_race_accuracy('log_model', 'Asian')
test_race_accuracy('nn_mlp', 'Asian')
test_race_accuracy('log_model', 'Native American')
test_race_accuracy('nn_mlp', 'Native American')
test_race_accuracy('log_model', 'Hispanic')
test_race_accuracy('nn_mlp', 'Hispanic')
test_race_accuracy('log_model', 'Other')
test_race_accuracy('nn_mlp', 'Other')