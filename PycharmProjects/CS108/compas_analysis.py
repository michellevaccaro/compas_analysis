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

            # arr.append([c_charge_degree, race, age_cat, sex, int(row['priors_count'])])
            arr.append([c_charge_degree, age_cat, sex, int(row['priors_count'])])
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
        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
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


def test_subgroup_accuracy(model, field, desc):
    if model == 'nn_mlp':
        clf = build_mlp_model()
    elif model == 'log_model':
        clf = build_logistic_model()
    else:
        print 'Invalid input'
        return

    # c_charge_degree
    if field == 0:
        if desc == 'M':
            subgroup = 0
        elif desc == 'F':
            subgroup = 1
        else:
            print 'Invalid description: ' + desc

    # race
    if field == 1:
        if desc == 'Caucasian':
            subgroup = 0
        elif desc == 'African-American':
            subgroup = 1
        elif desc == 'Hispanic':
            subgroup = 2
        elif desc == 'Asian':
            subgroup = 3
        elif desc == 'Native American':
            subgroup = 4
        elif desc == 'Other':
            subgroup = 5
        else:
            print 'Invalid description: ' + desc

    # age_cat
    if field == 2:
        if desc == 'Less than 25':
            subgroup = 0
        elif desc == '25 - 45':
            subgroup = 1
        elif desc == 'Greater than 45':
            subgroup = 2
        else:
            print 'Invalid description: ' + desc

    # sex
    if field == 3:
        if desc == 'Male':
            subgroup = 0
        if desc == 'Female':
            subgroup = 1
        else:
            print 'Invalid description: ' + desc

    # accuracy
    acc_count = 0
    acc_denom = 0
    fp_denom = 0
    fn_denom = 0
    fp = 0
    fn = 0

    for i in range(len(testing_data)):
        if int(testing_label[i]) == 1:  # individual did recidivate
            fn_denom += 1
        if int(testing_label[i]) == 0:  # individual did not recidivate
            fp_denom += 1

        if testing_data[i][field] == subgroup:
            acc_denom += 1
            if int(clf.predict([testing_data[i]])[0]) == int(testing_label[i]):     # clf returns an array with one elem
                acc_count += 1
            if int(clf.predict([testing_data[i]])[0]) == 0 and int(testing_label[i]) == 1:
                fn += 1
            if int(clf.predict([testing_data[i]])[0]) == 1 and int(testing_label[i]) == 0:
                fp += 1

    print model + ' accuracy for ' + desc + ': ' + str(acc_count / float(acc_denom))
    print model + ' FPR for ' + desc + ': ' + str(fp / float(fp_denom))
    print model + ' FNR for ' + desc + ': ' + str(fn / float(fn_denom))
    print '\n'


training_data, training_label = load_file('3split_COMPAS_training_data.csv')
testing_data, testing_label = load_file('3split_COMPAS_testing_data.csv')

test_gen_accuracy('log_model')
test_gen_accuracy('nn_mlp')

test_subgroup_accuracy('log_model', 1, 'African-American')
test_subgroup_accuracy('log_model', 1, 'Caucasian')

# test_subgroup_accuracy('log_model', 0, 'F')
# test_subgroup_accuracy('nn_mlp', 0, 'F')
# test_subgroup_accuracy('log_model', 0, 'M')
# test_subgroup_accuracy('nn_mlp', 0, 'M')

# test_subgroup_accuracy('log_model', 1, 'African-American')
# test_subgroup_accuracy('nn_mlp', 1, 'African-American')
# test_subgroup_accuracy('log_model', 1, 'Caucasian')
# test_subgroup_accuracy('nn_mlp', 1, 'Caucasian')
# test_subgroup_accuracy('log_model', 1, 'Asian')
# test_subgroup_accuracy('nn_mlp', 1, 'Asian')
# test_subgroup_accuracy('log_model', 1, 'Native American')
# test_subgroup_accuracy('nn_mlp', 1, 'Native American')
# test_subgroup_accuracy('log_model', 1, 'Hispanic')
# test_subgroup_accuracy('nn_mlp', 1, 'Hispanic')
# test_subgroup_accuracy('log_model', 1, 'Other')
# test_subgroup_accuracy('nn_mlp', 1, 'Other')

# test_subgroup_accuracy('log_model', 2, 'Less than 25')
# test_subgroup_accuracy('nn_mlp', 2, 'Less than 25')
# test_subgroup_accuracy('log_model', 2, '25 - 45')
# test_subgroup_accuracy('nn_mlp', 2, '25 - 45')
# test_subgroup_accuracy('log_model', 2, 'Greater than 45')
# test_subgroup_accuracy('nn_mlp', 2, 'Greater than 45')

# test_subgroup_accuracy('log_model', 3, 'Male')
# test_subgroup_accuracy('nn_mlp', 3, 'Male')
# test_subgroup_accuracy('log_model', 3, 'Female')
# test_subgroup_accuracy('nn_mlp', 3, 'Female')