import csv
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

training_data = []
testing_data = []
training_label = []
testing_label = []
training_race = []
testing_race = []
nn_preds = []
log_preds = []


def load_file(fname):
    with open(fname, "rb") as f:
        reader = csv.DictReader(f)

        arr = []
        labels = []
        arr_race = []
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

            arr.append([c_charge_degree, age_cat, sex, int(row['priors_count'])])
            labels.append(row['two_year_recid'])
            arr_race.append(row['race'])

    return arr, labels, arr_race


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


# def test_preds():
#     nn_clf = build_mlp_model()
#     log_clf = build_logistic_model()
#
#     diff_count = 0
#     diff_aa = 0
#     diff_white = 0
#
#     print diff_count


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

    # Consistency
    k = 10
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(testing_data)
    c = 0
    temp = 0

    for i in range(len(testing_data)):
        if testing_race[i] == desc:
            knn = neigh.kneighbors([testing_data[i]])  # indexes of knn in testing_data
            for i in range(0, len(knn[1][0])):
                pred = clf.predict([testing_data[knn[1][0][i]]])
                temp += abs(int(testing_label[i]) - int(pred[0]))
            c += temp/float(k)
            temp = 0

    c /= float(len(testing_data))

    # accuracy
    acc_denom = 0
    tn = 0
    tp = 0
    fp = 0
    fn = 0

    for i in range(len(testing_data)):
        if testing_race[i] == desc:
            acc_denom += 1

            if int(clf.predict([testing_data[i]])[0]) == 0 and int(testing_label[i]) == 0:     # clf returns an array with one elem
                tn += 1
            if int(clf.predict([testing_data[i]])[0]) == 1 and int(testing_label[i]) == 1:     # clf returns an array with one elem
                tp += 1
            if int(clf.predict([testing_data[i]])[0]) == 0 and int(testing_label[i]) == 1:
                fn += 1
            if int(clf.predict([testing_data[i]])[0]) == 1 and int(testing_label[i]) == 0:
                fp += 1

    print model + ' accuracy for ' + desc + ': ' + str((tn + tp) / float(acc_denom))
    print model + ' sensitivity for ' + desc + ': ' + str(tp / float(tp+fn))    # checked
    print model + ' specificity for ' + desc + ': ' + str(tn / float(tn+fp))    # checked
    print model + ' FPR for ' + desc + ': ' + str(fp / float(tn+fp))
    print model + ' FNR for ' + desc + ': ' + str(fn / float(tp+fn))
    print model + ' consistency ' + desc + ': ' + str(1-c)
    print '\n'


def graph_results():
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    x_labels = ['Accuracy', '', 'Consistency', '', 'Sensitivity', '', 'Specificity', '', 'FPR', '', 'FNR']

    # Plot 1: Logistic Regression
    # # Accuracy
    # bar1 = plt.bar(0, .65, width=0.4, color='b', align='center')
    # bar2 = plt.bar(.4, .69, width=0.4, color='g', align='center')
    # # Consistency
    # plt.bar(2, .75, width=0.4, color='b', align='center')
    # plt.bar(2.4, .78, width=0.4, color='g', align='center')
    # # Sensitivity
    # plt.bar(4, .64, width=0.4, color='b', align='center')
    # plt.bar(4.4, .40, width=0.4, color='g', align='center')
    # # Specificity
    # plt.bar(6, .66, width=0.4, color='b', align='center')
    # plt.bar(6.4, .86, width=0.4, color='g', align='center')
    # # FPR
    # plt.bar(8, .34, width=0.4, color='b', align='center')
    # plt.bar(8.4, .14, width=0.4, color='g', align='center')
    # # FNR
    # plt.bar(10, .36, width=0.4, color='b', align='center')
    # plt.bar(10.4, .6, width=0.4, color='g', align='center')
    # plt.tick_params(axis='x', bottom='off',  top='off')
    # plt.xticks(x, x_labels)
    # plt.ylabel('Value')
    # plt.xlabel('Fairness Metric')
    # plt.legend([bar1, bar2], ['African-American', 'Caucasian'])
    # plt.title('Logistic Regression')
    # plt.show()

    # Plot 2: Neural Network
    # Accuracy
    bar1 = plt.bar(0, .65, width=0.4, color='b', align='center')
    bar2 = plt.bar(.4, .68, width=0.4, color='g', align='center')
    # Consistency
    plt.bar(2, .74, width=0.4, color='b', align='center')
    plt.bar(2.4, .78, width=0.4, color='g', align='center')
    # Sensitivity
    plt.bar(4, .62, width=0.4, color='b', align='center')
    plt.bar(4.4, .39, width=0.4, color='g', align='center')
    # Specificity
    plt.bar(6, .68, width=0.4, color='b', align='center')
    plt.bar(6.4, .86, width=0.4, color='g', align='center')
    # FPR
    plt.bar(8, .32, width=0.4, color='b', align='center')
    plt.bar(8.4, .14, width=0.4, color='g', align='center')
    # FNR
    plt.bar(10, .38, width=0.4, color='b', align='center')
    plt.bar(10.4, .61, width=0.4, color='g', align='center')
    plt.tick_params(axis='x', bottom='off', top='off')
    plt.xticks(x, x_labels)
    plt.ylabel('Value')
    plt.xlabel('Fairness Metric')
    plt.legend([bar1, bar2], ['African-American', 'Caucasian'])
    plt.title('Neural Network')
    plt.show()

training_data, training_label, training_race = load_file('3split_COMPAS_training_data.csv')
testing_data, testing_label, testing_race = load_file('3split_COMPAS_testing_data.csv')

# test_preds()

# test_gen_accuracy('log_model')
# test_gen_accuracy('nn_mlp')
#
# test_subgroup_accuracy('log_model', 1, 'African-American')
# test_subgroup_accuracy('log_model', 1, 'Caucasian')
#
# test_subgroup_accuracy('nn_mlp', 1, 'African-American')
# test_subgroup_accuracy('nn_mlp', 1, 'Caucasian')

graph_results()