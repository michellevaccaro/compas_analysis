import csv
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

X = []
two_year_recid = []

def build_logistic_model(fname):
    with open(fname, "rb") as f:
        reader = csv.DictReader(f)

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

            X.append([c_charge_degree, race, age_cat, sex, int(row['priors_count'])])
            two_year_recid.append(row['two_year_recid'])

    clf = LogisticRegression()
    clf.fit(X, two_year_recid)
    print('Coefficients:  ', clf.coef_)


def build_mlp_model(fname):
    with open(fname, "rb") as f:
        reader = csv.DictReader(f)

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

            X.append([c_charge_degree, race, age_cat, sex, int(row['priors_count'])])
            two_year_recid.append(row['two_year_recid'])

    clf = MLPClassifier(
        hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
        random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    nn_mlp = clf.fit(X, two_year_recid)
    print clf.predict([[0, 0, 2, 0, 1]])

build_logistic_model('cleaned_COMPAS_data.csv')
build_mlp_model('cleaned_COMPAS_data.csv')
