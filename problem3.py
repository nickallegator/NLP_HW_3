import json
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from collections import Counter


def load_feature_data():
    with open('train_features.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open('test_features.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    return train_data, test_data

def prepare_data(feature_set, vectorizer=None, fit_vectorizer=True):
    features = [feat_dict for feat_dict, _ in feature_set]
    labels = [tag for _, tag in feature_set]

    if vectorizer is None:
        vectorizer = DictVectorizer(sparse=False)

    if fit_vectorizer:
        X = vectorizer.fit_transform(features)

    else:
        X = vectorizer.transform(features)

    return X, labels, vectorizer

def run_lda_experiment(X_train, y_train, X_test, y_test, experiment_name):
    # run lda classification experiment
    print(f'\n== {experiment_name} ==')

    # lda classifier
    lda = LinearDiscriminantAnalysis(solver='svd')

    print(f'training LDA on {X_train.shape[0]} samples with {X_train.shape[1]} features...')
    lda.fit(X_train, y_train)

    # predictions output
    train_predictions = lda.predict(X_train)
    test_predictions = lda.predict(X_test)

    # decision and confidence

    train_decision = lda.decision_function(X_train)
    test_decision = lda.decision_function(X_test)

    # score and accuracy

    train_score = lda.score(X_train, y_train)
    test_score = lda.score(X_test, y_test)

    print(f'Training accuracy: {train_score:.4f}')
    print(f'Test accuracy: {test_score:.4f}')

    # classification report
    print('\nClassification Report:')
    print(classification_report(y_test, test_predictions, zero_division=0))

    # common tags print

    print('\nMost common predicted tags in test:')
    pred_counter = Counter(test_predictions)
    for tag, count in pred_counter.most_common(10):
        print(f'  {tag}: {count}')

    return {
        'lda_model': lda,
        'train_decision': train_decision,
        'test_decision': test_decision,
        'train_score': train_score,
        'test_score': test_score,
        'train_predictions': train_predictions,
        'test_predictions': test_predictions
    }

def create_downcased_features(feature_set):
    downcased_set = []
    for feat_dict, tag in feature_set:
        new_feat = feat_dict.copy()
        if 'suffix' in new_feat:
            new_feat['suffix'] = new_feat['suffix'].lower()
        downcased_set.append((new_feat, tag))
    return downcased_set

def main():
    print('Loading feature data from problem 2')
    train_data, test_data = load_feature_data()

    train_feature_set = train_data['feature_set']
    test_feature_set = test_data['feature_set']

    print(f'Loaded {len(train_feature_set)} training features')
    print(f'Loaded {len(test_feature_set)} test features')

    # show examples

    print('\nExample features:')
    for i in range(min(5, len(train_feature_set))):
        feat, tag = train_feature_set[i]
        print(f' {feat} -> {tag}')

    # experiment 1 with original features
    print('\n' + '='*50)
    print('EXPERIMENT 1: Original Features')
    print('='*50)

    X_train, y_train, vectorizer = prepare_data(train_feature_set)
    X_test, y_test, _ = prepare_data(test_feature_set, vectorizer, fit_vectorizer=False)

    results_exp1 = run_lda_experiment(X_train, y_train, X_test, y_test, "Experiment 1: Original Features")

    # experiment 2 with downcased features

    print('\n' + '='*50)
    print('EXPERIMENT 2: Downcased Features')
    print('='*50)

    train_downcased = create_downcased_features(train_feature_set)
    test_downcased = create_downcased_features(test_feature_set)

    X_train_down, y_train_down, vectorizer_down = prepare_data(train_downcased)
    X_test_down, y_test_down, _ = prepare_data(test_downcased, vectorizer_down, fit_vectorizer=False)

    results_exp2 = run_lda_experiment(X_train_down, y_train_down, X_test_down, y_test_down, "Experiment 2: Downcased Features")

    # results comparison

    print('\n' + '='*50)
    print("RESULTS COMPARISON")
    print('='*50)

    print(f'Experiment 1 Test Accuracy: {results_exp1["test_score"]:.4f}')
    print(f'Experiment 2 Test Accuracy: {results_exp2["test_score"]:.4f}')

    improvement = results_exp2["test_score"] - results_exp1["test_score"]
    print(f'Accuracy Improvement: {improvement:.4f}')

    print(f'\nFeature space sizes:')
    print(f' Experiment 1: {X_train.shape[1]} features')
    print(f' Experiment 2: {X_train_down.shape[1]} features')

    print(f'\nDecision function output:')
    print(f'Original - Train: {results_exp1["train_decision"].shape}, Test: {results_exp1["test_decision"].shape}')
    print(f'Downcased - Train: {results_exp2["train_decision"].shape}, Test: {results_exp2["test_decision"].shape}')

    #discussion prompt
    print('\nDiscussion:')
    print(f'- Downcasing {"improved" if improvement > 0 else "decreased"} accuracy by {abs(improvement):.4f}')
    print(f'- Feature space was {"reduced" if X_train_down.shape[1] < X_train.shape[1] else "same size"}')
    print(f'- This suggests that case {"does" if abs(improvement) > 0.01 else "does not"} significantly affect suffix-based POS tagging')

    if X_train.shape[1] < X_train_down.shape[1]:
        print(f'- Reducing feature space from {X_train_down.shape[1]} to {X_train.shape[1]} features')
        print(f'   helps reducing overfitting and improves generalization')

    # saving
    results_summary = {
        'original': {
            'train_accuracy': results_exp1['train_score'],
            'test_accuracy': results_exp1['test_score'],
            'feature_count': X_train.shape[1]
        },
        'downcased': {
            'train_accuracy': results_exp2['train_score'],
            'test_accuracy': results_exp2['test_score'],
            'feature_count': X_train_down.shape[1]
        },
        'improvement': improvement
    }

    with open('lda_experiment_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2)

    print('\nExperiment results saved to lda_experiment_results.json')

if __name__ == '__main__':
    main()