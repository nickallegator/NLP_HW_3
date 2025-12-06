import json
import nltk
from collections import defaultdict

# extract last 3 letters as features and their tags
def extract_features_and_check_conflicts(tagged_words, dataset_name):
    feature_to_tags = defaultdict(set)
    feature_set = []


    # first pass
    for word, tag in tagged_words:
        suffix = word[-3:].lower()
        feature_to_tags[suffix].add(tag)
    
    # Find conflicting features (mapped to more than one tag)
    conflicting_features = {feat for feat, tags in feature_to_tags.items() if len(tags) > 1}

    print('Dataset: {}'.format(dataset_name))
    print('Total unique suffixes: {}'.format(len(feature_to_tags)))
    print('Conflicting suffixes: {}'.format(len(conflicting_features)))

    if conflicting_features:
        print('Examples of conflicting suffixes and their tags:')
        for feat in list(conflicting_features)[:5]:
            print('  Suffix: {}, Tags: {}'.format(feat, feature_to_tags[feat]))
    
    # second pass

    for word, tag in tagged_words:
        suffix = word[-3:].lower()

        if suffix in conflicting_features:
            final_tag = 'X'
        else:
            final_tag = tag

        features = {'suffix': suffix}
        feature_set.append((features, final_tag))
    
    return feature_set, conflicting_features 

def main():
    with open('brown_news_pos_splits.json', 'r') as f:
        data = json.load(f)

    train_data = [(item['word'], item['tag']) for item in data['train']]
    test_data = [(item['word'], item['tag']) for item in data['test']]

    print('Loaded {} training samples and {} test samples.'.format(len(train_data), len(test_data)))

    #process training set
    train_feature_set, train_conflicts = extract_features_and_check_conflicts(train_data, 'Training')

    test_feature_set, test_conflicts = extract_features_and_check_conflicts(test_data, 'Test')

    print('\nFeature set examples (first 5 from training):')
    for i, (features, tag) in enumerate(train_feature_set[:5]):
        print('  {} -> {}'.format(features, tag))

    train_output = {
        'feature_set': train_feature_set,
        'conflicting_suffixes': list(train_conflicts),
        'dataset': 'training'
    }

    test_output = {
        'feature_set': test_feature_set,
        'conflicting_features': list(test_conflicts),
        'dataset': 'test'
    }

    with open('train_features.json', 'w', encoding='utf-8') as f:
        #serialize
        serializable_train = {
            'feature_set': [(dict(feat), tag) for feat, tag in train_feature_set],
            'conflicting_features': list(train_conflicts),
            'dataset': 'training'
        }
        json.dump(serializable_train, f, indent=2)

    with open('test_features.json', 'w', encoding='utf-8') as f:
       # serializable 
        serializable_test = {
            'feature_set': [(dict(feat), tag) for feat, tag in test_feature_set],
            'conflicting_features': list(test_conflicts),
            'dataset': 'test'
        }
        json.dump(serializable_test, f, indent=2)

    print('\nFeature sets saved to train_features.json and test_features.json')

    #summary
    print('\nSummary:')
    print('training set: {} features, {} retagged as X'.format(len(train_feature_set), sum(1 for _, tag in train_feature_set if tag == 'X')))
    print('test set: {} features, {} retagged as X'.format(len(test_feature_set), sum(1 for _, tag in test_feature_set if tag == 'X')))


if __name__ == "__main__":
    main()
