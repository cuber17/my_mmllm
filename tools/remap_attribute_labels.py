import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path


REMAP_RULES = {
    'action_category': {
        'stationary_activity': 'gesture',
    },
    'posture': {
        'sitting': 'low_posture',
        'lying': 'low_posture',
    },
    'intensity': {
        'static': 'slow',
    },
    'active_part': {
        'head_neck': 'upper_body',
    },
    'trajectory': {},
}


TARGET_LABEL_ORDER = {
    'action_category': ['locomotion', 'gesture', 'exercise', 'transition'],
    'posture': ['upright', 'low_posture', 'crouching', 'bending'],
    'intensity': ['slow', 'normal', 'vigorous'],
    'active_part': ['full_body', 'upper_body', 'lower_body'],
    'trajectory': ['in_place', 'forwards', 'backwards', 'lateral_move', 'dynamic_turn'],
}


TASKS = ['action_category', 'posture', 'intensity', 'active_part', 'trajectory']


def normalize_label(value):
    if not isinstance(value, str):
        return ''
    return value.lower().strip()


def remap_label(task_name, raw_label):
    label = normalize_label(raw_label)
    return REMAP_RULES.get(task_name, {}).get(label, label)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def remap_split(items):
    remapped = []
    for item in items:
        new_item = deepcopy(item)
        labels = dict(new_item.get('labels', {}))
        for task in TASKS:
            labels[task] = remap_label(task, labels.get(task, ''))
        new_item['labels'] = labels
        remapped.append(new_item)
    return remapped


def build_label_maps(items):
    label_maps = {}
    for task in TASKS:
        observed = []
        seen = set()
        for item in items:
            label = normalize_label(item.get('labels', {}).get(task, ''))
            if label and label not in seen:
                seen.add(label)
                observed.append(label)

        ordered = [label for label in TARGET_LABEL_ORDER[task] if label in seen]
        extras = [label for label in observed if label not in ordered]
        final_labels = ordered + extras
        label_maps[task] = {label: idx for idx, label in enumerate(final_labels)}
    return label_maps


def print_distribution(title, items):
    print(f'\n=== {title} ({len(items)} samples) ===')
    for task in TASKS:
        counter = Counter(normalize_label(item.get('labels', {}).get(task, '')) for item in items)
        total = sum(counter.values())
        print(f'[{task}] classes={len(counter)} total={total}')
        for label, count in counter.most_common():
            pct = (count / total * 100) if total else 0.0
            print(f'  {label:20s} {count:5d}  ({pct:5.1f}%)')


def main():
    parser = argparse.ArgumentParser(description='Remap attribute labels for retraining.')
    parser.add_argument(
        '--input-dir',
        default='/root/jyz/my_mmLLM/processed_dataset',
        help='Directory containing the original train.json and test.json.',
    )
    parser.add_argument(
        '--output-dir',
        default='/root/jyz/my_mmLLM/processed_dataset_rebalanced',
        help='Directory to write the remapped train.json, test.json, and label_maps.json.',
    )
    parser.add_argument(
        '--overwrite-input',
        action='store_true',
        help='Write remapped files back to the input directory instead of output-dir.',
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = input_dir if args.overwrite_input else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = input_dir / 'train.json'
    test_path = input_dir / 'test.json'

    train_items = load_json(train_path)
    test_items = load_json(test_path)

    print_distribution('Original train', train_items)
    print_distribution('Original test', test_items)

    remapped_train = remap_split(train_items)
    remapped_test = remap_split(test_items)
    combined = remapped_train + remapped_test
    label_maps = build_label_maps(combined)

    save_json(output_dir / 'train.json', remapped_train)
    save_json(output_dir / 'test.json', remapped_test)
    save_json(output_dir / 'label_maps.json', label_maps)

    print_distribution('Remapped train', remapped_train)
    print_distribution('Remapped test', remapped_test)

    print('\nWritten files:')
    print(f'- {output_dir / "train.json"}')
    print(f'- {output_dir / "test.json"}')
    print(f'- {output_dir / "label_maps.json"}')


if __name__ == '__main__':
    main()