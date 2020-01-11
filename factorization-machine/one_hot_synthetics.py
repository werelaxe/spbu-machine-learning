from sklearn.preprocessing import OneHotEncoder


def get_dataset_with_encoder(sparse):
    xs = [('user-10', 'item-1'), ('user-8', 'item-0'), ('user-1', 'item-1'), ('user-9', 'item-1'), ('user-4', 'item-1'), ('user-3', 'item-5'), ('user-5', 'item-4'), ('user-8', 'item-1'), ('user-0', 'item-2'), ('user-5', 'item-5'), ('user-0', 'item-4'), ('user-7', 'item-4'), ('user-1', 'item-4'), ('user-6', 'item-5'), ('user-10', 'item-5'), ('user-7', 'item-2'), ('user-8', 'item-3'), ('user-4', 'item-4'), ('user-6', 'item-3'), ('user-6', 'item-2'), ('user-1', 'item-5'), ('user-5', 'item-3'), ('user-4', 'item-0')]
    xs = [('user-10', 'item-1'), ('user-8', 'item-0')]
    ys = [4, 4, 1, 3, 5, 4, 4, 3, 4, 3, 5, 2, 5, 2, 3, 2, 1, 5, 3, 5, 5, 2, 3]
    ys = [4, 3]
    encoder = OneHotEncoder(sparse=sparse)
    encoder.fit(xs)
    return encoder.transform(xs), ys, encoder
