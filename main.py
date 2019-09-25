import numpy as np
import multiprocessing


from dataset_reader import read_dataset, split_dataset, split_to_folds


FOLDS_COUNT = 5


def dist(xs1, xs2):
    return np.sqrt(np.dot(xs1, xs2))


def func(x):
    return 2 * x - 1


def f(ws, x):
    return ws[0] + np.dot(ws[1:], x)


def mse(xs, ys, ws):
    f_xs = np.array([f(ws, x) for x in xs])
    return ((ys - f_xs) ** 2).sum() / len(xs)


def r2(xs, ys, ws):
    avg = np.average(ys)
    f_xs = np.array([f(ws, x) for x in xs])
    ss_res = ((ys - f_xs) ** 2).sum()
    ss_tot = ((ys - avg) ** 2).sum()
    return 1 - ss_res / ss_tot


def rmse(xs, ys, ws):
    return np.sqrt(mse(xs, ys, ws))


def grad_of_mse(xs, ys, ws):
    grad = np.array([])

    f_0 = np.array([(f(ws, xs[i]) - ys[i]) * 1 for i in range(len(xs))])

    grad = np.append(grad, 2 * f_0.sum() / len(xs))

    for j in range(len(xs[0])):
        f_j = np.array([(f(ws, xs[i]) - ys[i]) * xs[i][j] for i in range(len(xs))])
        grad = np.append(grad, 2 * f_j.sum() / len(xs))

    return grad


def do_step(xs, ys, ws, coef):
    grad = grad_of_mse(xs, ys, ws)
    return ws - grad * coef


def get_ws(dataset, start_value):
    return np.array([start_value] * (len(dataset[0])))


def train(dataset, fold_index, mse_list, rmse_list, r2_list):
    print(f"Start train {fold_index}")
    train_set, test_set = split_to_folds(dataset, fold_index, FOLDS_COUNT)

    ws = get_ws(train_set, 0)
    train_xs, train_ys = split_dataset(train_set)

    iters = 0
    coef = 0.0405
    try:
        for _ in range(100):
            iters += 1
            print(f"{fold_index}: {iters}")
            ws = do_step(train_xs, train_ys, ws, coef)
            current = mse(train_xs, train_ys, ws)
            # print(f"{iterations}: {current}, {abs(prev - current)}, {coef}")
            if current > 10000:
                # print("reset all")
                iters = 0
                coef *= 0.9
                ws = get_ws(train_set, 0)
    except KeyboardInterrupt:
        pass

    test_xs, test_ys = split_dataset(test_set)

    rmse_result = rmse(test_xs, test_ys, ws)
    r2_result = r2(test_xs, test_ys, ws)
    mse_result = mse(test_xs, test_ys, ws)

    print(f"{fold_index}: rmse:", rmse_result)
    print(f"{fold_index}: r2:", r2_result)
    rmse_list.append(rmse_result)
    r2_list.append(r2_result)
    mse_list.append(mse_result)


def main():
    dataset = read_dataset()
    manager = multiprocessing.Manager()
    rmse_list = manager.list()
    r2_list = manager.list()
    mse_list = manager.list()
    jobs = []
    for i in range(FOLDS_COUNT):
        p = multiprocessing.Process(target=train, args=(dataset, i, mse_list, rmse_list, r2_list))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print("Avg rmse: ", sum(rmse_list) / FOLDS_COUNT)
    print("Avg r2:", sum(r2_list) / FOLDS_COUNT)
    print("Avg mse:", sum(mse_list) / FOLDS_COUNT)


if __name__ == '__main__':
    main()
