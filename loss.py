
def check_uniform_spacing_and_get_delta(vector):
    diff = np.diff(vector)
    if not np.all(np.isclose(diff[0], diff)):
        print("error")
    return diff[0]

def normalized_level_weights(data):
    level = data.coords["level"]
    return level/level.mean()

def sum_per_variable_losses(
        per_varibale_losses: Mapping[str, xarray.DataArray],
        per_varibale_weights: Mapping[str, float]
):
    weight_per_variable_losses = {
        name : loss*per_varibale_weights.get(name , 1)
        for name, loss in per_varibale_losses.items()
    }
    total = xarray.concat(weight_per_variable_losses.values(), dim="variable").sum(dim="variable")
    return total,per_varibale_losses

def mean_preserving_batch(x):
    return x.mean([d for d in x.dims if d!="batch"],skipna=False)

def weight_for_latitude_vector_without_ploes(latitude):
    delta_latitude = np.abs(check_uniform_spacing_and_get_delta(latitude))
    return np.cos(np.deg2rad(delta_latitude))

def weight_for_latitude_vector_with_poles(latitude):
    delta_latitude = np.abs(check_uniform_spacing_and_get_delta(latitude))
    weight = np.cos(np.deg2rad(delta_latitude)) * np.sin(np.deg2rad(delta_latitude/2))
    weight[[0,-1]] = np.sin(np.deg2rad(delta_latitude/4)) ** 2
    return weight


def normalized_latitude_weights(data):
    latitude = data.coords["latitude"]

    # 判断是否有极点
    if np.any(np.isclose(np.abs(latitude), 90.)):
        weight = weight_for_latitude_vector_with_poles(latitude)
    else:
        weight = weight_for_latitude_vector_without_ploes(latitude)
    return weight / weight.mean(skipna=False)

from collections.abc import Mapping
from typing import Callable, Any

def map_structure(func: Callable[..., Any], *structures):


    first = structures[0]

    # dict
    if isinstance(first, Mapping):
        return {
            k: map_structure(func, *(s[k] for s in structures))
            for k in first
        }

    # list
    if isinstance(first, list):
        return [
            map_structure(func, *items)
            for items in zip(*structures)
        ]

    # tuple
    if isinstance(first, tuple):
        return tuple(
            map_structure(func, *items)
            for items in zip(*structures)
        )

    # set
    if isinstance(first, set):
        return {
            map_structure(func, *items)
            for items in zip(*structures)
        }

    # leaf node
    return func(*structures)

def weighted_mse_per_level(
        prediction: xarray.DataArray,
        target: xarray.DataArray,
        per_varibale_weights: Mapping[str, float]
):
    def loss(prediction,target): # 输入每个变量的结果
        loss = (prediction - target) ** 2
        # 经度修正
        loss *= normlized_latitude_weights(target).astype(loss.dtype)
        if "level" in target.dims:
            # 增加level权重
            loss *= normlized_level_weights(target).astype(loss.dtype)

        return mean_preserving_batch(loss) # batch维度求均值

    losses = map_structure(loss, prediction, target) # 一个一个变量进
    return sum_per_variable_losses(losses, per_varibale_weights) # 求平均值

