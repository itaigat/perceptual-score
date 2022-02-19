Code for [Perceptual Score: What Data Modalities
Does Your Model Perceive?](https://arxiv.org/abs/2110.14375) presented in NeurIPS 2021.

## Installation

This repo only uses PyTorch and is tested on version 1.10.2.

## General

In this repository, we present two approaches to compute the denominator of the perceptual score (Eq. 2).

Note, as mentioned in the paper, it is essential to compute an expectation of the score.

## First method - permute samples within a batch

In `first_option.py` we implemented the calculation via the function that evaluates the model's performance.

```python
first_modality = first_modality[torch.randperm(first_modality.shape[0]), :]
```
Note, when using this approach, you must use a sufficiently large batch.

## Second method - wrap dataset class
The second method wraps the original dataset class and in the `__get_item__` function we randomly pick a different sample from the dataset:

```python
if self.permute_first_batch:
    random_idx = torch.randint(0, self.__len__(), (1, )).item()
    first = self.first[random_idx]
else:
    first = self.first[idx]
```