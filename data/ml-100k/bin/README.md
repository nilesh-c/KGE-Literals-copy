# Preprocessed MovieLens-100k

Preprocessed MovieLens-100k dataset.

## File Descriptions

### Index Files
All `idx2{something}` files are lookup tables which are useful to decode the preprocessed dataset. For example usage:

``` python
idx2occupation = np.load('data/ml-100k/bin/idx2occupation.npy')

# Suppose from the dataset we have these indexes
occs = [1, 5, 3, 1]

# Lookup
occs_idxs = [idx2occupation[o] for o in occs]

print(occs_idxs)
# ['artist', 'entertainment', 'educator', 'artist']
```

### Datatype Files
All `idx{userrel,movierel}2dtype` are lookup tables which are useful to get the information of particular user/movie relation's datatype. The input is relation index, and the output is a string of corresponding datatype. Possible datatypes are:

1. 'cat': categorical data, e.g. gender
2. 'num': numerical data, e.g. age
3. 'txt': textual data, e.g. movie title
4. 'img': image data, e.g. movie poster

### Dataset Files
The preprocessed dataset is divided into three: train, val, and test using 80-10-10 shuffle-split.

1. `rating_{train,val,test}` contains array triplets of (`user_id`, `rating`, `movie_id`).
2. `user_{train,val,test}` contains array triplets of (`user_id`, `user_rel`, `literal_val`). Please refer to `idx2userrel.npy` for the breakdown of `user_rel`
3. `movie_{train,val,test}` contains array triplets of (`movie_id`, `movie_rel`, `literal_val`). Please refer to `idx2movierel.npy` for the breakdown of `movie_rel`

Example:

```python
# Rating
X_train = np.load('data/ml-100k/bin/rating_train.npy')

X_train.shape
Out[330]: (80000, 3)

X_train[:3]
Out[331]:
array([[551,   2, 872],
       [324,   3, 960],
       [870,   3, 341]])

# Literals
X_train_movie = np.load('data/ml-100k/bin/movie_train.npy')

X_train_movie[:10]
Out[337]:
array([[944, 0, 16],
       [1023, 0, 14],
       [256, 0, 1],
       [1500, 2, 'Prisoner of the Mountains (Kavkazsky Plennik) (1996)'],
       [44, 2, 'Eat Drink Man Woman (1994)'],
       [428, 0, 15],
       [937, 0, 5],
       [1151, 0, 14],
       [341, 2, 'Man Who Knew Too Little, The (1997)'],
       [268, 0, 5]], dtype=object)
```