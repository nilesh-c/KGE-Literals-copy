# Preprocessed MovieLens-100k

Preprocessed MovieLens-100k dataset.

## File Descriptions

### Index Files
All `idx2{something}` files are lookup tables which are useful to decode the preprocessed dataset. For example usage:

``` python
idx2occupation = np.load('bin/idx2occupation.npy')

# Suppose from the dataset we have these indexes
occs = [1, 5, 3, 1]

# Lookup
occs_idxs = [idx2occupation[o] for o in occs]

print(occs_idxs)
# ['artist', 'entertainment', 'educator', 'artist']
```

### Dataset Files
The preprocessed dataset is divided into three: train, val, and test using 80-10-10 shuffle-split.

1. `rating_{train,val,test}` contains array triplets of `user_id rating movie_id`.
2. `user_{train,val,test}` contains array triplets of `user_id user_rel literal_val`. Please refer to `idx2userrel.npy` for the breakdown of `user_rel`
3. `movie_{train,val,test}` contains array triplets of `movie_id movie_rel literal_val`. Please refer to `idx2movierel.npy` for the breakdown of `movie_rel`
