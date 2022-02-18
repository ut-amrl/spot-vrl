# `naive_learning`

This package contains models and scripts forked from the `preference_learning`
repository. As the name implies, code was adapted naively to create an initial
proof-of-concept for learning representations of images taken by Spot.

Learning takes place in a supervised setting, where image patches taken from
trajectories are labeled as either concrete or grass. These labels are used to
generate training triplets.

## Usage Pipeline

`naive_patch.py` extracts image patches from concrete/grass trajectories.

`main_representation.py` trains the embedding space using extracted patches.

`collect_embeddings.py` generates clusters of the embedding space for the
proof-of-concept.

## Known Issues

- Lacking module/function documentation
- Lacking data organization documentation
- Failing linter checks
