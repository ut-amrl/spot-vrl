The general workflow is to train a model on dataset N, evaluate the trained
model on dataset N+1, train on N+1, etc.

## 00initial

Initial supervised training dataset with a simple set of terrains. The terrains
vary in visual and inertial characteristics.

The training and test sets have similar distributions.

Train:
+ concrete
+ grass
+ marble rocks

Test:
+ concrete
+ grass
+ marble rocks

2022-12-13 and 2022-12-14 dataset metadata:
- overcast
- wet/damp ground
- plant life: mix between gray/yellow/green

## 01pebble

Derived from 00initial

Adds pebble pavement as a terrain category. Pebble pavement has similar inertial
characteristics to concrete, but different visual characteristics to all of the
existing terrain categories.

Train:
* concrete
* grass
* marble rocks
+ pebble pavement

Test:
* concrete
* grass
* marble rocks
+ pebble pavement

The inclusion of the 2023-01-25 rosbags in the pebble pavement category add
shadows into the dataset. It may have been appropriate to introduce them in a
separate training generation, but due to color variations in Kinect images of
pebble pavement, it was kept as part of the training data for generation 1.

2023-01-25 dataset metadata:
- pebble on 24th street sidewalk
- bright + shadows

## 02shadows

Note: never used for February 2023 experiments

Derived from 01pebble

The rosbags in 01pebble were mainly taken on overcast days. This dataset
introduces rosbags with high contrast shadows. No additional labels are
introduced in the train set. The test set add explicit labels for shadows for
visualization purposes.

Train:
* concrete
  + shadows
* grass
* marble rocks
  + shadows
* pebble pavement
  + shadows

Test:
* concrete
* grass
* marble rocks
* pebble pavement
+ concrete with shadows             (as explicit terrain category)
+ pebble pavement with shadows      (as explicit terrain category)
+ marble rocks with shadows         (as explicit terrain category)

2023-01-18 dataset metadata:
- clear
- bright lighting conditions
- high contrast shadows

## 03ybrick

Note: never used for February 2023 experiments, see 05rybrick instead.

Derived from 01pebble

Adds yellow brick (brick_yellow) as a terrain category.

Train:
* concrete
* grass
* marble rocks
* pebble pavement
+ yellow brick

Test:
* concrete
* grass
* marble rocks
* pebble pavement
+ yellow brick

## 04rbrick

Note: never used for February 2023 experiments, see 05rybrick instead.

Derived from 01pebble

Adds red brick (brick_red) as a terrain category.

Train:
* concrete
* grass
* marble rocks
* pebble pavement
+ red brick

Test:
* concrete
* grass
* marble rocks
* pebble pavement
+ red brick

# 05rybrick

Derived from 01pebble

Adds red and yellow bricks (brick_{red,yellow}) as a terrain categories at the
same time instead of incrementally. The same rosbags are added as in 03ybrick and
04rbrick.

Train:
* concrete
* grass
* marble rocks
* pebble pavement
+ red brick
+ yellow brick

Test:
* concrete
* grass
* marble rocks
* pebble pavement
+ red brick
+ yellow brick

## 06winter

Derived from 00initial.

TODO: this should really be reordered to index 01 and extended by other datasets.

Adds winter foliage for bush and grass terrains.

Train:
* concrete
* grass
* marble rocks
+ grass_winter      (as separate terrain category)
+ bush_winter       (as separate terrain category)

Test:
* concrete
* grass
* marble rocks
+ grass_winter      (as separate terrain category)
+ bush_winter       (as separate terrain category)
