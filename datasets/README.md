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

## 02shadows

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
