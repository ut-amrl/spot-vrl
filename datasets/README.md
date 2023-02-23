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

## 05rybrick

The dependency tree gets messy here:

- Originally derived from 01pebble
- Partially derived from 02shadow (concrete_shadow) and 06winter (grass_winter)
  to add robustness to afternoon shadows and winter grass in the GDC area.

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
+ concrete_shadow   (as separate terrain category)
+ grass_winter      (as separate terrain category)

Test:
* concrete
* grass
* marble rocks
* pebble pavement
+ red brick
+ yellow brick
+ concrete_shadow   (as separate terrain category)
+ grass_winter      (as separate terrain category)

The inclusion of the 2022-12-16 and 2023-01-18 rosbags in the brick_yellow
category add shadows into the dataset. It may have been appropriate to
introduce them in a separate training generation, but they were kept as part of
the training data for generation 1.

## 06winter

The dependency tree is messy here

- Originally derived from 00initial.
- Partially derived from 02shadows (concrete_shadow). This was used to train a
  gen2 model, but it was not given a new dataset index and label. The gen1 model
  was trained without the concrete_shadow rosbag in the data.

Note: this should really be reordered to index 01 and extended by other
datasets, including the current 01pebble dataset.

Adds winter foliage for bush and grass terrains.

Train:
* concrete
* grass
* marble rocks
+ grass_winter      (as separate terrain category)
+ bush_winter       (as separate terrain category)
+ concrete_shadow   (gen2; as separate terrain category)

Test:
* concrete
* grass
* marble rocks
+ grass_winter      (as separate terrain category)
+ bush_winter       (as separate terrain category)
+ concrete_shadow   (gen2; as separate terrain category)

## 20initnight

Derived from 00initial

Adds night versions of concrete, grass, and bush.

Train:
* concrete
* grass
* marble rocks
+ concrete_night      (as separate terrain category)
+ grass_night         (as separate terrain category)
+ bush_night          (as separate terrain category)

Test:
* concrete
* grass
* marble rocks
+ concrete_night      (as separate terrain category)
+ grass_night         (as separate terrain category)
+ bush_night          (as separate terrain category)

## 21pebbnight

Derived from 01pebble and 20initnight

Adds night versions of pebble pavement and marble rock.

grass_night was accidentally omitted for the February 2023 experiments. The
agent was not able to assign higher costs to grass at night, which caused it to
occasionally overshoot the goal.

Train:
* concrete
* grass
* marble rocks
* concrete_night      (as separate terrain category)
- grass_night
- bush_night
+ marble_rocks_night  (as separate terrain category)
+ pebble_pvmt_night   (as separate terrain category)

Test:
* concrete
* grass
* marble rocks
* concrete_night      (as separate terrain category)
- grass_night
- bush_night
+ marble_rocks_night  (as separate terrain category)
+ pebble_pvmt_night   (as separate terrain category)

## 98night

Meta dataset derived from the union of some versions (possibly outdated) of the
gen1 01pebble, 06winter, and 05rybrick datasets. This dataset contains the
terrains contained in the parent datasets in lighting conditions after sunset.

This dataset should be split into appropriate gen 2 or gen 3 datasets for
specific deployment situations.

Train:
* concrete
* grass
* marble rocks
* pebble pavement
* red brick
* yellow brick
* grass_winter        (as separate terrain category)
* bush_winter         (as separate terrain category)

+ concrete_night      (as separate terrain category)
+ grass_night         (as separate terrain category)
+ pebble_pvmt_night   (as separate terrain category)
+ marble_rocks_night  (as separate terrain category)
+ red brick night     (as separate terrain category)
+ yellow brick night  (as separate terrain category)
+ bush_night          (as separate terrain category)

Test:
* concrete
* grass
* marble rocks
* pebble pavement
* red brick
* yellow brick
* grass_winter        (as separate terrain category)
* bush_winter         (as separate terrain category)

+ concrete_night      (as separate terrain category)
+ grass_night         (as separate terrain category)
+ pebble_pvmt_night   (as separate terrain category)
+ marble_rocks_night  (as separate terrain category)
+ red brick night     (as separate terrain category)
+ yellow brick night  (as separate terrain category)
+ bush_night          (as separate terrain category)
