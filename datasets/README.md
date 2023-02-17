Loose convention:

- Even numbers: training datasets
- Odd numbers: evaluation of trained models on OOD data

Exception:

= 99-*: fully supervised or other special cases

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

Parent: 00initial

Dataset to evaluate initial models on unseen pebble pavement terrain.

Pebble pavement has similar inertial characteristics to concrete, but different
visual characteristics to all of the training datasets.

Train (identical to 00initial):
* concrete
* grass
* marble rocks

Test:
* concrete
* grass
* marble rocks
+ pebble pavement

## 02pebble

Parent: 00initial

Dataset with pebble pavement in the training set.

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

## 03shadows

Parent: 02pebble

The training datasets were mainly taken on overcast days. The evaluation
datasets were taken on sunny days, which introduces high contrast shadows into
the visual data.

Train (identical to 02pebble):
* concrete
* grass
* marble rocks
* pebble pavement

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

## 04shadows

Parent: 02pebble

Dataset with additional lighting conditions. No additional terrains are
introduced.

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
