Dataset to evaluate 02pebble models on similar terrains with unseen lighting
conditions.

The training datasets were mainly taken on overcast days. The evaluation
datasets were taken on sunny days, which introduces high contrast shadows into
the visual data.

Train (identical to 02pebble):
- concrete
- grass
- marble rocks
- pebble pavement

Test:
- concrete
- grass
- marble rocks
- pebble pavement
* concrete with shadows             (as explicit terrain category)
* pebble pavement with shadows      (as explicit terrain category)
* marble rocks with shadows         (as explicit terrain category)


2022-12-13 and 2022-12-14 dataset metadata:
- overcast
- wet/damp ground
- plant life: mix between gray/yellow/green

2023-01-18 dataset metadata:
- clear
- bright lighting conditions
- high contrast shadows
