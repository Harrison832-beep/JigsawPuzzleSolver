# JigsawPuzzleSolver

## How to Run
```python jigsaw_solver.py```

Default puzzle is jigsaw1.png, need to modify in code (line 580 and 581) to solve different puzzles.

## 1. Preprocess Puzzle Pieces
* Identify edges and simplified to points
* Store edge info with puzzle piece as an object
![alt text](<demo-img/Preprocess1.png>)
![alt text](<demo-img/Preprocess2.png>)

## 2. Solve with Backtracking Algorthm
* Solve with backtracking algorithm, each each object will have a list of neighbors referring to corresponding puzzle piece objects
* Assemble and show result
 ![alt text](<demo-img/Solve1.png>)
 ![alt text](<demo-img/Solve2.png>)