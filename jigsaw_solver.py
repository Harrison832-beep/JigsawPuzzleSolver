import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys
import copy


FLAT = 4
INWARD = 5
OUTWARD = 6

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

sides_str = ["UP", "DOWN", "LEFT", "RIGHT"]
ans_img = None

class Piece:
    def __init__(self, img):
        self.img = img
        self.shapes = [None, None, None, None]  # up, down, left, right
        self.neighbors = [None, None, None, None]  # up, down, left, right
        self.shape_names = ["FLAT", "INWARD", "OUTWARD"]
    
    def get_fitted_sides(self, piece2):
        """
        Get fitted sides with another puzzle piece
        """
        sides = []
        
        for side in range(4):
            if self.neighbors[side] is None and piece2.neighbors[get_opposite(side)] is None:
                if self.shapes[side] == INWARD and piece2.shapes[get_opposite(side)] == OUTWARD or \
                    self.shapes[side] == OUTWARD and piece2.shapes[get_opposite(side)] == INWARD:
                        if side in [UP, DOWN]:
                            if self.valid_edges(piece2, LEFT) and self.valid_edges(piece2, RIGHT):
                                sides.append(side)
                        elif side in [LEFT, RIGHT]:
                            if self.valid_edges(piece2, UP) and self.valid_edges(piece2, DOWN):
                                sides.append(side)
        return sides
    
    def valid_edges(self, piece2, side):
        if side == LEFT:
            both_flat = self.shapes[LEFT] == FLAT and piece2.shapes[LEFT] == FLAT
            both_other = self.shapes[LEFT] != FLAT and piece2.shapes[LEFT] != FLAT
            return both_flat or both_other
        elif side == RIGHT:
            both_flat = self.shapes[RIGHT] == FLAT and piece2.shapes[RIGHT] == FLAT
            both_other = self.shapes[RIGHT] != FLAT and piece2.shapes[RIGHT] != FLAT
            return both_flat or both_other
        elif side == UP:
            both_flat = self.shapes[UP] == FLAT and piece2.shapes[UP] == FLAT
            both_other = self.shapes[UP] != FLAT and piece2.shapes[UP] != FLAT
            return both_flat or both_other
        elif side == DOWN:
            both_flat = self.shapes[DOWN] == FLAT and piece2.shapes[DOWN] == FLAT
            both_other = self.shapes[DOWN] != FLAT and piece2.shapes[DOWN] != FLAT
            return both_flat or both_other
        raise Exception("invalid side")
    
    def can_fit_with(self, piece2, side):
        if (self.shapes[side] == OUTWARD and piece2.shapes[get_opposite(side)] == INWARD)\
            or (self.shapes[side] == INWARD and piece2.shapes[get_opposite(side)] == OUTWARD):
                if side == LEFT or side == RIGHT:
                    if self.valid_edges(piece2, UP) and self.valid_edges(piece2, DOWN):
                        return True
                elif side == UP or side == DOWN:
                    if self.valid_edges(piece2, LEFT) and self.valid_edges(piece2, RIGHT):
                        return True
                else:
                    raise Exception("invalid side")
        return False
            
    def assemble_with(self, piece2, side):
        # Assemble first, might dissemble later
        self.neighbors[side] = piece2
        piece2.neighbors[get_opposite(side)] = self
        success = True
        
        # TODO: when assemble, also check if additional neighbors will be added
        # Check rest of the piece2 rest of the sides if additional neighbors exist
        if side == UP:  # If assemble to side "up" of piece1
            n_up = piece2.get_potential_neighbor([DOWN, LEFT, UP, UP, RIGHT])  # up
            if n_up is None:  # Try the other direction
                n_up = piece2.get_potential_neighbor([DOWN, RIGHT, UP, UP, LEFT])
            
            n_left = piece2.get_potential_neighbor([DOWN, LEFT, UP])  # left
            if n_left is None:
                n_left = piece2.get_potential_neighbor([DOWN, RIGHT, UP, UP, LEFT, LEFT, DOWN])
            n_right = piece2.get_potential_neighbor([DOWN, RIGHT, UP])  # right
            if n_right is None:
                n_right = piece2.get_potential_neighbor([DOWN, LEFT, UP, UP, RIGHT, RIGHT, DOWN])
            
            # =================================================================
            # Assemble with potential neighbors
            # =================================================================
            if n_up is not None:
                if piece2.can_fit_with(n_up, UP):
                    piece2.neighbors[UP] = n_up
                    n_up.neighbors[DOWN] = piece2
                else:
                    success = False
            if n_left is not None:
                if piece2.can_fit_with(n_left, LEFT):
                    piece2.neighbors[LEFT] = n_left
                    n_left.neighbors[RIGHT] = piece2
                else:
                    success = False
            if n_right is not None:
                if piece2.can_fit_with(n_right, RIGHT):
                    piece2.neighbors[RIGHT] = n_right
                    n_right.neighbors[LEFT] = piece2
                else:
                    success = False
        elif side == DOWN:
            n_down = piece2.get_potential_neighbor([UP, LEFT, DOWN, DOWN, RIGHT])  # down
            if n_down is None:
                n_down = piece2.get_potential_neighbor([UP, RIGHT, DOWN, DOWN, LEFT])
            n_left = piece2.get_potential_neighbor([UP, LEFT, DOWN])  # left
            if n_left is None:
                n_left = piece2.get_potential_neighbor([UP, RIGHT, DOWN, DOWN, LEFT, LEFT, UP])
            n_right = piece2.get_potential_neighbor([UP, RIGHT, DOWN])  # right
            if n_right is None:
                n_right = piece2.get_potential_neighbor([UP, LEFT, DOWN, DOWN, RIGHT, RIGHT, UP])
            # =================================================================
            # Assemble with potential neighbors
            # =================================================================
            if n_down is not None:
                if piece2.can_fit_with(n_down, DOWN):
                    piece2.neighbors[DOWN] = n_down
                    n_down.neighbors[UP] = piece2
                else:
                    success = False
            if n_left is not None:
                if piece2.can_fit_with(n_left, LEFT):
                    piece2.neighbors[LEFT] = n_left
                    n_left.neighbors[RIGHT] = piece2
                else:
                    success = False
            if n_right is not None:
                if piece2.can_fit_with(n_right, RIGHT):
                    piece2.neighbors[RIGHT] = n_right
                    n_right.neighbors[LEFT] = piece2
                else:
                    success = False
        elif side == LEFT:
            n_up = piece2.get_potential_neighbor([RIGHT, UP, LEFT])  # up
            if n_up is None:
                n_up = piece2.get_potential_neighbor([RIGHT, DOWN, LEFT, LEFT, UP, UP, RIGHT])
            n_down = piece2.get_potential_neighbor([RIGHT, DOWN, LEFT])  # down
            if n_down is None:
                n_down = piece2.get_potential_neighbor([RIGHT, UP, LEFT, LEFT, DOWN, DOWN, RIGHT])
            n_left = piece2.get_potential_neighbor([RIGHT, UP, LEFT, LEFT, DOWN])  # left
            if n_left is None:
                n_left = piece2.get_potential_neighbor([RIGHT, DOWN, LEFT, LEFT, UP])
            
            # =================================================================
            # Assemble with potential neighbors
            # =================================================================
            if n_up is not None:
                if piece2.can_fit_with(n_up, UP):
                    piece2.neighbors[UP] = n_up
                    n_up.neighbors[DOWN] = piece2
                else:
                    success = False
            if n_down is not None:
                if piece2.can_fit_with(n_down, DOWN):
                    piece2.neighbors[DOWN] = n_down
                    n_down.neighbors[UP] = piece2
                else:
                    success = False
            if n_left is not None:
                if piece2.can_fit_with(n_left, LEFT):
                    piece2.neighbors[LEFT] = n_left
                    n_left.neighbors[RIGHT] = piece2
                else:
                    success = False
        elif side == RIGHT:
            n_up = piece2.get_potential_neighbor([LEFT, UP, RIGHT])  # up
            if n_up is None:
                n_up = piece2.get_potential_neighbor([LEFT, DOWN, RIGHT, RIGHT, UP, UP, LEFT])
            n_down = piece2.get_potential_neighbor([LEFT, DOWN, RIGHT])  # down
            if n_down is None:
                n_down = piece2.get_potential_neighbor([LEFT, UP, RIGHT, RIGHT, DOWN, DOWN, LEFT])
            n_right = piece2.get_potential_neighbor([LEFT, UP, RIGHT, RIGHT, DOWN])  # right
            if n_right is None:
                n_right = piece2.get_potential_neighbor([LEFT, DOWN, RIGHT, RIGHT, UP])
            
            # =================================================================
            # Assemble with potential neighbors
            # =================================================================
            if n_up is not None:
                if piece2.can_fit_with(n_up, UP):
                    piece2.neighbors[UP] = n_up
                    n_up.neighbors[DOWN] = piece2
                else:
                    success = False
            if n_down is not None:
                if piece2.can_fit_with(n_down, DOWN):
                    piece2.neighbors[DOWN] = n_down
                    n_down.neighbors[UP] = piece2
                else:
                    success = False
            if n_right is not None:
                if piece2.can_fit_with(n_right, RIGHT):
                    piece2.neighbors[RIGHT] = n_right
                    n_right.neighbors[LEFT] = piece2
                else:
                    success = False
        else:
            raise Exception("Unrecognized side")
            
        if not success:
            # Dissemble
            # self.dissemble_with(piece2, side)
            piece2.dissemble()
        return success
            
    def dissemble_with(self, piece2, side):
        self.neighbors[side] = None
        piece2.neighbors[get_opposite(side)] = None
        
    def dissemble(self):
        """
        Dissemble from assembled pieces
        """
        for side, n in enumerate(self.neighbors):
            if n is not None:
                self.dissemble_with(n, side)
            
    def get_potential_neighbor(self, direction):
        """
        Get potential unassembled neighbor
        """
        try:
            piece = self
            for side in direction:
                piece = piece.neighbors[side]
            return piece
        except AttributeError:
            # print("No potential neighbor")
            return None
    
    def get_flat_side_count(self):
        return np.where(np.array(self.shapes)==FLAT)[0].shape[0]
        
    def is_full(self):
        """
        Check if all sides of this piece are assembled
        Flat side won't have neighbor, so subtract it.
        """
        return np.where(np.array(self.neighbors) != None)[0].shape[0] \
                + self.get_flat_side_count() == 4
    
    def sort_matching_sides(self, piece2, sides):
        kp1, kp2, gm1 = get_good_matches(self.img, ans_img)
        pos1 = get_position(kp1, kp2, gm1)
        kp1, kp2, gm2 = get_good_matches(piece2.img, ans_img)
        pos2 = get_position(kp1, kp2, gm2)
        
        
        if pos1 is None or pos2 is None:  # Not enough good matches to get relative position
            return sides
        
        pos_diff = pos1 - pos2
        # self.show()
        # piece2.show()
        
        # If pos_diff[1] < 0, above, DOWN first
        if pos_diff[1] < 0:  # Above piece2
            if DOWN in sides:
                sides.remove(DOWN)
                sides.insert(0, DOWN)
            # if UP in sides:
                # sides.remove(UP)
        elif pos_diff[1] > 0:  # Under piece2
            if UP in sides:
                sides.remove(UP)
                sides.insert(0, UP)
            # if DOWN in sides:
                # sides.remove(DOWN)
        # If pos_diff[0] < 0, on the left, RIGHT first
        elif pos_diff[0] < 0:  # Left of piece2
            if RIGHT in sides:
                sides.remove(RIGHT)
                sides.insert(0, RIGHT)
            # if LEFT in sides:
                # sides.remove(LEFT)
        elif pos_diff[0] > 0:
            if LEFT in sides:  # Right of piece2
                sides.remove(LEFT)
                sides.insert(0, LEFT)
            # if RIGHT in sides:
                # sides.remove(RIGHT)
        
        return sides
    
    def show(self, title=None):
        plt.imshow(cv.cvtColor(self.img, cv.COLOR_BGR2RGB), cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.title(title)
        plt.show()
    
    def __str__(self):
        return f'''Left: {self.shape_names[self.left]}
Right: {self.shape_names[self.right]}
Up: {self.shape_names[self.up]}
Down: {self.shape_names[self.down]}
    '''


def show(img, title=None):
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title(title)
    plt.show()


def process_puzzle_pieces(filename):
    jigsaw = cv.imread(filename)
    jigsaw_gray = cv.cvtColor(jigsaw, cv.COLOR_RGB2GRAY)
    
    thresh_val = 230
    maxval = 255
    ret, thresh = cv.threshold(jigsaw_gray, thresh_val, maxval, cv.THRESH_BINARY_INV)
    show(thresh)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    piececontour = jigsaw.copy()
    cv.drawContours(piececontour, contours, -1, (255,0,255), 3)
    show(piececontour)
    
    piecerect = jigsaw.copy()
    
    
    pieces = []
    areas = np.array([cv.contourArea(c) for c in contours])
    i = 0
    for contour in contours:
        a = cv.contourArea(contour)
        
        # if 500 < a < np.max(areas):
        if 10000 < a:
            mask = np.zeros_like(jigsaw_gray)
            cv.drawContours(mask, [contour], 0, 255, thickness=cv.FILLED)
            
            piece_img = cv.bitwise_and(jigsaw, jigsaw, mask=mask)

            (x,y,w,h) = cv.boundingRect(contour)
            cv.rectangle(piecerect, (x,y), (x+w, y+h), (255,0,255), 3)
            piece_img = piece_img[y:y+h,x:x+w]
            if np.all(piece_img==0):
                continue
            piece = Piece(piece_img)
            pieces.append(piece)
            i += 1
    assert len(pieces) == NUM_PIECE
    show(piecerect)
            
    # print(len(pieces))
    
    for i, piece in enumerate(pieces):
        piecegray = cv.cvtColor(piece.img, cv.COLOR_BGR2GRAY)
        
        thresh_val = 50
        maxval = 255
        ret, thresh = cv.threshold(piecegray, thresh_val, maxval, cv.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Find outer-most contour
        
        split_contours = []
        for contour in contours:
            epsilon = 0.01 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            split_contours.extend(approx)
        
        image_with_split_contours = piece.img.copy()
        split_contours_by_x = np.unique(split_contours, axis=0)

        image_with_split_contours = piece.img.copy()
        cv.drawContours(image_with_split_contours, split_contours_by_x, -1, (0, 255, 0), 10)
        show(image_with_split_contours)
        
        piececontour = piece.img.copy()
        cv.drawContours(piececontour, contours, -1, (255,0,255), 3)
        # show(piececontour)
    
        split_contours_by_y = split_contours_by_x[split_contours_by_x[:, 0, 1].argsort()]
        # Left
        if np.diff(split_contours_by_x[:4, 0, 0]).mean() < 5:
            piece.shapes[LEFT] = INWARD
        elif np.diff(split_contours_by_x[:2, 0, 0]).mean() < 5:
            piece.shapes[LEFT] = FLAT
        else:
            piece.shapes[LEFT] = OUTWARD
        
        # Right
        if np.diff(split_contours_by_x[-4:, 0, 0]).mean() < 5:
            piece.shapes[RIGHT] = INWARD
        elif np.diff(split_contours_by_x[-2:, 0, 0]).mean() < 5:
            piece.shapes[RIGHT] = FLAT
        else:
            piece.shapes[RIGHT] = OUTWARD
        
        # Up
        if np.diff(split_contours_by_y[:4, 0, 1]).mean() < 5:
            piece.shapes[UP] = INWARD
        elif np.diff(split_contours_by_y[:2, 0, 1]).mean() < 5:
            piece.shapes[UP] = FLAT
        else:
            piece.shapes[UP] = OUTWARD
        
        # Down
        # print(np.diff(split_contours_by_y[-4:, 0, 1]))
        if np.diff(split_contours_by_y[-4:, 0, 1]).mean() < 5:
            piece.shapes[DOWN] = INWARD
        elif np.diff(split_contours_by_y[-2:, 0, 1]).mean() < 5:
            piece.shapes[DOWN] = FLAT
        else:
            piece.shapes[DOWN] = OUTWARD
    return pieces


def get_good_matches(img1, img2):
    """
    img1: one puzzle piece
    img2: answer puzzle image
    """
    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray,None)
    kp2, des2 = sift.detectAndCompute(img2_gray,None)
    
    count2 = 0
    # """
    # find the keypoints and descriptors with SIFT
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(np.float32(des1),np.float32(des2),k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    good_matches = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            count2 += 1
            good_matches.append(m)
    # print("Good matches: ", count2)
    
    draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask, 
                        flags = cv.DrawMatchesFlags_DEFAULT)
    matches_img = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    # show(matches_img)
    return kp1, kp2, good_matches


def get_position(kp1, kp2, good_matches):
    list_kp1 = []
    list_kp2 = []
    if len(good_matches) <=0:
        return None
    for mat in good_matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Append to each list
        list_kp1.append((x1, y1))
        list_kp2.append((x2, y2))

    # print(np.mean(list_kp1, axis=0))
    # print(np.mean(list_kp2, axis=0))
    if len(list_kp2) > 0:
        ans_img_pos = np.mean(list_kp2, axis=0)
        return ans_img_pos.astype(np.int8)
    else:
        return None


# %%
# =============================================================================
# Solve jigsaw puzzle with backtracking algorithm
# =============================================================================
# unassembled = copy.deepcopy(pieces)


def get_opposite(side):
    if side == UP:
        return DOWN
    elif side == DOWN:
        return UP
    elif side == LEFT:
        return RIGHT
    elif side == RIGHT:
        return LEFT


def criteria(assembled, unassembled):
    assert len(assembled+unassembled) == NUM_PIECE
    for p in assembled+unassembled:
        if not p.is_full():
            return False
    return True


def show_pair(piece1, piece2, sides):
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(piece1.img, cv.COLOR_BGR2RGB), cmap='gray', vmin=0, vmax=255)
    plt.xticks([]),plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(piece2.img, cv.COLOR_BGR2RGB), cmap='gray', vmin=0, vmax=255)
    plt.xticks([]),plt.yticks([])
    
    if len(sides) > 0:
        plt.suptitle(f"Match {np.array(sides_str)[np.array(sides)]}")
    else:
        plt.suptitle(f"No Match")
        raise Exception("No side but passed if statement")
    plt.show()
    

def solve_jigsaw(assembled, unassembled):
    if criteria(assembled, unassembled):
        return True, assembled, unassembled
    
    assembled2 = assembled.copy()
    unassembled2 = unassembled.copy()
    
    for piece1 in assembled:
        for piece2 in unassembled:
            if not piece1.is_full() and not piece2.is_full():
                sides = piece1.get_fitted_sides(piece2)
                if len(sides) > 1:
                    # sides = piece1.sort_matching_sides(piece2, sides)
                    pass
                
                if len(sides) > 0:
                    # show_pair(piece1, piece2, sides)
                    # For all sides
                    for side in sides:
                        # Assemble and recursively solve jigsaw puzzle
                        if piece1.assemble_with(piece2, side):  # If success
                            assembled2.append(piece2)
                            unassembled2.remove(piece2)
                            
                            success, assembled3, unassembled3 = solve_jigsaw(assembled2.copy(), unassembled2.copy())
                            if success:
                                return True, assembled3, unassembled3
                            else: # If not successful, backtrack, dissemble and try another piece
                                # print("One solution failed...")
                                # piece1.dissemble_with(piece2, side)
                                piece2.dissemble()
                                assembled2.remove(piece2)
                                unassembled2.append(piece2)

            # raise NotImplementedError
    return False, assembled2, unassembled2

NUM_PIECE = 12
ans_img = cv.imread("jigsaw1-answer.png")
pieces = process_puzzle_pieces("jigsaw1.png")
assert len(pieces) == NUM_PIECE

assembled = [pieces[0]]
unassembled = [pieces[i] for i in range(1, len(pieces))]

import time
start_time = time.time()
success, assembled_puzzle, unassembled_puzzle = solve_jigsaw(assembled, unassembled)
print("Time spent:", time.time()-start_time)
if success:
    print("Puzzle is solved!")
else:
    print("Puzzle cannot be solved!")


# %%
# =============================================================================
# Show sovled puzzle
# =============================================================================

puzzles = list(assembled_puzzle)

def get_solution_board(puzzles):
    nrow= 1
    ncolumn = 1
    p1 = puzzles[0]
    while p1.neighbors[LEFT] is not None:
        p1 = p1.neighbors[LEFT]
    while p1.neighbors[UP] is not None:
        p1 = p1.neighbors[UP]
    
    while p1.neighbors[RIGHT] is not None:
        ncolumn += 1
        p1 = p1.neighbors[RIGHT]
    while p1.neighbors[DOWN] is not None:
        nrow += 1
        p1 = p1.neighbors[DOWN]
    
    sln_pieces = [[0]*ncolumn for _ in range(nrow)]
    
    # Reset p1 position
    while p1.neighbors[LEFT] is not None:
        p1 = p1.neighbors[LEFT]
    while p1.neighbors[UP] is not None:
        p1 = p1.neighbors[UP]
    
    row = 0
    col = 0
    while p1 is not None:
        sln_pieces[row][col] = p1
        
        col += 1
        p2 = p1.neighbors[RIGHT]
        while p2 is not None:
            sln_pieces[row][col] = p2
            p2 = p2.neighbors[RIGHT]
            col += 1
        p1 = p1.neighbors[DOWN]
        row += 1
        col = 0
    return sln_pieces
    

def show_solution(puzzles):
    p1 = puzzles[0]
    
    while p1.neighbors[LEFT] is not None:
        p1 = p1.neighbors[LEFT]
    
    while p1.neighbors[UP] is not None:
        p1 = p1.neighbors[UP]
    
    i = 1
    
    while p1 is not None:
        plt.subplot(3, 4, i)
        plt.imshow(cv.cvtColor(p1.img, cv.COLOR_BGR2RGB), cmap='gray', vmin=0, vmax=255)
        plt.xticks([]),plt.yticks([])
        
        i += 1
        p2 = p1.neighbors[RIGHT]
        while p2 is not None:
            plt.subplot(3, 4, i)
            plt.imshow(cv.cvtColor(p2.img, cv.COLOR_BGR2RGB), cmap='gray', vmin=0, vmax=255)
            plt.xticks([]),plt.yticks([])
            p2 = p2.neighbors[RIGHT]
            i += 1
            
        p1 = p1.neighbors[DOWN]
    plt.show()
# plt.subplot(3, 4)
show_solution(list(puzzles))
sln_pieces = get_solution_board(puzzles)

# %%
# =============================================================================
# Try to stitch solution
# =============================================================================
width = 0
height = 0

for row in range(len(sln_pieces)):
    maxh = 0
    row_width = 0
    for col in range(len(sln_pieces[0])):
        h,w,_ = sln_pieces[row][col].img.shape
        if h > maxh:
            maxh = h
        row_width += w
    if row_width > width:
        width = row_width
    height += maxh


stitched = np.zeros((height, width, 3), dtype=np.uint8)

width_start = 0
width_end = 0
height_start = 0
height_end = 0
for row in range(len(sln_pieces)):
    maxh = 0
    width_start = 0
    for col in range(len(sln_pieces[0])):
        h, w,_ = sln_pieces[row][col].img.shape
        if w > maxh:
            maxh = h
        stitched[height_start:height_start+h, width_start:width_start+w] = sln_pieces[row][col].img
        width_start += w
    height_start += maxh
    height_end += maxh

show(stitched)

# %%
PRE_DOWN = 7
CUR_DOWN = 8

if NUM_PIECE == 6:
    GAP = 76
if NUM_PIECE == 12:
    GAP = 56
# %%
def exist_outward(pieces, side):
    for p in pieces:
        if p.shapes[side] == OUTWARD:
            return True
    return False

# %%
def stitch_img_h(stitched, p, shift_command, is_expand=False):
    img = p.img
    # show(stitched)
    # show(img2)
    
    h1,w1,_ = stitched.shape
    h2,w2,_ = img.shape
    
    offset = 0
    if is_expand:
        offset = GAP
    new_frame1 = np.zeros((max(h1,h2) + offset,w1+w2,3), np.uint8)
    new_frame2 = np.zeros((max(h1,h2) + offset,w1+w2,3), np.uint8)
    new_frame1[:h1, :w1] = stitched
    new_frame2[:h2, w1:w1+w2] = img
    show(new_frame1)
    show(new_frame2)

    
    # =============================================================================
    # Translate
    # =============================================================================
    
    if shift_command == PRE_DOWN:
        M = np.float32([
                [1,0,0],
                [0,1,56]
            ])
        new_frame1 = cv.warpAffine(new_frame1, M, (new_frame1.shape[1], new_frame1.shape[0]))
    
    if shift_command == CUR_DOWN:
        M = np.float32([
            [1,0,-GAP],
            [0,1,GAP]
        ])
    else:
        M = np.float32([
                [1,0,-GAP],
                [0,1,0]
            ])

    shifted = cv.warpAffine(new_frame2, M, (new_frame2.shape[1], new_frame2.shape[0]))
    show(shifted)

    dest = cv.bitwise_or(new_frame1, shifted, mask=None)
    show(dest)
    
    new_height, new_width,_ = dest.shape

    for col in range(dest.shape[1]):
        if np.all(dest[:, col]==0):
            new_width = col - 1
            break
    for row in range(dest.shape[0]):
        if np.all(dest[row, :]==0):
            new_height = row - 1
            break

    stitched_frame = np.zeros((new_height, new_width, 3), np.uint8)
    stitched_frame[:] = dest[:new_height, :new_width]
    show(stitched_frame)
    return stitched_frame

# %%
def stitch_img_v(stitched, img, has_shift_down):
    # show(stitched)
    # show(img2)
    
    h1,w1,_ = stitched.shape
    h2,w2,_ = img.shape
    new_frame1 = np.zeros((h1+h2,max(w1,w2),3), np.uint8)
    new_frame2 = np.zeros((h1+h2,max(w1,w2),3), np.uint8)
    new_frame1[:h1, :w1] = stitched
    new_frame2[h1:h1+h2, :w2] = img
    show(new_frame1)
    show(new_frame2)

    
    # =============================================================================
    # Translate
    # =============================================================================
    if has_shift_down:
        M = np.float32([
                [1,0,0],
                [0,1,-GAP*2]
            ])
    else:
        M = np.float32([
                [1,0,0],
                [0,1,-GAP]
            ])
    

    shifted = cv.warpAffine(new_frame2, M, (new_frame2.shape[1], new_frame2.shape[0]))
    show(shifted)

    dest = cv.bitwise_or(new_frame1, shifted, mask=None)
    show(dest)
    
    new_height, new_width,_ = dest.shape
    
    for col in range(dest.shape[1]):
        if np.all(dest[:, col]==0):
            new_width = col - 1
            break
    for row in range(dest.shape[0]):
        if np.all(dest[row, :]==0):
            new_height = row - 1
            break

    stitched_frame = np.zeros((new_height, new_width, 3), np.uint8)
    stitched_frame[:] = dest[:new_height, :new_width]
    show(stitched_frame)
    return stitched_frame
# %%
# =============================================================================
# Do it for first row
# =============================================================================
first_row = sln_pieces[0]

stitched_l = []
is_shift_down_l = []
for row_imgs in sln_pieces:
    stitched = row_imgs[0].img
    is_shift_down = False
    for i in range(1, len(row_imgs)):
        p = row_imgs[i]
        
        if p.shapes[UP] == OUTWARD and p.shapes[DOWN] == OUTWARD:
            if exist_outward([row_imgs[i] for i in range(i)], UP) and \
                exist_outward([row_imgs[i] for i in range(i)], DOWN):
                    stitched = stitch_img_h(stitched, row_imgs[i], False, False)
            elif exist_outward([row_imgs[i] for i in range(i)], UP):
                    stitched = stitch_img_h(stitched, row_imgs[i], False, False)
            elif exist_outward([row_imgs[i] for i in range(i)], DOWN):
                stitched = stitch_img_h(stitched, row_imgs[i], PRE_DOWN, False)
                is_shift_down = True
            else:
                stitched = stitch_img_h(stitched, row_imgs[i], PRE_DOWN, False)
                is_shift_down = True
        elif p.shapes[UP] == OUTWARD:
            if exist_outward([row_imgs[i] for i in range(i)], UP) and \
                exist_outward([row_imgs[i] for i in range(i)], DOWN):
                    stitched = stitch_img_h(stitched, row_imgs[i], False, False)
            elif exist_outward([row_imgs[i] for i in range(i)], UP):
                stitched = stitch_img_h(stitched, row_imgs[i], False, True)
                is_shift_down = True
            else:
                stitched = stitch_img_h(stitched, row_imgs[i], PRE_DOWN, False)
                is_shift_down = True
        elif p.shapes[DOWN] == OUTWARD:
            if exist_outward([row_imgs[i] for i in range(i)], UP) and \
                exist_outward([row_imgs[i] for i in range(i)], DOWN):
                    stitched = stitch_img_h(stitched, row_imgs[i], CUR_DOWN, False)
                    is_shift_down = True
            elif exist_outward([row_imgs[i] for i in range(i)], UP):
                stitched = stitch_img_h(stitched, row_imgs[i], CUR_DOWN, True)
                is_shift_down = True
            else:
                stitched = stitch_img_h(stitched, row_imgs[i], False, False)
        else:
            if exist_outward([row_imgs[i] for i in range(i)], UP) and \
                exist_outward([row_imgs[i] for i in range(i)], DOWN):
                    stitched = stitch_img_h(stitched, row_imgs[i], CUR_DOWN, False)
                    is_shift_down = True
            elif exist_outward([row_imgs[i] for i in range(i)], UP):
                stitched = stitch_img_h(stitched, row_imgs[i], CUR_DOWN, False)
                is_shift_down = True
            else:
                stitched = stitch_img_h(stitched, row_imgs[i], False, False)
    stitched_l.append(stitched)
    is_shift_down_l.append(is_shift_down)
    show(stitched)

# %%
stitched = stitched_l[0]
for i in range(1, len(stitched_l)):
    stitched = stitch_img_v(stitched, stitched_l[i], is_shift_down_l[i])
    show(stitched)
    

