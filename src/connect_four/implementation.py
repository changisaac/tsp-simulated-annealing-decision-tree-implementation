"""
This is the only file you should change in your submission!
"""
from basicplayer import basic_evaluate, minimax, get_all_next_moves, is_terminal
from util import memoize, run_search_function, INFINITY, NEG_INFINITY
import time
import pdb

# TODO Uncomment and fill in your information here. Think of a creative name that's relatively unique.
# We may compete your agent against your classmates' agents as an experiment (not for marks).
# Are you interested in participating if this competition? Set COMPETE=TRUE if yes.

# STUDENT_ID = 20611091
# AGENT_NAME = Test
# COMPETE = False

# TODO Change this evaluation function so that it tries to win as soon as possible
# or lose as late as possible, when it decides that one side is certain to win.
# You don't have to change how it evaluates non-winning positions.

def focused_evaluate(board):
    """
    Given a board, return a numeric rating of how good
    that board is for the current player.
    A return value >= 1000 means that the current player has won;
    a return value <= -1000 means that the current player has lost
    """
    
    if board.is_game_over():
        score = -1000
    else:
        score = board.longest_chain(board.get_current_player_id()) * 10
        # Prefer having your pieces in the center of the board.
        for row in range(6):
            for col in range(7):
                if board.get_cell(row, col) == board.get_current_player_id():
                    score -= abs(3-col)
                elif board.get_cell(row, col) == board.get_other_player_id():
                    score += abs(3-col)
        # more tokens on board mean further in game
        # further in game is worse score so decrease frrom score
        score -= board.num_tokens_on_board() * 0.5

    return score


# Create a "player" function that uses the focused_evaluate function
# You can test this player by choosing 'quick' in the main program.
#quick_to_win_player = lambda board: minimax(board, depth=4, eval_fn=focused_evaluate)
quick_to_win_player = lambda board: test_alpha_beta_search(board)

# TODO Write an alpha-beta-search procedure that acts like the minimax-search
# procedure, but uses alpha-beta pruning to avoid searching bad ideas
# that can't improve the result. The tester will check your pruning by
# counting the number of static evaluations you make.

def test_alpha_beta_search(board):
    ab_ret = alpha_beta_search(board, depth=4, eval_fn=focused_evaluate)
    minimax_ret = minimax(board, depth=4, eval_fn=focused_evaluate)
    
    if ab_ret != minimax_ret:
        print("ERROR")
        raise

    return ab_ret

# You can use minimax() in basicplayer.py as an example.
# NOTE: You should use get_next_moves_fn when generating
# next board configurations, and is_terminal_fn when
# checking game termination.
# The default functions for get_next_moves_fn and is_terminal_fn set here will work for connect_four.
def alpha_beta_search(board, depth,
                      eval_fn,
                      get_next_moves_fn=get_all_next_moves,
                      is_terminal_fn=is_terminal,
                      verbose=True):
    """
     board is the current tree node.

     depth is the search depth.  If you specify depth as a very large number then your search will end at the leaves of trees.
     
     def eval_fn(board):
       a function that returns a score for a given board from the
       perspective of the state's current player.
    
     def get_next_moves(board):
       a function that takes a current node (board) and generates
       all next (move, newboard) tuples.
    
     def is_terminal_fn(depth, board):
       is a function that checks whether to statically evaluate
       a board/node (hence terminating a search branch).
    """
   
    s_time = time.time()
    
    best_val = None
    
    for move, new_board in get_next_moves_fn(board):
        val = -1 * alpha_beta_find_board_value(new_board, depth-1, eval_fn, NEG_INFINITY, INFINITY, False,
                                            get_next_moves_fn,
                                            is_terminal_fn)
        if best_val is None or val > best_val[0]:
            best_val = (val, move, new_board)
            
    if verbose:
        print("MINIMAX: Decided on column {} with rating {}".format(best_val[1], best_val[0]))
        print("Time alpha beta(s): {}".format(time.time() - s_time))

    return best_val[1]

def alpha_beta_find_board_value(board, depth, eval_fn, alpha, beta, is_max,
                             get_next_moves_fn=get_all_next_moves,
                             is_terminal_fn=is_terminal):
    """
    Minimax helper function: Return the minimax value of a particular board,
    given a particular depth to estimate to
    """

    if is_terminal_fn(depth, board):
        return eval_fn(board)

    best_val = None
    
    # swap states from min to max before recursing on children
    if is_max:
        next_state = False
    else:
        next_state = True
    
    for move, new_board in get_next_moves_fn(board):
        val = -1 * alpha_beta_find_board_value(new_board, depth-1, eval_fn, alpha, beta, next_state,
                                            get_next_moves_fn, is_terminal_fn)
        if best_val is None or val > best_val:
            best_val = val

        # update alpha and beta accordingly based on minmax stat
        if is_max:
            alpha = max(alpha, val)
        else:
            beta = min(beta, -val)

        # check if beta is ever smaller than alpha
        # if so then break from loop and skip rest of children
        if beta <= alpha:
            break

    return best_val

# Now you should be able to search twice as deep in the same amount of time.
# (Of course, this alpha-beta-player won't work until you've defined alpha_beta_search.)
def alpha_beta_player(board):
    return alpha_beta_search(board, depth=5, eval_fn=focused_evaluate)

# This player uses progressive deepening, so it can kick your ass while
# making efficient use of time:
def ab_iterative_player(board):
    return run_search_function(board, search_fn=alpha_beta_search, eval_fn=focused_evaluate, timeout=5)


# TODO Finally, come up with a better evaluation function than focused-evaluate.
# By providing a different function, you should be able to beat
# simple-evaluate (or focused-evaluate) while searching to the same depth.

def better_evaluate(board):
    if board.is_game_over():
        return -1000

    score = 0

    curr_chains = board.chain_cells(board.get_current_player_id())
    other_chains = board.chain_cells(board.get_other_player_id())

    other_3_chains = 0 
    other_2_chains = 0

    for chain in other_chains:
        if len(chain) == 3:
            other_3_chains += 1
        if len(chain) == 2:
            other_2_chains += 1
    
    curr_3_chains = 0
    curr_2_chains = 0

    for chain in curr_chains:
        if len(chain) == 3:
            curr_3_chains += 1
        if len(chain) == 2:
            curr_2_chains += 1

    score = board.longest_chain(board.get_current_player_id()) * 10
    
    score += curr_2_chains * 1
    score += curr_3_chains * 2
    score -= other_2_chains * 2
    score -= other_3_chains * 2
          
    score -= board.num_tokens_on_board() * 0.5

    return score
           
# Comment this line after you've fully implemented better_evaluate
#better_evaluate = memoize(basic_evaluate)

# Uncomment this line to make your better_evaluate run faster.
better_evaluate = memoize(better_evaluate)


# A player that uses alpha-beta and better_evaluate:
def my_player(board):
    return run_search_function(board, search_fn=alpha_beta_search, eval_fn=better_evaluate, timeout=5)

# my_player = lambda board: alpha_beta_search(board, depth=4, eval_fn=better_evaluate)
