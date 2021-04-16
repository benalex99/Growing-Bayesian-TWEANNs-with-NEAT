import chess
import chess.svg
class myChess():

    def __init__(self):
        print("yo")
        board = chess.Board()
        chess.svg.board(size= 350)