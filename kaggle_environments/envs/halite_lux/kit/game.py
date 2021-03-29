import constants
import game_map
INPUT_CONSTANTS = constants.Constants.INPUT_CONSTANTS

class Game():
    def __init__(self):
        pass
    def initialize(self, messages):
        self.id = int(messages[0])
        self.turn = 0;
        # get some other necessary initial input
        mapInfo = messages[1].split(" ")
        self.map_width = int(mapInfo[0])
        self.map_height = int(mapInfo[1])
        self.map = game_map.GameMap(map_width, map_height)
        self.players = [Player(0), Player(1)]
        # await this.retrieveUpdates();
        pass