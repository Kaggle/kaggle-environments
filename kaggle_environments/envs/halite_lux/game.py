import constants
INPUT_CONSTANTS = constants.INPUT_CONSTANTS.RESEARCH_POINTS

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
        this.map = GameMap(map_width, map_height);
        this.players = [new Player(0), new Player(1)];
        await this.retrieveUpdates();
        pass