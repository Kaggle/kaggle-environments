package kore;


public class Direction extends Point {

    public static final Direction NORTH = new Direction(0, 1);
    public static final Direction EAST = new Direction(1, 0);
    public static final Direction SOUTH = new Direction(0, -1);
    public static final Direction WEST = new Direction(-1, 0);
    
    private Direction(int x, int y) {
        super(x, y);
    }

    public boolean equals(Direction other) {
        return this.x == other.x && this.y == other.y;
    }

    public Direction rotateLeft() {
        if (this.equals(Direction.NORTH)) {
            return Direction.WEST;
        }
        if (this.equals(Direction.WEST)) {
            return Direction.SOUTH;
        }
        if (this.equals(Direction.SOUTH)) {
            return Direction.EAST;
        }
        if (this.equals(Direction.EAST)) {
            return Direction.NORTH;
        }
        throw new IllegalStateException("invalid direction");
    }

    public Direction rotateRight() {
        if (this.equals(Direction.NORTH)) {
            return Direction.EAST;
        }
        if (this.equals(Direction.EAST)) {
            return Direction.SOUTH;
        }
        if (this.equals(Direction.SOUTH)) {
            return Direction.WEST;
        }
        if (this.equals(Direction.WEST)) {
            return Direction.NORTH;
        }
        throw new IllegalStateException("invalid direction");
    }

    public Direction opposite() {
        if (this.equals(Direction.NORTH)) {
            return Direction.SOUTH;
        }
        if (this.equals(Direction.EAST)) {
            return Direction.WEST;
        }
        if (this.equals(Direction.SOUTH)) {
            return Direction.NORTH;
        }
        if (this.equals(Direction.WEST)) {
            return Direction.EAST;
        }
        throw new IllegalStateException("invalid direction");
    }

    public String toChar() {
        if (this.equals(Direction.NORTH)) {
            return "N";
        }
        if (this.equals(Direction.EAST)) {
            return "E";
        }
        if (this.equals(Direction.SOUTH)) {
            return "S";
        }
        if (this.equals(Direction.WEST)) {
            return "W";
        }
        throw new IllegalStateException("invalid direction");
    }

    public String toString() {
        if (this.equals(Direction.NORTH)) {
            return "NORTH";
        }
        if (this.equals(Direction.EAST)) {
            return "EAST";
        }
        if (this.equals(Direction.SOUTH)) {
            return "SOUTH";
        }
        if (this.equals(Direction.WEST)) {
            return "WEST";
        }
        throw new IllegalStateException("invalid direction");
    }

    public int toIndex() {
        if (this.equals(Direction.NORTH)) {
            return 0;
        }
        if (this.equals(Direction.EAST)) {
            return 1;
        }
        if (this.equals(Direction.SOUTH)) {
            return 2;
        }
        if (this.equals(Direction.WEST)) {
            return 3;
        }
        throw new IllegalStateException("invalid direction");
    }

    public static Direction fromString(String dirStr) {
        switch(dirStr) {
            case "NORTH":
                return Direction.NORTH;
            case "EAST":
                return Direction.EAST;
            case "SOUTH":
                return Direction.SOUTH;
            case "WEST":
                return Direction.WEST;
        }
        throw new IllegalStateException("invalid direction");
    }

    public static Direction fromChar(char dirChar) {
        switch(dirChar) {
            case 'N':
                return Direction.NORTH;
            case 'E':
                return Direction.EAST;
            case 'S':
                return Direction.SOUTH;
            case 'W':
                return Direction.WEST;
        }
        throw new IllegalStateException("invalid direction");
    }

    public static Direction fromIndex(int index) {
        switch(index) {
            case 0:
                return Direction.NORTH;
            case 1:
                return Direction.EAST;
            case 2:
                return Direction.SOUTH;
            case 3:
                return Direction.WEST;
        }
        throw new IllegalStateException("invalid direction");
    }

    public static Direction[] listDirections() {
        return new Direction[]{
            Direction.NORTH,
            Direction.EAST,
            Direction.SOUTH,
            Direction.WEST
        };
    }

}
