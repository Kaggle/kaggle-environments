import {Point} from "./Point";

export class Direction extends Point {

    public static readonly NORTH = new Direction(0, 1);
    public static readonly EAST = new Direction(1, 0);
    public static readonly SOUTH = new Direction(0, -1);
    public static readonly WEST = new Direction(-1, 0);
    
    private constructor(x: number, y: number) {
        super(x, y);
    }

    public equals(other: Direction): boolean {
        return this.x == other.x && this.y == other.y;
    }

    public rotateLeft(): Direction {
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
        throw new Error("invalid direction");
    }

    public rotateRight(): Direction {
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
        throw new Error("invalid direction");
    }

    public opposite(): Direction {
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
        throw new Error("invalid direction");
    }

    public toChar(): string {
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
        throw new Error("invalid direction");
    }

    public toString(): string {
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
        throw new Error("invalid direction");
    }

    public toIndex(): number {
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
        throw new Error("invalid direction");
    }

    public static fromString(dirStr: string): Direction {
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
        throw new Error("invalid direction");
    }

    public static fromChar(dirChar: string): Direction {
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
        throw new Error("invalid direction");
    }

    public static fromIndex(index: number): Direction {
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
        throw new Error("invalid direction");
    }

    public static listDirections(): Direction[] {
        return [
            Direction.NORTH,
            Direction.EAST,
            Direction.SOUTH,
            Direction.WEST
        ];
    }

}
