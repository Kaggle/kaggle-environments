export class Point {
    public readonly x: number; 
    public readonly y: number;

    public constructor(x: number, y: number) {
        this.x = x;
        this.y = y;
    }

    public  translate(offset: Point, size: number): Point {
        return this.add(offset).mod(size);
    }

    public  add(other: Point): Point {
        return new Point(this.x + other.x, this.y + other.y);
    }

    public mod(size: number): Point {
        return new Point(this.x % size, this.y % size);
    }

    /**
     * Gets the manhatten distance between two points
     */
    public distanceTo(other: Point, size: number): number {
        const abs_x = Math.abs(this.x - other.x);
        const dist_x = abs_x < size/2 ? abs_x : size - abs_x;
        const abs_y = Math.abs(this.y - other.y);
        const dist_y = abs_y < size/2 ? abs_y : size - abs_y;
        return dist_x + dist_y;
    }

    /**
     * Converts a 2d position in the form (x, y) to an index in the observation.kore list.
     * See fromIndex for the inverse.
     */
    public toIndex(size: number) {
        return (size - this.y - 1) * size + this.x;
    }

    public static fromIndex(index: number, size: number): Point {
        return new Point(index % size, size - Math.floor(index/size) - 1);
    }

    public abs(): Point {
        return new Point(Math.abs(this.x), Math.abs(this.y));
    }

    public equals(other: Point): boolean {
        return this.x == other.x && this.y == other.y;
    }

    public toString(): string {
        return "(" + this.x + "," + this.y + ")";
    }

    public multiply(factor: number): Point {
        return new Point(factor * this.x, factor * this.y);
    }

    public subtract(other: Point): Point {
        return new Point(this.x - other.x, this.y - other.y);
    }
}
