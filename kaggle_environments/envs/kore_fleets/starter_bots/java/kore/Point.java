package kore;

public class Point {
    public final int x, y;

    public Point(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public Point translate(Point offset, int size) {
        return this.add(offset).mod(size);
    }

    public Point add(Point other) {
        return new Point(this.x + other.x, this.y + other.y);
    }

    public Point mod(int size) {
        return new Point(this.x % size, this.y % size);
    }

    /**
     * Gets the manhatten distance between two points
     */
    public int distanceTo(Point other, int size) {
        int abs_x = Math.abs(this.x - other.x);
        int dist_x = abs_x < size/2 ? abs_x : size - abs_x;
        int abs_y = Math.abs(this.y - other.y);
        int dist_y = abs_y < size/2 ? abs_y : size - abs_y;
        return dist_x + dist_y;
    }

    /**
     * Converts a 2d position in the form (x, y) to an index in the observation.kore list.
     * See fromIndex for the inverse.
     */
    public int toIndex(int size) {
        return (size - this.y - 1) * size + this.x;
    }

    public static Point fromIndex(int index, int size) {
        return new Point(index % size, size - index/size - 1);
    }

    public Point abs() {
        return new Point(Math.abs(this.x), Math.abs(this.y));
    }

    public boolean equals(Point other) {
        return this.x == other.x && this.y == other.y;
    }

    public String toString() {
        return "(" + this.x + "," + this.y + ")";
    }

    public Point multiply(int factor) {
        return new Point(factor * this.x, factor * this.y);
    }

    public Point subtract(Point other) {
        return new Point(this.x - other.x, this.y - other.y);
    }
}
