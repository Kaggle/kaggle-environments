declare module 'tenuki' {
  // Core primitive types used by Tenuki
  export type Color = 'black' | 'white' | 'empty';

  export interface Point {
    y: number;
    x: number;
  }

  // Representation of a single intersection on the board
  export interface Intersection {
    y: number;
    x: number;
    value: Color;

    isOccupiedWith(color: Color): boolean;
    isBlack(): boolean;
    isWhite(): boolean;
    isEmpty(): boolean;
    sameColorAs(other: Intersection): boolean;
  }

  // Internal board state used by Game. We're exporting the interface so
  // callers can inspect or construct states if necessary, although the
  // runtime module does not export this class.
  export interface BoardStateOptions {
    moveNumber: number;
    playedPoint: Point | null;
    color: Color;
    pass: boolean;
    blackPassStones: number;
    whitePassStones: number;
    intersections: Intersection[];
    blackStonesCaptured: number;
    whiteStonesCaptured: number;
    capturedPositions: Point[];
    koPoint: Point | null;
    boardSize: number;
  }

  export interface BoardState {
    moveNumber: number;
    playedPoint: Point | null;
    color: Color;
    pass: boolean;
    blackPassStones: number;
    whitePassStones: number;
    intersections: Intersection[];
    blackStonesCaptured: number;
    whiteStonesCaptured: number;
    capturedPositions: Point[];
    koPoint: Point | null;
    boardSize: number;

    nextColor(): 'black' | 'white';
    yCoordinateFor(y: number): string;
    xCoordinateFor(x: number): string;
    playPass(color: Color): BoardState;
    playAt(y: number, x: number, color: Color): BoardState;
    intersectionAt(y: number, x: number): Intersection;
    groupAt(y: number, x: number): Intersection[];
    libertiesAt(y: number, x: number): number;
    inAtari(y: number, x: number): boolean;
    neighborsFor(y: number, x: number): Intersection[];
    positionSameAs(otherState: BoardState): boolean;
    partitionTraverse(
      startingPoint: Intersection,
      inclusionCondition: (neighbor: Intersection) => boolean
    ): [Intersection[], Intersection[]];
    copyWithAttributes(attrs: Partial<BoardStateOptions>): BoardState;
  }

  // Options that can be supplied when creating a new Game instance
  export interface GameOptions {
    element?: HTMLElement | null;
    boardSize?: number;
    scoring?: string;
    handicapStones?: number;
    koRule?: string;
    komi?: number;
    _hooks?: any;
    fuzzyStonePlacement?: boolean;
    renderer?: string;
    freeHandicapPlacement?: boolean;
  }

  /**
   * Primary class exported by the module.  Most code will interact with
   * Game instances when displaying or manipulating a board.
   */
  export class Game {
    constructor(options?: GameOptions);

    boardSize: number | null;
    handicapStones: number | null;
    callbacks: { postRender: (game: Game) => void };
    renderer: any;
    _deadPoints: Point[];
    _ruleset: any;
    _scorer: any;
    _moves: BoardState[];

    intersectionAt(y: number, x: number): Intersection;
    intersections(): Intersection[];
    deadStones(): Point[];
    coordinatesFor(y: number, x: number): string;
    currentPlayer(): 'black' | 'white';
    isWhitePlaying(): boolean;
    isBlackPlaying(): boolean;
    score(): any;
    currentState(): BoardState;
    moveNumber(): number;
    playAt(y: number, x: number, opts?: { render?: boolean }): boolean;
    pass(opts?: { render?: boolean }): boolean;
    isOver(): boolean;
    markDeadAt(y: number, x: number, opts?: { render?: boolean }): boolean;
    unmarkDeadAt(y: number, x: number, opts?: { render?: boolean }): boolean;
    toggleDeadAt(y: number, x: number, opts?: { render?: boolean }): boolean;
    isIllegalAt(y: number, x: number): boolean;
    territory(): { black: Point[]; white: Point[] };
    undo(): void;
    render(): void;
  }

  // Client class used for networked/controlled games
  export interface ClientHooks {
    handleClick?(y: number, x: number): any;
    hoverValue?(y: number, x: number): any;
    gameIsOver?(): any;
    submitPlay?(y: number, x: number, cb: (res: any) => void): void;
    submitPass?(cb: (res: any) => void): void;
    submitMarkDeadAt?(y: number, x: number, stones: Point[], cb: (res: any) => void): void;
  }

  export interface ClientOptions {
    element?: HTMLElement | null;
    player?: 'black' | 'white';
    gameOptions?: GameOptions;
    hooks?: ClientHooks;
  }

  export class Client {
    constructor(options?: ClientOptions);
    isOver(): boolean;
    currentPlayer(): 'black' | 'white';
    receivePlay(y: number, x: number): void;
    moveNumber(): number;
    receivePass(): void;
    receiveMarkDeadAt(y: number, x: number): void;
    deadStones(): Point[];
    setDeadStones(points: Point[]): void;
    pass(): void;
  }

  // miscellaneous helpers exposed by the library
  export const utils: any;
}
