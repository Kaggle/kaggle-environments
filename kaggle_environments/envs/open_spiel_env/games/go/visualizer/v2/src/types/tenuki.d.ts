/**
 * TypeScript type definitions for tenuki.js
 * A web-based board and JavaScript library for the game of go/baduk/weiqi
 * https://github.com/aprescott/tenuki
 */

declare module 'tenuki' {
  /**
   * Game option configurations
   */
  export interface GameOptions {
    /** DOM element where the board will be rendered */
    element?: HTMLElement;
    /** Board size (default: 19) - supports 1-19 */
    boardSize?: number;
    /** Scoring method: "territory", "area", or "equivalence" (default: "territory") */
    scoring?: 'territory' | 'area' | 'equivalence';
    /** Number of handicap stones (0, 2-9) (default: 0) */
    handicapStones?: number;
    /** Whether handicap stones can be placed freely (default: false) */
    freeHandicapPlacement?: boolean;
    /** Ko rule: "simple" or "superko" (default: "simple") */
    koRule?: 'simple' | 'superko';
    /** Komi (handicap points) (default: 0) */
    komi?: number;
    /** Renderer: "svg" or "dom" (default: "svg") */
    renderer?: 'svg' | 'dom';
    /** Whether to use fuzzy stone placement (default: false) */
    fuzzyStonePlacement?: boolean;
    /** Internal hooks for handling board events */
    _hooks?: GameHooks;
  }

  /**
   * Game event hooks
   */
  export interface GameHooks {
    handleClick?(y: number, x: number): void;
    hoverValue?(y: number, x: number): string | undefined;
    gameIsOver?(): boolean;
  }

  /**
   * Intersection on the board representing a position
   */
  export interface Intersection {
    value: string | null;
    y: number;
    x: number;
    isKoPoint(): boolean;
    isEmpty(): boolean;
    isBlack(): boolean;
    isWhite(): boolean;
    isFilled(): boolean;
  }

  /**
   * Board state representation
   */
  export interface BoardState {
    moveNumber: number;
    pass: boolean;
    koPoint: { y: number; x: number } | null;
    intersections: Intersection[];
    previousMove(): BoardState | null;
    nextColor(): 'black' | 'white';
    xCoordinateFor(x: number): string;
    yCoordinateFor(y: number): string;
    playAt(y: number, x: number, color: string): BoardState;
    playPass(color: string): BoardState;
    intersectionAt(y: number, x: number): Intersection;
    libertiesAt(x: number, y: number): number;
    inAtari(x: number, y: number): true | false;
  }

  /**
   * Score result
   */
  export interface ScoreResult {
    black: number;
    white: number;
    winner?: 'black' | 'white' | 'tie';
  }

  /**
   * Render options
   */
  export interface RenderOptions {
    render?: boolean;
  }

  /**
   * Main Game class for tenuki
   */
  export class Game {
    constructor(options?: GameOptions);

    /** Board size (9, 13, or 19) */
    boardSize: number;
    /** Number of handicap stones */
    handicapStones: number;
    /** Renderer instance */
    renderer: any;
    /** Callbacks for game events */
    callbacks: {
      postRender: () => void;
    };

    /** Get the intersection at coordinates (y, x) */
    intersectionAt(y: number, x: number): Intersection;

    /** Get all intersections on the board */
    intersections(): Intersection[];

    /** Get coordinates string representation (e.g., "A19") */
    coordinatesFor(y: number, x: number): string;

    /** Get the current player ("black" or "white") */
    currentPlayer(): 'black' | 'white';

    /** Check if white is currently playing */
    isWhitePlaying(): boolean;

    /** Check if black is currently playing */
    isBlackPlaying(): boolean;

    /** Get the current board state */
    currentState(): BoardState;

    /** Get the current move number */
    moveNumber(): number;

    /** Play a stone at (y, x) */
    playAt(y: number, x: number, options?: RenderOptions): boolean;

    /** Play a pass move */
    pass(options?: RenderOptions): boolean;

    /** Check if the game is over (two consecutive passes) */
    isOver(): boolean;

    /** Check if a move at (y, x) is illegal */
    isIllegalAt(y: number, x: number): boolean;

    /** Get territory for each player (only after game is over) */
    territory(): { black: []; white: [] };

    /** Mark a stone at (y, x) as dead during scoring */
    markDeadAt(y: number, x: number, options?: RenderOptions): boolean;

    /** Unmark a stone at (y, x) as dead */
    unmarkDeadAt(y: number, x: number, options?: RenderOptions): boolean;

    /** Toggle dead status at (y, x) */
    toggleDeadAt(y: number, x: number, options?: RenderOptions): boolean;

    /** Get all dead stones */
    deadStones(): Array<{ y: number; x: number }>;

    /** Get the current score */
    score(): ScoreResult;

    /** Render the board */
    render(): void;

    /** Undo the last move */
    undo(): boolean;

    /** Get all moves in the game */
    previousMoves(): BoardState[];

    _scorer: Scorer;
  }

  /**
   * Client for handling the board UI
   */
  export class Client {
    constructor(options?: GameOptions);
  }

  /**
   * Utility functions
   */
  export const utils: {
    flatten<T>(ary: T[][]): T[];
    flatMap<T, R>(ary: T[], lambda: (item: T) => R[]): R[];
    cartesianProduct<T, U>(ary1: T[], ary2: U[]): Array<[T, U]>;
    randomID(prefix: string): string;
    clone(element: Element): Element;
    createElement(elementName: string, options?: ElementOptions): HTMLElement;
    createSVGElement(elementName: string, options?: SVGElementOptions): SVGElement;
    appendElement(parent: Element, el: Element): void;
    addEventListener(el: Element, eventName: string, fn: (e: Event) => void): void;
    removeClass(el: Element, className: string): void;
    addClass(el: Element, className: string): void;
    hasClass(el: Element, className: string): boolean;
  };

  /**
   * Element creation options
   */
  export interface ElementOptions {
    class?: string;
  }

  /**
   * SVG element creation options
   */
  export interface SVGElementOptions {
    class?: string;
    attributes?: Record<string, string>;
    text?: string | number;
  }
}
