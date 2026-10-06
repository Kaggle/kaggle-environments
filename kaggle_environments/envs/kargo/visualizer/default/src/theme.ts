/** Every colour the map uses, and the check that let it in.
 *
 * Validated with the dataviz validator (OKLab/OKLCH, Machado CVD sim), not by
 * eye. The numbers quoted below are that script's output; re-run it before
 * changing any value here.
 */

/** Congestion is a magnitude, so it is ONE hue light->dark, never a rainbow.
 *
 * Measured OKLCH L: .991 .906 .810 .693 .548 .397 -- strictly monotonic, span
 * .594. That monotonicity is the whole check for a sequential ramp; the
 * categorical rules (chroma floor, adjacent CVD ΔE) do not apply, because
 * neighbouring steps are MEANT to be near each other. Ordering survives
 * greyscale and every CVD type because it is carried by lightness.
 */
export const CONGESTION: Record<string, string> = {
  FREE: '#fdfcf8',
  LIGHT: '#f6dcb4',
  MODERATE: '#eeb478',
  HEAVY: '#dd8244',
  SEVERE: '#b8491f',
  GRIDLOCK: '#7a2711',
};

export const CONGESTION_ORDER = ['FREE', 'LIGHT', 'MODERATE', 'HEAVY', 'SEVERE', 'GRIDLOCK'];

/** A shut road is a STATE, not a level of busy, so it is off the ramp entirely.
 *
 * Putting "closed" at the dark end would read as "very congested" and imply an
 * ordering it does not have -- a closed local street is not worse traffic than
 * a gridlocked arterial, it is a different thing. It gets ink-grey, a dashed
 * stroke and an icon: three encodings, none of them the ramp's hue.
 */
export const CLOSED_STROKE = '#55504a';

/** District grounds. Deliberately desaturated: they are the paper, not the ink.
 *
 * Worst road-vs-ground contrast is 1.04 (INDUSTRIAL under MODERATE), which is
 * far under 3:1 -- so the roads do NOT rely on the ground for separation. Each
 * road carries a dark casing stroke (8.7:1 against the lightest ground), and
 * that casing is what makes the network legible. The ramp only has to separate
 * roads from each other.
 */
export const DISTRICT_GROUND: Record<string, string> = {
  DOWNTOWN: '#ccc6ba',
  RIVERSIDE: '#bfd2c6',
  MIDTOWN: '#d4cbb5',
  SUBURBS_N: '#cbd8bb',
  SUBURBS_S: '#d6dcc6',
  INDUSTRIAL: '#c6c3b9',
};

export const ROAD_CASING = '#3a342c';

/** Player identity. Fixed order, assigned by player id and never cycled.
 *
 * All four validated together under `--pairs all`, not just adjacent pairs:
 * every seat can meet every other seat on one intersection, so a palette that
 * only separates neighbours in the list is no use here. Worst pair is
 * #8057d8 vs #0292a6 at ΔE 12.7 (deutan) / 12.2 (tritan), worst normal-vision
 * pair #a83a52 vs #c4741f at 17.1, chroma >= .1 and contrast >= 3:1 on all four.
 *
 * The sprites' own teal (#176c71) is greyer than the UI ink by design: same hue
 * family, so the pairing reads, while the ink stays above the chroma floor.
 */
export const PLAYER_INK = ['#0292a6', '#c4741f', '#8057d8', '#a83a52'];

/** Outcome status colours, shared with status-dots.css. Reserved: never a series. */
export const OUTCOME_INK: Record<string, string> = {
  delivered: '#2e7d4f',
  late: '#c8860d',
  refused: '#b23a30',
  failed: '#4a4d50',
  abandoned: '#6d6a78',
  pending: '#a9a49c',
};

export const SURFACE = '#efece4';
export const INK = '#231f1b';
export const INK_MUTED = '#6a635a';
