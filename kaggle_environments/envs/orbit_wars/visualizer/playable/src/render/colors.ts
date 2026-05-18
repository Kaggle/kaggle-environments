// Wong palette — colorblind-safe (blue, vermillion, teal, yellow)
export const PLAYER_COLORS = ['#0072B2', '#D55E00', '#009E73', '#F0E442'];
export const NEUTRAL_COLOR = '#666666';

export function ownerColor(owner: number): string {
  if (owner < 0 || owner >= PLAYER_COLORS.length) return NEUTRAL_COLOR;
  return PLAYER_COLORS[owner];
}
