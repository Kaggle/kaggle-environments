export type Vec2 = readonly [number, number];

export function distance(p1: Vec2, p2: Vec2): number {
  const dx = p1[0] - p2[0];
  const dy = p1[1] - p2[1];
  return Math.sqrt(dx * dx + dy * dy);
}

/** Minimum distance from point p to line segment v-w. */
export function pointToSegmentDistance(p: Vec2, v: Vec2, w: Vec2): number {
  const l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2;
  if (l2 === 0) return distance(p, v);
  const t = Math.max(0, Math.min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2));
  const projection: Vec2 = [v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1])];
  return distance(p, projection);
}
