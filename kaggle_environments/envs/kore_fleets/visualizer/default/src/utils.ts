const upgrade_times = [...Array(10).keys()].map((num) => num * num + 1).slice(1);
const SPAWN_VALUES = upgrade_times.slice(1).reduce((arr, num) => arr.concat(arr.at(-1) ?? 0 + num), [upgrade_times[0]]);

export const getSpawnValue = (turnsControlled: number) => {
  for (let i = 0; i < SPAWN_VALUES.length; i++) {
    if (turnsControlled < SPAWN_VALUES[i]) {
      return i + 1;
    }
  }
  return SPAWN_VALUES.length + 1;
};

export const data = function (selector: string | any, key: string, value?: any) {
  const el = typeof selector === 'string' ? document.querySelector(selector) : selector;
  if (arguments.length === 3) {
    el.setAttribute(`data-${key}`, JSON.stringify(value));
    return value;
  }
  if (el.hasAttribute(`data-${key}`)) {
    return JSON.parse(el.getAttribute(`data-${key}`));
  }
  return null;
};
