#!/usr/bin/env node
/**
 * Validates that all visualizer packages follow the project's naming and
 * placement conventions. This catches common mistakes such as:
 *
 *   1. An OpenSpiel game visualizer placed outside of open_spiel_env/games/
 *   2. A non-OpenSpiel visualizer placed inside open_spiel_env/games/
 *   3. Package names that don't match the expected convention for their location
 *   4. OpenSpiel game directories that don't match a registered game short_name
 *
 * Run standalone:  node web/scripts/validate-visualizer-conventions.js
 * Runs in CI via:  pnpm validate-conventions  (see root package.json)
 *
 * Exit codes:
 *   0 - All conventions followed
 *   1 - One or more violations found
 */

/* eslint-disable @typescript-eslint/no-require-imports */
const { execSync } = require('child_process');
const path = require('path');

const ROOT_DIR = path.resolve(__dirname, '../..');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getWorkspacePackages() {
  try {
    const output = execSync('pnpm m ls --json --depth -1', {
      encoding: 'utf8',
      cwd: ROOT_DIR,
    });
    return JSON.parse(output);
  } catch {
    console.error('Error: Could not list pnpm workspaces. Make sure pnpm is installed.');
    process.exit(1);
  }
}

/**
 * Determine whether a package path lives under the open_spiel_env/games/
 * subtree.
 */
function isOpenSpielPath(pkgPath) {
  const rel = path.relative(ROOT_DIR, pkgPath);
  return rel.includes(path.join('open_spiel_env', 'games'));
}

/**
 * Determine whether a package path lives under kaggle_environments/envs/ at
 * all (i.e. it is a game visualizer, not core/common/etc.)
 */
function isEnvVisualizerPath(pkgPath) {
  const rel = path.relative(ROOT_DIR, pkgPath);
  return rel.startsWith(path.join('kaggle_environments', 'envs'));
}

// ---------------------------------------------------------------------------
// Validation rules
// ---------------------------------------------------------------------------

// Allowlist of known standalone (non-OpenSpiel) game directories under envs/.
// If you are adding a NEW game visualizer, consider whether it belongs here or
// under envs/open_spiel_env/games/<game>/visualizer/ instead.
// Only add to this list if the game has its own Python environment (not OpenSpiel).
const KNOWN_STANDALONE_GAME_DIRS = new Set([
  'cabt',
  'chess',
  'codenames',
  'connectx',
  'halite',
  'hungry_geese',
  'kore_fleets',
  'llm_20_questions',
  'lux_ai_2021',
  'lux_ai_s2',
  'lux_ai_s3',
  'mab',
  'rps',
  'werewolf',
]);

const errors = [];

function fail(pkg, message) {
  errors.push({ pkg: pkg.name, path: path.relative(ROOT_DIR, pkg.path), message });
}

/**
 * Rule 1: Package name must end with "-visualizer".
 *         (Already enforced by find-games.js, but good to be explicit.)
 */
function ruleVisualizerSuffix(pkg) {
  if (!pkg.name.endsWith('-visualizer')) {
    fail(pkg, `Package name "${pkg.name}" must end with "-visualizer".`);
  }
}

/**
 * Rule 2: Visualizers inside open_spiel_env/games/ MUST have a package name
 *         starting with "@kaggle-environments/open-spiel-".
 */
function ruleOpenSpielPrefix(pkg) {
  if (isOpenSpielPath(pkg.path)) {
    const expectedPrefix = '@kaggle-environments/open-spiel-';
    if (!pkg.name.startsWith(expectedPrefix)) {
      fail(
        pkg,
        `OpenSpiel visualizer package name must start with "${expectedPrefix}", ` +
          `but got "${pkg.name}". ` +
          `Rename to follow the convention (e.g., "${expectedPrefix}<game>-visualizer").`
      );
    }
  }
}

/**
 * Rule 3: Visualizers outside open_spiel_env/games/ must NOT use the
 *         "open-spiel-" prefix — that would be misleading.
 */
function ruleNonOpenSpielNoPrefix(pkg) {
  if (!isOpenSpielPath(pkg.path) && isEnvVisualizerPath(pkg.path)) {
    if (pkg.name.includes('open-spiel-')) {
      fail(
        pkg,
        `Non-OpenSpiel visualizer "${pkg.name}" should not contain "open-spiel-" in its name. ` +
          `If this is an OpenSpiel game, move it to kaggle_environments/envs/open_spiel_env/games/<game>/visualizer/.`
      );
    }
  }
}

/**
 * Rule 4: For OpenSpiel visualizers, the game directory name (the directory
 *         under games/) must match the OpenSpiel game short_name registered
 *         in open_spiel_env.py's GAMES_LIST.
 */
function ruleOpenSpielGameDirMatchesRegistration(pkg, registeredShortNames) {
  if (!isOpenSpielPath(pkg.path)) return;

  // Extract the game directory name from the path:
  // .../open_spiel_env/games/<game_dir>/visualizer/<variant>/
  const rel = path.relative(ROOT_DIR, pkg.path);
  const parts = rel.split(path.sep);
  const gamesIdx = parts.indexOf('games');
  if (gamesIdx === -1 || gamesIdx + 1 >= parts.length) return;
  const gameDir = parts[gamesIdx + 1];

  if (!registeredShortNames.has(gameDir)) {
    fail(
      pkg,
      `Game directory "${gameDir}" does not match any registered OpenSpiel game short_name. ` +
        `Registered games: [${[...registeredShortNames].sort().join(', ')}]. ` +
        `The directory name must match the OpenSpiel game's short_name.`
    );
  }
}

/**
 * Rule 5: Standalone game directories (under envs/<game>/) must be in the
 *         KNOWN_STANDALONE_GAME_DIRS allowlist. This prevents accidentally
 *         placing an OpenSpiel game visualizer outside of open_spiel_env/games/.
 *
 *         If you are adding a genuinely new standalone game, add its directory
 *         name to the allowlist above. If this is an OpenSpiel game, move it
 *         to kaggle_environments/envs/open_spiel_env/games/<game>/visualizer/.
 */
function ruleStandaloneGameDirAllowlist(pkg) {
  if (isOpenSpielPath(pkg.path) || !isEnvVisualizerPath(pkg.path)) return;

  // Extract the game directory: envs/<game_dir>/visualizer/<variant>/
  const rel = path.relative(ROOT_DIR, pkg.path);
  const parts = rel.split(path.sep);
  const envsIdx = parts.indexOf('envs');
  if (envsIdx === -1 || envsIdx + 1 >= parts.length) return;
  const gameDir = parts[envsIdx + 1];

  if (gameDir === 'open_spiel_env') return; // Handled by other rules

  if (!KNOWN_STANDALONE_GAME_DIRS.has(gameDir)) {
    fail(
      pkg,
      `Game directory "${gameDir}" is not in the standalone game allowlist. ` +
        `If this is a new standalone (non-OpenSpiel) game, add "${gameDir}" to ` +
        `KNOWN_STANDALONE_GAME_DIRS in validate-visualizer-conventions.js. ` +
        `If this is an OpenSpiel game, move it to ` +
        `kaggle_environments/envs/open_spiel_env/games/<game>/visualizer/ instead.`
    );
  }
}

/**
 * Read GAMES_LIST from open_spiel_env.py and extract the short_name for each
 * game string. The short_name is the part before the first '(' (or the whole
 * string if there are no parameters).
 */
function getRegisteredOpenSpielShortNames() {
  const fs = require('fs');
  const envPath = path.join(ROOT_DIR, 'kaggle_environments', 'envs', 'open_spiel_env', 'open_spiel_env.py');

  if (!fs.existsSync(envPath)) {
    console.warn(`Warning: Could not find ${envPath} — skipping game registration check.`);
    return null;
  }

  const content = fs.readFileSync(envPath, 'utf8');

  // Find the GAMES_LIST block. It starts with "GAMES_LIST = [" and ends with "]"
  const listMatch = content.match(/GAMES_LIST\s*=\s*\[([\s\S]*?)\]/);
  if (!listMatch) {
    console.warn('Warning: Could not parse GAMES_LIST from open_spiel_env.py — skipping game registration check.');
    return null;
  }

  const shortNames = new Set();
  const listContent = listMatch[1];

  // Match quoted strings and variable references
  const stringEntries = listContent.matchAll(/"([^"]+)"/g);
  for (const match of stringEntries) {
    const gameString = match[1];
    // short_name is everything before the first '('
    const shortName = gameString.split('(')[0];
    shortNames.add(shortName);
  }

  // Also handle variable references like DEFAULT_REPEATED_POKER_GAME_STRING.
  // Extract the short_name from the variable's value in the file.
  const varRefs = listContent.matchAll(/^\s*(DEFAULT_\w+|[A-Z_]+GAME_STRING\w*)\s*,?\s*$/gm);
  for (const varMatch of varRefs) {
    const varName = varMatch[1];
    // Find the variable definition and extract the game name
    const varDefMatch = content.match(new RegExp(`${varName}\\s*=\\s*\\(?\\s*"(\\w+)\\(`));
    if (varDefMatch) {
      shortNames.add(varDefMatch[1]);
    }
  }

  return shortNames;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function main() {
  console.log('--- Validating Visualizer Conventions ---');

  const packages = getWorkspacePackages();
  const visualizers = packages.filter(
    (pkg) => pkg.name && pkg.name.startsWith('@kaggle-environments/') && pkg.name.endsWith('-visualizer')
  );

  if (visualizers.length === 0) {
    console.log('No visualizer packages found. Nothing to validate.');
    process.exit(0);
  }

  console.log(`Found ${visualizers.length} visualizer package(s).`);

  // Load registered OpenSpiel game short names for Rule 4
  const registeredShortNames = getRegisteredOpenSpielShortNames();

  for (const pkg of visualizers) {
    ruleVisualizerSuffix(pkg);
    ruleOpenSpielPrefix(pkg);
    ruleNonOpenSpielNoPrefix(pkg);
    ruleStandaloneGameDirAllowlist(pkg);
    if (registeredShortNames) {
      ruleOpenSpielGameDirMatchesRegistration(pkg, registeredShortNames);
    }
  }

  if (errors.length > 0) {
    console.error(`\n❌ Found ${errors.length} convention violation(s):\n`);
    for (const err of errors) {
      console.error(`  Package: ${err.pkg}`);
      console.error(`  Path:    ${err.path}`);
      console.error(`  Error:   ${err.message}\n`);
    }
    process.exit(1);
  }

  console.log('✅ All visualizer conventions are valid.');
  console.log('   Checked:');
  for (const pkg of visualizers) {
    const location = isOpenSpielPath(pkg.path) ? '(OpenSpiel)' : '(standalone)';
    console.log(`     - ${pkg.name} ${location}`);
  }
}

main();
